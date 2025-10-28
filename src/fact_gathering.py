import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from tqdm import tqdm

FACT_SOURCES = {
    'wikipedia': {
        'enabled': True,
        'priority': 1,
        'weight': 10
    },
    'wikivoyage': {
        'enabled': True,
        'priority': 2,
        'weight': 8
    },
    'openstreetmap': {
        'enabled': True,
        'priority': 3,
        'weight': 7
    },
    'geonames': {
        'enabled': True,
        'priority': 4,
        'weight': 6
    }
}

async def fetch_wikipedia_facts(session: aiohttp.ClientSession, url: str, user_agent: str) -> Optional[Dict]:
    """Fetch comprehensive facts from Wikipedia article."""
    if not url:
        return None
    
    try:
        headers = {"User-Agent": user_agent}
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                return None
            
            for element in content_div.find_all(['script', 'style', 'sup', 'div.reflist', 'div.navbox']):
                element.decompose()
            
            # Extract main paragraphs
            paragraphs = content_div.find_all('p')
            text_content = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            # Extract infobox data if available
            infobox_data = {}
            infobox = content_div.find('table', class_='infobox')
            if infobox:
                rows = infobox.find_all('tr')
                for row in rows:
                    header = row.find('th')
                    data = row.find('td')
                    if header and data:
                        key = header.get_text().strip()
                        value = data.get_text().strip()
                        infobox_data[key] = value
            
            if not text_content or len(text_content) < 50:
                return None
            
            # Truncate if too long
            max_chars = 12000
            if len(text_content) > max_chars:
                text_content = text_content[:max_chars] + "..."
            
            return {
                'source': 'wikipedia',
                'content': text_content,
                'infobox': infobox_data,
                'url': url,
                'weight': FACT_SOURCES['wikipedia']['weight']
            }
    
    except Exception as e:
        # Log the error for debugging
        import sys
        print(f"      ⚠️  Wikipedia fetch failed for {url}: {type(e).__name__}: {str(e)}", file=sys.stderr)
        return None

async def fetch_wikivoyage_facts(session: aiohttp.ClientSession, url: str, user_agent: str) -> Optional[Dict]:
    """Fetch travel-focused facts from Wikivoyage."""
    if not url:
        return None
    
    try:
        headers = {"User-Agent": user_agent}
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                return None
            
            # Wikivoyage has specific sections that are valuable
            sections_of_interest = ['See', 'Do', 'Understand', 'Get in', 'Background', 'History']
            facts = []
            
            for heading in content_div.find_all(['h2', 'h3', 'h4']):
                heading_text = heading.get_text().strip()
                if any(section.lower() in heading_text.lower() for section in sections_of_interest):
                    content = []
                    for sibling in heading.find_next_siblings():
                        if sibling.name in ['h2', 'h3', 'h4']:
                            break
                        if sibling.name == 'p':
                            text = sibling.get_text().strip()
                            if text and len(text) > 20:
                                content.append(text)
                    
                    if content:
                        facts.append(f"## {heading_text}\n" + '\n'.join(content))
            
            if not facts:
                # Try getting all paragraphs if no facts were retrieved from the previous step
                paragraphs = content_div.find_all('p')
                text_content = '\n\n'.join([p.get_text().strip() for p in paragraphs[:10] if p.get_text().strip()])
                if text_content and len(text_content) > 100:
                    facts.append(text_content)
            
            if not facts:
                return None
            
            combined = '\n\n'.join(facts)
            max_chars = 4000
            if len(combined) > max_chars:
                combined = combined[:max_chars] + "..."
            
            return {
                'source': 'wikivoyage',
                'content': combined,
                'url': url,
                'weight': FACT_SOURCES['wikivoyage']['weight']
            }
    
    except Exception as e:
        return None

async def fetch_osm_facts(session: aiohttp.ClientSession, name: str, lat: float, lon: float, osm_node_id: str = None) -> Optional[Dict]:
    """Fetch facts from OpenStreetMap tags and metadata."""
    try:
        tags = {}
        
        # Direct node lookup
        if osm_node_id:
            osm_url = f"https://www.openstreetmap.org/api/0.6/node/{osm_node_id}.json"
            try:
                async with session.get(osm_url) as response:
                    if response.ok:
                        data = await response.json()
                        if data.get('elements'):
                            tags = data['elements'][0].get('tags', {})
            except:
                pass
        
        # Overpass API search if OSM id wasn't present in the WikiData
        if not tags:
            overpass_query = f"""
            [out:json][timeout:15];
            (
              node["name"~"{name}",i](around:1000,{lat},{lon});
              way["name"~"{name}",i](around:1000,{lat},{lon});
            );
            out body 1;
            """
            
            async with session.get(
                "http://overpass-api.de/api/interpreter",
                params={'data': overpass_query}
            ) as response:
                if response.ok:
                    data = await response.json()
                    elements = data.get('elements', [])
                    if elements:
                        tags = elements[0].get('tags', {})
        
        if not tags:
            return None
        
        facts = []
        
        for desc_key in ['description', 'description:en', 'note']:
            if desc_key in tags:
                facts.append(f"Description: {tags[desc_key]}")
                break
        
        # The following code just retrieves some common values from OSM

        if 'ele' in tags:
            facts.append(f"Elevation: {tags['ele']}m")
        elif 'height' in tags:
            facts.append(f"Height: {tags['height']}m")
        
        if 'start_date' in tags:
            facts.append(f"Built/Established: {tags['start_date']}")
        elif 'year' in tags:
            facts.append(f"Year: {tags['year']}")
        
        if 'heritage' in tags:
            facts.append(f"Heritage Status: {tags['heritage']}")
        if 'historic' in tags:
            facts.append(f"Historic Type: {tags['historic']}")
        
        if 'religion' in tags:
            facts.append(f"Religion: {tags['religion']}")
        if 'denomination' in tags:
            facts.append(f"Denomination: {tags['denomination']}")
        
        if 'natural' in tags:
            facts.append(f"Natural Feature: {tags['natural']}")
        
        if 'website' in tags:
            facts.append(f"Official Website: {tags['website']}")
        
        # We probably already have this but why not
        if 'wikipedia' in tags:
            facts.append(f"Wikipedia Reference: {tags['wikipedia']}")
        
        if not facts:
            return None
        
        return {
            'source': 'openstreetmap',
            'content': '\n'.join(facts),
            'tags': tags,
            'weight': FACT_SOURCES['openstreetmap']['weight']
        }
    
    except Exception as e:
        return None

async def fetch_geonames_facts(session: aiohttp.ClientSession, name: str, lat: float, lon: float, username: str) -> Optional[Dict]:
    """Fetch facts from GeoNames database."""
    if not username:
        return None
    
    try:
        search_url = "http://api.geonames.org/searchJSON"
        params = {
            'q': name,
            'lat': lat,
            'lng': lon,
            'radius': 10,
            'maxRows': 1,
            'username': username,
            'lang': 'en',
            'style': 'FULL'
        }
        
        async with session.get(search_url, params=params) as response:
            if not response.ok:
                return None
            
            data = await response.json()
            geonames = data.get('geonames', [])
            
            if not geonames:
                return None
            
            info = geonames[0]
            facts = []
            
            if info.get('fcodeName'):
                facts.append(f"Geographic Classification: {info['fcodeName']}")
            
            if info.get('population') and info['population'] > 0:
                facts.append(f"Population: {info['population']:,}")
            
            if info.get('elevation'):
                facts.append(f"Elevation: {info['elevation']}m above sea level")
            
            if info.get('adminName1'):
                facts.append(f"Administrative Region: {info['adminName1']}")
            if info.get('adminName2'):
                facts.append(f"District: {info['adminName2']}")
            
            if info.get('timezone'):
                facts.append(f"Time Zone: {info['timezone']['timeZoneId']}")
            
            geonameId = info.get('geonameId')
            if geonameId:
                wiki_url = "http://api.geonames.org/wikipediaSearchJSON"
                wiki_params = {
                    'geonameId': geonameId,
                    'maxRows': 1,
                    'username': username,
                    'lang': 'en'
                }
                
                try:
                    async with session.get(wiki_url, params=wiki_params) as wiki_response:
                        if wiki_response.ok:
                            wiki_data = await wiki_response.json()
                            articles = wiki_data.get('geonames', [])
                            if articles and articles[0].get('summary'):
                                summary = articles[0]['summary']
                                # Geonames articles usually aren't very verbose
                                if len(summary) > 1000:
                                    summary = summary[:1000] + "..."
                                facts.append(f"\nSummary: {summary}")
                except:
                    pass
            
            if not facts:
                return None
            
            return {
                'source': 'geonames',
                'content': '\n'.join(facts),
                'geonames_data': info,
                'weight': FACT_SOURCES['geonames']['weight']
            }
    
    except Exception as e:
        return None

async def gather_facts_for_landmark(
    session: aiohttp.ClientSession,
    landmark: Dict,
    user_agent: str,
    geonames_username: Optional[str] = None
) -> Dict:
    """
    Gather facts from all available sources for a landmark.
    
    Returns a dict with:
    - facts: List of fact dictionaries from different sources
    - source_count: Number of sources that provided facts
    - sources: List of source names
    """
    
    tasks = []
    
    if landmark.get('wikipedia_url') and FACT_SOURCES['wikipedia']['enabled']:
        tasks.append(('wikipedia', fetch_wikipedia_facts(session, landmark['wikipedia_url'], user_agent)))
    
    if landmark.get('wikivoyage_url') and FACT_SOURCES['wikivoyage']['enabled']:
        tasks.append(('wikivoyage', fetch_wikivoyage_facts(session, landmark['wikivoyage_url'], user_agent)))
    
    if FACT_SOURCES['openstreetmap']['enabled']:
        tasks.append(('osm', fetch_osm_facts(
            session,
            landmark['name'],
            landmark['latitude'],
            landmark['longitude'],
            landmark.get('osm_node_id')
        )))
    
    if geonames_username and FACT_SOURCES['geonames']['enabled']:
        tasks.append(('geonames', fetch_geonames_facts(
            session,
            landmark['name'],
            landmark['latitude'],
            landmark['longitude'],
            geonames_username
        )))
    
    results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
    
    facts = []
    for (source_name, _), result in zip(tasks, results):
        if result and isinstance(result, dict):
            facts.append(result)
    
    facts.sort(key=lambda x: x['weight'], reverse=True)
    
    return {
        'facts': facts,
        'source_count': len(facts),
        'sources': [f['source'] for f in facts]
    }

async def gather_facts_for_city(
    session: aiohttp.ClientSession,
    city: Dict,
    user_agent: str,
    geonames_username: Optional[str] = None
) -> Dict:
    """
    Gather facts from all available sources for a city.
    
    Returns a dict with:
    - facts: List of fact dictionaries from different sources
    - source_count: Number of sources that provided facts
    - sources: List of source names
    """
    
    tasks = []
    
    if city.get('wikipedia_url') and FACT_SOURCES['wikipedia']['enabled']:
        tasks.append(('wikipedia', fetch_wikipedia_facts(session, city['wikipedia_url'], user_agent)))
    
    if city.get('wikivoyage_url') and FACT_SOURCES['wikivoyage']['enabled']:
        tasks.append(('wikivoyage', fetch_wikivoyage_facts(session, city['wikivoyage_url'], user_agent)))
    
    if FACT_SOURCES['openstreetmap']['enabled']:
        tasks.append(('osm', fetch_osm_facts(
            session,
            city['name'],
            city['latitude'],
            city['longitude'],
            city.get('osm_node_id')
        )))
    
    if geonames_username and FACT_SOURCES['geonames']['enabled']:
        tasks.append(('geonames', fetch_geonames_facts(
            session,
            city['name'],
            city['latitude'],
            city['longitude'],
            geonames_username
        )))
    
    results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
    
    facts = []
    for (source_name, _), result in zip(tasks, results):
        if result and isinstance(result, dict):
            facts.append(result)
    
    facts.sort(key=lambda x: x['weight'], reverse=True)
    
    return {
        'facts': facts,
        'source_count': len(facts),
        'sources': [f['source'] for f in facts]
    }

def format_facts_for_llm(gathered_facts: Dict, item: Dict, is_city: bool = False) -> str:
    """
    Format gathered facts into a structured prompt for LLM summarization.
    Uses the format: Web sources (### SOURCE_NAME \n INFO) and Custom facts (NULL/N/A).
    """
    
    name = item['name']
    location_type = "City" if is_city else item.get('specific_subtype', item.get('type', 'Unknown').title())
    prefecture = item.get('prefecture', 'Unknown Prefecture')
    
    facts = gathered_facts.get('facts', [])
    
    # Build web sources section
    web_sources = []
    
    if facts:
        for fact in facts:
            source_name = fact['source'].upper()
            content = fact['content']
            
            # Add infobox data for Wikipedia if available
            if fact['source'] == 'wikipedia' and fact.get('infobox'):
                infobox_text = "\n".join([f"{k}: {v}" for k, v in list(fact['infobox'].items())[:5]])
                if infobox_text:
                    content = f"[Key Facts]\n{infobox_text}\n\n{content}"
            
            # Truncate content if too long
            if len(content) > 2500:
                content = content[:2500] + "..."
            
            web_sources.append(f"### {source_name}\n{content}")
    
    # If no facts available, add a note
    if not web_sources:
        web_sources.append("### LIMITED INFORMATION\nNo detailed information available from external sources. Please create a brief summary based on the location name and type.")
    
    web_sources_text = "\n\n".join(web_sources)
    
    # Custom facts is always NULL/N/A as it's user-provided later
    custom_facts_text = "N/A"
    
    # Build the complete prompt using the new template format
    prompt = f"""# Instructions
You are tasked with creating a comprehensive and fun-to-read summary of the the following location: {name}. Any references to the location should use this name, and not ones stated in the **sources** section.
Below you have been given facts from a variety of sources which you can use to enrich your summary.

# Location Details
- Name: {name}
- Type: {location_type}
- Prefecture: {prefecture}, Japan

# Sources
Web sources is data parsed from websites. Custom sources is information provided by a user. You should assume everything provided below is a fact.

## Web sources
{web_sources_text}

## Custom facts
{custom_facts_text}

# Rules
Your summary should adhere to the following rules:
1. Highlight the most interesting and unique aspects
2. Include fun and unique facts about the location
3. Explain why it's significant and/or worth visiting
4. Avoid human unreadable data such as coordinates in your reply
5. Draw out any references to anime/manga (such as pilgrimage spots, in-real-life inspiration, etc.) if applicable.
6. Keep a more casual style of writing. This is not supposed to be an academic paper.
7. Keep historical elements brief. Only highlight major events. Your summary should always be <30% history.
8. Stay under 1024 characters"""
    
    return prompt

async def enrich_landmarks_with_facts(
    landmarks: List[Dict],
    user_agent: str,
    geonames_username: Optional[str] = None,
    max_concurrent: int = 10,
    batch_size: int = 20
) -> List[Dict]:
    """
    Enrich all landmarks with facts from multiple sources.
    """
    
    print(f"\n📚 Gathering facts from multiple sources...")
    
    active_sources = [name for name, config in FACT_SOURCES.items() if config['enabled']]
    
    print(f"   Active sources: {', '.join(active_sources)}")
    
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Process in batches to not get slapped with rate limits
        pbar = tqdm(total=len(landmarks), desc="Gathering facts", unit="landmark")
        
        for i in range(0, len(landmarks), batch_size):
            batch = landmarks[i:i + batch_size]
            
            tasks = [gather_facts_for_landmark(session, landmark, user_agent, geonames_username) for landmark in batch]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for landmark, result in zip(batch, results):
                if isinstance(result, dict):
                    landmark['facts'] = result['facts']
                    landmark['fact_sources'] = result['sources']
                    landmark['fact_source_count'] = result['source_count']
                pbar.update(1)
            
            if i + batch_size < len(landmarks):
                await asyncio.sleep(1)
    
    with_facts = sum(1 for lm in landmarks if lm.get('fact_source_count', 0) > 0)
    
    print(f"   ✓ Gathered facts for {with_facts} landmarks")
    
    return landmarks