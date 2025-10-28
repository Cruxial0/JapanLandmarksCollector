import requests
import json
import re
from geopy.distance import geodesic
from tqdm import tqdm
import time
from typing import List, Dict, Tuple, Optional
import argparse
import unicodedata
from bs4 import BeautifulSoup
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import os
import hashlib
import numpy as np
from scipy.spatial import KDTree
from pathlib import Path

from src.fact_gathering import (
    gather_facts_for_landmark,
    format_facts_for_llm,
    FACT_SOURCES
)
from src.image_gathering import (
    enrich_landmarks_with_images,
    IMAGE_SOURCES
)

# Wikidata SPARQL endpoint
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# User-Agent header to comply with Wikimedia User-Agent Policy
USER_AGENT = "JapanLandmarksCollector/1.0 (https://github.com/Cruxial0/JapanLandmarksCollector; your.email@example.com) python-requests/2.31"

HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": USER_AGENT
}

# Rate limiting (Wikimedia recommends max 200 requests per second)
REQUEST_DELAY = 0.2 

MAJOR_CITY_POPULATION_THRESHOLD = 290000

# OpenRouter configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "x-ai/grok-4-fast:online"

# Prompt template
DEFAULT_PROMPT_TEMPLATE = """Based on the comprehensive information provided from multiple verified sources, create a compelling one-paragraph summary (under 1024 characters) of this location.

Focus on:
- The most unique and interesting aspects
- Specific facts (dates, numbers, names, measurements)
- Why it's culturally or historically significant
- What makes it worth visiting
- Any anime/pop culture connections if mentioned in the sources

Important: Only include information that appears in the provided sources. Do not make anything up.

{formatted_facts}

Summary:"""

OPENROUTER_CONCURRENCY = 5  # Number of parallel requests

DEFAULT_WEB_SEARCH_PROMPT = """A web search was conducted. Use the following web search results to supplement your response with additional information. Integrate the information naturally into your summary WITHOUT including citations or source links."""

# Cache configuration
DEFAULT_CACHE_FILE = "cache/wikidata_cache.json"
CACHE_VERSION = "1.0"
DEFAULT_CACHE_EXPIRY_DAYS = 7

# Batching configuration
BATCH_SIZE = 5

# Landmark configuration
# Objects added here will automatically be queried and processed as a landmark (assuming they have valid coordinates)
LANDMARK_TYPES = {
    'mountain': {
        'wikidata_id': 'Q8502 Q8072 Q193457 Q842928 Q6022321 Q674775 Q1142381 Q332614 Q1197120 Q1200524 Q1242936 Q1325302 Q1330974 Q1368970 Q169358 Q11275390 Q11277298 Q1595289 Q11505921 Q190869 Q212057 Q2143039 Q3116906 Q478788 Q27963400',
        'display_name': 'Mountain',
        'name_patterns': {
            'prefixes': [r'^Mount\s+', r'^Mt\.?\s+'],
            'suffixes': [
                r'-yama$', r'-san$', r'-dake$', r'-take$', r'-mine$', r'-zan$',
                r'山$', r'岳$', r'峰$', r'\s+Mountain$', r'\s+Peak$'
            ]
        }
    },
    'lake': {
        'wikidata_id': 'Q23397 Q3511952 Q204324 Q100900880 Q11253318 Q3215290 Q188025 Q1048337 Q3215752 Q1165822 Q458063 Q10313934 Q1140477 Q10671833 Q11349558 Q11726978 Q11726974 Q11727001 Q1898470 Q13598778 Q444677 Q18378697 Q4904826 Q27966643 Q58236221 Q66716634 Q131681 Q187223 Q131581612 Q132656659 Q11726988 Q358976 Q1521583 Q6341928 Q317995 Q3488975 Q3738260 Q7205721 Q121782022 Q122292419',
        'display_name': 'Lake',
        'name_patterns': {
            'prefixes': [r'^Lake\s+'],
            'suffixes': [r'-ko$', r'-ike$', r'-numa$', r'湖$', r'池$', r'沼$', r'\s+Lake$']
        }
    },
    'shrine': {
        'wikidata_id': 'Q697295 Q845945',
        'display_name': 'Shinto Shrine',
        'name_patterns': {
            'prefixes': [],
            'suffixes': [r'-jinja$', r'-taisha$', r'-gū$', r'神社$', r'大社$', r'宮$', r'\s+Shrine$']
        }
    },
    'cave': {
        'wikidata_id': 'Q35509 Q89497 Q1131329 Q98446107 Q2455463 Q1149405 Q3571304 Q1266984 Q1526552 Q11665546 Q3480180 Q7558985 Q135741893 Q57732276 Q58214675 Q58215210 Q58215273 Q107480506',
        'display_name': 'Cave',
        'name_patterns': {
            'prefixes': [],
            'suffixes': [r'-do$', r'-dokutsu$', r'洞$', r'洞窟$', r'\s+Cave$', r'\s+Cavern$']
        }
    },
    'bridge': {
        'wikidata_id': 'Q12280 Q158555 Q12045874 Q428759 Q445800 Q3397526 Q158626 Q1494578 Q11552394 Q2104072 Q2297251 Q3396425 Q15980599 Q7577756 Q7661648 Q7850935 Q7883867 Q976622 Q12570 Q132775038 Q132775036 Q132775089 Q132795106 Q10513727 Q21494734 Q23701378 Q25105692 Q25556320 Q28976245 Q105501780 Q65045238 Q107040113 Q113866885 Q116188668 Q135995331 Q134303786 Q11111030 Q1143769 Q1415899 Q1250323 Q158448 Q99523097 Q11423692 Q11840152 Q17265036 Q474728 Q17105481 Q4868488 Q43514341 Q818882 Q25325210 Q50683289 Q1030403 Q1425971 Q157942 Q21170235 Q11396304 Q1704122 Q135994842 Q2933208 Q23383 Q64563393 Q108263503',
        'display_name': 'Bridge',
        'name_patterns': {
            'prefixes': [],
            'suffixes': [r'-bashi$', r'-hashi$', r'-kyō$', r'橋$', r'\s+Bridge$']
        }
    },
    'park': {
        'wikidata_id': 'Q22698 Q6629955 Q7138600 Q6063204 Q11649671 Q12327290 Q16363669 Q1470855 Q3363945 Q11409728 Q11546861 Q11559043 Q21164403 Q11637995 Q11665557 Q1939700 Q2244647 Q3364370 Q15982170 Q820084 Q7339681 Q136486169 Q136486243 Q8564897 Q30326995 Q46169 Q22746 Q110451841 Q96120838 Q11298806 Q1886911 Q1711697 Q1995305 Q3243966 Q3363934 Q642682 Q21550840 Q1443808 Q11665453 Q15060435 Q79979740 Q107691783 Q116823462 Q116823535 Q116823488 Q116823586 Q118256174 Q491675',
        'display_name': 'Park',
        'name_patterns': {
            'prefixes': [],
            'suffixes': [r'-kōen$', r'-en$', r'公園$', r'園$', r'\s+Park$', r'\s+Garden$']
        }
    }
}

TYPE_MAPPINGS = {
    'mountain': 'mountain',
    'volcano': 'mountain'
}

GENERIC_NAME_PATTERNS = {
    'prefixes': [r'^御', r'^小', r'^大', r'^新', r'^北', r'^南', r'^東', r'^西'],
    'suffixes': []
}

def build_landmark_query(values_clause: str, comment: str) -> str:
    """Dynamically build SPARQL query based on provided values clause and comment."""
    query = f"""
SELECT ?item ?itemLabel ?coordinate ?typeLabel ?wikipedia ?wikivoyage ?image ?osmNode ?wikidata WHERE {{
  VALUES ?type {{ {values_clause} }}  # {comment}
  ?item wdt:P31 ?type;
        wdt:P17 wd:Q17;
        wdt:P625 ?coordinate.
  
  # Get Wikipedia article
  OPTIONAL {{
    ?wikipedia schema:about ?item;
               schema:isPartOf <https://en.wikipedia.org/>.
  }}
  
  # Get Wikivoyage article
  OPTIONAL {{
    ?wikivoyage schema:about ?item;
                schema:isPartOf <https://en.wikivoyage.org/>.
  }}
  
  # Get image from Wikimedia Commons
  OPTIONAL {{ ?item wdt:P18 ?image. }}
  
  # Get OpenStreetMap node ID
  OPTIONAL {{ ?item wdt:P11693 ?osmNode. }}
  
  # Get Wikidata ID for cross-referencing
  BIND(?item AS ?wikidata)
  
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,ja". }}
}}
"""
    print(query)
    return query

TOWN_QUERY = """
SELECT DISTINCT ?town ?townLabel ?coordinate ?prefectureLabel ?population ?pointInTime ?isCapital ?wikipedia ?image ?wikidata WHERE {
  ?town wdt:P31/wdt:P279* wd:Q515;
        wdt:P17 wd:Q17;
        wdt:P625 ?coordinate.
  
  FILTER NOT EXISTS { ?town wdt:P576 ?dissolved }
  FILTER NOT EXISTS { ?town wdt:P582 ?endTime }
  
  OPTIONAL {
    ?town wdt:P131+ ?prefecture.
    ?prefecture wdt:P31 wd:Q50337.
  }
  
  OPTIONAL {
    ?prefecture wdt:P36 ?town.
    BIND(true AS ?isCapital)
  }
  
  OPTIONAL {
    ?wikipedia schema:about ?town;
               schema:isPartOf <https://en.wikipedia.org/>.
  }
  
  OPTIONAL { ?town wdt:P18 ?image. }
  
  OPTIONAL {
    ?town p:P1082 ?popStatement.
    ?popStatement ps:P1082 ?population.
    FILTER(DATATYPE(?population) = xsd:integer || DATATYPE(?population) = xsd:decimal)
    
    OPTIONAL {
      ?popStatement pq:P585 ?pointInTime.
    }
  }
  
  # Get Wikidata ID for cross-referencing
  BIND(?town AS ?wikidata)
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,ja". }
}
"""

def load_cache(cache_file: str) -> Dict:
    """Load cache from file if it exists and is valid."""
    if not os.path.exists(cache_file):
        return {
            "version": CACHE_VERSION,
            "created_at": datetime.now().isoformat(),
            "data": {},
            "image_cache": {}  # Dedicated cache for external image URLs
        }
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
            
        if cache.get("version") != CACHE_VERSION:
            print(f"  ⚠️  Cache version mismatch. Creating new cache.")
            return {
                "version": CACHE_VERSION,
                "created_at": datetime.now().isoformat(),
                "data": {},
                "image_cache": {}
            }
        
        # Ensure image_cache exists for older cache files
        if "image_cache" not in cache:
            cache["image_cache"] = {}
        
        return cache
    except Exception as e:
        print(f"  ⚠️  Failed to load cache: {e}. Creating new cache.")
        return {
            "version": CACHE_VERSION,
            "created_at": datetime.now().isoformat(),
            "data": {},
            "image_cache": {}
        }

def save_cache(cache: Dict, cache_file: str):
    """Save cache to file."""
    try:
        cache["updated_at"] = datetime.now().isoformat()
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  ⚠️  Failed to save cache: {e}")

def load_image_cache_for_landmarks(landmarks: List[Dict], cache: Dict, cache_expiry_days: int) -> int:
    """Load cached image URLs for landmarks. Returns count of cached images loaded."""
    if not cache or "image_cache" not in cache:
        return 0
    
    loaded_count = 0
    image_cache = cache["image_cache"]
    
    for landmark in landmarks:
        wikidata_id = landmark.get("wikidata_id")
        if not wikidata_id or wikidata_id not in image_cache:
            continue
        
        cache_entry = image_cache[wikidata_id]
        if not is_cache_valid(cache_entry, cache_expiry_days):
            continue
        
        # Load cached alternative images
        if "alternative_images" in cache_entry and cache_entry["alternative_images"]:
            landmark["alternative_images"] = cache_entry["alternative_images"]
            loaded_count += 1
    
    return loaded_count

def save_image_cache_for_landmarks(landmarks: List[Dict], cache: Dict):
    """Save image URLs from landmarks to cache."""
    if not cache:
        return
    
    if "image_cache" not in cache:
        cache["image_cache"] = {}
    
    image_cache = cache["image_cache"]
    
    for landmark in landmarks:
        wikidata_id = landmark.get("wikidata_id")
        if not wikidata_id:
            continue
        
        # Only cache if there are alternative images
        if landmark.get("alternative_images"):
            image_cache[wikidata_id] = {
                "cached_at": datetime.now().isoformat(),
                "alternative_images": landmark["alternative_images"]
            }

def is_cache_valid(cache_entry: Dict, expiry_days: int) -> bool:
    """Check if a cache entry is still valid."""
    if not cache_entry or "cached_at" not in cache_entry:
        return False
    
    try:
        cached_at = datetime.fromisoformat(cache_entry["cached_at"])
        expiry = timedelta(days=expiry_days)
        return datetime.now() - cached_at < expiry
    except Exception:
        return False

def get_cache_key(query: str) -> str:
    """Generate a cache key from a query string."""
    return hashlib.md5(query.encode('utf-8')).hexdigest()

def romanize_name(name: str) -> str:
    """Convert name to ASCII-only characters (romanization)."""
    normalized = unicodedata.normalize('NFKD', name)
    ascii_only = normalized.encode('ascii', 'ignore').decode('ascii')
    ascii_only = ' '.join(ascii_only.split())
    
    if not ascii_only or ascii_only.isspace():
        ascii_only = ''.join(c for c in name if ord(c) < 128)
        ascii_only = ' '.join(ascii_only.split())
    
    return ascii_only if ascii_only else name

def clean_prefecture_name(prefecture: str) -> str:
    """Clean and romanize prefecture names."""
    if not prefecture:
        return ""
    
    cleaned = prefecture
    suffixes_to_remove = [" Prefecture", " prefecture", "県", "-ken", " Ken"]
    for suffix in suffixes_to_remove:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
            break
    
    cleaned = romanize_name(cleaned.strip())
    
    return cleaned

def clean_landmark_name(name: str, landmark_type: str = None) -> str:
    """Remove common geographical identifiers from landmark names."""
    clean_name = name
    
    for pattern in GENERIC_NAME_PATTERNS['prefixes']:
        clean_name = re.sub(pattern, '', clean_name, flags=re.IGNORECASE)
    
    for pattern in GENERIC_NAME_PATTERNS['suffixes']:
        clean_name = re.sub(pattern, '', clean_name, flags=re.IGNORECASE)
    
    if landmark_type and landmark_type in LANDMARK_TYPES:
        type_patterns = LANDMARK_TYPES[landmark_type].get('name_patterns', {})
        
        for pattern in type_patterns.get('prefixes', []):
            clean_name = re.sub(pattern, '', clean_name, flags=re.IGNORECASE)
        
        for pattern in type_patterns.get('suffixes', []):
            clean_name = re.sub(pattern, '', clean_name, flags=re.IGNORECASE)
    
    clean_name = clean_name.strip()
    
    return clean_name if clean_name else name

def get_landmark_type_key(type_label: str) -> str:
    """Get the landmark type key from a Wikidata type label."""
    type_label_lower = type_label.lower()
    
    if type_label_lower in TYPE_MAPPINGS:
        return TYPE_MAPPINGS[type_label_lower]
    
    for key, config in LANDMARK_TYPES.items():
        if config['display_name'].lower() == type_label_lower:
            return key
    
    for key, config in LANDMARK_TYPES.items():
        if type_label_lower in config['display_name'].lower() or config['display_name'].lower() in type_label_lower:
            return key
    
    return type_label_lower.replace(' ', '_')

def fetch_wikidata(query: str, description: str = "Fetching data", delay: bool = True, 
                   cache: Optional[Dict] = None, cache_expiry_days: int = DEFAULT_CACHE_EXPIRY_DAYS,
                   force_refresh: bool = False) -> Dict:
    """Fetch data from Wikidata with retry logic, rate limiting, and caching."""
    
    if cache is not None and not force_refresh:
        cache_key = get_cache_key(query)
        if cache_key in cache["data"]:
            cache_entry = cache["data"][cache_key]
            if is_cache_valid(cache_entry, cache_expiry_days):
                print(f"  ✓ Using cached data for {description}")
                return cache_entry["results"]
    
    max_retries = 3
    
    if delay:
        time.sleep(REQUEST_DELAY)
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                WIKIDATA_SPARQL_URL, 
                params={"query": query}, 
                headers=HEADERS,
                timeout=30
            )
            response.raise_for_status()
            results = response.json()
            
            if cache is not None:
                cache_key = get_cache_key(query)
                cache["data"][cache_key] = {
                    "cached_at": datetime.now().isoformat(),
                    "description": description,
                    "results": results
                }
            
            return results
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = (attempt + 1) * 5
                print(f"\n⚠️  Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"\n⚠️  Error: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"\n⚠️  Error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise

async def call_openrouter_async(
    session: aiohttp.ClientSession,
    formatted_facts: str, 
    api_key: str, 
    prompt_template: str, 
    model: str = DEFAULT_OPENROUTER_MODEL,
    max_tokens: int = 500,
    semaphore: asyncio.Semaphore = None,
    web_search_prompt: Optional[str] = None
) -> Optional[str]:
    """Call OpenRouter API asynchronously to generate a summary."""
    if not formatted_facts:
        return None
    
    async with semaphore if semaphore else (contextlib.AsyncExitStack()):
        try:
            prompt = prompt_template.format(formatted_facts=formatted_facts)
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Cruxial0/japan-landmarks",
                "X-Title": "Japan Landmarks Collector"
            }
            
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens
            }
            
            if web_search_prompt and ":online" in model.lower():
                payload["plugins"] = [
                    {
                        "id": "web",
                        "search_prompt": web_search_prompt
                    }
                ]
            
            async with session.post(OPENROUTER_API_URL, headers=headers, json=payload) as response:
                if response.status == 429:
                    retry_after = response.headers.get('Retry-After', '5')
                    wait_time = int(retry_after) if retry_after.isdigit() else 5
                    print(f"\n⚠️  Rate limited. Waiting {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    async with session.post(OPENROUTER_API_URL, headers=headers, json=payload) as retry_response:
                        retry_response.raise_for_status()
                        result = await retry_response.json()
                        return result['choices'][0]['message']['content'].strip()
                
                if response.status >= 400:
                    error_text = await response.text()
                    try:
                        error_json = json.loads(error_text)
                        error_message = error_json.get('error', {}).get('message', error_text)
                        print(f"\n⚠️  API Error ({response.status}): {error_message}")
                    except json.JSONDecodeError:
                        print(f"\n⚠️  API Error ({response.status}): {error_text}")
                    return None
                
                response.raise_for_status()
                result = await response.json()
                summary = result['choices'][0]['message']['content'].strip()
                
                return summary
        
        except aiohttp.ClientError as e:
            print(f"\n⚠️  OpenRouter API call failed: {e}")
            return None
        except Exception as e:
            print(f"\n⚠️  Unexpected error during OpenRouter API call: {e}")
            return None

async def generate_summary_for_landmark(
    session: aiohttp.ClientSession,
    landmark: Dict,
    api_key: str,
    prompt_template: str,
    model: str,
    semaphore: asyncio.Semaphore,
    geonames_username: Optional[str] = None,
    web_search_prompt: Optional[str] = None
) -> Tuple[Dict, bool, Optional[Dict]]:
    """Generate a summary for a single landmark using multi-source fact gathering."""
    
    gathered_facts = await gather_facts_for_landmark(
        session,
        landmark,
        USER_AGENT,
        geonames_username
    )
    
    formatted_facts = format_facts_for_llm(gathered_facts, landmark)
    
    landmark['fact_sources'] = gathered_facts.get('sources', [])
    landmark['fact_source_count'] = gathered_facts.get('source_count', 0)
    
    fact_object = None
    wikidata_id = landmark.get('wikidata_id')
    
    if wikidata_id and gathered_facts.get('facts'):
        text_sources = {}
        url_sources = {}
        
        for fact in gathered_facts['facts']:
            source_name = fact['source']
            
            if fact.get('content'):
                text_sources[source_name] = fact['content']
            
            if fact.get('url'):
                url_sources[source_name] = fact['url']
            
            if fact.get('infobox'):
                text_sources[f'{source_name}_infobox'] = str(fact['infobox'])
            if fact.get('tags'):
                text_sources[f'{source_name}_tags'] = str(fact['tags'])
        
        fact_object = {
            'wikidata_id': wikidata_id,
            'name': landmark['name'],
            'text_sources': text_sources,
            'url_sources': url_sources,
            'formatted_prompt': formatted_facts,
            'llm_summary': ''  # Will be filled in if summary generation succeeds
        }
    
    if formatted_facts:
        summary = await call_openrouter_async(
            session,
            formatted_facts, 
            api_key, 
            prompt_template,
            model=model,
            semaphore=semaphore,
            web_search_prompt=web_search_prompt
        )
        
        if summary:
            landmark["llm_summary"] = summary
            if fact_object:
                fact_object['llm_summary'] = summary
            return landmark, True, fact_object
    
    return landmark, False, fact_object

async def generate_summary_for_city(
    session: aiohttp.ClientSession,
    city: Dict,
    api_key: str,
    prompt_template: str,
    model: str,
    semaphore: asyncio.Semaphore,
    geonames_username: Optional[str] = None,
    web_search_prompt: Optional[str] = None
) -> Tuple[Dict, bool, Optional[Dict]]:
    """Generate a summary for a city using multi-source fact gathering."""
    
    # For cities, we primarily use Wikipedia, as it has good coverage (797/798 cities in my testing. The missing city is "Q7473516" for those wondering lol)
    gathered_facts = await gather_facts_for_landmark(
        session,
        city,
        USER_AGENT,
        geonames_username
    )
    
    formatted_facts = format_facts_for_llm(gathered_facts, city, is_city=True)
    
    city['fact_sources'] = gathered_facts.get('sources', [])
    city['fact_source_count'] = gathered_facts.get('source_count', 0)
    
    fact_object = None
    wikidata_id = city.get('wikidata_id')
    
    if wikidata_id and gathered_facts.get('facts'):
        text_sources = {}
        url_sources = {}
        
        for fact in gathered_facts['facts']:
            source_name = fact['source']
            
            if fact.get('content'):
                text_sources[source_name] = fact['content']
            
            if fact.get('url'):
                url_sources[source_name] = fact['url']
            
            if fact.get('infobox'):
                text_sources[f'{source_name}_infobox'] = str(fact['infobox'])
            if fact.get('tags'):
                text_sources[f'{source_name}_tags'] = str(fact['tags'])
        
        fact_object = {
            'wikidata_id': wikidata_id,
            'name': city['name'],
            'text_sources': text_sources,
            'url_sources': url_sources,
            'formatted_prompt': formatted_facts,
            'llm_summary': ''
        }
    
    if formatted_facts:
        summary = await call_openrouter_async(
            session,
            formatted_facts, 
            api_key, 
            prompt_template,
            model=model,
            semaphore=semaphore,
            web_search_prompt=web_search_prompt
        )
        
        if summary:
            city["llm_summary"] = summary
            if fact_object:
                fact_object['llm_summary'] = summary
            return city, True, fact_object
    
    return city, False, fact_object

async def generate_summaries_parallel(
    items: List[Dict],
    api_key: str,
    prompt_template: str,
    model: str,
    concurrency: int,
    item_type: str = "landmarks",
    geonames_username: Optional[str] = None,
    web_search_prompt: Optional[str] = None
) -> Tuple[List[Dict], int, int, Dict[str, Dict]]:
    """Generate LLM summaries for multiple items in parallel using multi-source facts."""
    
    items_with_data = [item for item in items if item.get("wikipedia_url") or item.get("wikivoyage_url")]
    items_without_data = [item for item in items if not (item.get("wikipedia_url") or item.get("wikivoyage_url"))]
    
    if not items_with_data:
        print(f"  ⚠️  No {item_type} with Wikipedia/Wikivoyage URLs found")
        return items, 0, 0, {}
    
    print(f"  ℹ️  Processing {len(items_with_data)} {item_type} with source URLs (skipping {len(items_without_data)} without URLs)")
    
    if ":online" in model.lower():
        print(f"  🌐 Web search enabled (using :online model)")
    
    # Show active fact sources
    active_sources = [name for name, config in FACT_SOURCES.items() if config['enabled']]
    print(f"  📚 Active fact sources: {', '.join(active_sources)}")
    
    semaphore = asyncio.Semaphore(concurrency)
    
    success_count = 0
    failed_count = 0
    facts_dict = {}  # Will store fact objects keyed by wikidata_id or city_id
    
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        batch_size = 20
        pbar = tqdm(total=len(items_with_data), desc=f"Generating {item_type} summaries", unit="item")
        
        processed_items = []
        
        for i in range(0, len(items_with_data), batch_size):
            batch = items_with_data[i:i + batch_size]
            
            batch_tasks = []
            for item in batch:
                if item_type == "landmarks":
                    task = generate_summary_for_landmark(
                        session,
                        item,
                        api_key,
                        prompt_template,
                        model,
                        semaphore,
                        geonames_username,
                        web_search_prompt
                    )
                else:  # cities
                    task = generate_summary_for_city(
                        session,
                        item,
                        api_key,
                        prompt_template,
                        model,
                        semaphore,
                        geonames_username,
                        web_search_prompt
                    )
                batch_tasks.append(task)
            
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    failed_count += 1
                    pbar.update(1)
                    continue
                
                item, success, fact_object = result
                processed_items.append(item)
                if success:
                    success_count += 1
                else:
                    failed_count += 1
                
                if fact_object and fact_object.get('wikidata_id'):
                    facts_dict[fact_object['wikidata_id']] = fact_object
                
                pbar.update(1)
            
            if i + batch_size < len(items_with_data):
                await asyncio.sleep(1)
    
    all_results = processed_items + items_without_data
    
    return all_results, success_count, failed_count, facts_dict

def generate_llm_summaries(
    landmarks: List[Dict], 
    api_key: str,
    prompt_template: str,
    model: str = DEFAULT_OPENROUTER_MODEL,
    concurrency: int = OPENROUTER_CONCURRENCY,
    geonames_username: Optional[str] = None,
    web_search_prompt: Optional[str] = None 
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Generate LLM summaries for landmarks using multi-source fact gathering."""
    
    print(f"\n🤖 Generating LLM summaries for landmarks...")
    print(f"    Model: {model}")
    print(f"    Concurrency: {concurrency} parallel requests")
    print(f"    ⚠️  Press Ctrl+C to skip this step.")
    
    success_count = 0
    failed_count = 0
    facts_dict = {}
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            processed_landmarks, success_count, failed_count, facts_dict = loop.run_until_complete(
                generate_summaries_parallel(
                    landmarks,
                    api_key,
                    prompt_template,
                    model,
                    concurrency,
                    "landmarks",
                    geonames_username,
                    web_search_prompt
                )
            )
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
        
        processed_map = {lm.get("wikidata_id"): lm for lm in processed_landmarks if lm.get("wikidata_id")}
        
        for i, lm in enumerate(landmarks):
            wikidata_id = lm.get("wikidata_id")
            if wikidata_id and wikidata_id in processed_map:
                landmarks[i] = processed_map[wikidata_id]
        
        print(f"\n✅ Landmark summary generation complete:")
        print(f"    • Successful: {success_count}")
        print(f"    • Failed: {failed_count}")
        
        # Show fact source statistics
        source_stats = {}
        for lm in landmarks:
            for source in lm.get('fact_sources', []):
                source_stats[source] = source_stats.get(source, 0) + 1
        
        if source_stats:
            print(f"    • Fact sources used:")
            for source, count in sorted(source_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"      - {source}: {count} landmarks")
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Summary generation interrupted by user.")
        print(f"    Generated {success_count} summaries before interruption.")
    except Exception as e:
        print(f"\n❌ Error during summary generation: {e}")
    
    return landmarks, facts_dict

def generate_city_summaries(
    towns: List[Dict],
    api_key: str,
    prompt_template: str,
    model: str = DEFAULT_OPENROUTER_MODEL,
    concurrency: int = OPENROUTER_CONCURRENCY,
    geonames_username: Optional[str] = None,
    web_search_prompt: Optional[str] = None
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Generate LLM summaries for cities using multi-source fact gathering."""
    cities_dict = {}
    for town in towns:
        town_name = town["name"]
        if town_name not in cities_dict:
            cities_dict[town_name] = town
    
    unique_cities = list(cities_dict.values())
    
    print(f"\n🏙️  Generating LLM summaries for cities...")
    print(f"    Model: {model}")
    print(f"    Concurrency: {concurrency} parallel requests")
    print(f"    ⚠️  Press Ctrl+C to skip this step.")
    
    success_count = 0
    failed_count = 0
    facts_dict = {}
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            processed_cities, success_count, failed_count, facts_dict = loop.run_until_complete(
                generate_summaries_parallel(
                    unique_cities,
                    api_key,
                    prompt_template,
                    model,
                    concurrency,
                    "cities",
                    geonames_username,
                    web_search_prompt
                )
            )
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
        
        processed_map = {city["name"]: city for city in processed_cities}
        
        for i, town in enumerate(towns):
            town_name = town["name"]
            if town_name in processed_map:
                processed_city = processed_map[town_name]
                # Copy over the enriched fields
                town["fact_sources"] = processed_city.get("fact_sources", [])
                town["fact_source_count"] = processed_city.get("fact_source_count", 0)
                if processed_city.get("llm_summary"):
                    town["llm_summary"] = processed_city["llm_summary"]
                towns[i] = town
        
        print(f"\n✅ City summary generation complete:")
        print(f"    • Successful: {success_count}")
        print(f"    • Failed: {failed_count}")
    
    except KeyboardInterrupt:
        print(f"\n⚠️  City summary generation interrupted by user.")
        print(f"    Generated {success_count} summaries before interruption.")
    except Exception as e:
        print(f"\n❌ Error during city summary generation: {e}")
    
    return towns, facts_dict

def parse_landmarks(results: Dict) -> List[Dict]:
    """Parse landmark data from SPARQL results."""
    landmarks = []
    items = results["results"]["bindings"]
    skipped_coords = 0
    
    print("\n📝 Parsing landmark data...")
    for item in tqdm(items, desc="Processing landmarks", unit="landmark"):
        try:
            coord_str = item["coordinate"]["value"]
            coords = coord_str.replace("Point(", "").replace(")", "").split()
            
            if len(coords) != 2:
                skipped_coords += 1
                continue
            
            lon, lat = coords
            
            name = item["itemLabel"]["value"]
            type_label = item["typeLabel"]["value"]  # Specific WikiData type (e.g., "rock shelter")
            type_key = get_landmark_type_key(type_label)  # Broader category key (e.g., "cave")
            
            # Get the broader category display name from LANDMARK_TYPES
            if type_key in LANDMARK_TYPES:
                broader_type = LANDMARK_TYPES[type_key]['display_name'].lower()
            else:
                # Fallback to type_key if not in LANDMARK_TYPES
                broader_type = type_key.replace('_', ' ')
            
            clean_name = clean_landmark_name(name, type_key)
            romanized_name = romanize_name(name)
            
            landmark = {
                "wikidata_id": item["item"]["value"].split("/")[-1],
                "name": name,
                "name_clean": clean_name,
                "name_romanized": romanized_name,
                "type": broader_type,  # Broader category (e.g., "cave")
                "type_key": type_key,  # Category key (e.g., "cave")
                "specific_subtype": type_label,  # Detailed WikiData type (e.g., "rock shelter")
                "latitude": float(lat),
                "longitude": float(lon),
                "wikipedia_url": item.get("wikipedia", {}).get("value", ""),
                "wikivoyage_url": item.get("wikivoyage", {}).get("value", ""),
                "image_url": item.get("image", {}).get("value", ""),
                "osm_node_id": item.get("osmNode", {}).get("value", ""),
                "prefecture": "",
                "nearby_towns": []
            }
            landmarks.append(landmark)
        except (ValueError, KeyError, IndexError) as e:
            skipped_coords += 1
            continue
    
    if skipped_coords > 0:
        print(f"  ⚠️  Skipped {skipped_coords} landmarks with invalid coordinates")
    
    return landmarks

def parse_towns(results: Dict) -> List[Dict]:
    """Parse town/city data from SPARQL results and deduplicate."""
    towns_dict = {}
    items = results["results"]["bindings"]
    skipped_populations = 0
    skipped_null_pop = 0
    skipped_coords = 0
    
    print("\n🏘️  Parsing town/city data...")
    for item in tqdm(items, desc="Processing towns", unit="town"):
        try:
            coord_str = item["coordinate"]["value"]
            coords = coord_str.replace("Point(", "").replace(")", "").split()
            
            if len(coords) != 2:
                skipped_coords += 1
                continue
            
            lon, lat = coords
            
            town_name = item["townLabel"]["value"]
            wikidata_id = item["town"]["value"].split("/")[-1]
            prefecture_raw = item.get("prefectureLabel", {}).get("value", "")
            prefecture = clean_prefecture_name(prefecture_raw)
            
            is_capital = item.get("isCapital", {}).get("value", "") == "true"
            
            wikipedia_url = item.get("wikipedia", {}).get("value", "")
            
            image_url = item.get("image", {}).get("value", "")
            
            population = None
            point_in_time = None
            
            if "population" in item:
                pop_value = item["population"]["value"]
                try:
                    if not pop_value.startswith("http"):
                        population = int(float(pop_value))
                        if "pointInTime" in item:
                            point_in_time = item["pointInTime"]["value"]
                except (ValueError, TypeError):
                    skipped_populations += 1
                    pass
            
            if population is None:
                skipped_null_pop += 1
                continue
            
            # Use wikidata_id for deduplication
            town_key = wikidata_id
            
            if town_key in towns_dict:
                existing = towns_dict[town_key]
                
                should_replace = False
                
                if is_capital and not existing.get("is_capital"):
                    should_replace = True
                elif wikipedia_url and not existing.get("wikipedia_url"):
                    should_replace = True
                elif image_url and not existing.get("image_url"):
                    should_replace = True
                elif not existing.get("prefecture") and prefecture:
                    should_replace = True
                elif population is not None and existing.get("population") is not None:
                    if point_in_time and existing.get("point_in_time"):
                        should_replace = point_in_time > existing.get("point_in_time", "")
                    else:
                        should_replace = population > existing.get("population", 0)
                elif population is not None and not existing.get("population"):
                    should_replace = True
                
                if not should_replace:
                    continue
            
            town_data = {
                "wikidata_id": wikidata_id,
                "name": town_name,
                "latitude": float(lat),
                "longitude": float(lon),
                "prefecture": prefecture,
                "population": population,
                "is_capital": is_capital,
                "wikipedia_url": wikipedia_url,
                "image_url": image_url,
                "point_in_time": point_in_time
            }
            
            towns_dict[town_key] = town_data
        except (ValueError, KeyError, IndexError) as e:
            skipped_coords += 1
            continue
    
    towns = []
    for town in towns_dict.values():
        town.pop("point_in_time", None)
        towns.append(town)
    
    if skipped_coords > 0:
        print(f"  ⚠️  Skipped {skipped_coords} towns with invalid coordinates")
    if skipped_populations > 0:
        print(f"  ⚠️  Skipped {skipped_populations} non-numeric population values")
    if skipped_null_pop > 0:
        print(f"  ⚠️  Skipped {skipped_null_pop} towns with null population")
    
    major_cities = [t for t in towns if t.get("is_capital") or (t.get("population") is not None and t.get("population") > MAJOR_CITY_POPULATION_THRESHOLD)]
    cities_with_wiki = [t for t in towns if t.get("wikipedia_url")]
    cities_with_images = [t for t in towns if t.get("image_url")]
    
    print(f"  ℹ️  Deduplicated to {len(towns)} unique towns (from {len(items)} entries)")
    print(f"  ℹ️  Found {len(major_cities)} major cities (capitals or population > {MAJOR_CITY_POPULATION_THRESHOLD:,})")
    print(f"  ℹ️  Found {len(cities_with_wiki)} cities with Wikipedia articles")
    print(f"  ℹ️  Found {len(cities_with_images)} cities with images")
    
    return towns

def assign_nearby_towns(landmarks: List[Dict], towns: List[Dict]) -> List[Dict]:
    """Assign nearby towns and closest major city to each landmark."""
    print("\n🗺️  Calculating nearby towns for landmarks...")
    
    major_cities = [t for t in towns if t.get("is_capital") or (t.get("population") is not None and t.get("population") > MAJOR_CITY_POPULATION_THRESHOLD)]
    print(f"  ℹ️  Using {len(major_cities)} major cities for closest major city calculation")
    
    town_coords = np.array([[t["latitude"], t["longitude"]] for t in towns])
    town_tree = KDTree(town_coords)
    
    if major_cities:
        major_coords = np.array([[c["latitude"], c["longitude"]] for c in major_cities])
        major_tree = KDTree(major_coords)
    
    for lm in tqdm(landmarks, desc="Finding nearby towns", unit="landmark"):
        lm_coord = np.array([lm["latitude"], lm["longitude"]])
        
        closest_dist_deg, closest_idx = town_tree.query(lm_coord)
        closest_town = towns[closest_idx]
        
        closest_dist_km = geodesic(
            (lm["latitude"], lm["longitude"]),
            (closest_town["latitude"], closest_town["longitude"])
        ).km
        
        if closest_town["prefecture"]:
            lm["prefecture"] = closest_town["prefecture"]
        
        radius_km = closest_dist_km + 4
        radius_deg = radius_km / 85.0
        
        indices = town_tree.query_ball_point(lm_coord, radius_deg)
        
        nearby_towns_dict = {}
        for idx in indices:
            town = towns[idx]
            dist_km = geodesic(
                (lm["latitude"], lm["longitude"]),
                (town["latitude"], town["longitude"])
            ).km
            
            if dist_km <= radius_km:
                town_name = town["name"]
                town_id = town["wikidata_id"]
                if town_name not in nearby_towns_dict or dist_km < nearby_towns_dict[town_name]["distance_km"]:
                    nearby_town_data = {
                        "name": town_name,
                        "id": town_id,
                        "distance_km": round(dist_km, 2),
                        "prefecture": town["prefecture"],
                        "population": town.get("population")
                    }
                    if town.get("image_url"):
                        nearby_town_data["image_url"] = town["image_url"]
                    
                    nearby_towns_dict[town_name] = nearby_town_data
        
        nearby_towns = list(nearby_towns_dict.values())
        nearby_towns.sort(key=lambda x: x["distance_km"])
        
        lm["nearby_towns"] = nearby_towns[:10]
        lm["closest_town_distance_km"] = round(closest_dist_km, 2)
        
        if major_cities:
            _, major_idx = major_tree.query(lm_coord)
            closest_major = major_cities[major_idx]
            closest_major_dist = geodesic(
                (lm["latitude"], lm["longitude"]),
                (closest_major["latitude"], closest_major["longitude"])
            ).km
            
            closest_major_data = {
                "name": closest_major["name"],
                "id": closest_major["wikidata_id"],
                "distance_km": round(closest_major_dist, 2),
                "prefecture": closest_major["prefecture"],
                "population": closest_major.get("population"),
                "is_capital": closest_major.get("is_capital", False)
            }
            if closest_major.get("image_url"):
                closest_major_data["image_url"] = closest_major["image_url"]
            
            lm["closest_major_city"] = closest_major_data
        else:
            lm["closest_major_city"] = None
    
    return landmarks

def generate_statistics(landmarks: List[Dict], towns: List[Dict]) -> Dict:
    """Generate statistics about the collected data."""
    stats = {
        "total_landmarks": len(landmarks),
        "with_wikipedia": sum(1 for lm in landmarks if lm.get("wikipedia_url")),
        "with_wikivoyage": sum(1 for lm in landmarks if lm.get("wikivoyage_url")),
        "with_images": sum(1 for lm in landmarks if lm.get("image_url")),
        "with_alternative_images": sum(1 for lm in landmarks if lm.get("alternative_images")),
        "with_osm_node": sum(1 for lm in landmarks if lm.get("osm_node_id")),
        "with_llm_summary": sum(1 for lm in landmarks if lm.get("llm_summary")),
        "prefectures": len(set(lm["prefecture"] for lm in landmarks if lm["prefecture"])),
        "with_major_city": sum(1 for lm in landmarks if lm.get("closest_major_city")),
        "total_cities": len(towns),
        "cities_with_wikipedia": sum(1 for t in towns if t.get("wikipedia_url")),
        "cities_with_images": sum(1 for t in towns if t.get("image_url")),
        "cities_with_llm_summary": sum(1 for t in towns if t.get("llm_summary"))
    }
    
    type_counts = {}
    for lm in landmarks:
        type_key = lm.get("type_key", lm.get("type", "unknown"))
        type_counts[type_key] = type_counts.get(type_key, 0) + 1
    
    stats["by_type"] = type_counts
    
    # Multi-source statistics
    if any(lm.get("fact_sources") for lm in landmarks):
        stats["fact_sources_used"] = {}
        for lm in landmarks:
            for source in lm.get("fact_sources", []):
                stats["fact_sources_used"][source] = stats["fact_sources_used"].get(source, 0) + 1
    
    if any(lm.get("alternative_images") for lm in landmarks):
        total_alt_images = sum(len(lm.get("alternative_images", [])) for lm in landmarks)
        stats["total_alternative_images"] = total_alt_images
        
        image_sources = {}
        for lm in landmarks:
            for img in lm.get("alternative_images", []):
                source = img.get("source", "unknown")
                image_sources[source] = image_sources.get(source, 0) + 1
        stats["image_sources_used"] = image_sources
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Collect Japanese landmarks data from Wikidata with multi-source enrichment")
    parser.add_argument("--output", default="output/japan_landmarks.json",
                       help="Output filename (default: output/japan_landmarks.json)")
    parser.add_argument("--email", type=str,
                       help="Your email for the User-Agent header (required by Wikimedia)")
    
    # Landmark type selection
    parser.add_argument("--landmark-types", type=str, nargs='+',
                       choices=list(LANDMARK_TYPES.keys()),
                       help=f"Specific landmark types to fetch (default: all). Choices: {', '.join(LANDMARK_TYPES.keys())}")
    
    # LLM summarization arguments
    parser.add_argument("--generate-summaries", action="store_true",
                       help="Generate LLM summaries for landmarks using OpenRouter")
    parser.add_argument("--generate-city-summaries", action="store_true",
                    help="Generate LLM summaries for cities/towns (requires --generate-summaries)")
    parser.add_argument("--openrouter-api-key", type=str,
                       help="OpenRouter API key for LLM summarization")
    parser.add_argument("--llm-model", type=str, default=DEFAULT_OPENROUTER_MODEL,
                       help=f"OpenRouter model to use (default: {DEFAULT_OPENROUTER_MODEL})")
    parser.add_argument("--llm-prompt", type=str, default=DEFAULT_PROMPT_TEMPLATE,
                       help="Prompt template for LLM (use {formatted_facts} as placeholder)")
    parser.add_argument("--llm-concurrency", type=int, default=OPENROUTER_CONCURRENCY,
                       help=f"Number of parallel LLM requests (default: {OPENROUTER_CONCURRENCY})")
    parser.add_argument("--enable-web-search", action="store_true",
                       help="Enable OpenRouter web search plugin for additional research")
    parser.add_argument("--web-search-prompt", type=str, default=DEFAULT_WEB_SEARCH_PROMPT,
                       help="Custom prompt for web search results")
    
    # Multi-source API keys
    parser.add_argument("--geonames-username", type=str,
                       help="GeoNames username for fact gathering")
    parser.add_argument("--flickr-api-key", type=str,
                       help="Flickr API key for image gathering")
    parser.add_argument("--unsplash-api-key", type=str,
                       help="Unsplash API key for image gathering")
    
    # Image gathering control
    parser.add_argument("--gather-images", action="store_true",
                       help="Enable multi-source image gathering (Wikimedia Commons, Flickr, Unsplash)")
    
    # Cache arguments
    parser.add_argument("--cache-file", type=str, default=DEFAULT_CACHE_FILE,
                       help=f"Cache file location (default: {DEFAULT_CACHE_FILE})")
    parser.add_argument("--cache-expiry-days", type=int, default=DEFAULT_CACHE_EXPIRY_DAYS,
                       help=f"Number of days before cache expires (default: {DEFAULT_CACHE_EXPIRY_DAYS})")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching (fetch all data fresh)")
    parser.add_argument("--force-refresh", action="store_true",
                       help="Force refresh all cached data")
    
    args = parser.parse_args()
    
    # Scaffold directories
    cache_path = Path(args.cache_file)
    output_path = Path(args.output)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate LLM arguments
    if args.generate_summaries and not args.openrouter_api_key:
        print("\n❌ Error: --openrouter-api-key is required when --generate-summaries is enabled")
        print("   Get your API key from https://openrouter.ai/")
        return
    
    if args.generate_city_summaries and not args.generate_summaries:
        print("\n❌ Error: --generate-city-summaries requires --generate-summaries to be enabled")
        return
    
    # Update User-Agent if email is provided
    global USER_AGENT, HEADERS
    if args.email:
        USER_AGENT = f"JapanLandmarksCollector/1.0 (https://github.com/Cruxial0/JapanLandmarksCollector; {args.email}) python-requests/2.31"
        HEADERS["User-Agent"] = USER_AGENT
    
    # Determine which landmark types to fetch
    selected_types = LANDMARK_TYPES
    if args.landmark_types:
        selected_types = {k: v for k, v in LANDMARK_TYPES.items() if k in args.landmark_types}
        print(f"\n🔍 Fetching only selected types: {', '.join([v['display_name'] for v in selected_types.values()])}")
    
    print("=" * 60)
    print("🗾 Japanese Landmarks Data Collector")
    print("=" * 60)
    print(f"\n📋 Available landmark types:")
    for key, config in LANDMARK_TYPES.items():
        indicator = "✓" if key in selected_types else " "
        print(f"  [{indicator}] {config['display_name']} (Wikidata: {config['wikidata_id']})")
    
    # Show enabled features
    print(f"\n🔧 Enabled features:")
    print(f"  • Multi-source fact gathering: {'✓' if args.generate_summaries else '✗'}")
    if args.generate_summaries:
        active_sources = [name for name, config in FACT_SOURCES.items() if config['enabled']]
        print(f"    Sources: {', '.join(active_sources)}")
    print(f"  • Multi-source image gathering: {'✓' if args.gather_images else '✗'}")
    if args.gather_images:
        active_sources = [name for name, config in IMAGE_SOURCES.items() if config['enabled']]
        print(f"    Sources: {', '.join(active_sources)}")
    
    # Check if User-Agent is properly configured
    if "your.email@example.com" in USER_AGENT:
        print("\n⚠️  WARNING: Please provide your email for the User-Agent header!")
        print("    This is required by Wikimedia's User-Agent Policy.")
        print("    Use: python main.py --email your.email@example.com")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please provide email and try again.")
            return
    
    try:
        # Load cache
        cache = None
        if not args.no_cache:
            print(f"\n💾 Loading cache from {args.cache_file}...")
            cache = load_cache(args.cache_file)
            cache_age = "new cache"
            if "created_at" in cache:
                try:
                    created = datetime.fromisoformat(cache["created_at"])
                    age_days = (datetime.now() - created).days
                    cache_age = f"{age_days} days old"
                except:
                    pass
            cached_entries = len(cache.get("data", {}))
            print(f"  ✓ Cache loaded: {cached_entries} entries ({cache_age})")
            if args.force_refresh:
                print(f"  ⚠️  Force refresh enabled - all data will be fetched fresh")
        else:
            print(f"\n⚠️  Caching disabled - all data will be fetched fresh")
        
        # Determine number of steps
        total_steps = 4
        if args.gather_images:
            total_steps += 1
        if args.generate_summaries:
            total_steps += 1
        if args.generate_city_summaries:
            total_steps += 1
        
        # Build and fetch landmarks query in batches
        print(f"\n📊 Step 1/{total_steps}: Fetching landmarks from Wikidata...")
        
        all_type_ids = []
        for config in selected_types.values():
            parts = config['wikidata_id'].split()
            for part in parts:
                if part.startswith('wd:'):
                    all_type_ids.append(part)
                else:
                    all_type_ids.append(f'wd:{part}')
        
        batches = [all_type_ids[i:i + BATCH_SIZE] for i in range(0, len(all_type_ids), BATCH_SIZE)]
        
        landmark_results_list = []
        for batch_idx, batch in enumerate(batches, 1):
            print(f"Fetching landmarks batch {batch_idx}/{len(batches)}")
            values_clause = " ".join(batch)
            comment = "Batched landmark types"
            landmark_query = build_landmark_query(values_clause, comment)
            landmark_results = fetch_wikidata(
                landmark_query, 
                f"Fetching landmarks batch {batch_idx}", 
                delay=bool(batch_idx > 1),
                cache=cache,
                cache_expiry_days=args.cache_expiry_days if not args.no_cache else 0,
                force_refresh=args.force_refresh
            )
            landmark_results_list.append(landmark_results)
        
        combined_bindings = []
        for res in landmark_results_list:
            combined_bindings.extend(res['results']['bindings'])
        
        combined_results = {
            "head": landmark_results_list[0]['head'] if landmark_results_list else {},
            "results": {"bindings": combined_bindings}
        }
        
        landmarks = parse_landmarks(combined_results)
        print(f"✅ Fetched {len(landmarks)} landmarks")
        
        # Fetch towns
        print(f"\n📊 Step 2/{total_steps}: Fetching towns and cities...")
        town_results = fetch_wikidata(
            TOWN_QUERY, 
            "Fetching towns", 
            delay=True,
            cache=cache,
            cache_expiry_days=args.cache_expiry_days if not args.no_cache else 0,
            force_refresh=args.force_refresh
        )
        towns = parse_towns(town_results)
        print(f"✅ Fetched {len(towns)} unique towns/cities")
        
        # Calculate nearby towns
        print(f"\n📊 Step 3/{total_steps}: Calculating nearby towns...")
        landmarks = assign_nearby_towns(landmarks, towns)
        print("✅ Nearby towns calculated")
        
        # Save cache after Wikidata queries
        if cache is not None:
            print(f"\n💾 Saving cache to {args.cache_file}...")
            save_cache(cache, args.cache_file)
            print(f"  ✓ Cache saved: {len(cache.get('data', {}))} entries")
        
        current_step = 4
        
        # Multi-source image gathering (optional)
        if args.gather_images:
            print(f"\n📊 Step {current_step}/{total_steps}: Gathering images from multiple sources...")
            
            # Load cached images first
            if cache is not None and not args.force_refresh:
                cached_count = load_image_cache_for_landmarks(landmarks, cache, args.cache_expiry_days if not args.no_cache else 0)
                if cached_count > 0:
                    print(f"  ✓ Loaded cached images for {cached_count} landmarks")
            
            # Run async image gathering
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                landmarks = loop.run_until_complete(
                    enrich_landmarks_with_images(
                        landmarks,
                        flickr_api_key=args.flickr_api_key,
                        unsplash_api_key=args.unsplash_api_key
                    )
                )
            finally:
                loop.close()
            
            # Save newly fetched images to cache
            if cache is not None:
                save_image_cache_for_landmarks(landmarks, cache)
                print(f"  ✓ Saved image cache")
            
            print("✅ Multi-source image gathering completed")
            current_step += 1
        
        # Generate LLM summaries with multi-source facts (optional)
        landmark_facts = {}
        if args.generate_summaries:
            print(f"\n📊 Step {current_step}/{total_steps}: Generating landmark LLM summaries with multi-source facts...")
            
            search_prompt = args.web_search_prompt if ":online" in args.llm_model.lower() else None
            
            landmarks, landmark_facts = generate_llm_summaries(
                landmarks,
                api_key=args.openrouter_api_key,
                prompt_template=args.llm_prompt,
                model=args.llm_model,
                concurrency=args.llm_concurrency,
                geonames_username=args.geonames_username,
                web_search_prompt=search_prompt
            )
            print("✅ Landmark LLM summary generation completed")
            current_step += 1
        
        # Generate city summaries (optional)
        city_facts = {}
        if args.generate_city_summaries:
            print(f"\n📊 Step {current_step}/{total_steps}: Generating city LLM summaries...")
            towns, city_facts = generate_city_summaries(
                towns,
                api_key=args.openrouter_api_key,
                prompt_template=args.llm_prompt,
                model=args.llm_model,
                concurrency=args.llm_concurrency,
                geonames_username=args.geonames_username,
                web_search_prompt=search_prompt
            )
            print("✅ City LLM summary generation completed")
            current_step += 1
        
        # Generate statistics
        print(f"\n📊 Step {current_step}/{total_steps}: Generating statistics...")
        stats = generate_statistics(landmarks, towns)
        
        # Save to JSON
        # Combine landmark and city facts
        all_facts = {}
        all_facts.update(landmark_facts)
        all_facts.update(city_facts)
        
        output_data = {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "landmark_types": {k: v['display_name'] for k, v in selected_types.items()},
                "statistics": stats,
                "user_agent": USER_AGENT,
                "cache_info": {
                    "enabled": not args.no_cache,
                    "cache_file": args.cache_file if not args.no_cache else None,
                    "cache_entries": len(cache.get("data", {})) if cache else 0,
                    "image_cache_entries": len(cache.get("image_cache", {})) if cache else 0,
                    "cache_expiry_days": args.cache_expiry_days if not args.no_cache else None
                } if not args.no_cache else {"enabled": False},
                "llm_config": {
                    "enabled": args.generate_summaries,
                    "model": args.llm_model if args.generate_summaries else None,
                    "concurrency": args.llm_concurrency if args.generate_summaries else None,
                    "city_summaries_enabled": args.generate_city_summaries,
                    "fact_sources": list(FACT_SOURCES.keys()) if args.generate_summaries else None
                } if args.generate_summaries else None,
                "image_gathering": {
                    "enabled": args.gather_images,
                    "sources": list(IMAGE_SOURCES.keys()) if args.gather_images else None
                } if args.gather_images else None
            },
            "landmarks": landmarks,
            "cities": towns,
            "facts": all_facts
        }
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Save final cache (including images)
        if cache is not None:
            save_cache(cache, args.cache_file)
            print(f"\n💾 Final cache saved:")
            print(f"  • Query cache entries: {len(cache.get('data', {}))}")
            print(f"  • Image cache entries: {len(cache.get('image_cache', {}))}")
        
        print("\n" + "=" * 60)
        print("📈 Collection Statistics:")
        print(f"  • Total landmarks: {stats['total_landmarks']}")
        
        if stats.get('by_type'):
            print(f"\n  Landmarks by type:")
            for type_key, count in sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True):
                display_name = LANDMARK_TYPES.get(type_key, {}).get('display_name', type_key.title())
                print(f"    - {display_name}: {count}")
        
        print(f"\n  • With Wikipedia articles: {stats['with_wikipedia']}")
        print(f"  • With Wikivoyage articles: {stats['with_wikivoyage']}")
        print(f"  • With images: {stats['with_images']}")
        if stats.get('with_alternative_images'):
            print(f"  • With alternative images: {stats['with_alternative_images']}")
            print(f"  • Total alternative images: {stats.get('total_alternative_images', 0)}")
        if stats.get('image_sources_used'):
            print(f"\n  Image sources used:")
            for source, count in sorted(stats['image_sources_used'].items(), key=lambda x: x[1], reverse=True):
                print(f"    - {source}: {count} images")
        print(f"  • With OSM node IDs: {stats['with_osm_node']}")
        if args.generate_summaries:
            print(f"  • With LLM summaries: {stats['with_llm_summary']}")
            if stats.get('fact_sources_used'):
                print(f"\n  Fact sources used:")
                for source, count in sorted(stats['fact_sources_used'].items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {source}: {count} landmarks")
        print(f"  • With major city info: {stats['with_major_city']}")
        print(f"  • Prefectures covered: {stats['prefectures']}")
        print(f"\n  • Total cities/towns: {stats['total_cities']}")
        print(f"  • Cities with Wikipedia: {stats['cities_with_wikipedia']}")
        print(f"  • Cities with images: {stats['cities_with_images']}")
        if args.generate_city_summaries:
            print(f"  • Cities with LLM summaries: {stats['cities_with_llm_summary']}")
        print("=" * 60)
        print(f"\n✅ Data saved to: {args.output}")
        
        # Facts section info
        if all_facts:
            print(f"\n📚 Facts section:")
            print(f"  • Total fact entries: {len(all_facts)}")
            print(f"  • Landmark facts: {len(landmark_facts)}")
            print(f"  • City facts: {len(city_facts)}")
            
            # Count sources in facts
            text_source_counts = {}
            url_source_counts = {}
            for fact_key, fact_data in all_facts.items():
                for source in fact_data.get('text_sources', {}).keys():
                    # Remove _infobox and _tags suffixes for counting
                    base_source = source.replace('_infobox', '').replace('_tags', '')
                    text_source_counts[base_source] = text_source_counts.get(base_source, 0) + 1
                for source in fact_data.get('url_sources', {}).keys():
                    url_source_counts[source] = url_source_counts.get(source, 0) + 1
            
            if text_source_counts:
                print(f"  • Text sources distribution:")
                for source, count in sorted(text_source_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {source}: {count} entries")
        
        # Helpful tips
        if not args.gather_images:
            print(f"\n💡 Tip: Use --gather-images to fetch images from Wikimedia Commons, Flickr, and Unsplash")
        
        if not args.generate_summaries and stats['with_wikipedia'] > 0:
            print(f"\n💡 Tip: Use --generate-summaries to create AI-generated descriptions using multi-source facts")
        
        if args.generate_summaries and not args.geonames_username:
            print(f"\n💡 Tip: Use --geonames-username to enable GeoNames fact gathering (register at http://www.geonames.org/)")
        
        if args.gather_images and not args.flickr_api_key:
            print(f"\n💡 Tip: Use --flickr-api-key to enable Flickr image gathering")
        
        if args.gather_images and not args.unsplash_api_key:
            print(f"\n💡 Tip: Use --unsplash-api-key to enable Unsplash image gathering")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()