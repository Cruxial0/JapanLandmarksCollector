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

# Wikidata SPARQL endpoint
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# User-Agent header to comply with Wikimedia User-Agent Policy
# Format: <tool-name>/<version> (<contact-info>) <library>/<version>
# Replace YOUR_EMAIL with your actual email for the contact field
USER_AGENT = "JapanLandmarksCollector/1.0 (https://github.com/Cruxial0/JapanLandmarksCollector; your.email@example.com) python-requests/2.31"

# Request headers with proper User-Agent
HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": USER_AGENT
}

# Rate limiting (Wikimedia recommends max 200 requests per second, but we'll be much more conservative)
REQUEST_DELAY = 0.2  # seconds between requests

# Major city threshold (prefecture capitals or cities above this population)
MAJOR_CITY_POPULATION_THRESHOLD = 290000

# OpenRouter configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "x-ai/grok-4-fast:online"
DEFAULT_PROMPT_TEMPLATE = """Please provide a concise one-paragraph summary of this location based on the following Wikipedia content. Focus on highlighting fun/interesting facts, as well as some very brief history and why it might be worth visiting.

NOTE: THE SUMMARY HAS TO BE LESS THAN 1024 CHARACTERS! THIS IS EXTREMELY IMPORTANT!

Wikipedia content:
{wikipedia_content}

Summary:"""
OPENROUTER_CONCURRENCY = 5  # Number of parallel requests

DEFAULT_WEB_SEARCH_PROMPT = """A web search was conducted. Use the following web search results to supplement your response with additional information. Integrate the information naturally into your summary WITHOUT including citations or source links."""

# Cache configuration
DEFAULT_CACHE_FILE = "wikidata_cache.json"
CACHE_VERSION = "1.0"
DEFAULT_CACHE_EXPIRY_DAYS = 7  # Cache expires after 7 days

# Enhanced SPARQL query for mountains, lakes, and Shinto shrines with Wikipedia links, images, and OSM data
LANDMARK_QUERY = """
SELECT ?item ?itemLabel ?coordinate ?typeLabel ?wikipedia ?image ?osmNode WHERE {
  VALUES ?type { wd:Q8502 wd:Q23397 wd:Q845945 }  # mountain, lake, Shinto shrine
  ?item wdt:P31 ?type;
        wdt:P17 wd:Q17;
        wdt:P625 ?coordinate.
  
  # Get Wikipedia article
  OPTIONAL {
    ?wikipedia schema:about ?item;
               schema:isPartOf <https://en.wikipedia.org/>.
  }
  
  # Get image from Wikimedia Commons
  OPTIONAL { ?item wdt:P18 ?image. }
  
  # Get OpenStreetMap node ID (property P11693)
  OPTIONAL { ?item wdt:P11693 ?osmNode. }
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,ja". }
}
"""

# Enhanced SPARQL query for current towns/cities in Japan with Wikipedia links and images
TOWN_QUERY = """
SELECT DISTINCT ?town ?townLabel ?coordinate ?prefectureLabel ?population ?pointInTime ?isCapital ?wikipedia ?image WHERE {
  # Get current cities/towns only (not historical)
  ?town wdt:P31/wdt:P279* wd:Q515;  # instance of city/town
        wdt:P17 wd:Q17;              # in Japan
        wdt:P625 ?coordinate.         # has coordinates
  
  # Filter out dissolved/abolished entities
  FILTER NOT EXISTS { ?town wdt:P576 ?dissolved }  # no dissolved date
  FILTER NOT EXISTS { ?town wdt:P582 ?endTime }    # no end time
  
  # Get the prefecture (administrative division)
  OPTIONAL {
    ?town wdt:P131+ ?prefecture.
    ?prefecture wdt:P31 wd:Q50337.  # instance of prefecture of Japan
  }
  
  # Check if this city is a capital of a prefecture
  OPTIONAL {
    ?prefecture wdt:P36 ?town.
    BIND(true AS ?isCapital)
  }
  
  # Get Wikipedia article
  OPTIONAL {
    ?wikipedia schema:about ?town;
               schema:isPartOf <https://en.wikipedia.org/>.
  }
  
  # Get image from Wikimedia Commons
  OPTIONAL { ?town wdt:P18 ?image. }
  
  # Get the most recent population
  OPTIONAL {
    ?town p:P1082 ?popStatement.
    ?popStatement ps:P1082 ?population.
    FILTER(DATATYPE(?population) = xsd:integer || DATATYPE(?population) = xsd:decimal)
    
    # Get the date for this population figure
    OPTIONAL {
      ?popStatement pq:P585 ?pointInTime.
    }
  }
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,ja". }
}
"""

# Common identifiers to strip from names
NAME_PATTERNS = {
    'prefixes': [
        r'^Mount\s+', r'^Mt\.?\s+', r'^Lake\s+', 
        r'^御', r'^小', r'^大', r'^新', r'^北', r'^南', r'^東', r'^西'
    ],
    'suffixes': [
        r'-yama$', r'-san$', r'-dake$', r'-take$', r'-mine$', r'-zan$',
        r'-ko$', r'-ike$', r'-numa$', r'-gawa$', r'-kawa$',
        r'山$', r'岳$', r'峰$', r'湖$', r'池$', r'沼$', r'川$',
        r'\s+Mountain$', r'\s+Lake$', r'\s+River$', r'\s+Peak$'
    ]
}

def load_cache(cache_file: str) -> Dict:
    """Load cache from file if it exists and is valid."""
    if not os.path.exists(cache_file):
        return {
            "version": CACHE_VERSION,
            "created_at": datetime.now().isoformat(),
            "data": {}
        }
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
            
        # Validate cache version
        if cache.get("version") != CACHE_VERSION:
            print(f"  ⚠️  Cache version mismatch. Creating new cache.")
            return {
                "version": CACHE_VERSION,
                "created_at": datetime.now().isoformat(),
                "data": {}
            }
        
        return cache
    except Exception as e:
        print(f"  ⚠️  Failed to load cache: {e}. Creating new cache.")
        return {
            "version": CACHE_VERSION,
            "created_at": datetime.now().isoformat(),
            "data": {}
        }

def save_cache(cache: Dict, cache_file: str):
    """Save cache to file."""
    try:
        cache["updated_at"] = datetime.now().isoformat()
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  ⚠️  Failed to save cache: {e}")

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
    # First, normalize the unicode string
    normalized = unicodedata.normalize('NFKD', name)
    
    # Remove non-ASCII characters
    ascii_only = normalized.encode('ascii', 'ignore').decode('ascii')
    
    # Clean up extra spaces and return
    ascii_only = ' '.join(ascii_only.split())
    
    # If the result is empty, try a different approach
    if not ascii_only or ascii_only.isspace():
        # Keep only ASCII letters, numbers, spaces, and basic punctuation
        ascii_only = ''.join(c for c in name if ord(c) < 128)
        ascii_only = ' '.join(ascii_only.split())
    
    # If still empty, return the original name
    return ascii_only if ascii_only else name

def clean_prefecture_name(prefecture: str) -> str:
    """Clean and romanize prefecture names."""
    if not prefecture:
        return ""
    
    # Remove common suffixes
    cleaned = prefecture
    suffixes_to_remove = [" Prefecture", " prefecture", "県", "-ken", " Ken"]
    for suffix in suffixes_to_remove:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
            break
    
    # Romanize the cleaned name
    cleaned = romanize_name(cleaned.strip())
    
    return cleaned

def clean_landmark_name(name: str) -> str:
    """Remove common geographical identifiers from landmark names."""
    clean_name = name
    
    # Remove prefixes
    for pattern in NAME_PATTERNS['prefixes']:
        clean_name = re.sub(pattern, '', clean_name, flags=re.IGNORECASE)
    
    # Remove suffixes
    for pattern in NAME_PATTERNS['suffixes']:
        clean_name = re.sub(pattern, '', clean_name, flags=re.IGNORECASE)
    
    # Clean up whitespace
    clean_name = clean_name.strip()
    
    # If we completely stripped the name, return the original
    return clean_name if clean_name else name

def fetch_wikidata(query: str, description: str = "Fetching data", delay: bool = True, 
                   cache: Optional[Dict] = None, cache_expiry_days: int = DEFAULT_CACHE_EXPIRY_DAYS,
                   force_refresh: bool = False) -> Dict:
    """Fetch data from Wikidata with retry logic, rate limiting, and caching."""
    
    # Check cache first if provided
    if cache is not None and not force_refresh:
        cache_key = get_cache_key(query)
        if cache_key in cache["data"]:
            cache_entry = cache["data"][cache_key]
            if is_cache_valid(cache_entry, cache_expiry_days):
                print(f"  ✓ Using cached data for {description}")
                return cache_entry["results"]
    
    max_retries = 3
    
    # Add delay to respect rate limits
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
            
            # Cache the results if cache is provided
            if cache is not None:
                cache_key = get_cache_key(query)
                cache["data"][cache_key] = {
                    "cached_at": datetime.now().isoformat(),
                    "description": description,
                    "results": results
                }
            
            return results
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Too Many Requests
                wait_time = (attempt + 1) * 5  # Longer wait for rate limit errors
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

error_stats = {"timeout": 0, "http_error": 0, "empty_content": 0, "parse_error": 0}

async def fetch_wikipedia_content_async(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """Fetch and extract text content from a Wikipedia page asynchronously."""
    if not url:
        return None
    
    try:
        headers = {"User-Agent": USER_AGENT}
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            response.raise_for_status()
            html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                error_stats["empty_content"] += 1
                return None
            
            # Remove unwanted elements
            for element in content_div.find_all(['table', 'script', 'style', 'sup', 'div.reflist', 'div.navbox']):
                element.decompose()
            
            paragraphs = content_div.find_all('p')
            text_content = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            if not text_content or len(text_content) < 50:
                error_stats["empty_content"] += 1
                return None
            
            max_chars = 8000
            if len(text_content) > max_chars:
                text_content = text_content[:max_chars] + "..."
            
            return text_content
    
    except asyncio.TimeoutError:
        error_stats["timeout"] += 1
        return None
    except aiohttp.ClientError as e:
        error_stats["http_error"] += 1
        return None
    except Exception as e:
        error_stats["parse_error"] += 1
        return None

async def call_openrouter_async(
    session: aiohttp.ClientSession,
    content: str, 
    api_key: str, 
    prompt_template: str, 
    model: str = DEFAULT_OPENROUTER_MODEL,
    max_tokens: int = 500,
    semaphore: asyncio.Semaphore = None,
    web_search_prompt: Optional[str] = None  # Now this should work!
) -> Optional[str]:
    """Call OpenRouter API asynchronously to generate a summary."""
    if not content:
        return None
    
    async with semaphore if semaphore else asyncio.Semaphore(1):
        try:
            prompt = prompt_template.format(wikipedia_content=content)
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/yourusername/japan-landmarks",
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
            
            # Add web search plugin configuration if custom prompt provided
            if web_search_prompt and ":online" in model.lower():
                payload["plugins"] = [
                    {
                        "id": "web",
                        "search_prompt": web_search_prompt
                    }
                ]
            
            async with session.post(OPENROUTER_API_URL, headers=headers, json=payload) as response:
                # Check for rate limiting
                if response.status == 429:
                    retry_after = response.headers.get('Retry-After', '5')
                    wait_time = int(retry_after) if retry_after.isdigit() else 5
                    print(f"\n⚠️  Rate limited. Waiting {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    # Retry once
                    async with session.post(OPENROUTER_API_URL, headers=headers, json=payload) as retry_response:
                        retry_response.raise_for_status()
                        result = await retry_response.json()
                        return result['choices'][0]['message']['content'].strip()
                
                # Handle errors with detailed logging
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

async def generate_summary_for_item(
    session: aiohttp.ClientSession,
    item: Dict,
    api_key: str,
    prompt_template: str,
    model: str,
    semaphore: asyncio.Semaphore,
    is_city: bool = False,
    web_search_prompt: Optional[str] = None  # REMOVED enable_web_search
) -> Tuple[Dict, bool]:
    """Generate a summary for a single item (landmark or city)."""
    wiki_url = item.get("wikipedia_url")
    if not wiki_url:
        return item, False
    
    wiki_content = await fetch_wikipedia_content_async(session, wiki_url)
    
    if wiki_content:
        summary = await call_openrouter_async(
            session,
            wiki_content, 
            api_key, 
            prompt_template,
            model=model,
            semaphore=semaphore,
            web_search_prompt=web_search_prompt
        )
        
        if summary:
            item["llm_summary"] = summary
            return item, True
    
    return item, False


async def generate_summaries_parallel(
    items: List[Dict],
    api_key: str,
    prompt_template: str,
    model: str,
    concurrency: int,
    item_type: str = "landmarks",
    web_search_prompt: Optional[str] = None  # REMOVED enable_web_search
) -> Tuple[List[Dict], int, int]:
    """Generate LLM summaries for multiple items in parallel."""
    
    items_with_wiki = [item for item in items if item.get("wikipedia_url")]
    items_without_wiki = [item for item in items if not item.get("wikipedia_url")]
    
    if not items_with_wiki:
        print(f"  ⚠️  No {item_type} with Wikipedia URLs found")
        return items, 0, 0
    
    print(f"  ℹ️  Processing {len(items_with_wiki)} {item_type} with Wikipedia URLs (skipping {len(items_without_wiki)} without URLs)")
    
    # Check if model has :online suffix
    if ":online" in model.lower():
        print(f"  🌐 Web search enabled (using :online model)")
    
    semaphore = asyncio.Semaphore(concurrency)
    
    success_count = 0
    failed_count = 0
    
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for item in items_with_wiki:
            task = generate_summary_for_item(
                session,
                item,
                api_key,
                prompt_template,
                model,
                semaphore,
                is_city=(item_type == "cities"),
                web_search_prompt=web_search_prompt
            )
            tasks.append(task)
        
        processed_with_wiki = []
        try:
            for coro in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc=f"Generating {item_type} summaries",
                unit="item"
            ):
                try:
                    item, success = await coro
                    processed_with_wiki.append(item)
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
        except asyncio.CancelledError:
            print(f"\n⚠️  Processing cancelled. Completed {success_count} items.")
            raise
    
    all_results = processed_with_wiki + items_without_wiki
    
    return all_results, success_count, failed_count


def generate_llm_summaries(
    landmarks: List[Dict], 
    api_key: str,
    prompt_template: str,
    model: str = DEFAULT_OPENROUTER_MODEL,
    concurrency: int = OPENROUTER_CONCURRENCY,
    web_search_prompt: Optional[str] = None 
) -> List[Dict]:
    """Generate LLM summaries for landmarks with Wikipedia URLs using parallel processing."""
    
    print(f"\n🤖 Generating LLM summaries for landmarks...")
    print(f"    Model: {model}")
    print(f"    Concurrency: {concurrency} parallel requests")
    print(f"    ⚠️  Press Ctrl+C to skip this step.")
    
    success_count = 0
    failed_count = 0
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            processed_landmarks, success_count, failed_count = loop.run_until_complete(
                generate_summaries_parallel(
                    landmarks,
                    api_key,
                    prompt_template,
                    model,
                    concurrency,
                    "landmarks",
                    web_search_prompt=web_search_prompt  # PASS IT THROUGH
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
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Summary generation interrupted by user.")
        print(f"    Generated {success_count} summaries before interruption.")
    except Exception as e:
        print(f"\n❌ Error during summary generation: {e}")
    
    return landmarks


def generate_city_summaries(
    towns: List[Dict],
    api_key: str,
    prompt_template: str,
    model: str = DEFAULT_OPENROUTER_MODEL,
    concurrency: int = OPENROUTER_CONCURRENCY,
    web_search_prompt: Optional[str] = None  # NEW PARAMETER
) -> List[Dict]:
    """Generate LLM summaries for cities with Wikipedia URLs using parallel processing."""
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
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            processed_cities, success_count, failed_count = loop.run_until_complete(
                generate_summaries_parallel(
                    unique_cities,
                    api_key,
                    prompt_template,
                    model,
                    concurrency,
                    "cities",
                    web_search_prompt=web_search_prompt  # PASS IT THROUGH
                )
            )
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
        
        summary_map = {city["name"]: city.get("llm_summary") for city in processed_cities if city.get("llm_summary")}
        
        for town in towns:
            town_name = town["name"]
            if town_name in summary_map:
                town["llm_summary"] = summary_map[town_name]
        
        print(f"\n✅ City summary generation complete:")
        print(f"    • Successful: {success_count}")
        print(f"    • Failed: {failed_count}")
    
    except KeyboardInterrupt:
        print(f"\n⚠️  City summary generation interrupted by user.")
        print(f"    Generated {success_count} summaries before interruption.")
    except Exception as e:
        print(f"\n❌ Error during city summary generation: {e}")
    
    return towns

def parse_landmarks(results: Dict) -> List[Dict]:
    """Parse landmark data from SPARQL results."""
    landmarks = []
    items = results["results"]["bindings"]
    skipped_coords = 0
    
    print("\n📍 Parsing landmark data...")
    for item in tqdm(items, desc="Processing landmarks", unit="landmark"):
        try:
            coord_str = item["coordinate"]["value"]
            coords = coord_str.replace("Point(", "").replace(")", "").split()
            
            if len(coords) != 2:
                skipped_coords += 1
                continue
            
            lon, lat = coords
            
            name = item["itemLabel"]["value"]
            clean_name = clean_landmark_name(name)
            romanized_name = romanize_name(name)
            
            landmark = {
                "wikidata_id": item["item"]["value"].split("/")[-1],
                "name": name,
                "name_clean": clean_name,
                "name_romanized": romanized_name,
                "type": item["typeLabel"]["value"].lower(),
                "latitude": float(lat),
                "longitude": float(lon),
                "wikipedia_url": item.get("wikipedia", {}).get("value", ""),
                "image_url": item.get("image", {}).get("value", ""),
                "osm_node_id": item.get("osmNode", {}).get("value", ""),
                "prefecture": "",
                "nearby_towns": []
            }
            landmarks.append(landmark)
        except (ValueError, KeyError, IndexError) as e:
            # Skip landmarks with invalid or missing coordinates
            skipped_coords += 1
            continue
    
    if skipped_coords > 0:
        print(f"  ⚠️  Skipped {skipped_coords} landmarks with invalid coordinates")
    
    return landmarks

def parse_towns(results: Dict) -> List[Dict]:
    """Parse town/city data from SPARQL results and deduplicate."""
    towns_dict = {}  # Use dict to deduplicate by name
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
            # Clean prefecture name
            prefecture_raw = item.get("prefectureLabel", {}).get("value", "")
            prefecture = clean_prefecture_name(prefecture_raw)
            
            # Check if it's a capital
            is_capital = item.get("isCapital", {}).get("value", "") == "true"
            
            # Get Wikipedia URL
            wikipedia_url = item.get("wikipedia", {}).get("value", "")
            
            # Get image URL
            image_url = item.get("image", {}).get("value", "")
            
            # Parse population safely
            population = None
            point_in_time = None
            
            if "population" in item:
                pop_value = item["population"]["value"]
                try:
                    # Try to parse as float first (handles both int and decimal), then convert to int
                    if not pop_value.startswith("http"):  # Skip URIs
                        population = int(float(pop_value))
                        # Get the date if available
                        if "pointInTime" in item:
                            point_in_time = item["pointInTime"]["value"]
                except (ValueError, TypeError):
                    skipped_populations += 1
                    pass  # Keep population as None if conversion fails
            
            if population is None:
                skipped_null_pop += 1
                continue  # Skip items with null population
            
            # Create unique key for deduplication (name + prefecture + coordinates)
            town_key = f"{town_name}_{prefecture}_{round(float(lat), 3)}_{round(float(lon), 3)}"
            
            # If we already have this town, keep the one with the most recent/largest population
            if town_key in towns_dict:
                existing = towns_dict[town_key]
                # Prefer the entry with:
                # 1. Capital status over non-capital
                # 2. Wikipedia URL over no URL
                # 3. Image URL over no image
                # 4. More recent date (if both have dates)
                # 5. Higher population (if no dates or same date)
                # 6. Having a prefecture over not having one
                
                should_replace = False
                
                # Capital status takes precedence
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
                        # Compare dates (more recent is better)
                        should_replace = point_in_time > existing.get("point_in_time", "")
                    else:
                        # Compare population (higher is likely more recent)
                        should_replace = population > existing.get("population", 0)
                elif population is not None and not existing.get("population"):
                    should_replace = True
                
                if not should_replace:
                    continue
            
            town_data = {
                "name": town_name,
                "latitude": float(lat),
                "longitude": float(lon),
                "prefecture": prefecture,
                "population": population,
                "is_capital": is_capital,
                "wikipedia_url": wikipedia_url,
                "image_url": image_url,
                "point_in_time": point_in_time  # Keep for deduplication, won't include in output
            }
            
            towns_dict[town_key] = town_data
        except (ValueError, KeyError, IndexError) as e:
            # Skip towns with invalid or missing coordinates
            skipped_coords += 1
            continue
    
    # Convert back to list and remove point_in_time
    towns = []
    for town in towns_dict.values():
        town.pop("point_in_time", None)  # Remove internal field
        towns.append(town)
    
    if skipped_coords > 0:
        print(f"  ⚠️  Skipped {skipped_coords} towns with invalid coordinates")
    if skipped_populations > 0:
        print(f"  ⚠️  Skipped {skipped_populations} non-numeric population values")
    if skipped_null_pop > 0:
        print(f"  ⚠️  Skipped {skipped_null_pop} towns with null population")
    
    # Count major cities and those with Wikipedia/images
    major_cities = [t for t in towns if t.get("is_capital") or (t.get("population") is not None and t.get("population") > MAJOR_CITY_POPULATION_THRESHOLD)]
    cities_with_wiki = [t for t in towns if t.get("wikipedia_url")]
    cities_with_images = [t for t in towns if t.get("image_url")]
    
    print(f"  ℹ️  Deduplicated to {len(towns)} unique towns (from {len(items)} entries)")
    print(f"  ℹ️  Found {len(major_cities)} major cities (capitals or population > {MAJOR_CITY_POPULATION_THRESHOLD:,})")
    print(f"  ℹ️  Found {len(cities_with_wiki)} cities with Wikipedia articles")
    print(f"  ℹ️  Found {len(cities_with_images)} cities with images")
    
    return towns

def fetch_additional_images(landmarks: List[Dict], skip_additional: bool = False,
                           cache: Optional[Dict] = None, cache_expiry_days: int = DEFAULT_CACHE_EXPIRY_DAYS,
                           force_refresh: bool = False) -> List[Dict]:
    """Fetch additional images for landmarks without images."""
    landmarks_without_images = [lm for lm in landmarks if not lm.get("image_url")]
    
    if not landmarks_without_images or skip_additional:
        if skip_additional and landmarks_without_images:
            print(f"\n🖼️  Skipping additional image search for {len(landmarks_without_images)} landmarks (use --fetch-all-images to enable)")
        return landmarks
    
    print(f"\n🖼️  Fetching additional images for {len(landmarks_without_images)} landmarks...")
    print("    ⚠️  This may take a while due to rate limiting. Press Ctrl+C to skip this step.")
    
    batch_size = 20  # Process in batches to reduce queries
    batches = [landmarks_without_images[i:i + batch_size] for i in range(0, len(landmarks_without_images), batch_size)]
    
    try:
        for batch_idx, batch in enumerate(tqdm(batches, desc="Image batches", unit="batch")):
            wikidata_ids = [lm["wikidata_id"] for lm in batch]
            ids_str = " ".join([f"wd:{wid}" for wid in wikidata_ids])
            
            # Batch query for Commons categories
            query = f"""
            SELECT ?item ?category WHERE {{
              VALUES ?item {{ {ids_str} }}
              OPTIONAL {{ ?item wdt:P373 ?category. }}
            }}
            """
            
            try:
                # Check cache first
                use_cached = False
                if cache is not None and not force_refresh:
                    cache_key = get_cache_key(query)
                    if cache_key in cache["data"]:
                        cache_entry = cache["data"][cache_key]
                        if is_cache_valid(cache_entry, cache_expiry_days):
                            results = cache_entry["results"]
                            use_cached = True
                
                if not use_cached:
                    results = fetch_wikidata(query, delay=True, cache=cache, 
                                            cache_expiry_days=cache_expiry_days,
                                            force_refresh=force_refresh)
                
                # Map results back to landmarks
                for binding in results["results"]["bindings"]:
                    item_id = binding["item"]["value"].split("/")[-1]
                    if "category" in binding:
                        # Find the corresponding landmark
                        for lm in batch:
                            if lm["wikidata_id"] == item_id:
                                lm["commons_category"] = binding["category"]["value"]
                                break
                
                # Add delay between batches only if not using cache
                if not use_cached:
                    time.sleep(REQUEST_DELAY * 2)
                
            except Exception as e:
                print(f"\n⚠️  Failed to fetch images for batch {batch_idx + 1}: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\n⚠️  Image fetching interrupted by user. Continuing with available data...")
    
    return landmarks

from scipy.spatial import KDTree

def assign_nearby_towns(landmarks: List[Dict], towns: List[Dict]) -> List[Dict]:
    """Assign nearby towns and closest major city to each landmark"""
    print("\n🗺️  Calculating nearby towns for landmarks...")
    
    major_cities = [t for t in towns if t.get("is_capital") or (t.get("population") is not None and t.get("population") > MAJOR_CITY_POPULATION_THRESHOLD)]
    print(f"  ℹ️  Using {len(major_cities)} major cities for closest major city calculation")
    
    # Build KDTrees
    town_coords = np.array([[t["latitude"], t["longitude"]] for t in towns])
    town_tree = KDTree(town_coords)
    
    if major_cities:
        major_coords = np.array([[c["latitude"], c["longitude"]] for c in major_cities])
        major_tree = KDTree(major_coords)
    
    for lm in tqdm(landmarks, desc="Finding nearby towns", unit="landmark"):
        lm_coord = np.array([lm["latitude"], lm["longitude"]])
        
        # Find closest town using KDTree
        closest_dist_deg, closest_idx = town_tree.query(lm_coord)
        closest_town = towns[closest_idx]
        
        # Get kilometers from degrees
        closest_dist_km = geodesic(
            (lm["latitude"], lm["longitude"]),
            (closest_town["latitude"], closest_town["longitude"])
        ).km
        
        # Set prefecture from closest town
        if closest_town["prefecture"]:
            lm["prefecture"] = closest_town["prefecture"]
        
        # We consider every town within 4km of the closest to also be a nearby town
        radius_km = closest_dist_km + 4
        # Approximate conversion based on Japan's latitude (thanks ChatGPT)
        radius_deg = radius_km / 85.0
        
        # Query all towns within radius
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
                if town_name not in nearby_towns_dict or dist_km < nearby_towns_dict[town_name]["distance_km"]:
                    # Include image_url if available
                    nearby_town_data = {
                        "name": town_name,
                        "distance_km": round(dist_km, 2),
                        "prefecture": town["prefecture"],
                        "population": town.get("population")
                    }
                    if town.get("image_url"):
                        nearby_town_data["image_url"] = town["image_url"]
                    
                    nearby_towns_dict[town_name] = nearby_town_data
        
        nearby_towns = list(nearby_towns_dict.values())
        nearby_towns.sort(key=lambda x: x["distance_km"])
        
        # Limit to 10 nearest unique towns
        lm["nearby_towns"] = nearby_towns[:10]
        lm["closest_town_distance_km"] = round(closest_dist_km, 2)
        
        # Find closest major city
        if major_cities:
            _, major_idx = major_tree.query(lm_coord)
            closest_major = major_cities[major_idx]
            closest_major_dist = geodesic(
                (lm["latitude"], lm["longitude"]),
                (closest_major["latitude"], closest_major["longitude"])
            ).km
            
            closest_major_data = {
                "name": closest_major["name"],
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
        "mountains": sum(1 for lm in landmarks if "mountain" in lm["type"]),
        "lakes": sum(1 for lm in landmarks if "lake" in lm["type"]),
        "shrines": sum(1 for lm in landmarks if "shrine" in lm["type"].lower()),
        "with_wikipedia": sum(1 for lm in landmarks if lm.get("wikipedia_url")),
        "with_images": sum(1 for lm in landmarks if lm.get("image_url")),
        "with_osm_node": sum(1 for lm in landmarks if lm.get("osm_node_id")),
        "with_llm_summary": sum(1 for lm in landmarks if lm.get("llm_summary")),
        "prefectures": len(set(lm["prefecture"] for lm in landmarks if lm["prefecture"])),
        "with_major_city": sum(1 for lm in landmarks if lm.get("closest_major_city")),
        "total_cities": len(towns),
        "cities_with_wikipedia": sum(1 for t in towns if t.get("wikipedia_url")),
        "cities_with_images": sum(1 for t in towns if t.get("image_url")),
        "cities_with_llm_summary": sum(1 for t in towns if t.get("llm_summary"))
    }
    return stats

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Collect Japanese landmarks data from Wikidata")
    parser.add_argument("--skip-images", action="store_true", 
                       help="Skip the additional image fetching step")
    parser.add_argument("--output", default="japan_landmarks.json",
                       help="Output filename (default: japan_landmarks.json)")
    parser.add_argument("--email", type=str,
                       help="Your email for the User-Agent header (required by Wikimedia)")
    
    # LLM summarization arguments
    parser.add_argument("--generate-summaries", action="store_true",
                       help="Generate LLM summaries for landmarks using OpenRouter")
    parser.add_argument("--openrouter-api-key", type=str,
                       help="OpenRouter API key for LLM summarization")
    parser.add_argument("--llm-model", type=str, default=DEFAULT_OPENROUTER_MODEL,
                       help=f"OpenRouter model to use (default: {DEFAULT_OPENROUTER_MODEL})")
    parser.add_argument("--llm-prompt", type=str, default=DEFAULT_PROMPT_TEMPLATE,
                       help="Prompt template for LLM (use {wikipedia_content} as placeholder)")
    parser.add_argument("--llm-concurrency", type=int, default=OPENROUTER_CONCURRENCY,
                       help=f"Number of parallel LLM requests (default: {OPENROUTER_CONCURRENCY})")
    parser.add_argument("--generate-city-summaries", action="store_true",
                       help="Generate LLM summaries for cities/towns (requires --generate-summaries)")
    parser.add_argument("--enable-web-search", action="store_true",
                       help="Enable OpenRouter web search plugin for additional research")
    parser.add_argument("--web-search-prompt", type=str, default=DEFAULT_WEB_SEARCH_PROMPT,
                       help="Custom prompt for web search results (overrides OpenRouter default)")
    
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
        USER_AGENT = f"JapanLandmarksCollector/1.0 (https://github.com/yourusername/japan-landmarks; {args.email}) python-requests/2.31"
        HEADERS["User-Agent"] = USER_AGENT
    
    print("=" * 60)
    print("🗾 Japanese Landmarks Data Collector")
    print("=" * 60)
    
    # Check if User-Agent is properly configured
    if "your.email@example.com" in USER_AGENT:
        print("\n⚠️  WARNING: Please provide your email for the User-Agent header!")
        print("    This is required by Wikimedia's User-Agent Policy.")
        print("    Use: python main.py --email your.email@example.com")
        print("    Or edit the USER_AGENT variable in the script.")
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
        total_steps = 5
        if args.generate_summaries:
            total_steps += 1
        if args.generate_city_summaries:
            total_steps += 1
        
        # Fetch landmarks
        print(f"\n📊 Step 1/{total_steps}: Fetching landmarks from Wikidata...")
        landmark_results = fetch_wikidata(
            LANDMARK_QUERY, 
            "Fetching landmarks", 
            delay=False,
            cache=cache,
            cache_expiry_days=args.cache_expiry_days if not args.no_cache else 0,
            force_refresh=args.force_refresh
        )
        landmarks = parse_landmarks(landmark_results)
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
        
        # Try to fetch additional images (optional)
        print(f"\n📊 Step 4/{total_steps}: Fetching additional images...")
        landmarks = fetch_additional_images(
            landmarks, 
            skip_additional=args.skip_images,
            cache=cache,
            cache_expiry_days=args.cache_expiry_days if not args.no_cache else 0,
            force_refresh=args.force_refresh
        )
        print("✅ Image search completed")
        
        # Save cache after Wikidata queries
        if cache is not None:
            print(f"\n💾 Saving cache to {args.cache_file}...")
            save_cache(cache, args.cache_file)
            print(f"  ✓ Cache saved: {len(cache.get('data', {}))} entries")
        
        # Generate LLM summaries (optional)
        current_step = 5
        if args.generate_summaries:
            print(f"\n📊 Step {current_step}/{total_steps}: Generating landmark LLM summaries...")
            
            # Use web_search_prompt only if model has :online suffix
            search_prompt = args.web_search_prompt if ":online" in args.llm_model.lower() else None
            
            landmarks = generate_llm_summaries(
                landmarks,
                api_key=args.openrouter_api_key,
                prompt_template=args.llm_prompt,
                model=args.llm_model,
                concurrency=args.llm_concurrency,
                web_search_prompt=search_prompt
            )
            print("✅ Landmark LLM summary generation completed")
            current_step += 1
        
        # Generate city summaries (optional)
        if args.generate_city_summaries:
            print(f"\n📊 Step {current_step}/{total_steps}: Generating city LLM summaries...")
            towns = generate_city_summaries(
                towns,
                api_key=args.openrouter_api_key,
                prompt_template=args.llm_prompt,
                model=args.llm_model,
                concurrency=args.llm_concurrency,
                web_search_prompt=search_prompt
            )
            print("✅ City LLM summary generation completed")
            current_step += 1
        
        # Generate statistics
        print(f"\n📊 Step {current_step}/{total_steps}: Generating statistics...")
        stats = generate_statistics(landmarks, towns)
        
        # Save to JSON
        output_data = {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "statistics": stats,
                "user_agent": USER_AGENT,
                "cache_info": {
                    "enabled": not args.no_cache,
                    "cache_file": args.cache_file if not args.no_cache else None,
                    "cache_entries": len(cache.get("data", {})) if cache else 0,
                    "cache_expiry_days": args.cache_expiry_days if not args.no_cache else None
                } if not args.no_cache else {"enabled": False},
                "llm_config": {
                    "enabled": args.generate_summaries,
                    "model": args.llm_model if args.generate_summaries else None,
                    "concurrency": args.llm_concurrency if args.generate_summaries else None,
                    "city_summaries_enabled": args.generate_city_summaries
                } if args.generate_summaries else None
            },
            "landmarks": landmarks,
            "cities": towns
        }
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 60)
        print("📈 Collection Statistics:")
        print(f"  • Total landmarks: {stats['total_landmarks']}")
        print(f"  • Mountains: {stats['mountains']}")
        print(f"  • Lakes: {stats['lakes']}")
        print(f"  • Shinto Shrines: {stats['shrines']}")
        print(f"  • With Wikipedia articles: {stats['with_wikipedia']}")
        print(f"  • With images: {stats['with_images']}")
        print(f"  • With OSM node IDs: {stats['with_osm_node']}")
        if args.generate_summaries:
            print(f"  • With LLM summaries: {stats['with_llm_summary']}")
        print(f"  • With major city info: {stats['with_major_city']}")
        print(f"  • Prefectures covered: {stats['prefectures']}")
        print(f"\n  • Total cities/towns: {stats['total_cities']}")
        print(f"  • Cities with Wikipedia: {stats['cities_with_wikipedia']}")
        print(f"  • Cities with images: {stats['cities_with_images']}")
        if args.generate_city_summaries:
            print(f"  • Cities with LLM summaries: {stats['cities_with_llm_summary']}")
        print("=" * 60)
        print(f"\n✅ Data saved to: {args.output}")
        
        if args.skip_images and stats['with_images'] < stats['total_landmarks']:
            print(f"\n💡 Tip: Run without --skip-images to fetch images for remaining {stats['total_landmarks'] - stats['with_images']} landmarks")
        
        if not args.generate_summaries and stats['with_wikipedia'] > 0:
            print(f"\n💡 Tip: Use --generate-summaries to create AI-generated descriptions for {stats['with_wikipedia']} landmarks with Wikipedia articles")
        
        if args.generate_summaries and not args.generate_city_summaries and stats['cities_with_wikipedia'] > 0:
            print(f"\n💡 Tip: Use --generate-city-summaries to create AI-generated descriptions for {stats['cities_with_wikipedia']} cities with Wikipedia articles")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()