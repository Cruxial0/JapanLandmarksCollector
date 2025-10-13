import asyncio
import aiohttp
from typing import Dict, List, Optional
from tqdm import tqdm

IMAGE_SOURCES = {
    'wikimedia_commons': {
        'enabled': True,
        'priority': 1,  # Highest priority - free, well-curated
        'api_url': 'https://commons.wikimedia.org/w/api.php'
    },
    # Use with caution, never tried them
    'flickr': {
        'enabled': True,
        'priority': 2,
        'api_url': 'https://www.flickr.com/services/rest/',
        'requires_key': True
    },
    'unsplash': {
        'enabled': True,
        'priority': 3,
        'api_url': 'https://api.unsplash.com',
        'requires_key': True
    }
}

async def fetch_commons_images(
    session: aiohttp.ClientSession,
    landmark_name: str,
    category: str = None,
    wikidata_id: str = None
) -> List[Dict]:
    """
    Fetch images from Wikimedia Commons
    """
    images = []
    api_url = IMAGE_SOURCES['wikimedia_commons']['api_url']
    
    try:
        # Search by Commons category
        if category:
            params = {
                'action': 'query',
                'format': 'json',
                'generator': 'categorymembers',
                'gcmtitle': f'Category:{category}',
                'gcmtype': 'file',
                'gcmlimit': 10,
                'prop': 'imageinfo',
                'iiprop': 'url|extmetadata',
                'iiurlwidth': 1024
            }
            
            async with session.get(api_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.ok:
                    data = await response.json()
                    pages = data.get('query', {}).get('pages', {})
                    
                    for page in pages.values():
                        imageinfo = page.get('imageinfo', [])
                        if imageinfo:
                            info = imageinfo[0]
                            url = info.get('thumburl') or info.get('url')
                            if url:
                                # Extract metadata
                                metadata = info.get('extmetadata', {})
                                description = metadata.get('ImageDescription', {}).get('value', '')
                                artist = metadata.get('Artist', {}).get('value', '')
                                license_info = metadata.get('LicenseShortName', {}).get('value', '')
                                
                                images.append({
                                    'url': url,
                                    'full_url': info.get('url'),
                                    'source': 'wikimedia_commons',
                                    'description': description,
                                    'photographer': artist,
                                    'license': license_info,
                                    'priority': IMAGE_SOURCES['wikimedia_commons']['priority']
                                })
        
        # Search by Wikidata ID
        if wikidata_id and len(images) < 5:
            wdq_params = {
                'action': 'query',
                'format': 'json',
                'generator': 'images',
                'titles': f'Q{wikidata_id}' if not wikidata_id.startswith('Q') else wikidata_id,
                'gimlimit': 10,
                'prop': 'imageinfo',
                'iiprop': 'url|extmetadata',
                'iiurlwidth': 1024
            }
            
            async with session.get(api_url, params=wdq_params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.ok:
                    data = await response.json()
                    pages = data.get('query', {}).get('pages', {})
                    
                    for page in pages.values():
                        imageinfo = page.get('imageinfo', [])
                        if imageinfo:
                            info = imageinfo[0]
                            url = info.get('thumburl') or info.get('url')
                            
                            # Skip if we already have this URL
                            if url and not any(img['url'] == url for img in images):
                                metadata = info.get('extmetadata', {})
                                description = metadata.get('ImageDescription', {}).get('value', '')
                                
                                images.append({
                                    'url': url,
                                    'full_url': info.get('url'),
                                    'source': 'wikimedia_commons',
                                    'description': description,
                                    'priority': IMAGE_SOURCES['wikimedia_commons']['priority']
                                })
        
        # Text search (fallback)
        if len(images) < 3:
            search_params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': f'{landmark_name} Japan',
                'srnamespace': 6,  # File namespace
                'srlimit': 5
            }
            
            async with session.get(api_url, params=search_params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.ok:
                    data = await response.json()
                    results = data.get('query', {}).get('search', [])
                    
                    for result in results:
                        title = result.get('title', '').replace('File:', '')
                        
                        # Get image URL
                        info_params = {
                            'action': 'query',
                            'format': 'json',
                            'titles': f'File:{title}',
                            'prop': 'imageinfo',
                            'iiprop': 'url|extmetadata',
                            'iiurlwidth': 1024
                        }
                        
                        async with session.get(api_url, params=info_params, timeout=aiohttp.ClientTimeout(total=10)) as info_response:
                            if info_response.ok:
                                info_data = await info_response.json()
                                pages = info_data.get('query', {}).get('pages', {})
                                for page in pages.values():
                                    imageinfo = page.get('imageinfo', [])
                                    if imageinfo:
                                        info = imageinfo[0]
                                        url = info.get('thumburl') or info.get('url')
                                        
                                        if url and not any(img['url'] == url for img in images):
                                            metadata = info.get('extmetadata', {})
                                            description = metadata.get('ImageDescription', {}).get('value', '')
                                            
                                            images.append({
                                                'url': url,
                                                'full_url': info.get('url'),
                                                'source': 'wikimedia_commons',
                                                'description': description,
                                                'priority': IMAGE_SOURCES['wikimedia_commons']['priority']
                                            })
                                            
                                            if len(images) >= 5:
                                                break
    
    except Exception as e:
        pass
    
    return images[:10]  # Limit to 10 images per source

async def fetch_flickr_images(
    session: aiohttp.ClientSession,
    landmark_name: str,
    lat: float,
    lon: float,
    api_key: str
) -> List[Dict]:
    """
    Fetch images from Flickr.
    """
    if not api_key:
        return []
    
    images = []
    api_url = IMAGE_SOURCES['flickr']['api_url']
    
    try:
        # Search with geolocation
        params = {
            'method': 'flickr.photos.search',
            'api_key': api_key,
            'text': landmark_name,
            'lat': lat,
            'lon': lon,
            'radius': 1,
            'radius_units': 'km',
            'format': 'json',
            'nojsoncallback': 1,
            'per_page': 10,
            'sort': 'relevance',
            'content_type': 1,  # Photos
            'media': 'photos',
            'extras': 'description,owner_name,license,url_b,url_c,url_l,geo'  # Get multiple sizes
        }
        
        async with session.get(api_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.ok:
                data = await response.json()
                photos = data.get('photos', {}).get('photo', [])
                
                for photo in photos:
                    # Prefer larger sizes
                    url = photo.get('url_l') or photo.get('url_c') or photo.get('url_b')
                    
                    if not url:
                        url = f"https://live.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}_b.jpg"
                    
                    if photo.get('latitude') and photo.get('longitude'):
                        lat_diff = abs(float(photo['latitude']) - lat)
                        lon_diff = abs(float(photo['longitude']) - lon)
                        
                        if lat_diff < 0.01 and lon_diff < 0.01:
                            images.append({
                                'url': url,
                                'source': 'flickr',
                                'description': photo.get('description', {}).get('_content', photo.get('title', '')),
                                'photographer': photo.get('ownername', ''),
                                'license': photo.get('license', ''),
                                'flickr_id': photo['id'],
                                'priority': IMAGE_SOURCES['flickr']['priority']
                            })
                    else:
                        # If no geo data, include anyway but lower priority (filtered out if >10 images are found)
                        images.append({
                            'url': url,
                            'source': 'flickr',
                            'description': photo.get('description', {}).get('_content', photo.get('title', '')),
                            'photographer': photo.get('ownername', ''),
                            'priority': IMAGE_SOURCES['flickr']['priority'] + 1  # Lower priority
                        })
    
    except Exception as e:
        pass
    
    return images[:10]

async def fetch_unsplash_images(
    session: aiohttp.ClientSession,
    landmark_name: str,
    location_context: str = "Japan",
    api_key: str = None
) -> List[Dict]:
    """
    Fetch images from Unsplash.
    """
    if not api_key:
        return []
    
    images = []
    api_url = f"{IMAGE_SOURCES['unsplash']['api_url']}/search/photos"
    
    try:
        headers = {
            'Authorization': f'Client-ID {api_key}',
            'Accept-Version': 'v1'
        }
        
        queries = [
            f"{landmark_name} {location_context}",
            landmark_name,
            f"{landmark_name.split()[0]} {location_context}"
        ]
        
        for query in queries:
            params = {
                'query': query,
                'per_page': 5,
                'orientation': 'landscape',
                'content_filter': 'high'  # Should be fine for cities/landmarks?
            }
            
            async with session.get(api_url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.ok:
                    data = await response.json()
                    results = data.get('results', [])
                    
                    for result in results:
                        urls = result.get('urls', {})
                        url = urls.get('regular') or urls.get('small')
                        
                        if url:
                            # Check for duplicates
                            if not any(img['url'] == url for img in images):
                                images.append({
                                    'url': url,
                                    'full_url': urls.get('full'),
                                    'source': 'unsplash',
                                    'description': result.get('description') or result.get('alt_description', ''),
                                    'photographer': result.get('user', {}).get('name', ''),
                                    'photographer_url': result.get('user', {}).get('links', {}).get('html', ''),
                                    'unsplash_id': result.get('id'),
                                    'priority': IMAGE_SOURCES['unsplash']['priority']
                                })
                    
                    if images:
                        break
    
    except Exception as e:
        pass
    
    return images[:5]

async def gather_images_for_landmark(
    session: aiohttp.ClientSession,
    landmark: Dict,
    flickr_api_key: Optional[str] = None,
    unsplash_api_key: Optional[str] = None
) -> List[Dict]:
    """
    Gather images from all available sources for a landmark.
    
    Returns a sorted list of image dicts, prioritized by:
    1. Source priority (Commons > Flickr > Unsplash)
    2. Relevance to the landmark
    3. Image quality/metadata completeness
    """
    
    tasks = []
    
    if IMAGE_SOURCES['wikimedia_commons']['enabled']:
        tasks.append(fetch_commons_images(
            session,
            landmark['name'],
            landmark.get('commons_category'),
            landmark.get('wikidata_id')
        ))
    
    if flickr_api_key and IMAGE_SOURCES['flickr']['enabled']:
        tasks.append(fetch_flickr_images(
            session,
            landmark['name'],
            landmark['latitude'],
            landmark['longitude'],
            flickr_api_key
        ))
    
    if unsplash_api_key and IMAGE_SOURCES['unsplash']['enabled']:
        location_ctx = f"{landmark.get('prefecture', '')} Japan".strip()
        tasks.append(fetch_unsplash_images(
            session,
            landmark['name'],
            location_ctx,
            unsplash_api_key
        ))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Flatten results
    all_images = []
    for result in results:
        if isinstance(result, list):
            all_images.extend(result)
    
    # Deduplicate by URL
    seen_urls = set()
    unique_images = []
    for img in all_images:
        if img['url'] not in seen_urls:
            seen_urls.add(img['url'])
            unique_images.append(img)
    
    # Sort by priority (lower number = higher priority)
    unique_images.sort(key=lambda x: x['priority'])
    
    return unique_images

async def enrich_landmarks_with_images(
    landmarks: List[Dict],
    flickr_api_key: Optional[str] = None,
    unsplash_api_key: Optional[str] = None,
    max_concurrent: int = 10
) -> List[Dict]:
    """
    Enrich all landmarks with images from configured sources
    """
    
    print(f"\n📸 Gathering images from configured sources...")
    
    active_sources = []
    if IMAGE_SOURCES['wikimedia_commons']['enabled']:
        active_sources.append('Wikimedia Commons')
    if flickr_api_key and IMAGE_SOURCES['flickr']['enabled']:
        active_sources.append('Flickr')
    if unsplash_api_key and IMAGE_SOURCES['unsplash']['enabled']:
        active_sources.append('Unsplash')
    
    print(f"   Active sources: {', '.join(active_sources)}")
    
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Batch processing to not get slapped with rate limits
        batch_size = 20
        pbar = tqdm(total=len(landmarks), desc="Gathering images", unit="landmark")
        
        for i in range(0, len(landmarks), batch_size):
            batch = landmarks[i:i + batch_size]
            
            tasks = []
            for landmark in batch:
                if len(landmark.get('alternative_images', [])) >= 10:
                    pbar.update(1)
                    continue
                
                task = gather_images_for_landmark(
                    session,
                    landmark,
                    flickr_api_key,
                    unsplash_api_key
                )
                tasks.append((landmark, task))
            
            if not tasks:
                continue
            
            results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
            
            # Update landmarks
            for (landmark, _), images in zip(tasks, results):
                if isinstance(images, list) and images:
                    landmark['alternative_images'] = images
                    landmark['total_images'] = 1 + len(images) if landmark.get('image_url') else len(images)
                pbar.update(1)
            
            if i + batch_size < len(landmarks):
                await asyncio.sleep(1)
    
    with_alt_images = sum(1 for lm in landmarks if lm.get('alternative_images'))
    total_images = sum(len(lm.get('alternative_images', [])) for lm in landmarks)
    
    print(f"   ✓ Found {total_images} alternative images for {with_alt_images} landmarks")
    
    return landmarks