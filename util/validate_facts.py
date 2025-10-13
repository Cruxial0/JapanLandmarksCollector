"""
Validation script for the facts section in Japan Landmarks JSON output.
This utility is vibe-coded af xd

Usage:
    python validate_facts.py japan_landmarks.json
"""

import json
import sys
from collections import Counter
from typing import Dict, List, Set


def validate_facts_section(data: Dict) -> Dict[str, any]:
    """Validate the facts section and return statistics."""
    
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check if facts section exists
    if 'facts' not in data:
        results['errors'].append("Missing 'facts' section in JSON")
        results['valid'] = False
        return results
    
    facts = data['facts']
    
    # Basic stats
    results['stats']['total_entries'] = len(facts)
    
    # Validate each fact entry
    landmark_ids = set()
    city_ids = set()
    source_usage = Counter()
    url_source_usage = Counter()
    entries_with_summaries = 0
    entries_without_summaries = 0
    
    for loc_id, fact_data in facts.items():
        # Check required fields
        required_fields = ['wikidata_id', 'name', 'text_sources', 'url_sources', 'formatted_prompt', 'llm_summary']
        for field in required_fields:
            if field not in fact_data:
                results['errors'].append(f"Missing '{field}' in fact entry: {loc_id}")
                results['valid'] = False
        
        # All entries should have wikidata_id now
        if not fact_data.get('wikidata_id'):
            results['errors'].append(f"Missing wikidata_id in fact entry: {loc_id}")
            results['valid'] = False
        
        # Check if it's a city (has prefecture field) or landmark
        if 'prefecture' in fact_data and fact_data.get('prefecture'):
            city_ids.add(loc_id)
        else:
            landmark_ids.add(loc_id)
        
        # Count source usage
        for source in fact_data.get('text_sources', {}).keys():
            base_source = source.replace('_infobox', '').replace('_tags', '')
            source_usage[base_source] += 1
        
        for source in fact_data.get('url_sources', {}).keys():
            url_source_usage[source] += 1
        
        # Check if summary exists
        if fact_data.get('llm_summary'):
            entries_with_summaries += 1
        else:
            entries_without_summaries += 1
        
        # Validate formatted_prompt is not empty
        if not fact_data.get('formatted_prompt') or len(fact_data.get('formatted_prompt', '')) < 50:
            results['warnings'].append(f"Suspiciously short formatted_prompt for: {loc_id}")
        
        # Check text_sources and url_sources are dicts
        if not isinstance(fact_data.get('text_sources'), dict):
            results['errors'].append(f"text_sources is not a dict for: {loc_id}")
            results['valid'] = False
        
        if not isinstance(fact_data.get('url_sources'), dict):
            results['errors'].append(f"url_sources is not a dict for: {loc_id}")
            results['valid'] = False
    
    # Store detailed stats
    results['stats']['landmark_entries'] = len(landmark_ids)
    results['stats']['city_entries'] = len(city_ids)
    results['stats']['entries_with_summaries'] = entries_with_summaries
    results['stats']['entries_without_summaries'] = entries_without_summaries
    results['stats']['source_usage'] = dict(source_usage.most_common())
    results['stats']['url_source_usage'] = dict(url_source_usage.most_common())
    
    # Cross-reference with landmarks and cities
    if 'landmarks' in data:
        landmark_wikidata_ids = {lm.get('wikidata_id') for lm in data['landmarks'] if lm.get('wikidata_id')}
        facts_landmark_ids = {f.get('wikidata_id') for f in facts.values() if f.get('wikidata_id')}
        
        missing_in_facts = landmark_wikidata_ids - facts_landmark_ids
        extra_in_facts = facts_landmark_ids - landmark_wikidata_ids
        
        results['stats']['landmarks_in_data'] = len(landmark_wikidata_ids)
        results['stats']['landmarks_missing_facts'] = len(missing_in_facts)
        results['stats']['extra_facts_not_in_landmarks'] = len(extra_in_facts)
        
        if len(missing_in_facts) > 0:
            results['warnings'].append(f"{len(missing_in_facts)} landmarks don't have fact entries (this is normal if they lack Wikipedia/Wikivoyage URLs)")
    
    if 'cities' in data:
        results['stats']['cities_in_data'] = len(data['cities'])
        unique_city_names = len(set(city['name'] for city in data['cities']))
        results['stats']['unique_cities'] = unique_city_names
        
        if results['stats']['city_entries'] < unique_city_names:
            results['warnings'].append(f"Only {results['stats']['city_entries']} cities have fact entries out of {unique_city_names} unique cities")
    
    return results


def print_results(results: Dict):
    """Pretty print validation results."""
    
    print("\n" + "=" * 60)
    print("FACTS SECTION VALIDATION RESULTS")
    print("=" * 60)
    
    if results['valid']:
        print("\n✅ VALIDATION PASSED")
    else:
        print("\n❌ VALIDATION FAILED")
    
    # Print errors
    if results['errors']:
        print(f"\n🚨 Errors ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"  • {error}")
    
    # Print warnings
    if results['warnings']:
        print(f"\n⚠️  Warnings ({len(results['warnings'])}):")
        for warning in results['warnings'][:10]:  # Show first 10
            print(f"  • {warning}")
        if len(results['warnings']) > 10:
            print(f"  ... and {len(results['warnings']) - 10} more warnings")
    
    # Print statistics
    print("\n📊 Statistics:")
    stats = results['stats']
    
    print(f"\nOverall:")
    print(f"  • Total fact entries: {stats.get('total_entries', 0)}")
    print(f"  • Landmark entries: {stats.get('landmark_entries', 0)}")
    print(f"  • City entries: {stats.get('city_entries', 0)}")
    print(f"  • Entries with LLM summaries: {stats.get('entries_with_summaries', 0)}")
    print(f"  • Entries without summaries: {stats.get('entries_without_summaries', 0)}")
    
    if 'landmarks_in_data' in stats:
        print(f"\nLandmark Coverage:")
        print(f"  • Landmarks in data: {stats['landmarks_in_data']}")
        print(f"  • Landmarks with facts: {stats['landmark_entries']}")
        coverage = (stats['landmark_entries'] / stats['landmarks_in_data'] * 100) if stats['landmarks_in_data'] > 0 else 0
        print(f"  • Coverage: {coverage:.1f}%")
    
    if 'cities_in_data' in stats:
        print(f"\nCity Coverage:")
        print(f"  • Cities in data: {stats['cities_in_data']}")
        print(f"  • Unique cities: {stats['unique_cities']}")
        print(f"  • Cities with facts: {stats['city_entries']}")
        coverage = (stats['city_entries'] / stats['unique_cities'] * 100) if stats['unique_cities'] > 0 else 0
        print(f"  • Coverage: {coverage:.1f}%")
    
    if stats.get('source_usage'):
        print(f"\nText Source Usage:")
        for source, count in sorted(stats['source_usage'].items(), key=lambda x: x[1], reverse=True):
            print(f"  • {source}: {count} entries")
    
    if stats.get('url_source_usage'):
        print(f"\nURL Source Usage:")
        for source, count in sorted(stats['url_source_usage'].items(), key=lambda x: x[1], reverse=True):
            print(f"  • {source}: {count} URLs")
    
    print("\n" + "=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_facts.py <json_file>")
        print("\nExample:")
        print("  python validate_facts.py japan_landmarks.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    try:
        print(f"Loading {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("Validating facts section...")
        results = validate_facts_section(data)
        
        print_results(results)
        
        # Exit with appropriate code
        sys.exit(0 if results['valid'] else 1)
        
    except FileNotFoundError:
        print(f"❌ Error: File not found: {json_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()