# Japan Landmarks Collector
This is a small project I put together over the course of a few days. It uses a multi-step process to collect all landmarks of a certain type from WikiData, then does some calculation to find the closest city to those landmarks.

It was created to build a geography minigame where users compete to narrow down the location of a landmark by using Japanese regions, prefectures, major cities (arbitrarily defined at >290k population) and closest towns/cities.

## Features
The program is configurable through a set of commandline arguments. You can optionally enable an automated LLM step that fetches all the wikipedia articles for data entries that has them, then performs a series of parallelized LLM requests through OpenRouter to give you a summary of each landmark/city. 

You can optionally enable web search by appending `:online` to the model you use. OpenRouter states that they support web search for all their model. Your mileage may vary with this, so be careful. (I personally noticed some misinformation when enabling it)

## Arguments
```
-h, --help            show this help message and exit
--output OUTPUT       Output filename (default: output/japan_landmarks.json)
--email EMAIL         Your email for the User-Agent header (required by Wikimedia)
--landmark-types {mountain,lake,shrine,cave,bridge,park} [{mountain,lake,shrine,cave,bridge,park} ...]
                      Specific landmark types to fetch (default: all). Choices: mountain, lake, shrine, cave, bridge, park
--generate-summaries  Generate LLM summaries for landmarks using OpenRouter
--openrouter-api-key OPENROUTER_API_KEY
                      OpenRouter API key for LLM summarization
--llm-model LLM_MODEL
                      OpenRouter model to use (default: x-ai/grok-4-fast:online)
--llm-prompt LLM_PROMPT
                      Prompt template for LLM (use {formatted_facts} as placeholder)
--llm-concurrency LLM_CONCURRENCY
                      Number of parallel LLM requests (default: 5)
--generate-city-summaries
                      Generate LLM summaries for cities/towns (requires --generate-summaries)
--enable-web-search   Enable OpenRouter web search plugin for additional research
--web-search-prompt WEB_SEARCH_PROMPT
                      Custom prompt for web search results
--geonames-username GEONAMES_USERNAME
                      GeoNames username for fact gathering
--flickr-api-key FLICKR_API_KEY
                      Flickr API key for image gathering
--unsplash-api-key UNSPLASH_API_KEY
                      Unsplash API key for image gathering
--gather-images       Enable multi-source image gathering (Wikimedia Commons, Flickr, Unsplash)
--cache-file CACHE_FILE
                      Cache file location (default: cache/wikidata_cache.json)
--cache-expiry-days CACHE_EXPIRY_DAYS
                      Number of days before cache expires (default: 7)
--no-cache            Disable caching (fetch all data fresh)
--force-refresh       Force refresh all cached data
```

## Usage
1. Clone and enter the repository directory
```
git clone https://github.com/Cruxial0/JapanLandmarksCollector
cd JapanLandmarksCollector
```

2. Create and activate a virtual environment
```
python -m venv .venv
source ./.venv/bin/activate
# For windows: .\venv\Scripts\activate.bat
```

3. Install the requirements
```
pip install -r requirements.txt
```

4. Execute the script
**NOTE:** Due to WikiData's usage policy, setting an email is required. Use the `--email` argument for this.
Here's an example with LLM summarization:
```
python main.py --email myemail@domain.com \
    --generate-summaries \
    --enable-web-search \
    --llm-model x-ai/grok-4-fast:online \
    --openrouter-api-key sk-v1-... # Required for LLM summarization
```

### Statistics
Running the current state of the script:
- Takes about 2-2.5h (with default rate limits)
- Costs about $2 for LLM summaries (Using `grok-4-fast:online`)