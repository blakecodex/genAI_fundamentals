# news_utils.py
# pulls business headlines using NewsAPI; built for consulting-systems thinking/client services

import requests
import os
from dotenv import load_dotenv
load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

def get_business_news(query="Microsoft", max_articles=5):
    """Fetch recent business headlines and return them as client-ready bullet points."""
    if not NEWSAPI_KEY or NEWSAPI_KEY.strip() == "":
        return "Missing NEWSAPI_KEY. Please set it in your .env or environment."

    url = (
        f"https://newsapi.org/v2/everything?q={query}"
        f"&language=en&sortBy=publishedAt&pageSize={max_articles}&apiKey={NEWSAPI_KEY}"
    )

    try:
        resp = requests.get(url, timeout=5)
        if not resp.ok:
            return f"News API error: status {resp.status_code}"

        articles = resp.json().get("articles", [])
        if not articles:
            return "No news found for query."

        out = []
        for art in articles:
            headline = art.get("title", "")
            source = art.get("source", {}).get("name", "")
            if headline:
                out.append(f"- {headline} ({source})")

        return "\n".join(out)

    except Exception as e:
        return f"News fetch failed: {e}"
