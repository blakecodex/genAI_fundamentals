# rss_utils.py
# parses Yahoo Finance RSS feeds for NVIDIA and Intel
# why these? because it's topical — NVIDIA invested billions in Intel this week (third week of September 2025)

import requests
import xml.etree.ElementTree as ET

def fetch_rss_headlines(ticker="NVDA", max_items=5):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        resp = requests.get(url, timeout=5)
        if not resp.ok:
            print(f"failed to fetch feed for {ticker}: status {resp.status_code}")
            return []

        root = ET.fromstring(resp.content)
        items = root.findall(".//item")

        headlines = []
        for i, item in enumerate(items[:max_items]):
            title = item.find("title").text if item.find("title") is not None else None
            if title:
                headlines.append({"id": f"{ticker.lower()}_{i}", "text": title})
        return headlines

    except Exception as e:
        print(f"rss parse error for {ticker}: {e}")
        return []

def get_combined_headlines():
    nvda_news = fetch_rss_headlines("NVDA")
    intc_news = fetch_rss_headlines("INTC")
    all_news = nvda_news + intc_news
    if not all_news:
        print("warning: fallback activated — using static headlines")
        return [
            {"id": "nvda_0", "text": "NVIDIA invests $3B in Intel to boost foundry collaboration."},
            {"id": "nvda_1", "text": "AI chip race intensifies as NVIDIA expands datacenter reach."},
            {"id": "intc_0", "text": "Intel gains after major capital infusion from NVIDIA."},
            {"id": "intc_1", "text": "Intel Foundry ramps up production for AI-focused silicon."},
            {"id": "nvda_2", "text": "NVIDIA and Intel CEOs meet to outline AI strategy for 2026."}
        ]
    return all_news
