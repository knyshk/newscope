# news_fetcher.py
import os
import re
import requests
from datetime import datetime, timedelta
from typing import List, Dict

NEWSAPI_URL = "https://newsapi.org/v2/everything"
NEWSAPI_TOP_URL = "https://newsapi.org/v2/top-headlines"
NEWSDATA_URL = "https://newsdata.io/api/1/news"

def _normalize_article(article: Dict) -> Dict:
    source = article.get("source", {})
    return {
        "id": article.get("url") or f"no-url-{hash(article.get('title',''))}",
        "title": article.get("title", "") or "",
        "content": article.get("content") or article.get("description") or "",
        "url": article.get("url", ""),
        "published_at": article.get("publishedAt") or article.get("published_at") or "",
        "source": source.get("name") if isinstance(source, dict) else source or "",
    }

def _format_date(date_val, default_days=1):
    """Format date for API - extract date part from ISO strings"""
    if not date_val:
        return (datetime.utcnow() - timedelta(days=default_days)).strftime("%Y-%m-%d")
    if isinstance(date_val, str) and 'T' in date_val:
        return date_val.split('T')[0]
    return str(date_val)

def _clean_query(query):
    """Remove common words from query for better API compatibility"""
    cleaned = re.sub(r'\b(any|news|about|the|a|an)\s+', '', query, flags=re.IGNORECASE).strip()
    return cleaned if cleaned and len(cleaned) >= 2 else query

def _deduplicate_articles(articles):
    """Remove duplicate articles by URL"""
    seen, unique = set(), []
    for a in articles:
        uid = a.get("url") or a.get("id")
        if uid and uid not in seen:
            seen.add(uid)
            unique.append(a)
    return unique

class NewsFetcher:
    def __init__(self, newsapi_key: str = None, newdata_key: str = None):
        self.newsapi_key = newsapi_key or os.getenv("NEWSAPI_KEY")
        self.newdata_key = newdata_key or os.getenv("NEWDATA_API_KEY")

    def fetch_from_newsapi(self, q: str = "latest", from_days: int = 1, page_size: int = 100, 
                          from_date: str = None, to_date: str = None) -> List[Dict]:
        if not self.newsapi_key:
            return []
        params = {
            "q": q, "from": _format_date(from_date, from_days), "language": "en",
            "pageSize": page_size, "sortBy": "publishedAt", "apiKey": self.newsapi_key
        }
        if to_date:
            params["to"] = _format_date(to_date)
        r = requests.get(NEWSAPI_URL, params=params, timeout=20)
        r.raise_for_status()
        return [_normalize_article(a) for a in r.json().get("articles", [])]

    def fetch_top_from_newsapi(self, country: str = "us", category: str = None, page_size: int = 50) -> List[Dict]:
        if not self.newsapi_key:
            return []
        params = {"country": country, "pageSize": page_size, "apiKey": self.newsapi_key}
        if category:
            params["category"] = category
        r = requests.get(NEWSAPI_TOP_URL, params=params, timeout=20)
        r.raise_for_status()
        return [_normalize_article(a) for a in r.json().get("articles", [])]

    def fetch_from_newdata(self, q: str = "latest", days: int = 1, limit: int = 100, 
                          country: str = None, from_date: str = None, to_date: str = None) -> List[Dict]:
        if not self.newdata_key:
            return []
        cleaned_query = _clean_query(q)
        params = {"apikey": self.newdata_key, "language": "en", "size": min(limit, 50)}
        if cleaned_query and cleaned_query.lower() != "latest":
            params["q"] = cleaned_query
        if country:
            params["country"] = country
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date
        r = requests.get(NEWSDATA_URL, params=params, timeout=20)
        r.raise_for_status()
        items = r.json().get("results", [])
        return [_normalize_article({
            "title": it.get("title", ""),
            "content": it.get("content") or it.get("description", ""),
            "url": it.get("link") or it.get("url", ""),
            "publishedAt": it.get("pubDate") or it.get("published_at", ""),
            "source": it.get("source_id") or it.get("source_name") or "",
        }) for it in items]

    def fetch(self, q: str = "latest", days: int = 1, use_top: bool = False, fallback_to_newdata: bool = True) -> List[Dict]:
        all_articles = []
        try:
            func = self.fetch_top_from_newsapi if use_top else lambda: self.fetch_from_newsapi(q=q, from_days=days)
            all_articles.extend(func())
        except Exception as e:
            print("NewsAPI fetch failed:", e)
        if fallback_to_newdata and self.newdata_key:
            try:
                all_articles.extend(self.fetch_from_newdata(q=q, days=days))
            except Exception as e:
                print("NewData fetch failed:", e)
        return _deduplicate_articles(all_articles)
