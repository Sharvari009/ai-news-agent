"""
AI Business News Agent
----------------------
Runs once a day. Reads trusted news feeds, asks Gemini to judge and explain
the most important AI business stories, links them to related past stories,
and saves everything for the dashboard.

Where the agent makes its own decisions:
  1. Gemini decides which stories are really about the business of AI.
  2. Gemini merges duplicate stories covered by several sites.
  3. Gemini ranks, categorises, and writes a "why it matters" line.
  4. Gemini links each story to related stories from past briefs (history).
  5. If there is too little news, the agent widens its time window and retries.
  6. If one Gemini model fails, the agent retries and falls back to another.

Links always come from the original feeds (never written by the AI),
so every story and every history item points to a real article.
"""

import html
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
from google import genai
from google.genai import types

IST = timezone(timedelta(hours=5, minutes=30))
MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]  # tried in this order
NEWS_DIR = Path("news")
MAX_ARTICLES = 300          # how many articles Gemini reads per run
STORIES_WANTED = 15         # how many stories appear on the dashboard
HISTORY_DAYS = 60           # how far back the agent looks for related stories
MAX_RELATED = 3             # history items shown per story
CATEGORIES = ["Funding", "Product launch", "Big Tech", "India", "Policy"]

PROMPT = """You are the editor of a daily AI business news brief for MBA students and product managers in India.

TODAY'S ARTICLES ({n} items, JSON):
{articles}

PAST STORIES from earlier briefs ({h} items, JSON with "ref", "date", "title"):
{history}

Your job:
1. Keep only stories about the BUSINESS of AI: funding and acquisitions, AI product launches, Big Tech AI moves, Indian AI companies and startups, AI regulation and policy. Skip tutorials, opinion pieces, gadget reviews, and non-AI news.
2. If several articles cover the same story, keep only the best one.
3. Pick the {k} most important stories, most important first. Include India stories when they exist.
4. For each story write:
   - "summary": at most 35 words, plain language
   - "why": at most 20 words on the business or product-management angle
   - "category": exactly one of {cats}
   - "related": up to {r} "ref" values from PAST STORIES that give useful background to this story (same company, same deal, same product, same regulation, or a clear earlier chapter of the same story). Use [] if nothing is clearly related. Never link a story just because both are about AI.

Return only a JSON array like:
[{{"id": 12, "summary": "...", "why": "...", "category": "Funding", "related": ["2026-09-20#3"]}}]
Use only ids from TODAY'S ARTICLES and refs from PAST STORIES."""


def clean(text, limit=300):
    """Remove HTML tags and extra spaces, then shorten."""
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(text or "")))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def read_sources():
    lines = Path("sources.txt").read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def fetch_articles(hours):
    """Read every feed and keep articles published in the last `hours` hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    articles, seen_links, seen_titles, sources_ok = [], set(), set(), 0

    for url in read_sources():
        try:
            feed = feedparser.parse(url, agent="Mozilla/5.0 (AI News Agent)")
            if not feed.entries:
                raise ValueError(feed.get("bozo_exception", "no articles in feed"))
        except Exception as err:
            print(f"  x {url[:80]}: {err}")
            continue

        feed_name = clean(feed.feed.get("title", url), 60)
        count = 0
        for entry in feed.entries:
            link = entry.get("link", "")
            stamp = entry.get("published_parsed") or entry.get("updated_parsed")
            if not link.startswith("http") or link in seen_links or not stamp:
                continue
            published = datetime(*stamp[:6], tzinfo=timezone.utc)
            if published < cutoff:
                continue

            # Google News items name the real publisher in entry.source.
            publisher = clean((entry.get("source") or {}).get("title"), 60)
            title = clean(entry.get("title"), 200)
            if publisher and title.endswith(" - " + publisher):
                title = title[: -len(" - " + publisher)]
            key = re.sub(r"\W+", "", title.lower())
            if key in seen_titles:          # same headline from another feed
                continue

            seen_links.add(link)
            seen_titles.add(key)
            articles.append({
                "title": title,
                "summary": clean(entry.get("summary")),
                "source": publisher or feed_name,
                "url": link,
                "published": published.isoformat(),
            })
            count += 1
        sources_ok += 1
        print(f"  ok {feed_name}: {count} recent articles")

    # Newest first, cap the total, then number them for Gemini.
    articles.sort(key=lambda a: a["published"], reverse=True)
    articles = articles[:MAX_ARTICLES]
    for i, a in enumerate(articles):
        a["id"] = i
    return articles, sources_ok


def load_history(today):
    """Collect stories from past briefs so Gemini can link related ones."""
    history = {}
    cutoff = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    for path in sorted(NEWS_DIR.glob("20*.json"), reverse=True):
        date = path.stem
        if date >= today or date < cutoff:
            continue
        try:
            digest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for i, s in enumerate(digest.get("stories", [])):
            history[f"{date}#{i}"] = {
                "date": date, "title": s.get("title", ""),
                "source": s.get("source", ""), "url": s.get("url", ""),
            }
    return history


def ask_gemini(client, prompt):
    """Ask Gemini, retrying and falling back to another model if needed."""
    last_error = None
    for model in MODELS:
        for attempt in range(3):
            try:
                reply = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.3,
                    ),
                )
                return json.loads(reply.text), model
            except Exception as err:
                last_error = err
                print(f"  ! {model}, attempt {attempt + 1} failed: {err}")
                time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Gemini did not respond: {last_error}")


def build_stories(picks, articles, history):
    """Match Gemini's picks back to real articles and real past stories."""
    by_id = {a["id"]: a for a in articles}
    stories, used = [], set()
    for pick in picks if isinstance(picks, list) else []:
        try:
            article = by_id[int(pick["id"])]
        except (KeyError, ValueError, TypeError):
            continue
        if article["id"] in used:
            continue
        used.add(article["id"])

        related = []
        refs = pick.get("related") or []
        for ref in refs if isinstance(refs, list) else []:
            past = history.get(str(ref))
            if past and past["url"] != article["url"] and past not in related:
                related.append(past)
        related.sort(key=lambda p: p["date"], reverse=True)

        category = pick.get("category")
        stories.append({
            "title": article["title"],
            "summary": clean(pick.get("summary"), 400) or article["summary"],
            "why": clean(pick.get("why"), 250),
            "category": category if category in CATEGORIES else "Big Tech",
            "source": article["source"],
            "url": article["url"],
            "published": article["published"],
            "related": related[:MAX_RELATED],
        })
    return stories[:STORIES_WANTED]


def main():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is missing. Add it in GitHub: Settings > Secrets and variables > Actions.")

    client = genai.Client(api_key=key)
    today = datetime.now(IST).strftime("%Y-%m-%d")
    print(f"AI News Agent: run for {today}")

    history = load_history(today)
    print(f"Loaded {len(history)} past stories for history links")
    history_for_ai = [{"ref": r, "date": h["date"], "title": h["title"]} for r, h in history.items()]

    stories, articles, sources_ok = [], [], 0
    for hours in (36, 72):
        print(f"\nReading news from the last {hours} hours...")
        articles, sources_ok = fetch_articles(hours)
        print(f"Read {len(articles)} articles from {sources_ok} sources")
        if len(articles) < 5:
            print("Too few articles, widening the time window...")
            continue

        prompt = PROMPT.format(
            n=len(articles),
            h=len(history_for_ai),
            k=STORIES_WANTED,
            r=MAX_RELATED,
            cats=", ".join(CATEGORIES),
            articles=json.dumps(
                [{k: a[k] for k in ("id", "title", "summary", "source")} for a in articles],
                ensure_ascii=False,
            ),
            history=json.dumps(history_for_ai, ensure_ascii=False),
        )
        picks, model = ask_gemini(client, prompt)
        stories = build_stories(picks, articles, history)
        linked = sum(1 for s in stories if s["related"])
        print(f"{model} selected {len(stories)} stories, {linked} with history")
        if len(stories) >= 5:
            break
        print("Too few strong stories, widening the time window...")

    if not stories:
        sys.exit("No stories found today. The dashboard keeps showing the last brief.")

    NEWS_DIR.mkdir(exist_ok=True)
    digest = {
        "date": today,
        "generated_at": datetime.now(IST).isoformat(timespec="minutes"),
        "articles_read": len(articles),
        "sources_read": sources_ok,
        "stories": stories,
    }
    (NEWS_DIR / f"{today}.json").write_text(
        json.dumps(digest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    index_path = NEWS_DIR / "index.json"
    dates = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    dates = sorted(set(dates) | {today}, reverse=True)
    index_path.write_text(json.dumps(dates, indent=2), encoding="utf-8")

    print(f"\nSaved {len(stories)} stories to news/{today}.json")


if __name__ == "__main__":
    main()
