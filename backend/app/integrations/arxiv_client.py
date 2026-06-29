# This file handles all communication with the arXiv API.
# arXiv is a free, open academic preprint server — no API key needed,
# which is why we start with it. It exposes a simple HTTP API that
# returns paper metadata as XML, which we parse into Python dictionaries.
#
# By keeping all arXiv-specific code in this one file, the rest of the
# app never needs to know how arXiv works — it just calls search_arxiv()
# and gets back a clean list of papers. If we ever need to change how we
# talk to arXiv (or swap it for a different source), we only change THIS
# file, nowhere else.

import httpx
import xmltodict
from typing import Optional


# arXiv's public API endpoint. No authentication required.
ARXIV_API_URL = "https://export.arxiv.org/api/query"


async def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
) -> list[dict]:
    """
    Searches arXiv for papers matching the query string and returns a
    clean list of paper dictionaries.

    Args:
        query: The search query (e.g. "transformer neural networks NLP")
        max_results: How many papers to return (max 100 per arXiv's limits)
        sort_by: "relevance", "lastUpdatedDate", or "submittedDate"

    Returns:
        List of paper dicts with keys: external_id, title, authors,
        abstract, year, venue, source.

    Why async?
    This function makes a real network request to an external server,
    which can take seconds. Making it async means FastAPI can handle
    other requests while waiting for arXiv to respond, instead of
    blocking the entire server. This is a fundamental difference between
    a toy app and a production-grade one.
    """

    # httpx is an async-capable HTTP client — like the `requests` library
    # you may have used before, but designed to work with async/await.
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            ARXIV_API_URL,
            params={
                "search_query": f"all:{query}",
                # arXiv uses "all:" to search across title, abstract,
                # authors, and comments simultaneously.
                "max_results": max_results,
                "sortBy": sort_by,
                "sortOrder": "descending",
            },
        )
        response.raise_for_status()
        # raise_for_status() raises an exception if arXiv returns a
        # 4xx or 5xx status code, so we don't silently process bad data.

    # arXiv returns XML. xmltodict converts it to a Python dictionary
    # so we can work with it normally without writing XML parsing code.
    data = xmltodict.parse(response.text)

    # The actual list of entries lives at this path in the XML structure.
    feed = data.get("feed", {})
    entries = feed.get("entry", [])

    # arXiv returns a single dict (not a list) if there's only one result.
    # Normalise it to always be a list so the rest of our code is simpler.
    if isinstance(entries, dict):
        entries = [entries]

    papers = []
    for entry in entries:
        paper = _parse_arxiv_entry(entry)
        if paper:
            papers.append(paper)

    return papers


def _parse_arxiv_entry(entry: dict) -> Optional[dict]:
    """
    Converts a single raw arXiv XML entry (as a dict) into the clean
    shape we use everywhere in our app. Private function (leading _)
    means it's an internal helper — only search_arxiv() calls it.
    """
    try:
        # arXiv IDs look like "http://arxiv.org/abs/2301.12345v2"
        # We strip the URL prefix and version suffix to get "2301.12345"
        raw_id = entry.get("id", "")
        external_id = raw_id.split("/abs/")[-1].split("v")[0]

        # Title often has extra whitespace and newlines from the XML
        title = entry.get("title", "").replace("\n", " ").strip()

        # Abstract similarly needs cleanup
        abstract = entry.get("summary", "").replace("\n", " ").strip()

        # Authors can be a single dict or a list of dicts
        raw_authors = entry.get("author", [])
        if isinstance(raw_authors, dict):
            raw_authors = [raw_authors]
        authors = [a.get("name", "") for a in raw_authors if a.get("name")]

        # Published date looks like "2023-01-30T00:00:00Z"
        published = entry.get("published", "")
        year = int(published[:4]) if published else None

        return {
            "external_id": external_id,
            "source": "arxiv",
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "year": year,
            "venue": "arXiv",       # arXiv preprints aren't peer-reviewed
            "citation_count": None, # arXiv doesn't provide citation counts
        }

    except Exception as e:
        # If one paper fails to parse, log it and skip it rather than
        # crashing the whole search — better to return 9 papers than 0.
        print(f"Failed to parse arXiv entry: {e}")
        return None
