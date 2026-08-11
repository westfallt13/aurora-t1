import argparse
import json
import urllib.parse
import urllib.request


def search(query, base_url="http://localhost:8888", num_results=5, timeout=10):
    """Queries a self-hosted SearXNG instance's JSON API. Requires
    `formats: [html, json]` in its settings.yml (see tools/searxng/) --
    disabled by default on public instances to prevent scraping abuse, safe
    to enable on a private, self-hosted one. stdlib-only on purpose: this
    is the one piece of the project that talks to a network service outside
    the training pipeline, and it shouldn't need its own dependency for
    that."""
    url = f"{base_url.rstrip('/')}/search?" + urllib.parse.urlencode({"q": query, "format": "json"})
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.load(resp)

    results = []
    for r in data.get("results", [])[:num_results]:
        results.append({"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")})
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Query a self-hosted SearXNG instance.")
    p.add_argument("query")
    p.add_argument("--base-url", default="http://localhost:8888")
    p.add_argument("--num-results", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    results = search(args.query, args.base_url, args.num_results)
    if not results:
        print("No results (check that the SearXNG instance is reachable and JSON format is enabled).")
        return
    for r in results:
        print(f"- {r['title']}\n  {r['url']}\n  {r['snippet']}\n")


if __name__ == "__main__":
    main()
