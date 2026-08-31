"""
Fetches closed GitHub issues (title + body + top comments) from the FastAPI
repo and saves each as a small JSON file in data/issues/.

NOTE ON RATE LIMITS: GitHub's API allows 60 requests/hour without a token,
or 5,000/hour with one. Fetching issues WITH their comments uses one request
per issue for the comments, so without a token you can only fetch ~50 issues
per hour. Make sure GITHUB_TOKEN is set in your .env file.

Usage:
    python ingestion/fetch_issues.py
"""
import os
import sys
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

REPO = "fastapi/fastapi"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "issues"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}


def _check_rate_limit(resp: requests.Response):
    """Raise a clear, actionable error if we've hit GitHub's rate limit."""
    if resp.status_code == 403 and resp.headers.get("x-ratelimit-remaining") == "0":
        reset_ts = int(resp.headers.get("x-ratelimit-reset", 0))
        wait_min = max(0, round((reset_ts - time.time()) / 60))
        print(f"\nGitHub API rate limit hit (resets in ~{wait_min} min).")
        print("Fix: make sure GITHUB_TOKEN is set correctly in your .env file.\n")
        sys.exit(1)
    resp.raise_for_status()


def fetch_closed_issues(pages: int, per_page: int = 20):
    """Fetch several pages of closed issues (skips pull requests)."""
    issues = []
    for page in range(1, pages + 1):
        url = (
            f"https://api.github.com/repos/{REPO}/issues"
            f"?state=closed&per_page={per_page}&page={page}&sort=comments&direction=desc"
        )
        resp = requests.get(url, headers=HEADERS, timeout=30)
        _check_rate_limit(resp)
        batch = resp.json()
        if not batch:
            break
        for item in batch:
            if "pull_request" in item:
                continue  # skip PRs, we only want issues
            issues.append(item)
        print(f"Fetched page {page}, total issues so far: {len(issues)}")
        time.sleep(0.3)
    return issues


def fetch_top_comments(issue_number: int, max_comments: int = 3):
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/comments"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 403 and resp.headers.get("x-ratelimit-remaining") == "0":
        _check_rate_limit(resp)  # will print message and exit
    if resp.status_code != 200:
        return []
    return [c["body"] for c in resp.json()[:max_comments] if c.get("body")]


def main(max_issues: int = 100, pages: int = 6):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not GITHUB_TOKEN:
        print("WARNING: No GITHUB_TOKEN found in .env — limited to 60 requests/hour.\n")

    issues = fetch_closed_issues(pages=pages)
    issues = issues[:max_issues]

    for i, issue in enumerate(issues, 1):
        number = issue["number"]
        record = {
            "number": number,
            "title": issue.get("title", ""),
            "body": issue.get("body", "") or "",
            "comments": fetch_top_comments(number),
            "url": issue.get("html_url", ""),
            "labels": [l["name"] for l in issue.get("labels", [])],
        }
        out_path = OUTPUT_DIR / f"issue_{number}.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"[{i}/{len(issues)}] saved issue #{number}: {record['title'][:60]}")
        time.sleep(0.2)

    print(f"\nDone. Issues saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()