"""
Fetches FastAPI's official English documentation (.md files) by doing a
shallow git clone of the repo (avoids GitHub API rate limits entirely) and
copying the docs into data/docs/.

Usage:
    python ingestion/fetch_docs.py
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/fastapi/fastapi.git"
DOCS_PATH_IN_REPO = "docs/en/docs"   # English docs folder in the FastAPI repo
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


def main(limit: int | None = None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        print(f"Cloning {REPO_URL} (shallow, this may take a moment)...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, tmp],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        docs_src = Path(tmp) / DOCS_PATH_IN_REPO
        if not docs_src.exists():
            raise FileNotFoundError(f"Expected docs folder not found at {docs_src}")

        md_files = sorted(docs_src.rglob("*.md"))
        print(f"Found {len(md_files)} markdown files.")

        if limit:
            md_files = md_files[:limit]

        for i, path in enumerate(md_files, 1):
            relative = path.relative_to(docs_src)
            flat_name = str(relative).replace("\\", "__").replace("/", "__")
            out_path = OUTPUT_DIR / flat_name
            shutil.copy(path, out_path)
            print(f"[{i}/{len(md_files)}] saved {flat_name}")

    print(f"\nDone. Docs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()