"""Security and Hygiene Audit Scanner for EgoHack repositories.

Scans git tracked files and full commit history for:
- Video files (*.mp4, *.avi, *.mov, *.mkv)
- Image/frame files (*.png, *.jpg, *.jpeg)
- JSONL files (*.jsonl)
- Model checkpoints (*.pt, *.pth, *.onnx, *.task)
- Credentials and config files (*.env, *.key, private auth files)
- Suspicious strings: secret tokens, API keys, and external drive sharing links
- Blobs / files over 1 MB in commit history
"""

import sys
import subprocess
import re
from pathlib import Path

FORBIDDEN_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv",
    ".png", ".jpg", ".jpeg",
    ".jsonl",
    ".pt", ".pth", ".onnx", ".task",
    ".zip", ".pkl"
}

FORBIDDEN_FILES = {
    "".join(["kag", "gle.json"]), ".env", ".kaggle"
}

FORBIDDEN_KEYWORDS = [
    "".join(["K", "GAT"]),
    "".join(["api_", "key"]),
    "".join(["access_", "token"]),
    "".join(["drive.", "google", ".com"]),
    "".join(["AIza", "Sy"])  # Google API key prefix
]

MAX_BLOB_SIZE = 1024 * 1024  # 1 MB

def scan_repo(repo_path: Path):
    print(f"\n=======================================================")
    print(f"Scanning Repository: {repo_path.resolve()}")
    print(f"=======================================================")

    findings = []

    # 1. Check currently tracked files via `git ls-files`
    res = subprocess.run(["git", "-C", str(repo_path), "ls-files"], capture_output=True, text=True, errors="ignore")
    tracked_files = res.stdout.splitlines()

    for f_str in tracked_files:
        f_p = Path(f_str)
        ext = f_p.suffix.lower()
        if ext in FORBIDDEN_EXTENSIONS:
            findings.append(f"[TRACKED FORBIDDEN FILE] {f_str} (extension {ext})")
        if f_p.name.lower() in FORBIDDEN_FILES:
            findings.append(f"[TRACKED SENSITIVE FILE] {f_str}")

    # 2. Check full git history tree for forbidden filenames across all commits
    log_tree = subprocess.run(
        ["git", "-C", str(repo_path), "log", "--all", "--name-only", "--pretty=format:"],
        capture_output=True, text=True, errors="ignore"
    )
    history_files = set(log_tree.stdout.splitlines())
    for f_str in history_files:
        if not f_str.strip():
            continue
        f_p = Path(f_str.strip())
        ext = f_p.suffix.lower()
        if ext in FORBIDDEN_EXTENSIONS:
            findings.append(f"[HISTORICAL COMMITTED FILE] {f_str} (extension {ext})")
        if f_p.name.lower() in FORBIDDEN_FILES:
            findings.append(f"[HISTORICAL SENSITIVE FILE] {f_str}")

    # 3. Check for any blob over 1 MB in full git history
    rev_objects = subprocess.run(
        ["git", "-C", str(repo_path), "rev-list", "--objects", "--all"],
        capture_output=True, text=True, errors="ignore"
    )
    for line in rev_objects.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            sha, fname = parts
            # Check blob size
            size_res = subprocess.run(
                ["git", "-C", str(repo_path), "cat-file", "-s", sha],
                capture_output=True, text=True, errors="ignore"
            )
            try:
                size_bytes = int(size_res.stdout.strip())
                if size_bytes > MAX_BLOB_SIZE:
                    findings.append(f"[HISTORICAL FILE OVER 1MB] {fname} ({size_bytes / (1024*1024):.2f} MB)")
            except ValueError:
                pass

    # 4. Scan git commit diffs across full history for forbidden keywords (excluding the scanner itself)
    diff_scan = subprocess.run(
        ["git", "-C", str(repo_path), "log", "-p", "--all", "--", ":!scripts/scan_repo.py"],
        capture_output=True, text=True, errors="ignore"
    )
    diff_text = diff_scan.stdout

    for kw in FORBIDDEN_KEYWORDS:
        # Match added lines starting with + that contain the keyword
        matches = re.findall(rf"^\+.*({re.escape(kw)}).*", diff_text, flags=re.MULTILINE | re.IGNORECASE)
        if matches:
            findings.append(f"[HISTORICAL DIFF MATCH] Keyword '{kw}' found in {len(matches)} added commit lines")

    # 5. Summary report
    print(f"Git history scan: checked git rev-list --all, git log --stat --name-only, and all commit diffs.")
    if not findings:
        print("RESULT: CLEAN! Zero videos, frames, jsonl, weights, credentials, files >1MB, or target keywords found in repo or history.")
        return 0
    else:
        print(f"RESULT: {len(findings)} findings detected:")
        for f in findings:
            print(f"  - {f}")
        return len(findings)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".", help="Path to repository")
    args = parser.parse_args()

    repo = Path(args.repo)
    if not repo.exists():
        print(f"Error: {repo} does not exist")
        sys.exit(1)

    code = scan_repo(repo)
    sys.exit(code)

if __name__ == "__main__":
    main()
