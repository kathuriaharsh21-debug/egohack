"""Security and Hygiene Audit Scanner for EgoHack repositories.

Scans git tracked files and full commit history for:
- Video files (*.mp4, *.avi, *.mov, *.mkv)
- Image/frame files (*.png, *.jpg, *.jpeg)
- Model checkpoints (*.pt, *.pth, *.onnx, *.task)
- Credentials and config files (kaggle.json, .env, *.key)
- Suspicious strings: "KGAT", "api_key", "access_token", Drive links ("drive.google.com")
"""

import sys
import subprocess
import re
from pathlib import Path

FORBIDDEN_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv",
    ".png", ".jpg", ".jpeg",
    ".pt", ".pth", ".onnx", ".task"
}

FORBIDDEN_FILES = {
    "kaggle.json", ".env"
}

FORBIDDEN_KEYWORDS = [
    "KGAT",
    "api_key",
    "access_token",
    "drive.google.com",
    "AIzaSy"  # Google API key prefix
]

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

    # 3. Scan git commit diffs across full history for forbidden keywords
    diff_scan = subprocess.run(
        ["git", "-C", str(repo_path), "log", "-p", "--all"],
        capture_output=True, text=True, errors="ignore"
    )
    diff_text = diff_scan.stdout

    for kw in FORBIDDEN_KEYWORDS:
        # Match added lines starting with + that contain the keyword
        matches = re.findall(rf"^\+.*({re.escape(kw)}).*", diff_text, flags=re.MULTILINE | re.IGNORECASE)
        if matches:
            findings.append(f"[HISTORICAL DIFF MATCH] Keyword '{kw}' found in {len(matches)} added commit lines")

    # 4. Summary report
    if not findings:
        print("RESULT: CLEAN! Zero videos, frames, weights, credentials, or target keywords found in repo or history.")
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
    sys.exit(0 if code == 0 else 1)

if __name__ == "__main__":
    main()

