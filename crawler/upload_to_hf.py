#!/usr/bin/env python3
"""
Upload the HF Space folder to a Hugging Face Space repo.

Required env vars:
  HF_TOKEN     - a token with write access to the Space
  HF_SPACE_ID  - e.g. "username/space-name"

Optional:
  HF_SPACE_DIR - local folder to upload (default: "hf_space")
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    token = os.environ.get("HF_TOKEN", "").strip()
    repo_id = os.environ.get("HF_SPACE_ID", "").strip()
    folder = Path(os.environ.get("HF_SPACE_DIR", "hf_space")).resolve()

    if not token:
        raise SystemExit("HF_TOKEN is missing (set it as a GitHub Actions secret).")
    if not repo_id:
        raise SystemExit("HF_SPACE_ID is missing (set it as a GitHub Actions secret).")
    if not folder.exists():
        raise SystemExit(f"HF_SPACE_DIR not found: {folder}")

    api = HfApi(token=token)

    # Upload the Space root (contents of hf_space/)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(folder),
        path_in_repo="",
        commit_message="Auto-update Space (app + sources)",
        ignore_patterns=["**/__pycache__/**", "**/*.pyc", "**/.DS_Store"],
    )

    print(f"[upload_to_hf] Uploaded {folder} -> {repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
