import os
import time
from pathlib import Path

import httpx
from huggingface_hub import HfApi


def main():
    token = os.getenv("HF_TOKEN", "").strip()
    space_id = os.getenv("HF_SPACE_ID", "").strip()
    space_dir = os.getenv("HF_SPACE_DIR", "hf_space").strip()

    if not token:
        raise SystemExit("HF_TOKEN is missing (set it as a GitHub Actions secret).")
    if not space_id:
        raise SystemExit("HF_SPACE_ID is missing (set it as a GitHub Actions secret).")

    folder_path = Path(space_dir)
    if not folder_path.exists():
        raise SystemExit(f"Space folder not found: {folder_path.resolve()}")

    # Enable hf_transfer if provided via env in Actions
    # (no harm if the env is not set)
    api = HfApi(token=token)

    # ---- 핵심：多文件时必须用 multi_commits 分批提交，否则容易超时 ----
    # 再加自动重试，避免网络抖动导致失败
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_folder(
                repo_id=space_id,
                repo_type="space",
                folder_path=str(folder_path),
                commit_message="Automated update: crawl + sync",
                multi_commits=True,
                multi_commits_verbose=True,
            )
            print(f"Uploaded {folder_path} to HF Space: {space_id}")
            return
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            wait = 8 * attempt
            print(f"[upload_to_hf] Timeout on attempt {attempt}/{max_retries}: {repr(e)}")
            print(f"[upload_to_hf] Sleeping {wait}s then retry...")
            time.sleep(wait)
        except Exception as e:
            # 其它错误直接抛出（不要无限重试）
            raise

    raise SystemExit("[upload_to_hf] Failed after retries due to repeated timeouts.")


if __name__ == "__main__":
    main()
