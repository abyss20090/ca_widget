import os
import shutil
import time
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, CommitOperationAdd, upload_folder


def _sync_sources_into_space(repo_root: Path, space_dir: Path) -> None:
    """
    把 repo_root/sources 同步到 space_dir/sources（如果存在）
    """
    sources_dir = repo_root / "sources"
    if not sources_dir.exists():
        print("[upload_to_hf] No sources/ folder found, skip syncing.")
        return

    target = space_dir / "sources"
    target.mkdir(parents=True, exist_ok=True)

    # 清空旧 sources（保证删掉已不存在的页面）
    for item in target.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink(missing_ok=True)

    shutil.copytree(sources_dir, target, dirs_exist_ok=True)
    print("[upload_to_hf] Synced sources/ -> hf_space/sources")


def _iter_files(folder: Path) -> List[Path]:
    ignore_dirs = {".git", "__pycache__"}
    ignore_suffix = {".pyc"}

    files: List[Path] = []
    for p in folder.rglob("*"):
        if p.is_dir():
            continue
        if any(part in ignore_dirs for part in p.parts):
            continue
        if p.suffix.lower() in ignore_suffix:
            continue
        files.append(p)

    files.sort()
    return files


def _sleep_backoff(attempt: int, base: float = 5.0, cap: float = 120.0) -> None:
    sleep_s = min(base * attempt, cap)
    print(f"[upload_to_hf] retrying in {sleep_s:.0f}s ...")
    time.sleep(sleep_s)


def _try_upload_large_folder(api: HfApi, repo_id: str, token: str, space_dir: Path) -> bool:
    if not hasattr(api, "upload_large_folder"):
        return False

    max_retries = int(os.getenv("HF_MAX_RETRIES", "10"))
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[upload_to_hf] Trying api.upload_large_folder() (attempt {attempt}/{max_retries}) ...")
            api.upload_large_folder(
                repo_id=repo_id,
                repo_type="space",
                folder_path=str(space_dir),
                commit_message="Update Space (crawler + sources)",
                token=token,
            )
            print("[upload_to_hf] Done via upload_large_folder().")
            return True
        except Exception as e:
            msg = repr(e)
            print("[upload_to_hf] upload_large_folder failed ->", msg)
            if attempt >= max_retries:
                return False
            _sleep_backoff(attempt)
    return False


def _try_upload_folder(api: HfApi, repo_id: str, token: str, space_dir: Path) -> bool:
    """
    次优：upload_folder + multi_commits（自动拆分/可恢复）
    """
    max_retries = int(os.getenv("HF_MAX_RETRIES", "10"))
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[upload_to_hf] Trying upload_folder(multi_commits=True) (attempt {attempt}/{max_retries}) ...")
            upload_folder(
                folder_path=str(space_dir),
                repo_id=repo_id,
                repo_type="space",
                token=token,
                commit_message="Update Space (crawler + sources)",
                multi_commits=True,
                multi_commits_verbose=True,
                ignore_patterns=["**/.git/**", "**/__pycache__/**", "**/*.pyc"],
            )
            print("[upload_to_hf] Done via upload_folder(multi_commits=True).")
            return True
        except TypeError as e:
            # 旧版 huggingface_hub 没这些参数
            print("[upload_to_hf] upload_folder doesn't support multi_commits ->", repr(e))
            return False
        except Exception as e:
            msg = repr(e)
            # 无变化不算失败
            if "No files have been modified since last commit" in msg or "empty commit" in msg:
                print("[upload_to_hf] No changes since last commit. Nothing to upload.")
                return True

            print("[upload_to_hf] upload_folder failed ->", msg)
            if attempt >= max_retries:
                return False
            _sleep_backoff(attempt)
    return False


def _upload_with_batched_commits(space_dir: Path, repo_id: str, token: str) -> None:
    """
    兜底方案：小批量 commit（把默认 batch 降低，避免 ReadTimeout）
    """
    batch_size = int(os.getenv("HF_COMMIT_BATCH", "40"))  # 默认 40
    max_retries = int(os.getenv("HF_MAX_RETRIES", "10"))

    api = HfApi(token=token)
    files = _iter_files(space_dir)
    total = len(files)
    if total == 0:
        print("[upload_to_hf] Nothing to upload (space folder is empty).")
        return

    num_commits = (total + batch_size - 1) // batch_size
    print(f"[upload_to_hf] Fallback batched commits: {total} files / {num_commits} commits (batch={batch_size})")

    for idx in range(num_commits):
        start = idx * batch_size
        end = min((idx + 1) * batch_size, total)
        chunk = files[start:end]

        ops = []
        for fp in chunk:
            rel = fp.relative_to(space_dir).as_posix()
            ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(fp)))

        commit_msg = f"Update Space (batched) {idx+1}/{num_commits} ({len(chunk)} files)"

        attempt = 0
        while True:
            attempt += 1
            try:
                api.create_commit(
                    repo_id=repo_id,
                    repo_type="space",
                    operations=ops,
                    commit_message=commit_msg,
                    token=token,
                )
                print(f"[upload_to_hf] committed: {idx+1}/{num_commits} ({len(chunk)} files)")
                break
            except Exception as e:
                msg = repr(e)
                if "No files have been modified since last commit" in msg or "empty commit" in msg:
                    print(f"[upload_to_hf] no changes in batch {idx+1}/{num_commits}, skip.")
                    break

                if attempt >= max_retries:
                    raise

                print(f"[upload_to_hf] commit failed (attempt {attempt}/{max_retries}) -> {msg}")
                _sleep_backoff(attempt, base=10.0, cap=120.0)


def main() -> None:
    # Repo root = project root (scripts typically live in /scripts or similar).
    # Your previous version used parents[1]; keep it for compatibility.
    repo_root = Path(__file__).resolve().parents[1]

    space_dir_env = os.environ.get("HF_SPACE_DIR", "").strip()
    space_dir = (repo_root / space_dir_env).resolve() if space_dir_env else (repo_root / "hf_space")

    if not space_dir.exists():
        raise FileNotFoundError(f"hf_space folder not found: {space_dir}")

    _sync_sources_into_space(repo_root, space_dir)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError("Missing HF_TOKEN (or HUGGINGFACE_TOKEN) in env/secrets.")

    repo_id = os.environ.get("HF_SPACE_ID") or os.environ.get("HF_REPO_ID")
    if not repo_id:
        raise RuntimeError("Missing HF_SPACE_ID (or HF_REPO_ID) in env/secrets.")

    api = HfApi(token=hf_token)

    # 0) 首选：upload_large_folder（更适合大目录、可断点）
    if _try_upload_large_folder(api, repo_id, hf_token, space_dir):
        return

    # 1) 次选：upload_folder(multi_commits=True)
    if _try_upload_folder(api, repo_id, hf_token, space_dir):
        return

    # 2) 兜底：batched commits（最稳但最慢）
    _upload_with_batched_commits(space_dir, repo_id, hf_token)
    print("[upload_to_hf] Done (batched commits).")


if __name__ == "__main__":
    main()
