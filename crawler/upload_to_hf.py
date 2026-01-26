import os
import shutil
import time
import inspect
from pathlib import Path
from typing import List

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

    # 清空旧 sources
    for item in target.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink(missing_ok=True)

    shutil.copytree(sources_dir, target, dirs_exist_ok=True)
    print("[upload_to_hf] Synced sources/ -> hf_space/sources")


def _iter_files(folder: Path) -> List[Path]:
    """
    Collect all files under folder, excluding cache / git / pyc.
    """
    ignore_dirs = {".git", "__pycache__"}
    ignore_suffix = {".pyc"}

    files: List[Path] = []
    for p in folder.rglob("*"):
        if p.is_dir():
            # skip ignored dirs
            if p.name in ignore_dirs:
                continue
            continue

        if any(part in ignore_dirs for part in p.parts):
            continue
        if p.suffix.lower() in ignore_suffix:
            continue

        files.append(p)

    files.sort()
    return files


def _safe_create_commit(api: HfApi, **kwargs):
    """
    Call api.create_commit with optional timeout if supported.
    """
    sig = inspect.signature(api.create_commit)
    if "timeout" in sig.parameters and "timeout" not in kwargs:
        timeout_s = int(os.getenv("HF_HTTP_TIMEOUT", "300"))
        kwargs["timeout"] = timeout_s
    return api.create_commit(**kwargs)


def _upload_with_batched_commits(space_dir: Path, repo_id: str, token: str) -> None:
    """
    Upload via small commits to avoid a huge single commit timing out.
    """
    batch_size = int(os.getenv("HF_COMMIT_BATCH", "80"))
    max_retries = int(os.getenv("HF_MAX_RETRIES", "6"))

    api = HfApi(token=token)
    files = _iter_files(space_dir)
    total = len(files)

    if total == 0:
        print("[upload_to_hf] Nothing to upload (space folder is empty).")
        return

    num_commits = (total + batch_size - 1) // batch_size
    print(f"[upload_to_hf] Uploading {total} files in {num_commits} commits (batch_size={batch_size})...")

    for idx in range(num_commits):
        start = idx * batch_size
        end = min((idx + 1) * batch_size, total)
        chunk = files[start:end]

        ops = []
        for fp in chunk:
            rel = fp.relative_to(space_dir).as_posix()
            ops.append(
                CommitOperationAdd(
                    path_in_repo=rel,
                    path_or_fileobj=str(fp),
                )
            )

        commit_msg = f"Update Space (batched) {idx+1}/{num_commits} ({len(chunk)} files)"

        # retry loop (ReadTimeout / transient failures)
        attempt = 0
        while True:
            attempt += 1
            try:
                _safe_create_commit(
                    api,
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
                # empty commit / no change -> treat as success
                if "No files have been modified since last commit" in msg or "empty commit" in msg:
                    print(f"[upload_to_hf] no changes detected in batch {idx+1}/{num_commits}, skip.")
                    break

                if attempt >= max_retries:
                    raise

                sleep_s = min(5 * attempt, 30)
                print(f"[upload_to_hf] commit failed (attempt {attempt}/{max_retries}) -> {msg}")
                print(f"[upload_to_hf] retrying in {sleep_s}s ...")
                time.sleep(sleep_s)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    # allow workflow to pass HF_SPACE_DIR
    space_dir_env = os.environ.get("HF_SPACE_DIR", "").strip()
    if space_dir_env:
        space_dir = (repo_root / space_dir_env).resolve()
    else:
        space_dir = repo_root / "hf_space"

    if not space_dir.exists():
        raise FileNotFoundError(f"hf_space folder not found: {space_dir}")

    # 同步 sources
    _sync_sources_into_space(repo_root, space_dir)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError("Missing HF_TOKEN (or HUGGINGFACE_TOKEN) in env/secrets.")

    repo_id = os.environ.get("HF_SPACE_ID") or os.environ.get("HF_REPO_ID")
    if not repo_id:
        raise RuntimeError("Missing HF_SPACE_ID (or HF_REPO_ID) in env/secrets.")

    # 1) Try upload_folder first (fast path)
    try:
        print("[upload_to_hf] Trying upload_folder() ...")
        upload_folder(
            folder_path=str(space_dir),
            repo_id=repo_id,
            repo_type="space",
            token=hf_token,
            commit_message="Update Space (crawler + sources)",
        )
        print("[upload_to_hf] Done via upload_folder().")
        return
    except Exception as e:
        print("[upload_to_hf] upload_folder failed -> fallback to batched commits:", repr(e))

    # 2) Fallback: batched commits with retries + timeout
    _upload_with_batched_commits(space_dir, repo_id, hf_token)
    print("[upload_to_hf] Done (batched commits).")


if __name__ == "__main__":
    main()
