import os
import time
import random
import traceback
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _log(msg: str) -> None:
    print(f"[upload_to_hf] {msg}", flush=True)


def _iter_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def _should_ignore(rel: str) -> bool:
    rel_l = rel.lower()
    ignore_parts = [
        ".git/", ".github/", ".pytest_cache/", "__pycache__/", ".venv/",
        "node_modules/", ".ds_store"
    ]
    return any(x in rel_l for x in ignore_parts)


def _retry(fn, *, max_retries: int, base_sleep: float = 2.0, jitter: float = 0.6, label: str = "op"):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            sleep_s = base_sleep * (1.6 ** (attempt - 1))
            sleep_s = min(sleep_s, 60.0) + random.random() * jitter
            _log(f"{label} failed (attempt {attempt}/{max_retries}): {type(e).__name__}: {e}")
            if attempt < max_retries:
                _log(f"sleep {sleep_s:.1f}s then retry ...")
                time.sleep(sleep_s)
    raise last_err


def _upload_large_folder(api: HfApi, repo_id: str, token: str, space_dir: Path, commit_message: str, max_retries: int):
    if not hasattr(api, "upload_large_folder"):
        raise RuntimeError("HfApi.upload_large_folder not available in this huggingface_hub version.")

    def _do():
        _log("Using upload_large_folder ...")
        # 有的版本 token 不在签名里，但传进去一般没问题；如果 TypeError 会被外层捕获并 fallback
        return api.upload_large_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=str(space_dir),
            token=token,
            commit_message=commit_message,
        )

    return _retry(_do, max_retries=max_retries, label="upload_large_folder")


def _safe_create_commit(api: HfApi, **kwargs):
    """
    兼容不同 huggingface_hub 版本的 create_commit 签名（有的版本不支持 timeout 参数）
    """
    timeout = kwargs.pop("timeout", None)
    if timeout is not None:
        try:
            return api.create_commit(timeout=timeout, **kwargs)
        except TypeError:
            # older versions
            return api.create_commit(**kwargs)
    return api.create_commit(**kwargs)


def _upload_in_batches(api: HfApi, repo_id: str, space_dir: Path, batch_size: int, timeout: int, max_retries: int):
    files = []
    for p in _iter_files(space_dir):
        rel = p.relative_to(space_dir).as_posix()
        if _should_ignore(rel):
            continue
        files.append((p, rel))

    files.sort(key=lambda x: x[1])
    total = len(files)
    _log(f"batch upload: {total} files, batch_size={batch_size}, timeout={timeout}s")

    if total == 0:
        _log("No files to upload (space_dir empty after ignore filters).")
        return

    batches = [files[i:i + batch_size] for i in range(0, total, batch_size)]

    for i, batch in enumerate(batches, start=1):
        ops = []
        for p, rel in batch:
            ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(p)))

        def _do_commit():
            _log(f"Committing batch {i}/{len(batches)} ({len(batch)} files) ...")
            return _safe_create_commit(
                repo_id=repo_id,
                repo_type="space",
                operations=ops,
                commit_message=f"Update Space (batch {i}/{len(batches)})",
                timeout=timeout,
            )

        _retry(_do_commit, max_retries=max_retries, label=f"create_commit batch {i}")

    _log("batch upload done.")


def main():
    hf_token = os.getenv("HF_TOKEN", "").strip()
    repo_id = os.getenv("HF_SPACE_REPO", "").strip()
    space_dir = Path(os.getenv("HF_SPACE_DIR", "hf_space")).resolve()

    upload_mode = os.getenv("HF_UPLOAD_MODE", "large_then_batch").strip().lower()
    batch_size = _env_int("HF_COMMIT_BATCH", 25)
    timeout = _env_int("HF_HTTP_TIMEOUT", 1200)
    max_retries = _env_int("HF_MAX_RETRIES", 10)

    if not hf_token:
        raise RuntimeError("HF_TOKEN is missing (set GitHub secret HF_TOKEN).")
    if not repo_id:
        raise RuntimeError("HF_SPACE_REPO is missing (e.g., 'username/space_name').")
    if not space_dir.exists():
        raise RuntimeError(f"HF_SPACE_DIR not found: {space_dir}")

    api = HfApi(token=hf_token)

    # ensure repo exists
    def _ensure_repo():
        _log(f"Ensuring Space repo exists: {repo_id}")
        try:
            api.repo_info(repo_id=repo_id, repo_type="space")
            _log("Space repo exists.")
        except Exception:
            _log("Space repo not found, creating ...")
            api.create_repo(repo_id=repo_id, repo_type="space", private=True, exist_ok=True)

    _retry(_ensure_repo, max_retries=max_retries, label="ensure_repo")

    # quick stats
    file_count = 0
    total_bytes = 0
    for p in _iter_files(space_dir):
        rel = p.relative_to(space_dir).as_posix()
        if _should_ignore(rel):
            continue
        file_count += 1
        try:
            total_bytes += p.stat().st_size
        except Exception:
            pass
    _log(f"Space dir: {space_dir}")
    _log(f"Files to upload (after ignore): {file_count}, total ~ {total_bytes/1024/1024:.2f} MB")

    commit_message = "Update Space (crawl + upload)"

    # 1) try upload_large_folder if requested
    if upload_mode in ("large_then_batch", "large", "auto"):
        try:
            _upload_large_folder(api, repo_id, hf_token, space_dir, commit_message, max_retries=max_retries)
            _log("upload_large_folder succeeded.")
            return
        except Exception as e:
            _log(f"upload_large_folder failed -> fallback to batch commits. Reason: {type(e).__name__}: {e}")
            _log(traceback.format_exc())

    # 2) fallback: batch commits
    _upload_in_batches(api, repo_id, space_dir, batch_size=batch_size, timeout=timeout, max_retries=max_retries)


if __name__ == "__main__":
    main()
