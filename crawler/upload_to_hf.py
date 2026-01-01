import os
import shutil
import inspect
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, upload_folder

try:
    # 新版 huggingface_hub 一般都有
    from huggingface_hub import CommitOperationAdd
except Exception:
    CommitOperationAdd = None


def _sync_sources_into_space(repo_root: Path, space_dir: Path) -> None:
    """
    把 repo_root/sources 同步到 repo_root/hf_space/sources（如果存在）
    这样 upload 只需要上传 hf_space 目录即可。
    """
    sources_dir = repo_root / "sources"
    if not sources_dir.exists():
        # 如果你爬虫直接输出到了 hf_space/sources，就不用同步
        return

    target = space_dir / "sources"
    target.mkdir(parents=True, exist_ok=True)

    # 清空旧 sources（避免删除/新增不一致）
    for item in target.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink(missing_ok=True)

    # 复制新的 sources
    shutil.copytree(sources_dir, target, dirs_exist_ok=True)


def _list_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for p in folder.rglob("*"):
        if p.is_file():
            files.append(p)
    return files


def _upload_with_multi_commits(space_dir: Path, repo_id: str, token: str) -> None:
    """
    如果当前 huggingface_hub 的 upload_folder 支持 multi_commits，就用它（最省事）。
    """
    sig = inspect.signature(upload_folder)
    kwargs = dict(
        folder_path=str(space_dir),
        repo_id=repo_id,
        repo_type="space",
        token=token,
        commit_message="Update Space (crawler + i18n + sources)",
    )

    if "multi_commits" in sig.parameters:
        kwargs["multi_commits"] = True
    if "multi_commits_verbose" in sig.parameters:
        kwargs["multi_commits_verbose"] = True

    upload_folder(**kwargs)


def _upload_with_batched_commits(space_dir: Path, repo_id: str, token: str) -> None:
    """
    兼容旧版本：不用 multi_commits，改用 create_commit 分批提交，避免超时。
    """
    if CommitOperationAdd is None:
        raise RuntimeError(
            "Your huggingface_hub version is too old: CommitOperationAdd not available. "
            "Please bump huggingface_hub to >= 0.20+ (recommended: >= 0.26.2)."
        )

    api = HfApi(token=token)

    files = _list_files(space_dir)
    if not files:
        print("[upload_to_hf] No files found in hf_space. Nothing to upload.")
        return

    # 每个 commit 放多少文件：200 一般很稳（你也可以用 env 改）
    batch_size = int(os.getenv("HF_COMMIT_BATCH", "200"))

    # 统一用 posix 路径
    ops: List[CommitOperationAdd] = []
    for f in files:
        rel = f.relative_to(space_dir).as_posix()
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(f)))

    total = len(ops)
    batch_count = (total + batch_size - 1) // batch_size
    print(f"[upload_to_hf] Uploading {total} files in {batch_count} commits (batch_size={batch_size})...")

    for i in range(0, total, batch_size):
        part = ops[i : i + batch_size]
        idx = i // batch_size + 1
        msg = f"Update Space (batch {idx}/{batch_count})"
        api.create_commit(
            repo_id=repo_id,
            repo_type="space",
            token=token,
            operations=part,
            commit_message=msg,
        )
        print(f"[upload_to_hf] committed: {idx}/{batch_count} ({len(part)} files)")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    space_dir = repo_root / "hf_space"
    if not space_dir.exists():
        raise FileNotFoundError(f"hf_space folder not found: {space_dir}")

    # 可选：同步 sources -> hf_space/sources
    _sync_sources_into_space(repo_root, space_dir)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError("Missing HF_TOKEN (or HUGGINGFACE_TOKEN) in env/secrets.")

    repo_id = os.environ.get("HF_SPACE_ID") or os.environ.get("HF_REPO_ID")
    if not repo_id:
        raise RuntimeError("Missing HF_SPACE_ID (or HF_REPO_ID) in env/secrets.")

    # 尝试 multi_commits（如果版本支持）；否则自动退回 batched commits（稳定不炸）
    try:
        _upload_with_multi_commits(space_dir, repo_id, hf_token)
        print("[upload_to_hf] Done via upload_folder (multi_commits if supported).")
    except TypeError as e:
        # 典型：不支持 multi_commits / multi_commits_verbose
        print(f"[upload_to_hf] upload_folder TypeError -> fallback to batched commits: {e}")
        _upload_with_batched_commits(space_dir, repo_id, hf_token)
        print("[upload_to_hf] Done via batched commits.")
    except Exception as e:
        # 其他异常也可退回，避免 workflow 直接死
        print(f"[upload_to_hf] upload_folder failed -> fallback to batched commits: {e}")
        _upload_with_batched_commits(space_dir, repo_id, hf_token)
        print("[upload_to_hf] Done via batched commits.")


if __name__ == "__main__":
    main()
