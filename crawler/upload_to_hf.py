import os
import shutil
from pathlib import Path

from huggingface_hub import upload_folder


def _sync_sources_into_space(repo_root: Path, space_dir: Path) -> None:
    """
    把 repo_root/sources 同步到 repo_root/hf_space/sources（如果存在）
    这样 upload_folder 只需要上传 hf_space 目录即可。
    """
    sources_dir = repo_root / "sources"
    if not sources_dir.exists():
        return

    target = space_dir / "sources"
    target.mkdir(parents=True, exist_ok=True)

    # 先清空旧 sources（避免删除/新增不一致）
    for item in target.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink(missing_ok=True)

    # 再复制新的 sources
    # Python 3.11 支持 dirs_exist_ok
    shutil.copytree(sources_dir, target, dirs_exist_ok=True)


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

    # 你可以在 Actions secrets 里放 HF_SPACE_ID，例如 "abyss20090/cheshireacademy-chatbot"
    repo_id = os.environ.get("HF_SPACE_ID") or os.environ.get("HF_REPO_ID")
    if not repo_id:
        raise RuntimeError("Missing HF_SPACE_ID (or HF_REPO_ID) in env/secrets.")

    # ⚠️ 关键：multi_commits=True 自动拆分成多个 commit，避免一次 commit 操作太多文件导致超时
    upload_folder(
        folder_path=str(space_dir),
        repo_id=repo_id,
        repo_type="space",
        token=hf_token,
        commit_message="Update Space (crawler + i18n + sources)",
        multi_commits=True,
        multi_commits_verbose=True,
    )

    print("[upload_to_hf] Done.")


if __name__ == "__main__":
    main()
