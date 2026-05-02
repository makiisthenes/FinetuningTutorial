"""Upload the Gradio demo in this folder to a Hugging Face Space.

The Space is separate from the model repo. The model itself stays at
``makiisthebes/gemma-3-270M-Instruct-FoodExtract`` and is loaded at runtime
by ``app.py`` via the ``transformers`` pipeline. This script only pushes the
demo files (``app.py``, ``requirements.txt``, ``README.md``) to the Space.

Re-running is safe: ``exist_ok=True`` on ``create_repo`` and ``upload_folder``
will simply push a new commit to the existing Space.

Authentication:
    Either run ``huggingface-cli login`` once, or set the ``HF_TOKEN``
    environment variable before running this script.
"""

import os
from pathlib import Path

from huggingface_hub import create_repo, upload_folder

HF_USERNAME = "makiisthebes"
HF_SPACE_NAME = "gemma-3-270M-FoodExtract-demo"
HF_REPO_ID = f"{HF_USERNAME}/{HF_SPACE_NAME}"
HF_REPO_TYPE = "space"
HF_SPACE_SDK = "gradio"

LOCAL_DEMO_FOLDER = Path(__file__).resolve().parent

HF_TOKEN = os.environ.get("HF_TOKEN")


def main() -> None:
    print(f"[INFO] Ensuring Space exists: {HF_REPO_ID}")
    create_repo(
        repo_id=HF_REPO_ID,
        token=HF_TOKEN,
        repo_type=HF_REPO_TYPE,
        space_sdk=HF_SPACE_SDK,
        private=False,
        exist_ok=True,
    )

    print(f"[INFO] Uploading {LOCAL_DEMO_FOLDER} -> {HF_REPO_ID}")
    commit_url = upload_folder(
        repo_id=HF_REPO_ID,
        folder_path=str(LOCAL_DEMO_FOLDER),
        path_in_repo=".",
        token=HF_TOKEN,
        repo_type=HF_REPO_TYPE,
        commit_message="Upload FoodExtract Gradio demo",
        ignore_patterns=[
            "upload_to_hf.py",
            "__pycache__/*",
            "*.pyc",
            ".venv/*",
            ".env",
            ".DS_Store",
        ],
    )
    print(f"[INFO] Upload complete: {commit_url}")
    print(f"[INFO] Space URL: https://huggingface.co/spaces/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
