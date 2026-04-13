# -*- coding: utf-8 -*-
# file: hf_upload.py
# author: YANG, HENG <hy345@exeter.ac.uk>
# Copyright (C) 2024. All Rights Reserved.
#
# Publish checkpoints and datasets to the Hugging Face Hub using the
# official SDK. Designed to replace the old "zip up and push to a Space"
# flow which is no longer supported by the Hub.
#
# Entry points (also exposed as console scripts; see setup.py):
#   publish_checkpoint  -> pyabsa-upload-ckpt
#   publish_dataset     -> pyabsa-upload-data
#   update_index        -> pyabsa-index-update
#
# Auth: uses the standard HF auth chain. Run `huggingface-cli login`
# once, or pass `hf_token=` / set `HF_TOKEN` in the environment.
import json
import os
import tempfile
from typing import Any, Dict, Optional

from pyabsa.framework.checkpoint_class.hf_checkpoint_utils import (
    INDEX_REPO_ID,
    INDEX_REPO_TYPE,
    INDEX_FILENAME,
)


def _api(hf_token: Optional[str] = None):
    from huggingface_hub import HfApi

    return HfApi(token=hf_token or os.environ.get("HF_TOKEN"))


def publish_checkpoint(
    local_dir: str,
    repo_id: str,
    revision: str = "main",
    private: bool = False,
    commit_message: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> str:
    """Upload a local checkpoint directory to a HF Model repo.

    The directory should contain the unzipped checkpoint files
    (`.config`, `.state_dict`, `.tokenizer`, ...) at its root, exactly
    as consumed by the predictor classes.
    """
    from huggingface_hub import create_repo

    if not os.path.isdir(local_dir):
        raise ValueError("local_dir does not exist or is not a directory: " + local_dir)

    api = _api(hf_token)
    create_repo(
        repo_id, repo_type="model", exist_ok=True, private=private, token=api.token
    )
    # If revision is not "main", create/update the branch after the push.
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message or "Publish PyABSA checkpoint",
        revision=revision if revision and revision != "main" else None,
    )
    return "https://huggingface.co/{}/tree/{}".format(repo_id, revision or "main")


def publish_dataset(
    local_dir: str,
    repo_id: str,
    private: bool = False,
    commit_message: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> str:
    """Upload a local dataset directory to a HF Dataset repo."""
    from huggingface_hub import create_repo

    if not os.path.isdir(local_dir):
        raise ValueError("local_dir does not exist or is not a directory: " + local_dir)

    api = _api(hf_token)
    create_repo(
        repo_id, repo_type="dataset", exist_ok=True, private=private, token=api.token
    )
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message or "Publish PyABSA dataset",
    )
    return "https://huggingface.co/datasets/{}".format(repo_id)


def update_index(
    index: Dict[str, Any],
    repo_id: str = INDEX_REPO_ID,
    repo_type: str = INDEX_REPO_TYPE,
    filename: str = INDEX_FILENAME,
    commit_message: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> str:
    """Push a new `checkpoints.json` to the index repo.

    `index` must already conform to schema v2 and include `schema_version: 2`
    at the top level.
    """
    from huggingface_hub import create_repo

    if index.get("schema_version") != 2:
        raise ValueError(
            "Index must declare 'schema_version': 2 to be published via the new flow"
        )

    api = _api(hf_token)
    create_repo(repo_id, repo_type=repo_type, exist_ok=True, token=api.token)

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf8"
    ) as tf:
        json.dump(index, tf, ensure_ascii=False, indent=2)
        tmp_path = tf.name
    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message or "Update checkpoint index",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return "https://huggingface.co/{}/blob/main/{}".format(repo_id, filename)


# ---- CLI ----


def _cli_publish_checkpoint():
    import argparse

    p = argparse.ArgumentParser(description="Upload a PyABSA checkpoint to HF Hub")
    p.add_argument("local_dir", help="Unzipped checkpoint directory")
    p.add_argument("--repo", required=True, help="HF repo id, e.g. yangheng/pyabsa-apc-english")
    p.add_argument("--revision", default="main")
    p.add_argument("--private", action="store_true")
    p.add_argument("--message", default=None)
    args = p.parse_args()
    url = publish_checkpoint(
        args.local_dir,
        repo_id=args.repo,
        revision=args.revision,
        private=args.private,
        commit_message=args.message,
    )
    print(url)


def _cli_publish_dataset():
    import argparse

    p = argparse.ArgumentParser(description="Upload a PyABSA dataset to HF Hub")
    p.add_argument("local_dir", help="Local dataset directory")
    p.add_argument("--repo", required=True, help="HF dataset repo id")
    p.add_argument("--private", action="store_true")
    p.add_argument("--message", default=None)
    args = p.parse_args()
    url = publish_dataset(
        args.local_dir,
        repo_id=args.repo,
        private=args.private,
        commit_message=args.message,
    )
    print(url)


def _cli_update_index():
    import argparse

    p = argparse.ArgumentParser(description="Publish checkpoints.json to the index repo")
    p.add_argument("index_file", help="Local checkpoints.json (schema v2)")
    p.add_argument("--repo", default=INDEX_REPO_ID)
    p.add_argument("--message", default=None)
    args = p.parse_args()
    with open(args.index_file, "r", encoding="utf8") as f:
        index = json.load(f)
    url = update_index(index, repo_id=args.repo, commit_message=args.message)
    print(url)
