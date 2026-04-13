# -*- coding: utf-8 -*-
# file: hf_checkpoint_utils.py
# author: YANG, HENG <hy345@exeter.ac.uk>
# Copyright (C) 2024. All Rights Reserved.
#
# Hugging Face Hub-native checkpoint and index loading for PyABSA.
#
# The legacy mechanism stored zipped checkpoints on a HF Space and relied on
# raw URL downloads. This module replaces that with the official
# huggingface_hub APIs: `hf_hub_download` for the JSON index file and
# `snapshot_download` for whole checkpoint repos. Each entry with a
# `repo_id` is resolved to a local directory in the HF cache.
#
# Schema v2 index entry:
#   {
#     "repo_id": "yangheng/pyabsa-apc-english",
#     "revision": "v2.4.3",            # optional; defaults to "main"
#     "training_model": "FAST-LSA-T-V2-Deberta",
#     "training_dataset": "APCDatasetList.English",
#     "language": "English",
#     "pyabsa_min_version": "2.4.0",
#     "pyabsa_max_version": null,
#     "author": "H, Yang"
#   }
import json
import os
from typing import Optional, Dict, Any

from pyabsa.utils.pyabsa_utils import fprint

# Override points for private mirrors or forks.
INDEX_REPO_ID = os.environ.get("PYABSA_INDEX_REPO", "yangheng/pyabsa-index")
INDEX_REPO_TYPE = os.environ.get("PYABSA_INDEX_REPO_TYPE", "model")
INDEX_FILENAME = os.environ.get("PYABSA_INDEX_FILE", "checkpoints.json")


def _hf():
    # Lazy import so the rest of pyabsa keeps working if huggingface_hub
    # is ever missing (e.g. stripped-down deploys).
    from huggingface_hub import hf_hub_download, snapshot_download

    return hf_hub_download, snapshot_download


def fetch_index_from_hub() -> Optional[Dict[str, Any]]:
    """Download `checkpoints.json` from the index repo on the HF Hub.

    Returns parsed JSON or None if fetch failed. Caller decides fallback.
    """
    try:
        hf_hub_download, _ = _hf()
        path = hf_hub_download(
            repo_id=INDEX_REPO_ID,
            filename=INDEX_FILENAME,
            repo_type=INDEX_REPO_TYPE,
        )
        with open(path, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception as e:
        fprint("HF Hub index fetch failed ({}): {}".format(INDEX_REPO_ID, e))
        return None


def is_hub_entry(entry: Dict[str, Any]) -> bool:
    """Schema v2 entries carry a HF `repo_id`."""
    return isinstance(entry, dict) and bool(entry.get("repo_id"))


def download_checkpoint_from_hub(entry: Dict[str, Any]) -> str:
    """Snapshot-download a checkpoint repo; return the local directory.

    Honours the HF cache, supports resume and revision pinning. The
    returned directory contains the flat checkpoint files (e.g. `.config`,
    `.state_dict`, `.tokenizer`) just like the unzipped legacy bundle.
    """
    _, snapshot_download = _hf()
    return snapshot_download(
        repo_id=entry["repo_id"],
        revision=entry.get("revision") or "main",
        repo_type=entry.get("repo_type") or "model",
    )


def download_dataset_from_hub(repo_id: str, revision: Optional[str] = None) -> str:
    _, snapshot_download = _hf()
    return snapshot_download(
        repo_id=repo_id,
        revision=revision or "main",
        repo_type="dataset",
    )
