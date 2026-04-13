# -*- coding: utf-8 -*-
# file: checkpoint_utils.py
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Copyright (C) 2022. All Rights Reserved.
#
# Resolution layer between PyABSA's CheckpointManager classes and the
# canonical checkpoint index hosted on the Hugging Face Hub.
#
# Schema v2 only — the legacy zip-on-Space distribution mechanism has been
# removed. See pyabsa.framework.checkpoint_class.hf_checkpoint_utils for
# the actual Hub I/O.
import sys
from typing import Any, Dict, Union

from packaging import version
from termcolor import colored

from pyabsa.framework.checkpoint_class.hf_checkpoint_utils import (
    download_checkpoint_from_hub,
    fetch_index_from_hub,
    is_hub_entry,
)
from pyabsa.framework.flag_class import TaskCodeOption
from pyabsa.utils.pyabsa_utils import fprint

try:  # Prefer installed package metadata to avoid importing top-level pyabsa
    from importlib.metadata import version as _pkg_version
except Exception:  # pragma: no cover
    try:
        from importlib_metadata import version as _pkg_version
    except Exception:
        _pkg_version = None

try:
    current_version = _pkg_version("pyabsa") if _pkg_version else None
except Exception:
    current_version = None

if not current_version:
    current_version = getattr(sys.modules.get("pyabsa"), "__version__", "0.0.0")


def parse_checkpoint_info(t_checkpoint_map, task_code, show_ckpts=False):
    """Print available checkpoints for a task. Schema v2 entries only."""
    fprint(
        "*" * 10,
        colored(
            "Available {} model checkpoints for Version:{} (this version)".format(
                task_code, current_version
            ),
            "green",
        ),
        "*" * 10,
    )
    if not show_ckpts:
        return t_checkpoint_map
    for checkpoint_name, entry in t_checkpoint_map.items():
        fprint("-" * 100)
        fprint("Checkpoint Name: {}".format(checkpoint_name))
        for k, v in entry.items():
            fprint("{}: {}".format(k, v))
        fprint("-" * 100)
    return t_checkpoint_map


def available_checkpoints(
    task_code: TaskCodeOption = None, show_ckpts: bool = False
) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Return the available checkpoint entries for a task code.

    The index is fetched from the HF Hub index repo (configurable via
    `PYABSA_INDEX_REPO`). Each entry is a schema-v2 dict carrying at
    least a `repo_id`. Entries are filtered by the running pyabsa
    version against optional `pyabsa_min_version` / `pyabsa_max_version`.

    Returns:
        - the full index dict if `task_code` is None
        - the per-task `{name: entry}` map otherwise
        - `{}` if the index cannot be fetched or no entries match
    """
    if task_code is None:
        fprint("Please specify the task code, e.g. from pyabsa import TaskCodeOption")

    checkpoint_map = fetch_index_from_hub()
    if checkpoint_map is None:
        fprint(
            colored(
                "Could not fetch the PyABSA checkpoint index from the HF Hub. "
                "Set PYABSA_INDEX_REPO if you use a private mirror.",
                "red",
            )
        )
        return {} if task_code else {}

    if checkpoint_map.get("schema_version") != 2:
        raise RuntimeError(
            "Unsupported checkpoint index schema. Expected schema_version=2, "
            "got {}. Please update pyabsa or repoint PYABSA_INDEX_REPO.".format(
                checkpoint_map.get("schema_version")
            )
        )

    if not task_code:
        return checkpoint_map

    task_map = checkpoint_map.get(task_code.upper(), {})
    cur = version.parse(current_version)
    t_checkpoint_map = {}
    for name, entry in task_map.items():
        min_ver = entry.get("pyabsa_min_version") or "0.0.0"
        max_ver = entry.get("pyabsa_max_version")
        if cur < version.parse(min_ver):
            continue
        if max_ver is not None and cur > version.parse(max_ver):
            continue
        t_checkpoint_map[name] = entry

    parse_checkpoint_info(t_checkpoint_map, task_code, show_ckpts)
    return t_checkpoint_map


def download_checkpoint(task: str, language: str, checkpoint: dict) -> str:
    """Resolve a schema-v2 checkpoint entry to a local directory.

    The directory comes from huggingface_hub's snapshot cache and contains
    the unpacked checkpoint files (e.g. `.config`, `.state_dict`,
    `.tokenizer`) at its root.
    """
    fprint(
        colored(
            "Notice: The pretrained model are used for testing, "
            "it is recommended to train the model on your own custom datasets",
            "red",
        )
    )

    if not is_hub_entry(checkpoint):
        raise ValueError(
            "Checkpoint entry has no 'repo_id'. The legacy zip-based "
            "distribution has been removed in pyabsa 2.5.0; please update "
            "the checkpoint index to schema v2 (see pyabsa.utils.hf_upload)."
        )

    local_dir = download_checkpoint_from_hub(checkpoint)
    fprint(
        colored(
            "Resolved checkpoint '{}/{}' from HF Hub repo '{}' (revision: {}) -> {}".format(
                task.upper(),
                language,
                checkpoint["repo_id"],
                checkpoint.get("revision") or "main",
                local_dir,
            ),
            "green",
        )
    )
    return local_dir
