"""Loading documents and their gold labels off disk.

A corpus directory holds one ``.txt`` per document plus a ``gold.json`` mapping
document id to labelled fields. Document id is the filename stem, which is what
ties the two together and what the Batch API's ``custom_id`` carries.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import SensorSpec

GOLD_FILENAME = "gold.json"


def load_documents(directory: Path) -> dict[str, str]:
    """Read every ``.txt`` in ``directory``, keyed by filename stem."""
    if not directory.is_dir():
        raise NotADirectoryError(f"no corpus directory at {directory}")
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(directory.glob("*.txt"))
    }


def load_gold(directory: Path) -> dict[str, SensorSpec]:
    """Read ``gold.json``, validating each entry against the schema.

    Validation here is the point: a typo in a hand-written label would otherwise
    surface as a mysterious model error, and a label the schema can't represent
    is a label the model can never match.
    """
    path = directory / GOLD_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"no {GOLD_FILENAME} in {directory}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    gold = {}
    for doc_id, fields in raw.items():
        try:
            gold[doc_id] = SensorSpec.model_validate(fields)
        except Exception as exc:
            raise ValueError(f"gold label for {doc_id!r} is invalid: {exc}") from None
    return gold


def load_corpus(directory: Path) -> tuple[dict[str, str], dict[str, SensorSpec]]:
    """Load documents and gold labels, checking that they line up.

    An unlabelled document or a label with no document is a corpus bug, and
    silently dropping either would quietly change what the reported score means.
    """
    documents = load_documents(directory)
    gold = load_gold(directory)

    unlabelled = set(documents) - set(gold)
    missing_docs = set(gold) - set(documents)
    if unlabelled:
        raise ValueError(f"documents with no gold label: {', '.join(sorted(unlabelled))}")
    if missing_docs:
        raise ValueError(f"gold labels with no document: {', '.join(sorted(missing_docs))}")

    return documents, gold
