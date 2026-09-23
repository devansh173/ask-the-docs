"""The golden dataset.

Each entry pairs a question with a reference answer and the documentation pages
that actually contain the answer. Relevance is recorded at *page* level
(``relevant_urls``) rather than chunk level on purpose: chunk ids change every
time the chunk size or the scraper changes, so chunk-level labels would silently
rot and quietly invalidate the eval. Page URLs are stable.

Recording relevant pages is what makes the retrieval half of the evaluation
possible without an LLM judge - recall, MRR and nDCG are computed by comparing
retrieved URLs against these labels, which costs nothing and is deterministic.
The reference answers drive the generation metrics (faithfulness, answer
relevancy), which do need a judge model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import GOLDEN_DIR

log = logging.getLogger(__name__)

GOLDEN_PATH = GOLDEN_DIR / "golden_set.jsonl"


@dataclass
class GoldenItem:
    id: str
    question: str
    reference_answer: str
    doc_set: str
    relevant_urls: list[str] = field(default_factory=list)
    # "lookup" questions have one right answer; "howto" and "comparison" are
    # more diffuse. Tracked so scores can be read per difficulty rather than
    # as one average that hides where the pipeline actually struggles.
    kind: str = "lookup"
    # Questions deliberately outside the corpus. The pipeline should decline
    # these, not answer them - without negatives, a system that answers
    # everything confidently scores the same as one that knows its limits.
    answerable: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def load(path: Path | None = None) -> list[GoldenItem]:
    path = path or GOLDEN_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No golden set at {path}. Run `python -m askthedocs.evals.build_golden`."
        )
    items = [
        GoldenItem(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    log.info(
        "loaded %d golden questions (%d answerable, %d out-of-scope)",
        len(items),
        sum(i.answerable for i in items),
        sum(not i.answerable for i in items),
    )
    return items


def save(items: list[GoldenItem], path: Path | None = None) -> Path:
    path = path or GOLDEN_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(item.to_json() for item in items) + "\n", encoding="utf-8"
    )
    log.info("wrote %d golden questions to %s", len(items), path)
    return path


def sample(items: list[GoldenItem], limit: int | None) -> list[GoldenItem]:
    """Take a representative slice, not the first N.

    Slicing the list head-first is wrong here: the golden set is grouped by doc
    set with the out-of-scope questions last, so `--limit 8` yields eight Qdrant
    questions and zero negatives. Abstention then scores over an empty set and
    the run silently measures one third of the corpus.

    This spreads the budget across doc sets and always keeps at least one
    out-of-scope question, so a small run stays a miniature of the full one.
    Selection is deterministic (evenly spaced within each group) so repeat runs
    are comparable.
    """
    if not limit or limit >= len(items):
        return items

    answerable = [i for i in items if i.answerable]
    negatives = [i for i in items if not i.answerable]

    # Keep negatives in proportion, but never zero when any exist.
    n_neg = max(1, round(limit * len(negatives) / len(items))) if negatives else 0
    n_neg = min(n_neg, len(negatives), max(0, limit - 1))
    n_pos = limit - n_neg

    groups: dict[str, list[GoldenItem]] = {}
    for item in answerable:
        groups.setdefault(item.doc_set, []).append(item)

    picked: list[GoldenItem] = []
    names = sorted(groups)
    # Round-robin across doc sets so every set is represented before any is
    # sampled twice.
    cursors = {name: 0 for name in names}
    while len(picked) < n_pos and any(cursors[n] < len(groups[n]) for n in names):
        for name in names:
            if len(picked) >= n_pos:
                break
            idx = cursors[name]
            if idx < len(groups[name]):
                picked.append(groups[name][idx])
                cursors[name] = idx + 1

    picked.extend(negatives[:n_neg])
    # Restore the original ordering so ids stay readable in reports.
    order = {item.id: n for n, item in enumerate(items)}
    return sorted(picked, key=lambda i: order[i.id])


def stats(items: list[GoldenItem]) -> dict:
    by_set: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for item in items:
        by_set[item.doc_set] = by_set.get(item.doc_set, 0) + 1
        by_kind[item.kind] = by_kind.get(item.kind, 0) + 1
    return {
        "total": len(items),
        "answerable": sum(i.answerable for i in items),
        "out_of_scope": sum(not i.answerable for i in items),
        "by_doc_set": by_set,
        "by_kind": by_kind,
    }
