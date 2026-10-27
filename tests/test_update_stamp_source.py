"""Regression tests for #4811: the merged extract must not stamp a failed doc.

``--update`` reads this run's extraction from ``.graphify_extract.json``, then
overwrites that same path with the whole merged graph so Step 4 can build from
it. Anything that reads the file afterwards as "what this run produced" sees
every doc's old nodes as fresh output:

* re-running the merge block stamps a changed doc whose chunk failed;
* running core Step 9 after an update does the same.

Both paths mark the doc done, so ``detect_incremental`` never re-queues it and
the graph keeps serving the pre-edit nodes.

The fix stamps from ``.graphify_semantic.json`` instead: Part B writes it with
this run's semantic output (cache hits included) and the merge never overwrites
it. These tests run the real rendered blocks through bash, so the shell wrapper
is exercised too. A first version of the sibling #4807 fix passed its test by
running the extracted Python tail directly and hid an unescaped quote that broke
the block whenever the partial list existed (#4811 harness finding).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.skillgen import gen  # noqa: E402


def _claude_artifacts() -> dict[str, str]:
    return {
        a.path: a.content
        for a in gen.render_all(gen.load_platforms(), only="claude")
    }


_ARTIFACTS = _claude_artifacts()
_CORE = _ARTIFACTS["graphify/skill.md"]
_UPDATE = next(
    content for path, content in _ARTIFACTS.items()
    if path.endswith("references/update.md")
)


def _bash_blocks(text: str) -> list[str]:
    return re.findall(r"```bash\n(.*?)```", text, re.S)


def _block(text: str, anchor: str) -> str:
    return next(b for b in _bash_blocks(text) if anchor in b)


BLOCKS = {
    "start": _block(_UPDATE, "rm -f graphify-out/.graphify_partial.txt"),
    "detect": _block(_UPDATE, "result = detect_incremental("),
    "populate": _block(_UPDATE, "'all_files': r.get('files'"),
    "merge": _block(_UPDATE, "build_merge("),
    "step9": _block(_CORE, "Update cumulative cost tracker"),
}


class Project:
    """A one-project checkout driven by the real rendered runbook blocks."""

    def __init__(self, root: Path):
        self.root = root
        self.out = root / "graphify-out"
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / ".graphify_python").write_text(sys.executable, encoding="utf-8")
        self.manifest = self.out / "manifest.json"

    def _env(self) -> dict[str, str]:
        return {
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT),
            "GRAPHIFY_OUT": "graphify-out",
        }

    def py(self, source: str) -> str:
        proc = subprocess.run(
            [sys.executable, "-c", source],
            cwd=self.root,
            env=self._env(),
            capture_output=True,
            text=True,
        )
        if proc.returncode:
            raise AssertionError(f"{proc.stdout}\n{proc.stderr}")
        return proc.stdout

    def bash(self, block: str) -> None:
        proc = subprocess.run(
            [
                "bash",
                "-e",
                "-c",
                block.replace("INPUT_PATH", ".").replace("IS_DIRECTED", "False"),
            ],
            cwd=self.root,
            env=self._env(),
            capture_output=True,
            text=True,
        )
        if proc.returncode:
            raise AssertionError(f"{proc.stdout}\n{proc.stderr}")

    def run(self, name: str) -> None:
        self.bash(BLOCKS[name])

    # -- state ---------------------------------------------------------
    def doc(self, name: str, body: str) -> str:
        (self.root / name).write_text(body, encoding="utf-8")
        return name

    def node(self, name: str, suffix: str = "widget") -> dict:
        return {
            "id": f"{Path(name).stem}_{suffix}",
            "label": "X",
            "file_type": "document",
            "source_file": name,
        }

    def seed_built(self, docs: list[str], *, failed: list[str] | None = None) -> None:
        """A prior successful build: both hashes set, nodes in graph.json."""
        for d in docs:
            self.doc(d, f"# {d}\nWidget\n")
        self.py(
            "import json\nfrom pathlib import Path\n"
            "from graphify.cli import _stamped_manifest_files\n"
            "from graphify.detect import detect, save_manifest\n"
            f"x = {{'nodes': [{{'id': Path(d).stem + '_widget', 'source_file': d}} for d in {docs!r}], "
            "'edges': [], 'hyperedges': []}\n"
            "save_manifest(_stamped_manifest_files(detect(Path('.'))['files'], x, Path('.')), root='.')\n"
        )
        (self.out / "graph.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {"id": f"{Path(d).stem}_widget", "label": "Widget", "source_file": d}
                        for d in docs
                    ],
                    "edges": [],
                    "hyperedges": [],
                }
            ),
            encoding="utf-8",
        )
        if failed:
            manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
            for d in failed:
                row = manifest[d]
                row["ast_hash"] = ""
                row["semantic_hash"] = ""
            self.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    def start_update(self) -> None:
        self.run("start")

    def update_pipeline(self) -> None:
        self.run("detect")
        self.run("populate")

    def part_b(self, nodes: list[dict]) -> None:
        (self.out / ".graphify_semantic.json").write_text(
            json.dumps(
                {
                    "nodes": nodes,
                    "edges": [],
                    "hyperedges": [],
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            ),
            encoding="utf-8",
        )

    def part_c(self, nodes: list[dict]) -> None:
        (self.out / ".graphify_extract.json").write_text(
            json.dumps(
                {
                    "nodes": nodes,
                    "edges": [],
                    "hyperedges": [],
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            ),
            encoding="utf-8",
        )

    def merge(self) -> None:
        self.run("merge")

    def step9(self) -> None:
        self.run("step9")

    def queue(self) -> set[str]:
        raw = self.py(
            "import json\nfrom pathlib import Path\nfrom graphify.detect import detect_incremental\n"
            "print(json.dumps(sorted(Path(f).name for fl in "
            "detect_incremental(Path('.'))['new_files'].values() for f in fl)))"
        )
        return set(json.loads(raw))


def _failed_chunk_setup(p: Project) -> tuple[str, str]:
    """A built doc, changed, with this run's chunk failing (no semantic output)."""
    notes = p.doc("notes.md", "# Notes\nWidget\n")
    p.seed_built([notes])
    p.doc(notes, "# Notes\nWidget and Sprocket\n")
    p.start_update()
    p.update_pipeline()
    p.part_b([])
    p.part_c([])
    return notes, "notes.md"


def _partial_setup(p: Project) -> tuple[str, str]:
    """A changed doc B3.5 only read partway: head node + partial list entry."""
    long = p.doc("long.md", "".join(f"line {i} Widget\n" for i in range(3000)))
    p.seed_built([long])
    p.doc(long, "# Long\nWidget and Sprocket\n")
    p.start_update()
    p.update_pipeline()
    (p.out / ".graphify_partial.txt").write_text(long + "\n", encoding="utf-8")
    p.part_b([p.node(long, "head")])
    p.part_c([p.node(long, "head")])
    return long, "long.md"


# --------------------------------------------------------------- the bug paths


def test_rerunning_the_update_merge_keeps_a_failed_doc_queued(tmp_path):
    p = Project(tmp_path)
    _failed_chunk_setup(p)

    p.merge()
    assert p.queue() == {"notes.md"}, "the first merge must leave the failed doc queued"

    p.merge()
    assert p.queue() == {"notes.md"}, (
        "#4811: the second merge read the merged graph and stamped notes.md"
    )


def test_step9_after_an_update_keeps_a_failed_doc_queued(tmp_path):
    p = Project(tmp_path)
    _failed_chunk_setup(p)

    p.merge()
    p.step9()
    assert p.queue() == {"notes.md"}, (
        "#4811: Step 9 read the merged graph and stamped notes.md"
    )


def test_step9_after_an_update_does_not_stamp_an_earlier_failure(tmp_path):
    p = Project(tmp_path)
    notes, pending = "notes.md", "pending.md"
    p.doc(notes, "# Notes\nWidget\n")
    p.doc(pending, "# Pending\nSprocket\n")
    p.seed_built([notes, pending], failed=[pending])
    p.doc(notes, "# Notes\nWidget and Sprocket\n")
    p.start_update()
    p.update_pipeline()
    p.part_b([])
    p.part_c([])

    p.merge()
    p.step9()
    assert p.queue() == {notes, pending}, (
        "#4811: Step 9 stamped a doc that failed in an earlier run and was not "
        "even dispatched this run"
    )


def test_step9_without_a_semantic_file_fails_closed(tmp_path):
    p = Project(tmp_path)
    _failed_chunk_setup(p)
    (p.out / ".graphify_semantic.json").unlink()

    p.merge()
    p.step9()
    assert p.queue() == {"notes.md"}, (
        "a missing semantic file must not fall back to the merged extract; "
        "fail closed and re-queue the doc"
    )


# ------------------------------------------------------- the partial-list paths


def test_a_partly_read_doc_survives_a_merge_rerun(tmp_path):
    p = Project(tmp_path)
    _partial_setup(p)

    p.merge()
    assert p.queue() == {"long.md"}, "the first merge must not stamp a partial read"
    assert (p.out / ".graphify_partial.txt").exists(), (
        "the merge deleted the partial list, so a rerun can no longer honour it"
    )

    p.merge()
    assert p.queue() == {"long.md"}, (
        "#4811: the rerun stamped a partly-read doc after the list was deleted"
    )


def test_a_partly_read_doc_survives_step9_after_an_update(tmp_path):
    p = Project(tmp_path)
    _partial_setup(p)

    p.merge()
    p.step9()
    assert p.queue() == {"long.md"}, (
        "#4811: Step 9 stamped a partly-read doc because the merge had already "
        "deleted the partial list"
    )


def test_the_next_run_cleans_up_the_partial_list(tmp_path):
    p = Project(tmp_path)
    _partial_setup(p)
    p.merge()
    assert (p.out / ".graphify_partial.txt").exists()

    (p.out / ".graphify_partial.txt").write_text("long.md\n", encoding="utf-8")
    p.start_update()
    assert not (p.out / ".graphify_partial.txt").exists(), (
        "the update start hook must drop a list a prior update left behind (#4814)"
    )


# ------------------------------------------------ full-build equivalence guards


def test_step9_after_a_full_build_still_stamps_a_cached_doc(tmp_path):
    p = Project(tmp_path)
    doc = p.doc("guide.md", "# Guide\nWidget\n")
    p.py(
        "import json\nfrom pathlib import Path\nfrom graphify.detect import detect\n"
        "Path('graphify-out/.graphify_detect.json').write_text("
        "json.dumps({'files': detect(Path('.'))['files']}))"
    )
    p.part_b([p.node(doc)])
    p.part_c([p.node(doc)])

    p.step9()
    assert p.queue() == set(), (
        "a doc served from cache in a full build produced output this run and "
        "must stay stamped"
    )


def test_step9_after_a_full_build_leaves_a_failed_doc_queued(tmp_path):
    p = Project(tmp_path)
    doc = p.doc("guide.md", "# Guide\nWidget\n")
    p.py(
        "import json\nfrom pathlib import Path\nfrom graphify.detect import detect\n"
        "Path('graphify-out/.graphify_detect.json').write_text("
        "json.dumps({'files': detect(Path('.'))['files']}))"
    )
    p.part_b([])
    p.part_c([])

    p.step9()
    assert p.queue() == {doc}, "a failed full-build doc must stay queued"


def test_step9_after_an_update_keeps_a_healthy_doc_stamped(tmp_path):
    p = Project(tmp_path)
    notes = p.doc("notes.md", "# Notes\nWidget\n")
    guide = p.doc("guide.md", "# Guide\nSprocket\n")
    p.seed_built([notes, guide])
    p.doc(notes, "# Notes\nWidget and Sprocket\n")
    p.start_update()
    p.update_pipeline()
    p.part_b([p.node(notes)])
    p.part_c([p.node(notes)])

    p.merge()
    p.step9()
    assert p.queue() == set(), "the untouched healthy doc must stay stamped"


# --------------------------------------------------------------- code contract


def test_the_shipped_blocks_stamp_from_the_semantic_file():
    """Pin the seam, not just the outcome, so a later refactor cannot undo it."""
    merge = BLOCKS["merge"]
    step9 = BLOCKS["step9"]

    assert ".graphify_semantic.json" in merge
    assert "_stamped_manifest_files(incremental['files'], _semantic," in merge
    assert "_stamped_manifest_files(_corpus, _semantic," in step9
    assert "_pf.unlink(missing_ok=True)" not in merge, (
        "the merge must keep the partial list until the next run starts (#4811)"
    )
