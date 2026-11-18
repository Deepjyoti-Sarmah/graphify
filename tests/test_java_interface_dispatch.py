"""Java member-level interface dispatch (#5319).

`this.worker.run()` on a `Worker` field resolves to the interface's own
`.run()`; the implementer's `.run()` is a separate node, so a directed walk
stops at the interface. `resolve_java_interface_dispatch` adds the same
`dispatches_to` edge the C#/Swift/Go passes do.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from graphify.extract import extract


def _extract(tmp_path, files):
    for name, body in files.items():
        (tmp_path / name).write_text(body)
    old = os.getcwd()
    try:
        os.chdir(tmp_path)
        r = extract([Path(n) for n in files], cache_root=Path(tempfile.mkdtemp()))
    finally:
        os.chdir(old)
    return [e for e in r["edges"] if e["relation"] == "dispatches_to"], r


def _id(r, label, file):
    return next(n["id"] for n in r["nodes"]
                if n["label"] == label and n.get("source_file", "").endswith(file))


ISSUE = {
    "Worker.java": "public interface Worker { void run(); }",
    "EmailWorker.java": "public class EmailWorker implements Worker {\n"
                        "    public void run() { send(); }\n"
                        "    private void send() {}\n}",
    "Job.java": "public class Job {\n    private final Worker worker;\n"
                "    public Job(Worker worker) { this.worker = worker; }\n"
                "    public void execute() { this.worker.run(); }\n}",
}


def test_issue_repro_links_interface_method_to_implementer(tmp_path):
    d, r = _extract(tmp_path, ISSUE)
    assert [(e["source"], e["target"]) for e in d] == [
        (_id(r, ".run()", "Worker.java"), _id(r, ".run()", "EmailWorker.java"))]
    assert d[0]["confidence"] == "INFERRED" and d[0]["confidence_score"] == 0.85


def test_execute_reaches_implementer_by_directed_walk(tmp_path):
    _, r = _extract(tmp_path, ISSUE)
    adj = {}
    for e in r["edges"]:
        adj.setdefault(e["source"], []).append(e["target"])
    seen, todo = set(), [_id(r, ".execute()", "Job.java")]
    while todo:
        n = todo.pop()
        if n not in seen:
            seen.add(n); todo += adj.get(n, [])
    assert _id(r, ".send()", "EmailWorker.java") in seen


def test_two_implementers_both_linked_at_075(tmp_path):
    d, _ = _extract(tmp_path, {
        "W.java": "public interface W { void run(); }",
        "A.java": "public class A implements W { public void run() {} }",
        "B.java": "public class B implements W { public void run() {} }"})
    assert sorted(e["confidence_score"] for e in d) == [0.75, 0.75]


def test_overloads_pair_by_signature(tmp_path):
    d, r = _extract(tmp_path, {
        "W.java": "public interface W { void run(); void run(int x); }",
        "A.java": "public class A implements W { public void run() {} public void run(int x) {} }"})
    meta = {n["id"]: n.get("metadata") or {} for n in r["nodes"]}
    assert {(meta[e["source"]]["arity"], meta[e["target"]]["arity"]) for e in d} == {(0, 0), (1, 1)}


def test_enum_and_record_implementers_link(tmp_path):
    d, _ = _extract(tmp_path, {
        "W.java": "public interface W { void run(); }",
        "E.java": "public enum E implements W { X; public void run() {} }",
        "R.java": "public record R(int a) implements W { public void run() {} }"})
    assert {Path(e["source_file"]).name for e in d} == {"E.java", "R.java"}


def test_constructor_named_like_interface_method_is_not_a_target(tmp_path):
    d, r = _extract(tmp_path, {
        "F.java": "public interface F { void Widget(); }",
        "Widget.java": "public class Widget implements F { public Widget() {} }"})
    assert d == []
    ctor = _id(r, ".Widget()", "Widget.java")
    assert next(n for n in r["nodes"] if n["id"] == ctor)["metadata"]["kind"] == "constructor"


def test_default_method_not_overridden_gets_no_edge(tmp_path):
    d, _ = _extract(tmp_path, {
        "W.java": "public interface W { default void run() {} }",
        "A.java": "public class A implements W {}"})
    assert d == []


def test_kotlin_class_with_same_named_interface_is_not_dispatched(tmp_path):
    d, _ = _extract(tmp_path, {
        "W.java": "public interface W { void run(); }",
        "K.kt": "class K : W { override fun run() {} }"})
    assert d == []


def test_java_pass_is_registered_for_java_only():
    import graphify.extract  # noqa: F401  (registers resolvers)
    from graphify.resolver_registry import registered_resolvers
    [r] = [r for r in registered_resolvers() if r.name == "java_interface_dispatch"]
    assert r.suffixes == frozenset({".java"})
