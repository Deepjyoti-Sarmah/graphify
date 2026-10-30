"""#4859: extracted fact owns an ID over an inference that also spells it.

A doc beside a code folder (`pkg.md` beside `pkg/test_x.py`) can mint the code
file's id. Both paths define the id, so the existing definer key ties and the
shorter label used to decide the survivor. The doc then owned `source_file`,
and the next doc re-extract deleted the code node. Provenance is the missing key.
"""
from __future__ import annotations

import copy
import json

import pytest
from networkx.readwrite import json_graph

from graphify.build import build_merge
from graphify.dedup import deduplicate_entities


CODE = {
    "id": "pkg_test_x",
    "label": "test_x.py",
    "file_type": "code",
    "source_file": "pkg/test_x.py",
    "source_location": "L1",
    "_origin": "ast",
}
CODE_METHOD = {
    "id": "pkg_test_x_run",
    "label": "run()",
    "file_type": "code",
    "source_file": "pkg/test_x.py",
    "source_location": "L3",
    "_origin": "ast",
}
DOC_NODE = {
    "id": "pkg_test_x",
    "label": "Tests",
    "file_type": "code",
    "source_file": "pkg.md",
    "_origin": "semantic",
}
DOC_OVERVIEW = {
    "id": "pkg_overview",
    "label": "Overview",
    "file_type": "document",
    "source_file": "pkg.md",
    "_origin": "semantic",
}
DOC_EDGE = {
    "source": "pkg_overview",
    "target": "pkg_test_x",
    "relation": "describes",
    "confidence": "EXTRACTED",
    "source_file": "pkg.md",
    "_origin": "semantic",
}


@pytest.mark.parametrize(
    "doc_label",
    ["Tests", "test_x.py", "test_x.py (tests)"],
    ids=["short-label", "equal-label", "long-label"],
)
@pytest.mark.parametrize("doc_first", [False, True], ids=["code-first", "doc-first"])
def test_ast_node_keeps_id_for_every_doc_label_length(doc_label, doc_first):
    doc = dict(copy.deepcopy(DOC_NODE), label=doc_label)
    code = copy.deepcopy(CODE)
    inputs = [doc, code] if doc_first else [code, doc]

    nodes, edges = deduplicate_entities(
        inputs, [copy.deepcopy(DOC_EDGE)], communities={}
    )

    survivor = next(n for n in nodes if n["id"] == "pkg_test_x")
    assert survivor["source_file"] == "pkg/test_x.py"
    assert survivor["label"] == "test_x.py"
    assert survivor["_origin"] == "ast"
    assert any(e["target"] == "pkg_test_x" for e in edges)


def test_ast_owner_survives_a_doc_reextract(tmp_path):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "directed": False,
                "multigraph": False,
                "graph": {},
                "hyperedges": [],
                "nodes": [copy.deepcopy(CODE), copy.deepcopy(CODE_METHOD)],
                "links": [
                    {
                        "source": "pkg_test_x",
                        "target": "pkg_test_x_run",
                        "relation": "contains",
                        "confidence": "EXTRACTED",
                        "source_file": "pkg/test_x.py",
                        "_origin": "ast",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    doc = {
        "nodes": [copy.deepcopy(DOC_NODE), copy.deepcopy(DOC_OVERVIEW)],
        "edges": [copy.deepcopy(DOC_EDGE)],
        "hyperedges": [],
    }

    merged = build_merge([doc], graph_path=graph_path)
    assert merged.nodes["pkg_test_x"]["source_file"] == "pkg/test_x.py"
    assert merged.nodes["pkg_test_x"]["label"] == "test_x.py"

    graph2 = tmp_path / "graph2.json"
    graph2.write_text(
        json.dumps(json_graph.node_link_data(merged, edges="links")),
        encoding="utf-8",
    )
    doc_without_claim = {
        "nodes": [copy.deepcopy(DOC_OVERVIEW)],
        "edges": [],
        "hyperedges": [],
    }
    merged2 = build_merge([doc_without_claim], graph_path=graph2)
    assert "pkg_test_x" in merged2
    assert merged2.nodes["pkg_test_x"]["source_file"] == "pkg/test_x.py"
