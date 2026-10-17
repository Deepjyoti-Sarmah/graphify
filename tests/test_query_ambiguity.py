"""`graphify query` refuses a name defined in several files instead of picking one (#4511).

`explain` and `path` already refuse such a name with exit 3 and the candidate list. `query` seeds from the
best-scoring node and `_pick_seeds` keeps one seed per label (#1766), so two same-named nodes in different files
that tie on score were decided by graph iteration order: the answer silently described one file and exited 0.
`--seed <name>` reported the same case as "No unique node match" and exited 0.
"""
from __future__ import annotations

import json

import graphify.__main__ as mainmod
from graphify.affected import format_affected
from graphify.benchmark import _query_subgraph_tokens
from graphify.cli import EXIT_AMBIGUOUS, EXIT_NOT_FOUND
from graphify.serve import _load_graph, _query_graph_text, _resolve_single_node


def _node(nid, label, source, loc, community=0):
    return {"id": nid, "label": label, "source_file": source, "source_location": loc, "community": community}


def _nodes():
    """`Response` is defined in both server.py and train.py; `Trainer` and `HttpServer` are unique."""
    return [
        _node("web::server_httpserver", "HttpServer", "server.py", "L1"),
        _node("web::server_response", "Response", "server.py", "L3"),
        _node("web::server_response_render", ".render()", "server.py", "L4"),
        _node("ml::train_response", "Response", "train.py", "L1", 1),
        _node("ml::train_response_compute", ".compute()", "train.py", "L2", 1),
        _node("ml::train_trainer", "Trainer", "train.py", "L3", 1),
    ]


_LINKS = [
    {"source": "web::server_response", "target": "web::server_response_render", "relation": "method"},
    {"source": "ml::train_response", "target": "ml::train_response_compute", "relation": "method"},
]


def _write_graph(tmp_path, nodes=None, reverse=False, name="graph.json", links=None):
    nodes = list(nodes if nodes is not None else _nodes())
    graph = {"directed": False, "multigraph": False, "graph": {},
             "nodes": list(reversed(nodes)) if reverse else nodes,
             "links": _LINKS if links is None else links}
    path = tmp_path / name
    path.write_text(json.dumps(graph))
    return path


def _run(monkeypatch, capsys, graph_path, *args):
    """Run `graphify query <args> --graph <path>`; return (stdout, exit code or None for a normal return)."""
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(mainmod.sys, "argv", ["graphify", "query", *args, "--graph", str(graph_path)])
    try:
        mainmod.main()
    except SystemExit as exc:
        return capsys.readouterr().out, exc.code
    return capsys.readouterr().out, None


# --- change 1: the keyword path refuses an auto-selected, ambiguous seed ---

def test_query_refuses_a_name_defined_in_two_files(monkeypatch, tmp_path, capsys):
    out, code = _run(monkeypatch, capsys, _write_graph(tmp_path), "Response")
    assert code == EXIT_AMBIGUOUS
    assert out.startswith("Ambiguous: 'Response' matches 2 nodes in different files.")
    for needle in ("server.py", "web::server_response", "train.py", "ml::train_response"):
        assert needle in out, needle
    assert "Traversal:" not in out, "a refusal must not come with a partial answer"


def test_query_refusal_does_not_depend_on_node_order(monkeypatch, tmp_path, capsys):
    """The bug: which file answered was decided by graph iteration order."""
    forward, fcode = _run(monkeypatch, capsys, _write_graph(tmp_path), "Response")
    reverse, rcode = _run(monkeypatch, capsys, _write_graph(tmp_path, reverse=True, name="rev.json"), "Response")
    assert fcode == rcode == EXIT_AMBIGUOUS
    assert sorted(forward.splitlines()) == sorted(reverse.splitlines())


def test_query_refuses_when_the_name_sits_inside_a_longer_question(monkeypatch, tmp_path, capsys):
    out, code = _run(monkeypatch, capsys, _write_graph(tmp_path), "who calls Response")
    assert code == EXIT_AMBIGUOUS and out.startswith("Ambiguous: ")


def test_query_path_scoped_name_answers_from_that_file(monkeypatch, tmp_path, capsys):
    """The retry the refusal asks for: `<path>::<symbol>` lifts one file's node above the other."""
    out, code = _run(monkeypatch, capsys, _write_graph(tmp_path), "train.py::Response")
    assert code is None
    assert "src=train.py" in out and "src=server.py" not in out


def test_query_a_path_word_in_the_question_breaks_the_tie(monkeypatch, tmp_path, capsys):
    out, code = _run(monkeypatch, capsys, _write_graph(tmp_path), "Response train")
    assert code is None
    assert "src=train.py" in out and "src=server.py" not in out


def test_query_unique_name_is_not_refused(monkeypatch, tmp_path, capsys):
    out, code = _run(monkeypatch, capsys, _write_graph(tmp_path), "Trainer")
    assert code is None and "NODE Trainer" in out


def test_query_without_a_match_still_exits_not_found(monkeypatch, tmp_path, capsys):
    out, code = _run(monkeypatch, capsys, _write_graph(tmp_path), "zzzznothing")
    assert code == EXIT_NOT_FOUND and out.strip() == "No matching nodes found."


def test_query_same_file_duplicates_are_not_refused(monkeypatch, tmp_path, capsys):
    """#1766's one-seed-per-label rule stays for duplicates inside one file: not a choice between files."""
    nodes = [_node("a::h1", "handler", "routes.py", "L1"), _node("a::h2", "handler", "routes.py", "L9"),
             _node("a::u", "users_model", "models.py", "L1")]
    out, code = _run(monkeypatch, capsys, _write_graph(tmp_path, nodes), "handler")
    assert code is None and "Start: ['handler']" in out


def test_query_graph_text_on_the_mcp_graph_returns_the_same_refusal(tmp_path):
    """The MCP tool returns `_query_graph_text` unchanged, on the directed graph `_load_graph` builds."""
    text = _query_graph_text(_load_graph(str(_write_graph(tmp_path))), "Response")
    assert text.startswith("Ambiguous: 'Response' matches 2 nodes in different files.")


def test_benchmark_does_not_count_a_refusal_as_an_answer(tmp_path):
    """A refusal is a few lines long; counting it as the answer would inflate the reduction ratio."""
    G = _load_graph(str(_write_graph(tmp_path)))
    assert _query_subgraph_tokens(G, "Response") == 0
    assert _query_subgraph_tokens(G, "Trainer") > 0


# --- change 4: `--seed <name>` separates "ambiguous" from "no match" ---

def test_seed_naming_two_files_is_refused_with_the_candidates(monkeypatch, tmp_path, capsys):
    out, code = _run(monkeypatch, capsys, _write_graph(tmp_path), "anything", "--seed", "Response")
    assert code == EXIT_AMBIGUOUS
    assert out.startswith("Ambiguous: 'Response' matches 2 nodes in different files.")
    assert "No unique node match" not in out
    assert "server.py" in out and "train.py" in out


def test_seed_path_scoped_form_resolves_to_that_file(monkeypatch, tmp_path, capsys):
    out, code = _run(monkeypatch, capsys, _write_graph(tmp_path), "anything", "--seed", "train.py::Response")
    assert code is None
    assert "Start (given): ['Response']" in out
    assert "src=train.py" in out and "src=server.py" not in out


def test_seed_node_id_still_resolves_the_ambiguous_name(monkeypatch, tmp_path, capsys):
    out, code = _run(monkeypatch, capsys, _write_graph(tmp_path), "anything", "--seed", "web::server_response")
    assert code is None and "src=server.py" in out and "src=train.py" not in out


def test_seed_refusal_wins_over_a_resolvable_seed_beside_it(tmp_path):
    """Skipping the ambiguous value would answer a narrower question than the one asked."""
    G = _load_graph(str(_write_graph(tmp_path)))
    text = _query_graph_text(G, "anything", seeds=["Trainer", "Response"])
    assert text.startswith("Ambiguous: 'Response'") and "Start" not in text


def test_seed_that_matches_nothing_is_still_skipped_not_refused(tmp_path):
    G = _load_graph(str(_write_graph(tmp_path)))
    text = _query_graph_text(G, "anything", seeds=["zzzznothing", "Trainer"])
    assert "Start (given): ['Trainer']" in text
    assert "Unresolved seeds (skipped): ['zzzznothing']" in text


def test_seed_that_matches_nothing_alone_keeps_its_message(tmp_path):
    G = _load_graph(str(_write_graph(tmp_path)))
    assert _query_graph_text(G, "anything", seeds=["zzzznothing"]) == "No unique node match for --seed: zzzznothing"


def test_affected_answers_an_ambiguous_label_exactly_as_before(tmp_path):
    """`resolve_seed` is shared with `affected` and must not change: still no unique match, no refusal text."""
    G = _load_graph(str(_write_graph(tmp_path)))
    assert format_affected(G, "Response") == "No unique node match for Response"


# --- the same relative path in two repos of a merged graph (#4512) ---

def _dup_nodes():
    """`dup.py` exists in both repos. A merged graph keeps `source_file` bare and tells the repos apart
    only by the node id and the `repo` attribute."""
    return [
        {**_node("web::dup_config", "Config", "dup.py", "L1", 0), "repo": "web"},
        {**_node("ml::dup_config", "Config", "dup.py", "L1", 1), "repo": "ml"},
    ]


def _dup_graph(tmp_path):
    return _write_graph(tmp_path, _dup_nodes(), links=[])


def test_query_refuses_same_basename_files_from_two_repos(monkeypatch, tmp_path, capsys):
    out, code = _run(monkeypatch, capsys, _dup_graph(tmp_path), "Config")
    assert code == EXIT_AMBIGUOUS
    assert "web/dup.py" in out and "ml/dup.py" in out
    assert "in dup.py" not in out, "the two repos must not read as one file"
    assert "web::dup_config" in out and "ml::dup_config" in out


def test_explain_same_basename_message_names_both_repos(tmp_path):
    G = _load_graph(str(_dup_graph(tmp_path)))
    nid, refusal = _resolve_single_node(G, "Config")
    assert nid is None and refusal.startswith("Ambiguous: ")
    assert "web/dup.py" in refusal and "ml/dup.py" in refusal
    assert "web::dup_config" in refusal and "ml::dup_config" in refusal


def test_full_node_id_answers_same_basename_query(monkeypatch, tmp_path, capsys):
    out, code = _run(monkeypatch, capsys, _dup_graph(tmp_path), "anything", "--seed", "web::dup_config")
    assert code is None
    node_lines = [line for line in out.splitlines() if line.startswith("NODE ")]
    # NODE lines carry no id, so the repos are told apart by community: web is 0, ml is 1.
    assert len(node_lines) == 1 and "src=dup.py" in node_lines[0] and "community=0" in node_lines[0]


def test_refusal_hint_for_same_basename_names_the_full_node_id(monkeypatch, tmp_path, capsys):
    """`web/dup.py::Config` does not resolve, so no path::symbol retry may be suggested."""
    _nid, explain_text = _resolve_single_node(_load_graph(str(_dup_graph(tmp_path))), "Config")
    query_text, _code = _run(monkeypatch, capsys, _dup_graph(tmp_path), "Config")
    for text in (explain_text, query_text):
        assert text.startswith("Ambiguous: "), text
        assert "web::dup_config" in text
        assert "::Config" not in text
        assert "path::symbol" not in text


def test_same_repo_different_paths_message_is_unchanged(monkeypatch, tmp_path, capsys):
    plain, plain_code = _run(monkeypatch, capsys, _write_graph(tmp_path), "Response")
    tagged_nodes = [{**n, "repo": "web"} for n in _nodes()]
    tagged, tagged_code = _run(monkeypatch, capsys, _write_graph(tmp_path, tagged_nodes, name="repo.json"), "Response")
    assert plain_code == tagged_code == EXIT_AMBIGUOUS
    assert "matches 2 nodes in different files." in tagged
    assert tagged == plain


def test_seed_naming_same_basename_files_is_refused(monkeypatch, tmp_path, capsys):
    out, code = _run(monkeypatch, capsys, _dup_graph(tmp_path), "anything", "--seed", "Config")
    assert code == EXIT_AMBIGUOUS and out.startswith("Ambiguous: ")
    assert "web/dup.py" in out and "ml/dup.py" in out
