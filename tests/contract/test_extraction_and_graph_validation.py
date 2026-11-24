"""Дополнительные проверки извлечения и сборки графа."""

import json

import networkx as nx

from assistant.graphrag.extraction import extract_chunk
from assistant.graphrag.graph_builder import build_graph
from assistant.infra.schemas import Chunk, Entity, ExtractionResult, Relation
from assistant.infra.testing import FakeLLM
from tests.contract.test_extraction_and_graph import reply, triples

MANUAL = Chunk(
    "manual_AR-210.md#0",
    "manual_AR-210.md",
    "# Руководство по эксплуатации: витрина AR-210\n\nВитрина AR-210 входит в линейку «Арктика».\n\n"
    "- пусковое реле – деталь P-1042, поставщик «ХолодКомплект»",
)


def test_reversed_relation_is_turned_by_entity_types():
    llm = FakeLLM([reply(relations=[("Component", "пусковое реле", "IS_A", "Part", "P-1042")])])

    result = extract_chunk(MANUAL, llm)

    assert triples(result) == {("Part:P-1042", "IS_A", "Component:пусковое реле")}


def test_code_with_wrong_type_gets_the_type_of_its_form():
    llm = FakeLLM([reply(relations=[("Model", "AR-210", "HAS_PART", "Model", "p1042")])])

    result = extract_chunk(MANUAL, llm)

    assert triples(result) == {("Model:AR-210", "HAS_PART", "Part:P-1042")}
    assert "Part:P-1042" in {entity.node_id for entity in result.entities}


def test_failed_outside_a_report_is_dropped():
    llm = FakeLLM([reply(relations=[("Model", "AR-210", "FAILED", "Component", "пусковое реле")])])

    result = extract_chunk(MANUAL, llm)

    assert triples(result) == set() and result.dropped == 1


def test_answer_wrapped_in_reasoning_and_fence_is_parsed():
    llm = FakeLLM(["<think>рассуждения</think>\n```json\n" + reply(entities=[("ProductLine", "Арктика")]) + "\n```"])

    result = extract_chunk(MANUAL, llm)

    assert len(llm.requests) == 1
    assert {entity.node_id for entity in result.entities} == {"ProductLine:Арктика"}


def test_retry_request_differs_from_the_first_one():
    llm = FakeLLM(["не JSON", reply()])

    extract_chunk(MANUAL, llm)

    assert llm.requests[0]["messages"] != llm.requests[1]["messages"]


def test_duplicates_inside_a_chunk_are_merged():
    relation = ("Model", "AR-210", "BELONGS_TO", "ProductLine", "Арктика")
    llm = FakeLLM([reply(relations=[relation, relation])])

    result = extract_chunk(MANUAL, llm)

    assert len(result.relations) == 1 and result.dropped == 0


def test_graph_keeps_its_attributes_after_json_round_trip():
    old, new = Entity("Part", "P-1042"), Entity("Part", "P-1042B")
    graph = build_graph([ExtractionResult("b.md#0", "b.md", [old, new], [Relation(old, "REPLACED_BY", new)])])

    restored = nx.node_link_graph(json.loads(json.dumps(nx.node_link_data(graph, edges="edges"))), edges="edges")

    assert dict(restored.nodes(data=True)) == dict(graph.nodes(data=True))
    assert restored.edges["Part:P-1042", "Part:P-1042B"] == {"type": "REPLACED_BY", "weight": 1, "sources": ["b.md#0"]}
