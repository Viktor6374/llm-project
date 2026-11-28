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
PROCEDURE = Chunk(
    "procedure_SP-112.md#0",
    "procedure_SP-112.md",
    "# Регламент SP-112: Замена циркуляционного насоса\n\n"
    "Регламент SP-112 применяется при ошибке E-22.",
)


def test_reversed_relation_is_turned_by_entity_types():
    llm = FakeLLM([reply(relations=[("Component", "пусковое реле", "IS_A", "Part", "P-1042")])])

    result = extract_chunk(MANUAL, llm)

    assert ("Part:P-1042", "IS_A", "Component:пусковое реле") in triples(result)


def test_code_with_wrong_type_gets_the_type_of_its_form():
    llm = FakeLLM([reply(relations=[("Model", "AR-210", "HAS_PART", "Model", "p1042")])])

    result = extract_chunk(MANUAL, llm)

    assert ("Model:AR-210", "HAS_PART", "Part:P-1042") in triples(result)
    assert "Part:P-1042" in {entity.node_id for entity in result.entities}


def test_failed_outside_a_report_is_dropped():
    llm = FakeLLM([reply(relations=[("Model", "AR-210", "FAILED", "Component", "пусковое реле")])])

    result = extract_chunk(MANUAL, llm)

    assert not any(relation == "FAILED" for _, relation, _ in triples(result))
    assert result.dropped == 1


def test_answer_wrapped_in_reasoning_and_fence_is_parsed():
    llm = FakeLLM(["<think>рассуждения</think>\n```json\n" + reply(entities=[("ProductLine", "Арктика")]) + "\n```"])

    result = extract_chunk(MANUAL, llm)

    assert len(llm.requests) == 1
    assert "ProductLine:Арктика" in {entity.node_id for entity in result.entities}


def test_retry_request_differs_from_the_first_one():
    llm = FakeLLM(["не JSON", reply()])

    extract_chunk(MANUAL, llm)

    assert llm.requests[0]["messages"] != llm.requests[1]["messages"]


def test_duplicates_inside_a_chunk_are_merged():
    relation = ("Model", "AR-210", "BELONGS_TO", "ProductLine", "Арктика")
    llm = FakeLLM([reply(relations=[relation, relation])])

    result = extract_chunk(MANUAL, llm)

    assert sum(relation.type == "BELONGS_TO" for relation in result.relations) == 1
    assert result.dropped == 0


def test_manual_structure_recovers_model_part_component_and_supplier_links():
    result = extract_chunk(MANUAL, FakeLLM([reply()]))

    assert triples(result) == {
        ("Model:AR-210", "HAS_PART", "Part:P-1042"),
        ("Part:P-1042", "IS_A", "Component:пусковое реле"),
        ("Part:P-1042", "SUPPLIED_BY", "Supplier:ХолодКомплект"),
    }


def test_procedure_structure_recovers_explicit_error_mapping():
    result = extract_chunk(PROCEDURE, FakeLLM([reply()]))

    assert triples(result) == {("ErrorCode:E-22", "FIXED_BY", "Procedure:SP-112")}


def test_graph_keeps_its_attributes_after_json_round_trip():
    old, new = Entity("Part", "P-1042"), Entity("Part", "P-1042B")
    graph = build_graph([ExtractionResult("b.md#0", "b.md", [old, new], [Relation(old, "REPLACED_BY", new)])])

    restored = nx.node_link_graph(json.loads(json.dumps(nx.node_link_data(graph, edges="edges"))), edges="edges")

    assert dict(restored.nodes(data=True)) == dict(graph.nodes(data=True))
    assert restored.edges["Part:P-1042", "Part:P-1042B"] == {"type": "REPLACED_BY", "weight": 1, "sources": ["b.md#0"]}
