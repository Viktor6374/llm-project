"""Дополнительные проверки границ поиска и происхождения фактов."""

import networkx as nx

from assistant.graphrag.communities import detect_communities, summarize_community
from assistant.graphrag.search import global_search, local_search
from assistant.infra.schemas import MAP_SCHEMA, Community, KnowledgeBase
from assistant.infra.testing import FakeLLM
from tests.contract.test_communities_and_search import said_to_model


def add_fact(graph, subject, relation, target, source, weight=1):
    for node in (subject, target):
        if node not in graph:
            kind, name = node.split(":", 1)
            graph.add_node(node, type=kind, name=name, sources=[])
        graph.nodes[node]["sources"].append(source)
    graph.add_edge(subject, target, type=relation, weight=weight, sources=[source])


def test_community_detection_handles_empty_and_edgeless_graphs():
    graph = nx.DiGraph()
    assert detect_communities(graph) == []
    graph.add_nodes_from(["Part:P-4010", "Part:P-4020"])

    communities = detect_communities(graph)

    assert [item.community_id for item in communities] == [0, 1]
    assert sorted(item.nodes for item in communities) == [["Part:P-4010"], ["Part:P-4020"]]


def test_summary_receives_precomputed_failure_totals():
    graph = nx.DiGraph()
    component = "Component:пусковое реле"
    add_fact(graph, "Model:AR-210", "FAILED", component, "report_a.md#0", weight=17)
    add_fact(graph, "Model:AR-310", "FAILED", component, "report_b.md#0", weight=29)
    llm = FakeLLM(["Сводка отказов."])

    summarize_community(Community(0, list(graph)), graph, llm)

    assert len(llm.requests) == 1
    assert "46" in said_to_model(llm)


def test_local_uses_three_hop_path_and_its_replacement_chain():
    graph = nx.DiGraph()
    model = "Model:AR-210"
    procedure = "Procedure:SP-104"
    add_fact(graph, "Model:AR-310", "BELONGS_TO", "ProductLine:Арктика", "off_path.md#0")
    add_fact(graph, model, "BELONGS_TO", "ProductLine:Арктика", "manual.md#0")
    add_fact(graph, model, "HAS_PART", "Part:P-4010", "manual.md#1")
    add_fact(graph, "Part:P-4010", "IS_A", "Component:пусковое реле", "manual.md#2")
    add_fact(graph, "ErrorCode:E-17", "INDICATES", "Component:пусковое реле", "error.md#0")
    add_fact(graph, "ErrorCode:E-17", "FIXED_BY", procedure, "off_path.md#1")
    add_fact(graph, "Part:P-4010", "REPLACED_BY", "Part:P-4010B", "bulletin_a.md#1")
    add_fact(graph, "Part:P-4010B", "REPLACED_BY", "Part:P-4010C", "bulletin_b.md#2")
    llm = FakeLLM(["Актуальная деталь P-4010C."])

    answer = local_search("Какая деталь нужна для AR-210 при E-17?", KnowledgeBase(graph, [], {}), llm)

    assert not answer.refused
    assert len(llm.requests) == 1
    assert any("IS_A" in fact for fact in answer.facts)
    assert any("P-4010C" in fact for fact in answer.facts)
    assert set(answer.sources) == {"manual.md", "error.md", "bulletin_a.md", "bulletin_b.md"}
    assert len(answer.sources) == 4
    assert all("SP-104" not in fact and "AR-310" not in fact for fact in answer.facts)


def test_local_keeps_direct_indicates_fact_for_a_component_question(kb):
    llm = FakeLLM(["Ошибка E-17 указывает на пусковое реле."])

    answer = local_search("На отказ какого узла указывает ошибка E-17?", kb, llm)

    assert not answer.refused
    assert answer.facts == ["E-17 -INDICATES-> пусковое реле"]


def test_local_disconnected_anchors_do_not_fall_back_to_neighbours():
    graph = nx.DiGraph()
    add_fact(graph, "Model:AR-210", "HAS_PART", "Part:P-4010", "manual_a.md#0")
    add_fact(graph, "Model:AR-310", "HAS_PART", "Part:P-4020", "manual_b.md#0")
    llm = FakeLLM([])

    answer = local_search("Какая общая деталь у AR-210 и AR-310?", KnowledgeBase(graph, [], {}), llm)

    assert answer.refused and not answer.sources and not llm.requests
    assert "в базе знаний не найдено" in answer.text.lower()


def test_local_respects_prompt_budget_and_cites_only_presented_facts():
    graph = nx.DiGraph()
    for index in range(100):
        name = f"{index:02d} " + "узел " * 30 + "отказал"
        add_fact(graph, "Model:AR-210", "FAILED", f"Component:{name}", f"report_{index:02d}.md#0")
    llm = FakeLLM(["У модели AR-210 зарегистрированы отказы."])

    answer = local_search("Что отказывало у AR-210?", KnowledgeBase(graph, [], {}), llm)

    assert not answer.refused
    assert 0 < len(answer.facts) < graph.number_of_edges()
    numbered_facts = "\n".join(f"{number}. {fact}" for number, fact in enumerate(answer.facts, start=1))
    assert numbered_facts in said_to_model(llm)
    assert len(numbered_facts) <= 8000
    shown_components = {fact.split("-> ", 1)[1] for fact in answer.facts}
    assert set(answer.sources) == {f"report_{component[:2]}.md" for component in shown_components}


def test_global_reports_only_the_sources_of_useful_partials():
    graph = nx.DiGraph()
    communities = []
    for index, (model, part) in enumerate(
        [("AR-210", "P-4010"), ("AR-310", "P-4020"), ("AR-320", "P-4030"), ("AR-220", "P-4011")]
    ):
        nodes = [f"Model:{model}", f"Part:{part}"]
        add_fact(graph, nodes[0], "HAS_PART", nodes[1], f"manual_{model}.md#0")
        summary = {
            "AR-210": "У первой модели из линейки «Арктика» было 37 отказов пускового реле.",
            "AR-220": "У второй модели из линейки «Арктика» было 12 отказов пускового реле.",
        }.get(model, f"Модель {model} использует деталь {part}.")
        communities.append(Community(index, nodes, summary))
    llm = FakeLLM(
        [
            '{"answer": "У AR-210 было 37 отказов пускового реле", "score": 8}',
            '{"answer": "Не относится к вопросу", "score": 0}',
            "неразбираемый ответ",
            '{"answer": "У AR-220 было 12 отказов пускового реле", "score": 6}',
            "У AR-210 и AR-220 было 49 отказов пускового реле.",
        ]
    )

    answer = global_search(
        "Сколько отказов пускового реле у AR-210 и AR-220 вместе?", KnowledgeBase(graph, communities, {}), llm
    )

    assert not answer.refused and set(answer.sources) == {"manual_AR-210.md", "manual_AR-220.md"}
    assert "49 отказов" in answer.text
    assert [request["json_schema"] for request in llm.requests] == [MAP_SCHEMA] * 4 + [None]
    reduce_prompt = said_to_model(llm)
    assert "У AR-210 было 37 отказов" in reduce_prompt
    assert "У AR-220 было 12 отказов" in reduce_prompt
    assert "Не относится к вопросу" not in reduce_prompt


def test_global_model_ranking_is_checked_against_failed_edge_weights():
    graph = nx.DiGraph()
    add_fact(graph, "Model:AR-210", "BELONGS_TO", "ProductLine:Арктика", "manual_AR-210.md#0")
    add_fact(graph, "Model:AR-310", "BELONGS_TO", "ProductLine:Арктика", "manual_AR-310.md#0")
    add_fact(graph, "Model:AR-210", "FAILED", "Component:пусковое реле", "report_a.md#0", weight=7)
    add_fact(graph, "Model:AR-310", "FAILED", "Component:пусковое реле", "report_b.md#0", weight=2)
    summary = Community(0, sorted(graph.nodes), "У AR-210 7 отказов, у AR-310 2 отказа.")
    llm = FakeLLM(['{"answer": "AR-310", "score": 9}', "У AR-310 больше отказов."])
    kb = KnowledgeBase(graph, [summary], {"ProductLine": ["Арктика"]})

    answer = global_search("У какой модели линейки «Арктика» больше всего отказов?", kb, llm)

    assert "AR-210" in answer.text and "7" in answer.text
    assert answer.sources == ["report_a.md"]


def test_global_discontinued_parts_are_checked_against_model_and_replacement_edges():
    graph = nx.DiGraph()
    add_fact(graph, "Model:BZ-110", "BELONGS_TO", "ProductLine:Бриз", "manual_BZ-110.md#0")
    add_fact(graph, "Model:BZ-110", "HAS_PART", "Part:P-2054", "manual_BZ-110.md#0")
    add_fact(graph, "Model:BZ-210", "BELONGS_TO", "ProductLine:Бриз", "manual_BZ-210.md#0")
    add_fact(graph, "Model:BZ-210", "HAS_PART", "Part:P-2110", "manual_BZ-210.md#0")
    add_fact(graph, "Part:P-2054", "REPLACED_BY", "Part:P-2054B", "bulletin_a.md#0")
    add_fact(graph, "Part:P-2054B", "REPLACED_BY", "Part:P-2054C", "bulletin_b.md#0")
    add_fact(graph, "Part:P-2110", "REPLACED_BY", "Part:P-2110B", "bulletin_c.md#0")
    summary = Community(0, sorted(graph.nodes), "P-2054 и P-2110 заменены.")
    llm = FakeLLM(['{"answer": "P-2054 и P-2110 заменены.", "score": 9}', "Деталь P-9999 снята с производства."])
    kb = KnowledgeBase(graph, [summary], {"ProductLine": ["Бриз"]})

    answer = global_search("Какие детали линейки «Бриз» сняты с производства?", kb, llm)

    assert "P-2054" in answer.text and "P-2110" in answer.text
    assert "P-9999" not in answer.text
    assert answer.sources == ["bulletin_a.md", "bulletin_b.md", "bulletin_c.md", "manual_BZ-110.md", "manual_BZ-210.md"]
