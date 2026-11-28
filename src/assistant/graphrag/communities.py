"""Сообщества графа и их сводки. Требование FR-3."""

import re
from collections import Counter

import networkx as nx

from assistant.infra.grounding import find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    projection = nx.Graph()
    projection.add_nodes_from(sorted(graph))
    for source, target in sorted(graph.edges):
        weight = graph[source][target].get("weight", 1)
        if projection.has_edge(source, target):
            weight += projection[source][target]["weight"]
        projection.add_edge(source, target, weight=weight)

    if not projection.size(weight="weight"):
        return [Community(index, [node]) for index, node in enumerate(projection)]

    partition = nx.community.louvain_communities(projection, weight="weight", seed=42)
    groups = sorted(sorted(group) for group in partition)
    return [Community(index, nodes) for index, nodes in enumerate(groups)]


def community_edges(community: Community, graph: nx.DiGraph) -> list[tuple[str, str]]:
    """Возвращает рёбра, начало которых принадлежит сообществу."""
    return sorted(graph.out_edges(community.nodes))


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """

    def name(node: str) -> str:
        return graph.nodes[node]["name"]

    members = [f"{graph.nodes[node]['type']}: {name(node)}" for node in sorted(community.nodes)]
    edges = community_edges(community, graph)
    model_lines = {source: name(target) for source, target in edges if graph[source][target]["type"] == "BELONGS_TO"}
    facts = []
    relation_weights: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    by_component: Counter[tuple[str, str]] = Counter()
    for source, target in edges:
        edge = graph[source][target]
        kind, weight = edge["type"], edge["weight"]
        line = model_lines.get(source)
        facts.append(f"{name(source)} -{kind}-> {name(target)} (вес {weight})")
        relation_weights[kind] += weight
        if kind == "FAILED":
            model = f"{name(source)} (линейка {line})" if line else name(source)
            by_model[model] += weight
            by_component[line or "линейка неизвестна", name(target)] += weight

    statistics = [f"{kind}: {weight}" for kind, weight in sorted(relation_weights.items())]
    statistics.extend(f"Отказы модели {model}: {total}" for model, total in by_model.most_common())
    statistics.extend(
        f"Отказы узла {component}, линейка {line}: {total}" for (line, component), total in by_component.most_common()
    )
    context = "\n".join(["Участники:", *members, "Связи:", *facts, "Подсчёты по весам связей:", *statistics])
    messages = [
        {
            "role": "system",
            "content": (
                "Составь сводку до 150 слов, используя только переданные узлы, связи и готовые подсчёты. "
                "Не придумывай коды, названия, связи и числа. Указывай модель и линейку только если они есть "
                "среди участников или в связях. Число отказов бери только из строк «Отказы модели» и "
                "«Отказы узла»; веса INDICATES, IS_A и SUPPLIED_BY не являются числом отказов. "
                "Если сообщество не содержит связей, кратко сообщи, что фактов для сводки нет."
            ),
        },
        {"role": "user", "content": context},
    ]
    generated = " ".join(llm.chat(messages).content.split()[:150])

    def numeric_values(value: str) -> set[str]:
        without_codes = value
        for code in find_codes(value):
            without_codes = without_codes.replace(code, " ")
        return set(re.findall(r"(?<![\w-])\d+(?![\w-])", without_codes))

    claims_no_facts = "фактов для сводки нет" in generated.lower()
    if (
        not (claims_no_facts and facts)
        and set(find_codes(generated)) <= set(find_codes(context))
        and numeric_values(generated) <= numeric_values(context)
    ):
        return generated

    # Не допускаем коды и числа, которых нет в графе.
    fallback = "Узлы: " + ", ".join(members) + ". Связи: " + "; ".join(facts) + ". Подсчёты: " + "; ".join(
        statistics
    )
    return " ".join(fallback.split()[:150])
