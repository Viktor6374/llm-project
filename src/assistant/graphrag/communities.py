"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

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
                "Напиши сводку сообщества до 150 слов по переданным данным. Укажи линейки, модели и замены деталей. "
                "Приведи готовые числа отказов по моделям и узлам каждой линейки, начиная с наибольших значений. "
                "Число отказов определяется весом связей FAILED."
            ),
        },
        {"role": "user", "content": context},
    ]
    return " ".join(llm.chat(messages).content.split()[:150])
