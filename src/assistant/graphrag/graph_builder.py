"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import Entity, ExtractionResult


def _add_node(nodes: dict[str, dict], entity: Entity, chunk_id: str) -> None:
    node = nodes.setdefault(entity.node_id, {"type": entity.type, "name": entity.name, "sources": set()})
    node["sources"].add(chunk_id)


def _resolve_replacements(edges: dict[tuple[str, str], dict]) -> int:
    conflicts = 0
    for source, target in sorted(edges):
        forward, backward = edges.get((source, target)), edges.get((target, source))
        if forward is None or backward is None or {forward["type"], backward["type"]} != {"REPLACED_BY"}:
            continue
        conflicts += 1
        forward_weight, backward_weight = len(forward["sources"]), len(backward["sources"])
        if forward_weight >= backward_weight:
            del edges[target, source]
        if forward_weight <= backward_weight:
            del edges[source, target]
    return conflicts


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str], dict] = {}
    for result in results:
        for entity in result.entities:
            _add_node(nodes, entity, result.chunk_id)
        for relation in result.relations:
            if not relation.matches_schema():
                continue
            _add_node(nodes, relation.subject, result.chunk_id)
            _add_node(nodes, relation.object, result.chunk_id)
            edge = edges.setdefault(
                (relation.subject.node_id, relation.object.node_id), {"type": relation.type, "sources": set()}
            )
            edge["sources"].add(result.chunk_id)

    graph = nx.DiGraph(conflicts=_resolve_replacements(edges))
    for node, attributes in sorted(nodes.items()):
        graph.add_node(node, type=attributes["type"], name=attributes["name"], sources=sorted(attributes["sources"]))
    for (source, target), attributes in sorted(edges.items()):
        sources = sorted(attributes["sources"])
        graph.add_edge(source, target, type=attributes["type"], weight=len(sources), sources=sources)
    return graph
