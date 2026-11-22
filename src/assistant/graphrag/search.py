"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations, pairwise

import networkx as nx

from assistant.graphrag.communities import community_edges
from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

REFUSAL = "В базе знаний не найдено."
FACT_LIMIT = 8000
LOCAL_SYSTEM = (
    "Отвечай только по фактам из графа и сохраняй точное написание кодов. "
    "Если спрашивают об актуальной детали или её поставщике, используй последнюю деталь в цепочке REPLACED_BY. "
    f"Если фактов недостаточно, ответь «{REFUSAL}»"
)
MAP_SYSTEM = (
    "Ответь на вопрос только по этой сводке сообщества. Верни JSON с полями answer и score. "
    "Оцени полезность ответа целым числом от 0 до 10. Если сводка не помогает, верни пустой ответ и 0. "
    "Сохрани точные показатели и контекст, нужные для сравнения частичных ответов."
)
REDUCE_SYSTEM = (
    "Объедини частичные ответы сообществ. При подсчёте отказов одного узла в одной линейке "
    "сложи показатели разных сообществ и учитывай каждый факт один раз. "
    f"Если сведений недостаточно, ответь «{REFUSAL}»"
)


def _ask(llm: LLMClient, system: str, context: str, *, json_schema: dict | None = None) -> str:
    return llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": context}],
        json_schema=json_schema,
    ).content.strip()


def _answers_question(text: str) -> bool:
    return bool(text) and "в базе знаний не найдено" not in text.lower()


def _path_edges(graph: nx.DiGraph, view: nx.Graph, anchors: list[str]) -> list[tuple[str, str]]:
    edges: dict[tuple[str, str], None] = {}
    for first, second in combinations(anchors, 2):
        for path in nx.all_simple_paths(view, first, second, cutoff=3):
            for start, end in pairwise(path):
                if graph.has_edge(start, end):
                    edges[start, end] = None
                if graph.has_edge(end, start):
                    edges[end, start] = None
    return list(edges)


def _expand_replacements(
    graph: nx.DiGraph, edges: list[tuple[str, str]], *, include_suppliers: bool
) -> list[tuple[str, str]]:
    parts = {node for edge in edges for node in edge if graph.nodes[node]["type"] == "Part"}
    expanded = dict.fromkeys(edges)
    replaced_parts: set[str] = set()
    replacement_targets: set[str] = set()
    pending = sorted(parts, reverse=True)
    while pending:
        part = pending.pop()
        for target, data in graph[part].items():
            if data["type"] != "REPLACED_BY":
                continue
            replaced_parts.add(part)
            replacement_targets.add(target)
            expanded[part, target] = None
            if target not in parts:
                parts.add(target)
                pending.append(target)
    if include_suppliers:
        for part in sorted(replacement_targets - replaced_parts):
            for target, data in graph[part].items():
                if data["type"] == "SUPPLIED_BY":
                    expanded[part, target] = None
    return list(expanded)


def _add_sources(sources: set[str], attributes: dict) -> None:
    sources.update(chunk.split("#", 1)[0] for chunk in attributes.get("sources", []))


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. Для вопросов о поставщике добавляются связи
    с поставщиками конечных деталей этих цепочек. При одной опорной сущности берётся окрестность
    на глубину до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    graph = kb.graph
    codes = set(find_codes(question))
    code_nodes = {f"{code_type(code)}:{code}" for code in codes}
    if any(node not in graph for node in code_nodes):
        return Answer(REFUSAL, refused=True)

    name_nodes = {f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary)}
    anchors = sorted(code_nodes | {node for node in name_nodes if node in graph})
    if not anchors:
        return Answer(REFUSAL, refused=True)

    view = graph.to_undirected(as_view=True)
    if len(anchors) == 1:
        distances = nx.single_source_shortest_path_length(view, anchors[0], cutoff=2)
        edges = sorted(graph.subgraph(distances).edges, key=lambda edge: (max(distances[node] for node in edge), edge))
    else:
        edges = _path_edges(graph, view, anchors)
        lowered = question.lower()
        include_suppliers = "поставщик" in lowered or "поставля" in lowered
        edges = _expand_replacements(graph, edges, include_suppliers=include_suppliers)

    facts: list[str] = []
    lines: list[str] = []
    sources: set[str] = set()
    remaining = FACT_LIMIT
    for start, end in edges:
        attributes = graph.edges[start, end]
        fact = f"{graph.nodes[start]['name']} -{attributes['type']}-> {graph.nodes[end]['name']}"
        line = f"{len(facts) + 1}. {fact}"
        cost = len(line) + bool(lines)
        if cost > remaining:
            continue
        facts.append(fact)
        lines.append(line)
        _add_sources(sources, attributes)
        remaining -= cost

    if not facts:
        return Answer(REFUSAL, refused=True)

    context = "\n".join(lines)
    text = _ask(llm, LOCAL_SYSTEM, f"Вопрос: {question}\n\nФакты:\n{context}")
    supported_codes = codes.union(find_codes(context))
    if not _answers_question(text) or not set(find_codes(text)) <= supported_codes:
        return Answer(REFUSAL, refused=True)
    return Answer(text, sources=sorted(sources), facts=facts)


def _parse_partial(text: str) -> str | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    answer, score = value.get("answer"), value.get("score")
    if not isinstance(answer, str) or type(score) is not int or not 0 < score <= 10:
        return None
    answer = answer.strip()
    return answer if _answers_question(answer) else None


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой непустой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: корректные частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    partials: list[str] = []
    sources: set[str] = set()
    for community in kb.communities:
        if not community.summary.strip():
            continue
        text = _ask(llm, MAP_SYSTEM, f"Вопрос: {question}\n\nСводка:\n{community.summary}", json_schema=MAP_SCHEMA)
        answer = _parse_partial(text)
        if answer is None:
            continue
        partials.append(answer)
        for edge in community_edges(community, kb.graph):
            _add_sources(sources, kb.graph.edges[edge])
        for node in community.nodes:
            if node in kb.graph and kb.graph.degree(node) == 0:
                _add_sources(sources, kb.graph.nodes[node])

    if not partials:
        return Answer(REFUSAL, refused=True)

    context = "\n\n".join(partials)
    text = _ask(llm, REDUCE_SYSTEM, f"Вопрос: {question}\n\nЧастичные ответы:\n{context}")
    if not _answers_question(text):
        return Answer(REFUSAL, refused=True)
    return Answer(text, sources=sorted(sources))
