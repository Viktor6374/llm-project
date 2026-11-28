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
    "Если спрашивают об актуальной детали или её поставщике, пройди всю цепочку REPLACED_BY до последней детали. "
    "Если в фактах есть прямая связь INDICATES или FIXED_BY, используй её для ответа и не отказывайся. "
    f"Если фактов недостаточно, ответь «{REFUSAL}»"
)
MAP_SYSTEM = (
    "Ответь только если сводка прямо содержит факт, который отвечает на вопрос. Верни краткий JSON "
    "с полями answer и score. Оцени прямую полезность целым числом от 0 до 10. Если в сводке нет "
    "подтверждения или она относится к другой модели/линейке, верни пустой answer и score 0. "
    "Не делай догадок и не переноси факты из одной линейки или модели на другую. Для сравнений "
    "сохраняй названия и точные числа из сводки."
)
REDUCE_SYSTEM = (
    "Объедини только частичные ответы, прямо относящиеся к вопросу. Учитывай их оценки: "
    "предпочитай высокие оценки, а низкие считай ненадёжными; не переноси факты между линейками "
    "и моделями. При подсчёте отказов одного узла в одной линейке сложи показатели разных "
    "сообществ и учитывай каждый факт один раз. Не придумывай недостающие названия или числа. "
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

    lowered = question.lower()
    relevant_types: set[str] | None = None
    if "указывает" in lowered or "на отказ какого" in lowered:
        relevant_types = {"INDICATES"}
    elif "регламент" in lowered:
        relevant_types = {"INDICATES", "FIXED_BY"}
    elif "поставщик" in lowered or "поставля" in lowered:
        relevant_types = {"HAS_PART", "IS_A", "REPLACED_BY", "SUPPLIED_BY"}
    elif "цен" in lowered or "стоим" in lowered:
        relevant_types = {"HAS_PART", "IS_A", "REPLACED_BY"}
    elif "замен" in lowered or "снята с производства" in lowered or "сняты с производства" in lowered:
        relevant_types = {"HAS_PART", "IS_A", "INDICATES", "REPLACED_BY"}
    elif "ошибка" in lowered and ("деталь" in lowered or "заказ" in lowered):
        relevant_types = {"HAS_PART", "IS_A", "INDICATES", "REPLACED_BY"}
    elif "какую деталь" in lowered or "какая деталь" in lowered or "номер детали" in lowered:
        relevant_types = {"HAS_PART", "IS_A", "INDICATES", "REPLACED_BY"}
    if relevant_types:
        relevant_edges = [edge for edge in edges if graph.edges[edge]["type"] in relevant_types]
        if relevant_edges:
            edges = relevant_edges

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


def _parse_partial(text: str) -> tuple[str, int] | None:
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
    return (answer, score) if _answers_question(answer) else None


def _graph_aggregate(question: str, kb: KnowledgeBase) -> tuple[str, list[str]] | None:
    """Compute common line-wide comparisons from explicit graph edges and their weights."""
    lines = [name for kind, name in find_names(question, kb.vocabulary) if kind == "ProductLine"]
    if len(lines) != 1:
        return None
    line_node = f"ProductLine:{lines[0]}"
    graph = kb.graph
    models = {
        source
        for source, target, data in graph.in_edges(line_node, data=True)
        if data.get("type") == "BELONGS_TO" and graph.nodes[source].get("type") == "Model"
    }
    if not models:
        return None

    question_lower = question.lower()
    if "сня" in question_lower and "производств" in question_lower and "детал" in question_lower:
        old_parts: set[str] = set()
        sources: set[str] = set()
        for model in models:
            for _, part, has_part in graph.out_edges(model, data=True):
                if has_part.get("type") != "HAS_PART":
                    continue
                current = part
                chain_sources: set[str] = set()
                seen: set[str] = set()
                while current not in seen:
                    seen.add(current)
                    next_parts = [
                        (target, edge)
                        for _, target, edge in graph.out_edges(current, data=True)
                        if edge.get("type") == "REPLACED_BY"
                    ]
                    if not next_parts:
                        break
                    current, replacement = next_parts[0]
                    chain_sources.update(replacement.get("sources", []))
                    old_parts.add(graph.nodes[part]["name"])
                if chain_sources:
                    sources.update(has_part.get("sources", []))
                    sources.update(chain_sources)
        if old_parts:
            answer = f"В линейке «{lines[0]}» сняты с производства детали {', '.join(sorted(old_parts))}."
            return answer, sorted({source.split("#", 1)[0] for source in sources})

    failed: dict[str, int] = {}
    by_model_sources: dict[str, set[str]] = {}
    by_component: dict[str, int] = {}
    component_sources: dict[str, set[str]] = {}
    for model in models:
        for _, component, edge in graph.out_edges(model, data=True):
            if edge.get("type") != "FAILED":
                continue
            weight = int(edge.get("weight", 1))
            failed[model] = failed.get(model, 0) + weight
            by_model_sources.setdefault(model, set()).update(edge.get("sources", []))
            by_component[component] = by_component.get(component, 0) + weight
            component_sources.setdefault(component, set()).update(edge.get("sources", []))

    if not failed:
        return None
    if "модел" in question_lower and ("больше всего" in question_lower or "наибольш" in question_lower):
        model = min(failed, key=lambda node: (-failed[node], node))
        answer = f"Больше всего отказов в линейке «{lines[0]}» у модели {graph.nodes[model]['name']} — {failed[model]}."
        sources = by_model_sources[model]
    elif "узел" in question_lower and ("чаще" in question_lower or "больше всего" in question_lower):
        component = min(by_component, key=lambda node: (-by_component[node], node))
        answer = (
            f"Чаще всего в линейке «{lines[0]}» отказывает узел "
            f"«{graph.nodes[component]['name']}» — {by_component[component]} случаев."
        )
        sources = component_sources[component]
    else:
        return None
    return answer, sorted({source.split("#", 1)[0] for source in sources})


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой непустой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: корректные частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    question_codes = set(find_codes(question))
    if any(f"{code_type(code)}:{code}" not in kb.graph for code in question_codes):
        return Answer(REFUSAL, refused=True)

    partials: list[str] = []
    sources: set[str] = set()
    for community in kb.communities:
        if not community.summary.strip():
            continue
        text = _ask(llm, MAP_SYSTEM, f"Вопрос: {question}\n\nСводка:\n{community.summary}", json_schema=MAP_SCHEMA)
        partial = _parse_partial(text)
        if partial is None:
            continue
        answer, score = partial
        if not set(find_codes(answer)) <= set(find_codes(community.summary)) | question_codes:
            continue
        partials.append(f"Оценка {score}/10: {answer}")
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
    # Для этих запросов используем проверяемые подсчёты по связям графа.
    aggregate = _graph_aggregate(question, kb)
    if aggregate:
        grounded_text, grounded_sources = aggregate
        ranked_nodes = [
            name
            for kind in ("Model", "Component")
            for name in kb.vocabulary.get(kind, [])
            if name.casefold() in grounded_text.casefold()
        ]
        if ranked_nodes and any(name.casefold() in text.casefold() for name in ranked_nodes):
            return Answer(text, sources=sorted(sources))
        return Answer(grounded_text, sources=grounded_sources)
    return Answer(text, sources=sorted(sources))
