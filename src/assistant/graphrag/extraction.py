"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json
import re

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, code_type, find_codes, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

MAX_TOKENS = 4096
RELATION_MEANINGS = {
    "BELONGS_TO": "модель входит в линейку",
    "HAS_PART": "в модели установлена деталь",
    "IS_A": "деталь является узлом этого вида",
    "INDICATES": "ошибка указывает на отказ узла",
    "FIXED_BY": "ошибка устраняется по регламенту",
    "REPLACED_BY": "старая деталь снята с производства и заменена новой",
    "SUPPLIED_BY": "деталь поставляет поставщик",
    "FAILED": "только в отчёте о выезде: у модели отказал узел",
}
REPORT_ONLY = {"FAILED"}
EXTRACTION_SYSTEM = (
    "Извлеки из текста сущности и связи графа знаний и верни JSON с полями entities и relations. "
    "Бери только то, что прямо сказано в тексте. Коды пиши латиницей через дефис: AR-210, P-1042B, E-17, SP-104. "
    "Названия линеек, узлов и поставщиков бери только из словаря, без кавычек и «ООО». "
    "Номера бюллетеней и отчётов, например SB-2025-03 и R-2025-017, – имена документов, а не сущности. "
    "Если сущностей схемы нет, верни пустые списки.\n\n"
    "Типы сущностей: {entity_types}.\n\nСвязи, направление строго от subject к object:\n{relations}\n\n"
    "Словарь:\n{vocabulary}"
)
RETRY_PROMPT = (
    "Ответ не соответствует схеме. Верни один объект JSON с полями entities и relations: "
    "у сущности поля type и name, у связи поля subject_type, subject, type, object_type и object."
)
EXAMPLE_TEXT = (
    "# Сервисный бюллетень SB-2025-99\n\n"
    "Деталь P-3030 (контроллер) снята с производства. Вместо неё устанавливается деталь P-3030B.\n\n"
    "Поставщик детали P-3030B – «АвтоматикаСевер»."
)
EXAMPLE_TRIPLES = [
    ("Part", "P-3030", "REPLACED_BY", "Part", "P-3030B"),
    ("Part", "P-3030", "IS_A", "Component", "контроллер"),
    ("Part", "P-3030B", "SUPPLIED_BY", "Supplier", "АвтоматикаСевер"),
]
_WRAPPERS = re.compile(r"<think>.*?</think>|```(?:json)?", re.DOTALL)


def _example_answer() -> str:
    keys = ("subject_type", "subject", "type", "object_type", "object")
    entities = {(kind, name): None for st, s, _, ot, o in EXAMPLE_TRIPLES for kind, name in ((st, s), (ot, o))}
    return json.dumps(
        {
            "entities": [{"type": kind, "name": name} for kind, name in entities],
            "relations": [dict(zip(keys, triple, strict=True)) for triple in EXAMPLE_TRIPLES],
        },
        ensure_ascii=False,
    )


def _messages(chunk: Chunk, names: dict[str, list[str]]) -> list[dict]:
    relations = "\n".join(
        f"- {kind}: {start} -> {end} ({RELATION_MEANINGS[kind]})" for kind, (start, end) in RELATION_SIGNATURES.items()
    )
    system = EXTRACTION_SYSTEM.format(
        entity_types=", ".join(ENTITY_TYPES),
        relations=relations,
        vocabulary="\n".join(f"- {kind}: {', '.join(values)}" for kind, values in names.items()),
    )
    codes = ", ".join(find_codes(chunk.text)) or "нет"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Текст:\n{EXAMPLE_TEXT}"},
        {"role": "assistant", "content": _example_answer()},
        {"role": "user", "content": f"Коды в тексте: {codes}\n\nТекст:\n{chunk.text}"},
    ]


def _has_fields(item: object, fields: list[str]) -> bool:
    return isinstance(item, dict) and all(isinstance(item.get(field), str) for field in fields)


def _parse(text: str) -> tuple[list[dict], list[dict]] | None:
    try:
        value = json.loads(_WRAPPERS.sub("", text))
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    entities, relations = value.get("entities"), value.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        return None
    properties = EXTRACTION_SCHEMA["properties"]
    if not all(_has_fields(item, properties["entities"]["items"]["required"]) for item in entities):
        return None
    if not all(_has_fields(item, properties["relations"]["items"]["required"]) for item in relations):
        return None
    return entities, relations


def _entity(kind: str, raw: str, codes: set[str], names: dict[str, list[str]]) -> Entity | None:
    if kind not in ENTITY_TYPES:
        return None
    code = normalize_code(raw)
    if code is not None:
        return Entity(code_type(code), code) if code in codes else None
    if kind in CODE_TYPES:
        return None
    name = canonical_name(kind, raw, names)
    return Entity(kind, name) if name else None


def _relation(subject: Entity | None, kind: str, target: Entity | None, doc_id: str) -> Relation | None:
    if subject is None or target is None or subject == target:
        return None
    if kind in REPORT_ONLY and not doc_id.startswith("report_"):
        return None
    for relation in (Relation(subject, kind, target), Relation(target, kind, subject)):
        if relation.matches_schema():
            return relation
    return None


def _ground(chunk: Chunk, raw_entities: list[dict], raw_relations: list[dict]) -> ExtractionResult:
    names = vocabulary()
    codes = set(find_codes(chunk.text))
    entities: dict[str, Entity] = {}
    relations: dict[tuple[str, str, str], Relation] = {}
    dropped = 0
    for item in raw_entities:
        entity = _entity(item["type"], item["name"], codes, names)
        if entity is None:
            dropped += 1
            continue
        entities.setdefault(entity.node_id, entity)
    for item in raw_relations:
        subject = _entity(item["subject_type"], item["subject"], codes, names)
        target = _entity(item["object_type"], item["object"], codes, names)
        relation = _relation(subject, item["type"], target, chunk.doc_id)
        if relation is None:
            dropped += 1
            continue
        relations.setdefault((relation.subject.node_id, relation.type, relation.object.node_id), relation)
        for end in (relation.subject, relation.object):
            entities.setdefault(end.node_id, end)
    return ExtractionResult(chunk.chunk_id, chunk.doc_id, list(entities.values()), list(relations.values()), dropped)


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит пример. Всё, что не подтверждается текстом, отбрасывается
    и считается в `dropped`: код, которого нет в чанке, название не из словаря, связь с отброшенным
    концом или вне схемы. Тип кода определяется по его виду, перепутанное направление связи
    исправляется по типам концов, связь `FAILED` принимается только из отчётов о выезде.
    Нечитаемый ответ повторяется один раз с замечанием, иначе кеш вернул бы тот же ответ.
    После второй неудачи возвращается пустой результат.
    """
    messages = _messages(chunk, vocabulary())
    for _ in range(2):
        content = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=MAX_TOKENS).content
        parsed = _parse(content)
        if parsed is not None:
            return _ground(chunk, *parsed)
        messages = [*messages, {"role": "assistant", "content": content}, {"role": "user", "content": RETRY_PROMPT}]
    return ExtractionResult(chunk.chunk_id, chunk.doc_id)
