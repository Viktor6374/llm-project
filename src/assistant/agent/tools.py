"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import code_type, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


_PART = {"type": "string", "description": "Код детали, например P-1042B"}
_WAREHOUSE = {"type": "string", "enum": WAREHOUSES}

TOOL_SCHEMAS: list[dict] = [
    _schema(
        "search_knowledge",
        "Найти факты о модели, ошибке, узле, детали и её замене в документах.",
        {"question": {"type": "string"}},
        ["question"],
    ),
    _schema(
        "summarize_topic",
        "Ответить на обзорный вопрос по сводкам сообществ графа.",
        {"question": {"type": "string"}},
        ["question"],
    ),
    _schema(
        "check_stock",
        "Узнать остаток детали на складе в ERP.",
        {"part_id": _PART, "warehouse": _WAREHOUSE},
        ["part_id", "warehouse"],
    ),
    _schema(
        "get_price",
        "Узнать актуальную цену детали в ERP.",
        {"part_id": _PART},
        ["part_id"],
    ),
    _schema(
        "create_service_order",
        "Оформить явно запрошенный пользователем заказ. При confirm=False система остановит этот вызов "
        "и вернёт параметры заказа для подтверждения, не создавая заказ.",
        {
            "part_id": _PART,
            "warehouse": _WAREHOUSE,
            "qty": {"type": "integer", "minimum": 1},
            "model": {"type": "string", "description": "Код модели, например AR-210"},
        },
        ["part_id", "warehouse", "qty", "model"],
    ),
]


def _part_id(value: object) -> str | None:
    code = normalize_code(value) if isinstance(value, str) else None
    return code if code and code_type(code) == "Part" else None


def _model_id(value: object) -> str | None:
    code = normalize_code(value) if isinstance(value, str) else None
    return code if code and code_type(code) == "Model" else None


def _erp_error(exc: ErpNotFound | ErpRejected) -> dict:
    if isinstance(exc, ErpNotFound):
        return {"error": f"В ERP данных нет: {exc}"}
    return {"error": f"ERP отклонила запрос: {exc}"}


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """

    def search_knowledge(question: str | None = None) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Нужен непустой вопрос."}
        answer = local_search(question, kb, llm)
        return {
            "answer": answer.text,
            "facts": answer.facts,
            "sources": answer.sources,
            "refused": answer.refused,
        }

    def summarize_topic(question: str | None = None) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Нужен непустой вопрос."}
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def check_stock(part_id: str | None = None, warehouse: str | None = None) -> dict:
        part = _part_id(part_id)
        if part is None:
            return {"error": "Неверный номер детали."}
        if warehouse not in WAREHOUSES:
            return {"error": "Неизвестный склад."}
        try:
            return erp.get_stock(part, warehouse)
        except (ErpNotFound, ErpRejected) as exc:
            return _erp_error(exc)

    def get_price(part_id: str | None = None) -> dict:
        part = _part_id(part_id)
        if part is None:
            return {"error": "Неверный номер детали."}
        try:
            return erp.get_price(part)
        except (ErpNotFound, ErpRejected) as exc:
            return _erp_error(exc)

    def create_service_order(
        part_id: str | None = None,
        warehouse: str | None = None,
        qty: int | None = None,
        model: str | None = None,
    ) -> dict:
        part = _part_id(part_id)
        model_code = _model_id(model)
        if part is None:
            return {"error": "Неверный номер детали."}
        if warehouse not in WAREHOUSES:
            return {"error": "Неизвестный склад."}
        if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
            return {"error": "Количество должно быть положительным целым числом."}
        if model_code is None:
            return {"error": "Неверный номер модели."}
        try:
            return erp.create_order(part, warehouse, qty, model_code)
        except (ErpNotFound, ErpRejected) as exc:
            return _erp_error(exc)

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
