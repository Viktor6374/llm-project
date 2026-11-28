"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS, WAREHOUSES
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import code_type, find_codes, normalize_code
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

SYSTEM = (
    "Ты ассистент сервисной службы компании \"ПолярТех\". Для сведений из документов используй "
    "search_knowledge для конкретной модели, кода или одного факта; summarize_topic — для частоты "
    "отказов, сравнений моделей и списков по линейке или корпусу. Актуальные остатки и цены узнавай только через ERP. "
    "Не угадывай номера деталей, цены и остатки. Для заказа нужно явно названное пользователем "
    "количество. Если пользователь прямо просит оформить заказ, вызови create_service_order после поиска "
    "подтверждённой детали; при confirm=False система остановит вызов и вернёт его на подтверждение, "
    "не создавая заказ. Указывай только подтверждённые факты. "
    "Если данных нет, ответь: \"В базе знаний не найдено\"."
)
REFUSAL = "В базе знаний не найдено."
ERP_UNAVAILABLE = "Данные ERP получить не удалось."
_NUMBER = re.compile(r"(?<![\w-])\d+(?:[ \u00a0]\d{3})*(?![\w-])")
_QTY_WORDS = {"один": 1, "одну": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5}
_QTY = re.compile(
    r"(?<![\w-])(\d+|" + "|".join(_QTY_WORDS) + r")\s*(?:шт\b|штук|единиц)|количеств\w*\s*:?\s*(\d+)",
    re.IGNORECASE,
)
_PRICE_UNIT = re.compile(r"\s*(?:руб|р\.|\u20bd)", re.IGNORECASE)
_STOCK_UNIT = re.compile(r"\s*(?:шт\b|штук)", re.IGNORECASE)
_PRICE_WORD = re.compile(r"цен|стои", re.IGNORECASE)
_STOCK_WORD = re.compile(r"остат|налич|склад", re.IGNORECASE)


def _result_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _known_codes(question: str, steps: list[ToolStep]) -> set[str]:
    codes = set(find_codes(question))
    for step in steps:
        if step.result and "error" not in step.result:
            codes.update(find_codes(_result_text(step.result)))
    return codes


def _numbers(text: str) -> set[int]:
    without_codes = text
    for code in find_codes(text):
        without_codes = re.sub(re.escape(code), " ", without_codes, flags=re.IGNORECASE)
    return {int(match.group().replace(" ", "").replace("\u00a0", "")) for match in _NUMBER.finditer(without_codes)}


def _supported_answer(answer: str, question: str, steps: list[ToolStep]) -> bool:
    evidence = question + " " + " ".join(
        _result_text(step.result) for step in steps if step.result and "error" not in step.result
    )
    if not (set(find_codes(answer)) <= set(find_codes(evidence)) and _numbers(answer) <= _numbers(evidence)):
        return False
    prices, stock = _erp_values(question, steps)
    for match in _NUMBER.finditer(answer):
        value = int(match.group().replace(" ", "").replace("\u00a0", ""))
        role = _number_role(answer, match)
        if role == "price" and value not in prices or role == "stock" and value not in stock:
            return False
    return True


def _number_role(text: str, match: re.Match) -> str | None:
    after = text[match.end():]
    if _PRICE_UNIT.match(after):
        return "price"
    if _STOCK_UNIT.match(after):
        return "stock"
    before = re.split(r"[,;:.!?\n]| и | а ", text[max(0, match.start() - 40):match.start()])[-1]
    price = [found.start() for found in _PRICE_WORD.finditer(before)]
    stock = [found.start() for found in _STOCK_WORD.finditer(before)]
    if price and (not stock or price[-1] > stock[-1]):
        return "price"
    return "stock" if stock else None


def _erp_values(question: str, steps: list[ToolStep]) -> tuple[set[int], set[int]]:
    prices: set[int] = set()
    stock = _requested_quantities(question)
    for step in steps:
        if step.result and "error" not in step.result:
            if isinstance(step.result.get("price_rub"), int | float):
                prices.add(int(step.result["price_rub"]))
            if isinstance(step.result.get("qty"), int):
                stock.add(step.result["qty"])
    return prices, stock


def _sources(steps: list[ToolStep]) -> list[str]:
    return list(
        dict.fromkeys(
            source
            for step in steps
            if step.result and "error" not in step.result
            for source in step.result.get("sources", [])
            if isinstance(source, str)
        )
    )


def _requested_quantities(question: str) -> set[int]:
    found = set()
    for match in _QTY.finditer(question):
        value = (match.group(1) or match.group(2)).lower()
        found.add(_QTY_WORDS.get(value) or int(value))
    return found


def _quantity_in_question(question: str, qty: object) -> bool:
    return isinstance(qty, int) and not isinstance(qty, bool) and _requested_quantities(question) == {qty}


def _order_error(args: dict) -> str | None:
    part = normalize_code(args.get("part_id")) if isinstance(args.get("part_id"), str) else None
    model = normalize_code(args.get("model")) if isinstance(args.get("model"), str) else None
    if part is None or code_type(part) != "Part":
        return "Неверный номер детали."
    if args.get("warehouse") not in WAREHOUSES:
        return "Неизвестный склад."
    qty = args.get("qty")
    if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
        return "Количество должно быть положительным целым числом."
    if model is None or code_type(model) != "Model":
        return "Неверный номер модели."
    return None


def _with_erp_error(answer: str, steps: list[ToolStep]) -> str:
    failed = any(step.error and step.error.startswith(ERP_UNAVAILABLE) for step in steps)
    if failed and ERP_UNAVAILABLE not in answer:
        return f"{answer} {ERP_UNAVAILABLE}".strip()
    return answer


def _with_order_id(answer: str, steps: list[ToolStep]) -> str:
    for step in steps:
        order_id = (step.result or {}).get("order_id") if step.tool == "create_service_order" else None
        if order_id and order_id not in answer:
            answer = f"{answer} Номер заказа: {order_id}."
    return answer


def _result(answer: str, steps: list[ToolStep], **kwargs) -> AgentResult:
    answer = _with_erp_error(_with_order_id(answer, steps), steps)
    return AgentResult(answer, steps=steps, sources=_sources(steps), **kwargs)


def _partial_result(steps: list[ToolStep]) -> AgentResult:
    facts: list[str] = []
    for step in steps:
        result = step.result
        if step.error or not result or "error" in result:
            continue
        part = result.get("part_id") or step.args.get("part_id")
        if isinstance(result.get("answer"), str):
            if not result.get("refused"):
                facts.append(result["answer"])
        elif step.tool == "check_stock" and part and result.get("qty") is not None:
            warehouse = result.get("warehouse") or step.args.get("warehouse")
            facts.append(f"{part}: {result['qty']} шт. на складе {warehouse}.")
        elif step.tool == "get_price" and part and result.get("price_rub") is not None:
            facts.append(f"{part}: {result['price_rub']} руб.")
    prefix = " ".join(facts)
    return _result(f"{prefix} Работа прервана: достигнут лимит шагов.".strip(), steps)


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    steps: list[ToolStep] = []
    cache: dict[tuple[str, str], tuple[dict | None, str | None]] = {}
    corrections = 0

    while True:
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())
        if not response.tool_calls:
            answer = response.content.strip()
            refused = "не найдено" in answer.lower()
            if not answer or not steps and not refused and not answer.endswith("?"):
                return _result(REFUSAL, steps, refused=True)
            if not _supported_answer(answer, question, steps):
                if corrections:
                    return _result(REFUSAL, steps, refused=True)
                corrections += 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "В ответе есть код или число, которого нет в вопросе и результатах инструментов, "
                            "либо цена и остаток перепутаны. Исправь ответ, используя только подтверждённые "
                            "данные. Если их нет, откажись."
                        ),
                    }
                )
                continue
            return _result(answer, steps, refused=refused)

        for call in response.tool_calls:
            if len(steps) >= max_steps:
                return _partial_result(steps)

            args = call.arguments
            step = ToolStep(call.name, args if isinstance(args, dict) else {})
            steps.append(step)
            if not isinstance(args, dict) or "_raw" in args:
                step.error = "Не удалось разобрать аргументы инструмента."
            elif call.name not in tools:
                step.error = f"Неизвестный инструмент: {call.name}."
            else:
                known = _known_codes(question, steps[:-1])
                part_id = args.get("part_id")
                part = normalize_code(part_id) if isinstance(part_id, str) else None
                if part_id is not None and (part is None or part not in known):
                    step.error = "Номер детали не подтверждён. Сначала вызови search_knowledge."
                elif call.name == "create_service_order" and (error := _order_error(args)):
                    step.error = error
                elif call.name == "create_service_order" and normalize_code(args["model"]) not in known:
                    step.error = "Модель не подтверждена вопросом или результатами инструментов."
                elif call.name == "create_service_order" and not _quantity_in_question(question, args.get("qty")):
                    step.error = "Количество не названо пользователем."
                    return _result("Сколько штук заказать? Укажите количество в вопросе.", steps)
                elif call.name == "create_service_order" and not confirm:
                    step.error = "Заказ ожидает подтверждения пользователя."
                    args = {**args, "part_id": part, "model": normalize_code(args["model"])}
                    return _result(
                        f"Подтвердите заказ: деталь {args['part_id']}, модель {args['model']}, "
                        f"склад {args['warehouse']}, количество {args['qty']}. Заказ пока не создан.",
                        steps,
                        needs_confirmation=True,
                        pending_action={"tool": call.name, "args": args},
                    )
                else:
                    key = (call.name, json.dumps(args, ensure_ascii=False, sort_keys=True, default=str))
                    if key in cache:
                        step.result, step.error = cache[key]
                    else:
                        try:
                            for attempt in range(3):
                                try:
                                    step.result = tools[call.name](**args)
                                    break
                                except ErpUnavailable as exc:
                                    if attempt == 2:
                                        step.error = f"{ERP_UNAVAILABLE} {exc}".strip()
                        except (TypeError, ValueError) as exc:
                            step.error = f"Неверные аргументы инструмента: {exc}"
                        if step.result and "error" in step.result:
                            step.error = str(step.result["error"])
                        cache[key] = (step.result, step.error)

            if step.result and "error" in step.result:
                step.error = str(step.result["error"])
            messages.append(tool_message(call, step.result if step.result is not None else {"error": step.error}))
            if len(steps) >= max_steps:
                return _partial_result(steps)
