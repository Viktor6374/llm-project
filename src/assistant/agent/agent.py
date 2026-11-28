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
_NUMBER = re.compile(r"(?<![\w-])\d+(?:[ \u00a0]\d{3})*(?![\w-])")


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
    return set(find_codes(answer)) <= set(find_codes(evidence)) and _numbers(answer) <= _numbers(evidence)


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


def _quantity_in_question(question: str, qty: object) -> bool:
    return isinstance(qty, int) and not isinstance(qty, bool) and qty > 0 and qty in _numbers(question)


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


def _partial_result(steps: list[ToolStep]) -> AgentResult:
    facts: list[str] = []
    for step in steps:
        result = step.result or {}
        if "error" in result:
            continue
        if isinstance(result.get("answer"), str) and not result.get("refused"):
            facts.append(result["answer"])
        elif step.tool == "check_stock":
            facts.append(f"{result.get('part_id')}: {result.get('qty')} шт. на складе {result.get('warehouse')}.")
        elif step.tool == "get_price":
            facts.append(f"{result.get('part_id')}: {result.get('price_rub')} руб.")
    prefix = " ".join(facts)
    answer = f"{prefix} Работа прервана: достигнут лимит шагов.".strip()
    return AgentResult(answer, steps=steps, sources=_sources(steps))


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
            if not _supported_answer(answer, question, steps):
                if corrections:
                    return AgentResult(REFUSAL, steps=steps, sources=_sources(steps), refused=True)
                corrections += 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "В ответе есть код или число, которого нет в вопросе и результатах инструментов. "
                            "Исправь ответ, используя только подтверждённые данные. Если их нет, откажись."
                        ),
                    }
                )
                continue
            refused = "не найдено" in answer.lower()
            if not answer:
                return AgentResult(REFUSAL, steps=steps, sources=_sources(steps), refused=True)
            return AgentResult(answer, steps=steps, sources=_sources(steps), refused=refused)

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
                part_id = args.get("part_id")
                part = normalize_code(part_id) if isinstance(part_id, str) else None
                if part_id is not None and (part is None or part not in _known_codes(question, steps[:-1])):
                    step.error = "Номер детали не подтверждён. Сначала вызови search_knowledge."
                elif call.name == "create_service_order" and (error := _order_error(args)):
                    step.error = error
                elif call.name == "create_service_order" and not _quantity_in_question(question, args.get("qty")):
                    return AgentResult(
                        "Сколько штук заказать? Укажите количество в вопросе.",
                        steps=steps,
                        sources=_sources(steps),
                    )
                elif call.name == "create_service_order" and not confirm:
                    confirmation = (
                        f"Подтвердите заказ: деталь {args['part_id']}, модель {args['model']}, "
                        f"склад {args['warehouse']}, количество {args['qty']}. Заказ пока не создан."
                    )
                    # Заказ ожидает подтверждения и ещё не был выполнен.
                    steps.pop()
                    return AgentResult(
                        confirmation,
                        steps=steps,
                        sources=_sources(steps),
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
                                        step.error = str(exc) or "Данные ERP получить не удалось."
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
