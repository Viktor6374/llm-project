"""Дополнительные проверки агента: количество заказа, цены и остатки, отказы ERP."""

import pytest

from assistant.agent.agent import run_agent
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.llm import LLMResponse
from assistant.infra.testing import FakeLLM
from tests.contract.test_tools_and_agent import Tools, call

PRICE = {"part_id": "P-1042B", "price_rub": 500, "discontinued": False}
STOCK = {"part_id": "P-1042B", "warehouse": "Казань", "qty": 7, "discontinued": False}
ORDER = {"part_id": "P-1042B", "warehouse": "Казань", "qty": 2, "model": "AR-210"}
QUESTION = "Сколько стоит P-1042B и сколько её на складе в Казани?"


def price_and_stock(*answers: str) -> FakeLLM:
    return FakeLLM(
        [call("get_price", part_id="P-1042B"), call("check_stock", part_id="P-1042B", warehouse="Казань"), *answers]
    )


def dead(**arguments) -> dict:
    raise ErpUnavailable("ERP недоступна")


@pytest.mark.parametrize(
    "question",
    [
        "Закажи P-1042B для AR-210 на склад Казань к 25 ноября.",
        "Закажи P-1042B для AR-210, склад Казань, до 25.11.",
    ],
)
def test_number_from_a_date_is_not_a_quantity(question):
    tools = Tools()
    llm = FakeLLM([call("create_service_order", **{**ORDER, "qty": 25})])

    result = run_agent(question, llm, tools.as_dict(), confirm=True)

    assert tools.count("create_service_order") == 0
    assert not result.needs_confirmation and "Сколько штук" in result.answer


@pytest.mark.parametrize(
    "question",
    [
        "Закажи две штуки P-1042B для AR-210, склад Казань.",
        "Закажи 2 шт. P-1042B для AR-210 на склад Казань к 25 ноября.",
        "Закажи P-1042B для AR-210, склад Казань, количество: 2.",
    ],
)
def test_quantity_with_a_unit_is_accepted(question):
    llm = FakeLLM([call("create_service_order", **ORDER)])

    result = run_agent(question, llm, Tools().as_dict())

    assert result.needs_confirmation
    assert result.pending_action == {"tool": "create_service_order", "args": ORDER}


def test_invented_model_is_not_sent_to_the_order():
    tools = Tools()
    llm = FakeLLM([call("create_service_order", **{**ORDER, "model": "AR-999"}), "Модель не подтверждена."])

    result = run_agent("Закажи 2 шт. P-1042B на склад Казань.", llm, tools.as_dict(), confirm=True)

    assert tools.count("create_service_order") == 0
    assert result.steps[0].error and not result.needs_confirmation


def test_order_number_is_added_when_the_model_forgets_it():
    llm = FakeLLM([call("create_service_order", **ORDER), "Заказ создан."])

    result = run_agent("Оформи заказ: 2 штуки P-1042B для AR-210, склад Казань.", llm, Tools().as_dict(), confirm=True)

    assert "SO-000001" in result.answer


@pytest.mark.parametrize("answer", ["7 руб., 500 шт.", "Цена 7 руб., остаток 500 шт.", "Цена P-1042B 7, остаток 500."])
def test_swapped_price_and_stock_become_a_refusal(answer):
    tools = Tools(get_price=PRICE, check_stock=STOCK)

    result = run_agent(QUESTION, price_and_stock(answer, answer), tools.as_dict())

    assert result.refused and "500" not in result.answer


@pytest.mark.parametrize("answer", ["Цена 500 руб., на складе в Казани 7 шт.", "Остаток 7, цена 500."])
def test_correct_price_and_stock_are_accepted(answer):
    tools = Tools(get_price=PRICE, check_stock=STOCK)

    result = run_agent(QUESTION, price_and_stock(answer), tools.as_dict())

    assert result.answer == answer and not result.refused


def test_step_limit_with_dead_erp_does_not_print_none():
    tools = Tools(check_stock=dead)
    script = [call("check_stock", part_id="P-1042B", warehouse=warehouse) for warehouse in ["Казань", "Москва"]]

    result = run_agent("Где есть P-1042B?", FakeLLM(script), tools.as_dict(), max_steps=2)

    assert "None" not in result.answer
    assert "Данные ERP получить не удалось" in result.answer and "прервана" in result.answer


def test_step_limit_keeps_what_was_found():
    tools = Tools(get_price=PRICE, check_stock=STOCK)

    result = run_agent(QUESTION, price_and_stock(), tools.as_dict(), max_steps=2)

    assert "P-1042B: 500 руб." in result.answer and "P-1042B: 7 шт. на складе Казань." in result.answer


def test_dead_erp_is_reported_even_if_the_model_keeps_silent():
    llm = FakeLLM([call("check_stock", part_id="P-1042B", warehouse="Казань"), "Остаток неизвестен."])

    result = run_agent("Есть ли P-1042B в Казани?", llm, Tools(check_stock=dead).as_dict())

    assert "Данные ERP получить не удалось" in result.answer


def test_answer_without_tools_is_refused():
    llm = FakeLLM([LLMResponse("Завтра в Москве солнечно, +5.")])

    result = run_agent("Какая погода завтра в Москве?", llm, Tools().as_dict())

    assert result.refused and result.answer == "В базе знаний не найдено."


def test_clarifying_question_without_tools_is_passed_on():
    llm = FakeLLM(["Для какой модели нужна деталь?"])

    result = run_agent("Сколько стоит пусковое реле?", llm, Tools().as_dict())

    assert result.answer == "Для какой модели нужна деталь?" and not result.refused
