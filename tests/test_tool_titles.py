import pytest

from local_agent_chat.tool_titles import normalize_tool_title


def test_normalize_tool_title_enforces_three_to_five_words() -> None:
    assert (
        normalize_tool_title('**Заголовок:** "Проверка параметров рабочей системы"')
        == "Проверка параметров рабочей системы"
    )
    assert (
        normalize_tool_title(
            "Инвентаризация всех установленных системных пакетов завершена"
        )
        == "Инвентаризация всех установленных системных пакетов"
    )
    assert normalize_tool_title("Проверяю файлы") is None


@pytest.mark.parametrize(
    "model_title",
    [
        "Изучаю рабочую директорию",
        "Я изучил содержимое родительских каталогов",
        "Устанавливаю Python-библиотеку requests",
    ],
)
def test_normalize_tool_title_rejects_first_person_process_narration(
    model_title: str,
) -> None:
    assert normalize_tool_title(model_title) is None
