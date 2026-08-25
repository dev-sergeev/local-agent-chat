from local_agent_chat.tool_titles import normalize_tool_title


def test_normalize_tool_title_enforces_three_to_five_words() -> None:
    assert (
        normalize_tool_title('**Заголовок:** "Изучаю параметры рабочей системы"')
        == "Изучаю параметры рабочей системы"
    )
    assert (
        normalize_tool_title("Проверяю версии всех системных пакетов в окружении")
        == "Проверяю версии всех системных пакетов"
    )
    assert normalize_tool_title("Проверяю файлы") is None
