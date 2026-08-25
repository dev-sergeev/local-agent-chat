from local_agent_chat.chat_titles import normalize_chat_title


def test_normalize_chat_title_enforces_plain_three_to_five_word_title() -> None:
    assert (
        normalize_chat_title("**Название:** «Аудит проекта перед публикацией».")
        == "Аудит проекта перед публикацией"
    )
    assert (
        normalize_chat_title(
            "Настройка полностью изолированного окружения для песочницы проекта"
        )
        == "Настройка полностью изолированного окружения для"
    )
    assert normalize_chat_title("Настройка песочницы") is None


def test_normalize_chat_title_rejects_oversized_first_words() -> None:
    long_word = "а" * 63

    assert normalize_chat_title(f"{long_word} второе третье") is None
