from local_agent_chat.chat_titles import (
    chat_title_source,
    fallback_chat_title,
    normalize_chat_title,
)


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


def test_fallback_chat_title_uses_request_words_instead_of_generic_name() -> None:
    assert (
        fallback_chat_title("Исправь загрузку файлов в агенте после revision")
        == "Исправь загрузку файлов в агенте"
    )
    assert fallback_chat_title("  **Привет**  ") == "Привет"


def test_fallback_chat_title_uses_default_only_for_empty_request() -> None:
    assert fallback_chat_title(" \n\t ") == "Новый диалог"


def test_file_only_request_gets_a_semantic_title_source() -> None:
    source = chat_title_source(" \n", ["reports/quarterly results.pdf"])

    assert source == "Файлы: quarterly results.pdf"
    assert fallback_chat_title(source) == "Файлы: quarterly results.pdf"
