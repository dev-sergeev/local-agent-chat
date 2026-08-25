from chainlit.config import config


def test_tool_steps_have_no_usage_prefix() -> None:
    translation = config.load_translation("ru-RU")
    status = translation["chat"]["messages"]["status"]

    assert status["using"] == ""
    assert status["used"] == ""
