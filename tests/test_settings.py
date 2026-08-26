import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_agent_chat.settings import LLMRetryConfig, load_settings


class SettingsTest(unittest.TestCase):
    def test_loads_public_root_and_model_profiles_without_secrets_in_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models.yaml").write_text(
                "models:\n"
                "  - id: local\n"
                "    label: Local model\n"
                "    model: openai-compatible:local\n"
                "    api_key_env: LOCAL_MODEL_KEY\n"
                "    streaming: false\n",
                encoding="utf-8",
            )
            env = {
                "APP_ROOT_PATH": "user/alice/vscode/proxy/8000/",
                "APP_DATA_DIR": str(root / "data"),
                "MODEL_PROFILES_FILE": str(root / "models.yaml"),
                "LOCAL_MODEL_KEY": "secret-value",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = load_settings()

            yaml_text = (root / "models.yaml").read_text(encoding="utf-8")

        self.assertEqual(settings.root_path, "/user/alice/vscode/proxy/8000")
        self.assertEqual(settings.models[0].id, "local")
        self.assertEqual(settings.models[0].api_key, "secret-value")
        self.assertFalse(settings.models[0].streaming)
        self.assertEqual(settings.llm_retry, LLMRetryConfig())
        self.assertNotIn("secret-value", yaml_text)

    def test_loads_llm_retry_configuration_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models.yaml").write_text(
                "models:\n"
                "  - id: local\n"
                "    label: Local model\n"
                "    model: openai:local\n"
                "    api_key_env: LOCAL_MODEL_KEY\n",
                encoding="utf-8",
            )
            env = {
                "MODEL_PROFILES_FILE": str(root / "models.yaml"),
                "LLM_MAX_RETRIES": "7",
                "LLM_STREAM_RETRIES": "2",
                "LLM_REQUEST_TIMEOUT_SECONDS": "12.5",
                "LLM_STREAM_CHUNK_TIMEOUT_SECONDS": "34.5",
                "LLM_AUXILIARY_TIMEOUT_SECONDS": "8",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = load_settings()

        self.assertEqual(
            settings.llm_retry,
            LLMRetryConfig(
                max_retries=7,
                stream_retries=2,
                request_timeout_seconds=12.5,
                stream_chunk_timeout_seconds=34.5,
                auxiliary_timeout_seconds=8.0,
            ),
        )

    def test_rejects_invalid_llm_max_retries(self) -> None:
        invalid_values = ("-1", "11", "1.5", "many", "")
        with patch.dict(os.environ, {}, clear=True):
            for value in invalid_values:
                with self.subTest(value=value):
                    with patch.dict(os.environ, {"LLM_MAX_RETRIES": value}, clear=True):
                        with self.assertRaisesRegex(ValueError, "LLM_MAX_RETRIES"):
                            load_settings()

    def test_rejects_invalid_stream_retries(self) -> None:
        invalid_values = ("-1", "11", "1.5", "many", "")
        for value in invalid_values:
            with self.subTest(value=value):
                with patch.dict(
                    os.environ,
                    {"LLM_STREAM_RETRIES": value},
                    clear=True,
                ):
                    with self.assertRaisesRegex(ValueError, "LLM_STREAM_RETRIES"):
                        load_settings()

    def test_rejects_non_positive_or_non_finite_llm_timeouts(self) -> None:
        names = (
            "LLM_REQUEST_TIMEOUT_SECONDS",
            "LLM_STREAM_CHUNK_TIMEOUT_SECONDS",
            "LLM_AUXILIARY_TIMEOUT_SECONDS",
        )
        invalid_values = ("0", "-1", "nan", "inf", "-inf", "forever", "")
        for name in names:
            for value in invalid_values:
                with self.subTest(name=name, value=value):
                    with patch.dict(os.environ, {name: value}, clear=True):
                        with self.assertRaisesRegex(ValueError, name):
                            load_settings()


if __name__ == "__main__":
    unittest.main()
