import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_agent_chat.settings import load_settings


class SettingsTest(unittest.TestCase):
    def test_loads_public_root_and_model_profiles_without_secrets_in_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models.yaml").write_text(
                "models:\n"
                "  - id: local\n"
                "    label: Local model\n"
                "    model: openai-compatible:local\n"
                "    api_key_env: LOCAL_MODEL_KEY\n",
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
        self.assertNotIn("secret-value", yaml_text)


if __name__ == "__main__":
    unittest.main()
