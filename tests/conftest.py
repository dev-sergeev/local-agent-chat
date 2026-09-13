"""Load the shipped Chainlit configuration before test modules import Chainlit."""

import os
import tempfile
from pathlib import Path

from local_agent_chat.installation import copy_ui

_workspace = tempfile.TemporaryDirectory(prefix="localchat-tests-")
copy_ui(Path(_workspace.name))
os.environ["CHAINLIT_APP_ROOT"] = _workspace.name
os.environ["CHAINLIT_ENV_FILE"] = os.devnull


def pytest_unconfigure(config):
    _workspace.cleanup()
