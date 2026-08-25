# Local Agent Chat

Однопользовательский Chainlit UI для Deep Agents в JupyterHub: история чатов, возобновление после перезапуска, файлы, revision с откатом состояния и наблюдаемые tool calls.

## Возможности

- OpenAI-compatible Model Profiles из YAML; ключи берутся только из environment.
- Read-only Agent Mode по умолчанию читает и ищет по абсолютным путям всего диска, но не может изменять файлы или выполнять команды.
- Extended Agent Mode включается до первого сообщения и добавляет создание, изменение, удаление и команды; после начала Chat режим фиксируется.
- Отдельные файлы, память и SQLite-история для каждого Chat; Extended Chat также получает собственное Python-окружение.
- Global Memory по необходимости ищет актуальные Turn прошлых Chat и раскрывает только выбранный ограниченный контекст.
- Короткие LLM-названия диалогов и вызовов инструментов вместо сырого текста.
- Работа под URL-префиксом JupyterHub / VS Code Proxy.

## Быстрый старт

Требуются Python 3.12–3.13 с рабочими `venv`/`ensurepip` (на Debian/Ubuntu — пакет `python3-venv`) и доступ к OpenAI-compatible API.

```bash
python -m pip install -e '.[test]'
cp models.example.yaml models.yaml
cp .env.example .env
# Укажите OPENAI_API_KEY и случайный CHAINLIT_AUTH_SECRET в .env
./scripts/run.sh
```

По умолчанию UI доступен на `http://127.0.0.1:8765`. Для JupyterHub:

```bash
export APP_ROOT_PATH="${JUPYTERHUB_SERVICE_PREFIX%/}/vscode/proxy/8765"
./scripts/run.sh
```

Основные настройки: `models.yaml` — модели, `.env` — секреты и пути, `.chainlit/config.toml` — UI. Примеры: `models.example.yaml` и `.env.example`.
Если модель не поддерживает streaming, добавьте в её профиль `streaming: false`: UI покажет полный ответ после завершения вызова.

## Где менять проект

| Что | Где |
| --- | --- |
| System prompt агента, названий диалогов и инструментов | [`local_agent_chat/prompts.py`](local_agent_chat/prompts.py) |
| Стартовые запросы и Chainlit callbacks | `app.py` |
| Приветствие | `chainlit.md` |
| Вид tool steps | `local_agent_chat/chainlit_ui.py` |
| Запуск агента и модели | `local_agent_chat/agent_service.py` |
| Agent Mode и файловые capabilities | `agent_modes.py`, `sandbox_provider.py` |
| Хранение, Global Memory, revision и Sandbox | `sqlite_history.py`, `memory_tools.py`, `runtime.py`, `sandbox_*.py` |

Термины закреплены в [`CONTEXT.md`](CONTEXT.md), а границы модулей — в [`docs/architecture.md`](docs/architecture.md) и [`docs/adr/`](docs/adr/).

## Безопасность

Read-only не предоставляет Agent инструменты записи и исполнения, но разрешает читать доступные процессу host-файлы. Extended Chat получает собственный `venv`, `HOME`, cache и temp, однако shell и файловые инструменты всё равно работают с правами процесса приложения. Agent Mode не является границей безопасности: используйте отдельный контейнер/VM и только доверенного одиночного пользователя. Подробности: [`SECURITY.md`](SECURITY.md).

## Разработка

```bash
pytest -q
ruff check .
ruff format --check .
python -m compileall -q local_agent_chat app.py
```

Правила и PR-checklist: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Лицензия

[MIT](LICENSE).
