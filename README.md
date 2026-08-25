# Local Agent Chat

Однопользовательский Chainlit UI для Deep Agents в JupyterHub: история чатов, возобновление после перезапуска, файлы, revision с откатом состояния и наблюдаемые tool calls.

## Возможности

- OpenAI-compatible Model Profiles из YAML; ключи берутся только из environment.
- Отдельные файлы, память и SQLite-история для каждого Chat.
- Shell/Python и файловые инструменты с компактными LLM-заголовками и ограниченными логами.
- Работа под URL-префиксом JupyterHub / VS Code Proxy.

## Быстрый старт

Требуются Python 3.12–3.13 и доступ к OpenAI-compatible API.

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

## Где менять проект

| Что | Где |
| --- | --- |
| Основной system prompt и prompt LLM-заголовков | [`local_agent_chat/prompts.py`](local_agent_chat/prompts.py) |
| Стартовые запросы и Chainlit callbacks | `app.py` |
| Приветствие | `chainlit.md` |
| Вид tool steps | `local_agent_chat/chainlit_ui.py` |
| Запуск агента и модели | `local_agent_chat/agent_service.py` |
| Хранение, revision и Sandbox | `chainlit_data.py`, `runtime.py`, `sandbox_*.py` |

Термины закреплены в [`CONTEXT.md`](CONTEXT.md), а границы модулей — в [`docs/architecture.md`](docs/architecture.md) и [`docs/adr/`](docs/adr/).

## Безопасность

`LocalShellBackend` запускает команды с правами процесса приложения. Sandbox — рабочий каталог, а не граница безопасности. Запускайте сервис в отдельном контейнере/VM и только для доверенного одиночного пользователя. Подробности: [`SECURITY.md`](SECURITY.md).

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
