# Local Agent Chat

Однопользовательский Chainlit UI для Deep Agents в JupyterHub: история чатов, возобновление после перезапуска, файлы, revision с откатом состояния и наблюдаемые tool calls.

## Возможности

- OpenAI-compatible Model Profiles из YAML; ключи берутся только из environment.
- Read-only Agent Mode по умолчанию читает и ищет по абсолютным путям всего диска, но не может изменять файлы или выполнять команды.
- Extended Agent Mode включается до первого сообщения и добавляет создание, изменение, удаление и команды; после начала Chat режим фиксируется.
- Отдельные файлы, память и SQLite-история для каждого Chat; Extended Chat также получает собственное Python-окружение.
- Загрузка файлов любого типа, включая пустые; одинаковые имена сохраняются рядом с безопасным числовым суффиксом, без молчаливой перезаписи.
- Global Memory по необходимости ищет актуальные Turn прошлых Chat и раскрывает только выбранный ограниченный контекст.
- Короткие LLM-названия диалогов и вызовов инструментов вместо сырого текста.
- Provider-native retry каждого отдельного LLM-запроса с timeout, exponential backoff, jitter и поддержкой `Retry-After`.
- Проектные навыки с progressive disclosure из отдельной папки `skills/`.
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
UI принимает до 20 файлов по 100 MiB за выбор; общий объём файлов одного Chat ограничен 1 GiB.

### Отказоустойчивость LLM

| Переменная | По умолчанию | Назначение |
| --- | ---: | --- |
| `LLM_MAX_RETRIES` | `3` | Дополнительные попытки после первой, от `0` до `10` (по умолчанию до четырёх HTTP-запросов); `0` отключает retry. |
| `LLM_REQUEST_TIMEOUT_SECONDS` | `60` | Timeout одного HTTP-запроса к LLM. |
| `LLM_STREAM_CHUNK_TIMEOUT_SECONDS` | `120` | Максимальная пауза между частями streaming-ответа. |
| `LLM_STREAM_RETRIES` | `1` | Дополнительные попытки только model handler после zero-chunk stream timeout, от `0` до `10`; это отдельный budget. |
| `LLM_AUXILIARY_TIMEOUT_SECONDS` | `30` | Общий timeout одного вспомогательного LLM-вызова для названия Chat или tool call. |

Retry выполняет provider SDK только для transient-сбоев: ошибок соединения и timeout, rate limit, request/lock timeout и ответов сервера 5xx. Задержка определяется provider-native exponential backoff с jitter; `Retry-After` и явные provider-заголовки имеют приоритет.

Одна и та же provider-политика действует для основного Agent, default subagent, summarization и генерации названий. Встроенный дополнительный retry summarization отключён, поэтому provider-повторы не вкладываются друг в друга.

Граница provider-retry — один HTTP inference. После первой полученной части streaming повтора нет, чтобы не дублировать уже показанный текст. Если provider не прислал ни одной части до stream timeout, `LLM_STREAM_RETRIES` повторяет только внутренний model handler. Граф Agent, summarization/offload, Turn и tool call заново не запускаются, поэтому их побочные эффекты не воспроизводятся.

### Навыки агента

Чтобы добавить навык, создайте папку `skills/<skill-name>/` и обязательный файл `SKILL.md` с YAML-полями `name` и `description`. Навыки автоматически доступны основному Agent и стандартному subagent; после добавления или изменения откройте новый диалог.

Формат, шаблон, дополнительные файлы и правила проверки описаны в [`skills/README.md`](skills/README.md).

## Где менять проект

| Что | Где |
| --- | --- |
| System prompt агента, названий диалогов и инструментов | [`local_agent_chat/prompts.py`](local_agent_chat/prompts.py) |
| Сборка приложения, стартовые запросы и Chainlit callbacks | [`app.py`](app.py) |
| Выбор профиля и режима Chat | [`local_agent_chat/chat_configuration.py`](local_agent_chat/chat_configuration.py), [`local_agent_chat/chat_bindings.py`](local_agent_chat/chat_bindings.py) |
| Приветствие | [`chainlit.md`](chainlit.md) |
| Вид tool steps | [`local_agent_chat/chainlit_ui.py`](local_agent_chat/chainlit_ui.py) |
| Граф Deep Agents, Agent Memory и события | [`local_agent_chat/deep_agent_execution.py`](local_agent_chat/deep_agent_execution.py) |
| Retry всех моделей и вспомогательные LLM-названия | [`local_agent_chat/llm_retry.py`](local_agent_chat/llm_retry.py), [`local_agent_chat/auxiliary_labels.py`](local_agent_chat/auxiliary_labels.py) |
| Навыки агента и шаблон нового навыка | [`skills/README.md`](skills/README.md) |
| Agent Mode и файловые capabilities | [`local_agent_chat/agent_modes.py`](local_agent_chat/agent_modes.py), [`local_agent_chat/sandbox_provider.py`](local_agent_chat/sandbox_provider.py) |
| Turn rollback и Sandbox | [`local_agent_chat/runtime.py`](local_agent_chat/runtime.py), [`local_agent_chat/sandbox_files.py`](local_agent_chat/sandbox_files.py) |
| Chainlit History и атомарная Revision | [`local_agent_chat/chainlit_data.py`](local_agent_chat/chainlit_data.py) |
| Каноническая история и Global Memory | [`local_agent_chat/sqlite_history.py`](local_agent_chat/sqlite_history.py), [`local_agent_chat/memory_tools.py`](local_agent_chat/memory_tools.py) |

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
