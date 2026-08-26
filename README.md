# LocalChat

Однопользовательский Chainlit UI для Deep Agents в JupyterHub: история чатов, возобновление после перезапуска, файлы, revision с откатом состояния и наблюдаемые tool calls.

## Возможности

- OpenAI-compatible Model Profiles из YAML; ключи берутся только из environment.
- Chat Files Agent Mode по умолчанию читает и ищет только среди файлов, загруженных в текущий Chat.
- Host Files Agent Mode можно выбрать до первого сообщения: он читает доступные процессу host-файлы по абсолютным путям; после начала Chat режим фиксируется.
- Оба режима предоставляют один read-only набор `ls`, `read_file`, `glob`, `grep`; у Agent нет инструментов изменения файлов, запуска команд или кода.
- Отдельные загруженные файлы, память и SQLite-история для каждого Chat.
- Загрузка файлов любого типа, включая пустые; одинаковые имена сохраняются рядом с безопасным числовым суффиксом, без молчаливой перезаписи.
- Global Memory по необходимости ищет актуальные Turn прошлых Chat и раскрывает только выбранный ограниченный контекст.
- С Long-term Memory Agent может проактивно сохранять устойчивые факты, предпочтения и проверенные результаты в компактном Markdown и использовать их в других Chat без поиска по ключевым словам.
- Короткие релевантные LLM-названия диалогов; при недоступности модели используется детерминированный заголовок из текста запроса вместо «Новый диалог».
- Provider-native retry каждого отдельного LLM-запроса с timeout, exponential backoff, jitter и поддержкой `Retry-After`.
- Проектные навыки с progressive disclosure из отдельной папки `skills/`.
- Работа под URL-префиксом JupyterHub / VS Code Proxy.

## Быстрый старт

Требуются Python 3.12–3.13 и доступ к OpenAI-compatible API.
Поддерживаемый формат поставки этой версии — checkout исходного репозитория:
приложение использует верхнеуровневые `app.py`, `.chainlit/`, `public/`, `skills/`
и `scripts/`. Собранный Python wheel не является автономным дистрибутивом сервиса.

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
| `LLM_AUXILIARY_TIMEOUT_SECONDS` | `30` | Общий timeout одного вспомогательного LLM-вызова для названия Chat. |

Retry выполняет provider SDK только для transient-сбоев: ошибок соединения и timeout, rate limit, request/lock timeout и ответов сервера 5xx. Задержка определяется provider-native exponential backoff с jitter; `Retry-After` и явные provider-заголовки имеют приоритет.

Одна и та же provider-политика действует для основного Agent, default subagent, summarization и генерации названий. Встроенный дополнительный retry summarization отключён, поэтому provider-повторы не вкладываются друг в друга.

Граница provider-retry — один HTTP inference. После первой полученной части streaming повтора нет, чтобы не дублировать уже показанный текст. Если provider не прислал ни одной части до stream timeout, `LLM_STREAM_RETRIES` повторяет только внутренний model handler. Граф Agent, summarization/offload, Turn и tool call заново не запускаются, поэтому их побочные эффекты не воспроизводятся.

### Навыки агента

Чтобы добавить навык, создайте папку `skills/<skill-name>/` и обязательный файл `SKILL.md` с YAML-полями `name` и `description`. Навыки автоматически доступны основному Agent и стандартному subagent в обоих Agent Mode; после добавления или изменения откройте новый диалог. Навык задаёт инструкции, но не расширяет файловую область и не возвращает удалённые инструменты записи или исполнения.

Формат, шаблон, дополнительные файлы и правила проверки описаны в [`skills/README.md`](skills/README.md).

### Долговременная память

Deep Agents `MemoryMiddleware` загружает долговременную память перед каждым Turn. Когда основной Agent узнаёт явно сообщённый устойчивый факт, стабильное предпочтение, повторяющееся ограничение или проверенный результат, он может вызвать `remember_context`. Например, после фразы «Меня зовут Анна» Agent может сохранить имя под ключом `user.name`, и новый Chat получит его напрямую из памяти, не вызывая `search_past_chats`.

Файл находится в `APP_DATA_DIR/memory/MEMORY.md` — по умолчанию `.local-agent-chat/memory/MEMORY.md`. Это обычный читаемый Markdown:

```markdown
# Long-term Memory

Compact durable facts shared by all chats. This file is managed by the application.

<!-- local-agent-chat-memory:v1 -->

## Facts

- **user.name**: Анна
- **user.preference.language**: Предпочитает русский язык
```

Один стабильный ключ содержит один текущий факт: повтор не создаёт дубликат, а новое значение заменяет старое. Команда пользователя «забудь моё имя» должна привести к `forget_context("user.name")`. Память общая для всех Chat, поэтому Revision или удаление исходного Chat её не откатывает.

Это управляемый приложением Markdown-файл. При ручном редактировании сохраняйте показанные заголовок, пояснение и marker без изменений; располагайте записи лексикографически по key, разделяйте пустой строкой и используйте однострочный формат `- **key**: fact`. Key содержит от 1 до 80 ASCII-символов, начинается с lowercase-буквы или цифры, а затем может содержать lowercase-буквы, цифры, `.`, `_` и `-`; пробелы недопустимы.

Лимиты: до 128 записей, 500 символов на нормализованный факт и 32 KiB как на файл, так и на безопасно экранированный снимок для prompt. Обновления из параллельных Chat и процессов защищены файловой блокировкой и публикуются атомарно. Если файл повреждён, не соответствует формату, превышает лимиты или записан не в UTF-8, он не передаётся модели, а `remember_context` и `forget_context` отказываются его перезаписывать; обычный Turn при этом продолжается. Исправьте файл вручную, чтобы вернуть память в рабочее состояние.

Принятая мутация памяти доводится до атомарной публикации даже при отмене Turn. После публикации ошибка или отмена Turn, Revision и удаление исходного Chat не откатывают этот независимый commit. Исправление повторно использует тот же key, а удаление — `forget_context`. Credentials и распознаваемые секреты запрещены. Поскольку актуальный снимок отправляется выбранному model provider в каждом Turn, не помещайте в него чувствительные данные без явной необходимости и никогда не сохраняйте пароли, токены или API keys.

Это дополняет Global Memory, а не заменяет её: Markdown содержит краткие актуальные выводы, а `search_past_chats` / `read_past_chat` по-прежнему находят первоисточник и подробности в истории. Отдельного extraction-вызова LLM нет — решение запомнить принимает основной Agent в рамках обычного Deep Agents шага.

## Где менять проект

| Что | Где |
| --- | --- |
| System prompt агента и названий диалогов | [`local_agent_chat/prompts.py`](local_agent_chat/prompts.py) |
| Сборка приложения, стартовые запросы и Chainlit callbacks | [`app.py`](app.py) |
| Выбор профиля и режима Chat | [`local_agent_chat/chat_configuration.py`](local_agent_chat/chat_configuration.py), [`local_agent_chat/chat_bindings.py`](local_agent_chat/chat_bindings.py) |
| Приветствие | [`chainlit.md`](chainlit.md) |
| Вид tool steps | [`local_agent_chat/chainlit_ui.py`](local_agent_chat/chainlit_ui.py) |
| Граф Deep Agents, Agent Memory и события | [`local_agent_chat/deep_agent_execution.py`](local_agent_chat/deep_agent_execution.py) |
| Retry всех моделей и вспомогательные LLM-названия Chat | [`local_agent_chat/llm_retry.py`](local_agent_chat/llm_retry.py), [`local_agent_chat/auxiliary_labels.py`](local_agent_chat/auxiliary_labels.py) |
| Долговременная Markdown-память и Agent tools | [`local_agent_chat/long_term_memory.py`](local_agent_chat/long_term_memory.py) |
| Навыки агента и шаблон нового навыка | [`skills/README.md`](skills/README.md) |
| Agent Mode и файловые capabilities | [`local_agent_chat/agent_modes.py`](local_agent_chat/agent_modes.py), [`local_agent_chat/sandbox_provider.py`](local_agent_chat/sandbox_provider.py) |
| Turn rollback и Sandbox | [`local_agent_chat/runtime.py`](local_agent_chat/runtime.py), [`local_agent_chat/sandbox_files.py`](local_agent_chat/sandbox_files.py) |
| Chainlit History и атомарная Revision | [`local_agent_chat/chainlit_data.py`](local_agent_chat/chainlit_data.py) |
| Каноническая история и Global Memory | [`local_agent_chat/sqlite_history.py`](local_agent_chat/sqlite_history.py), [`local_agent_chat/memory_tools.py`](local_agent_chat/memory_tools.py) |

Термины закреплены в [`CONTEXT.md`](CONTEXT.md), а границы модулей — в [`docs/architecture.md`](docs/architecture.md) и [`docs/adr/`](docs/adr/).

## Безопасность

Chat Files ограничивает файловые инструменты загруженными в текущий Chat файлами и доверенными Project Skills. Host Files разрешает чтение любых host-путей, доступных процессу приложения, поэтому их содержимое может попасть к model provider. Ни один режим не даёт Agent общих инструментов записи, удаления, запуска команд или кода. Используйте отдельный контейнер/VM и только доверенного одиночного пользователя. Подробности: [`SECURITY.md`](SECURITY.md).

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
