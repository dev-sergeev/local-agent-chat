# Local Agent Chat

Chainlit UI для Deep Agents в JupyterHub. Поддерживаются URL-префикс, тихая локальная идентификация без формы входа, SQLite-история, возобновление чатов, Model Profile на Chat, файлы, локальные Python/shell-команды и нативный Revision.

Интерфейс показывает единый хронологический поток текста и наблюдаемых шагов Agent: чтение и изменение файлов, поиск, shell/Python, подзадачи, результаты и ошибки. Каждый текстовый фрагмент после инструмента сохраняется отдельно, поэтому после перезагрузки объяснения, ошибки и следующие вызовы остаются в исходном порядке. Вывод инструментов отображается как моноширинный preformatted log: переносы строк сохраняются, ANSI-коды удаляются, а при сокращении длинного результата остаются его начало и конец. Успешные шаги автоматически сворачиваются, ошибки остаются раскрытыми, а Stop отменяет Turn и восстанавливает pre-turn состояние. Созданные изображения показываются inline, остальные файлы доступны из боковой панели и сохраняются в истории.

## Установка

```bash
python -m pip install -e '.[test]'
cp models.example.yaml models.yaml
cp .env.example .env
```

По умолчанию настроена OpenAI-compatible модель `deepseek/deepseek-v4-flash-0731` через OpenRouter. Model Profile хранится в `models.yaml`, а endpoint и ключ — в `.env` как `OPENAI_BASE_URL` и `OPENAI_API_KEY`.

```bash
export CHAINLIT_AUTH_SECRET="$(openssl rand -hex 32)"
```

Chainlit автоматически идентифицирует фиксированного Local User через header-auth: формы логина нет, но встроенная история доступна. Для каждого Chat создаётся отдельный локальный каталог Sandbox, а backend хранится в памяти процесса. Python и shell-команды выполняются в Docker-контейнере с этим каталогом в качестве рабочей директории; ключи моделей в окружение команд не передаются.

## Запуск через JupyterHub

```bash
export APP_PORT=8765
export APP_ROOT_PATH="${JUPYTERHUB_SERVICE_PREFIX%/}/vscode/proxy/$APP_PORT"
./scripts/run.sh
```

Открывайте приложение по адресу `${APP_ROOT_PATH}/` на внешнем адресе VS Code/JupyterHub. Публичный путь должен совпадать с Chainlit `--root-path` и указываться без завершающего `/`. VS Code proxy удаляет `/proxy/$APP_PORT` перед передачей запроса; встроенный ASGI-адаптер восстанавливает этот префикс для HTTP и WebSocket. Для прямого запуска без JupyterHub оставьте `APP_ROOT_PATH` пустым.

Путь находится под защитой самого JupyterHub, поэтому пользователь без активной Hub-сессии увидит его форму входа. Авторизация внутри Chainlit при этом остается отключенной.

## Хранилища

- `chainlit.sqlite3` — UI Threads, Steps и метаданные.
- `checkpoints.sqlite3` — LangGraph checkpoints и активная ветка.
- `runtime-history.sqlite3` — Turn и audit superseded-веток.
- `sandboxes/` — Uploaded File и pre-turn snapshots.
- `blobs/` — элементы Chainlit для возобновлённого Chat.

Revision архивирует UI-ветку, восстанавливает checkpoint и снимок файлов, повторно запускает Agent и после успеха заменяет активную runtime-историю. При ошибке восстанавливаются прежние UI Steps, Agent Memory и Sandbox.

## Проверка

```bash
pytest -q
python -m compileall -q local_agent_chat app.py
```
