# Chainlit как платформа интерактивного UI

Дата проверки: **2026-08-21**. Цель: определить, насколько Chainlit подходит для удобного интерактивного интерфейса Local Agent Chat и какой уровень кастомизации рационален для этого репозитория.

## Краткий вывод

Chainlit 2.11.1 уже содержит почти весь набор, нужный для качественного agent UI: потоковые сообщения и шаги, древовидные tool steps, действия-кнопки, блокирующие `Ask*`-диалоги, произвольные JSX-элементы, task list, файлы/изображения/PDF/аудио, профили, настройки, starters, история, feedback и локализация. Всё это работает поверх Python/FastAPI backend и штатного React-клиента, связанного с backend через Socket.IO/WebSocket; готовый frontend поставляется внутри Python-пакета. [Архитектура репозитория Chainlit](https://github.com/Chainlit/chainlit/blob/main/AGENTS.md#architecture), [deployment overview](https://docs.chainlit.io/deploy/overview).

Для этого проекта лучший путь — **сохранить штатный Chainlit Web App**, сначала раскрыть его нативные примитивы, а для специализированных панелей добавить `CustomElement`. Полный React frontend стоит рассматривать только тогда, когда потребуется постоянная трёхпанельная компоновка, собственная навигация/маршрутизация, сложный файловый explorer или иной UX, который не укладывается в message/step/element model. React client действительно покрывает messages, streaming, elements, audio, Ask User, history, profiles и feedback, но переносит на проект ответственность за весь UI и совместимость. [React client: supported features](https://docs.chainlit.io/deploy/react/overview#supported-features), [React hooks](https://docs.chainlit.io/deploy/react/usage#react-hooks).

Текущий проект уже совместим с актуальной стабильной линией: установлен `chainlit==2.11.1`, а зависимость ограничена `chainlit>=2.11,<2.12`; Python 3.12 входит в официальный диапазон Chainlit `>=3.10,<3.14`. [Локальный `pyproject.toml`](../../pyproject.toml), [Chainlit 2.11.1 `pyproject.toml`](https://github.com/Chainlit/chainlit/blob/2.11.1/backend/pyproject.toml#L1-L35). На дату проверки последним GitHub release является **2.11.1 от 2026-04-22**. [GitHub Releases](https://github.com/Chainlit/chainlit/releases/tag/2.11.1), [CHANGELOG](https://github.com/Chainlit/chainlit/blob/main/CHANGELOG.md#2111---2026-04-22).

## Метод и локальный контекст

Использованы только первичные источники: официальная документация Chainlit, официальный репозиторий `Chainlit/chainlit`, его исходный код, changelog и releases. Полный официальный индекс документации находится в [`llms.txt`](https://docs.chainlit.io/llms.txt).

В репозитории нет отдельной конвенции для research notes: `docs/` содержит архитектуру и ADR, а корневой [`chainlit.md`](../../chainlit.md) является пользовательской welcome/readme-страницей Chainlit. Поэтому отчёт размещён в `docs/research/`.

Локальная реализация сегодня:

- Использует штатный Web App и lifecycle hooks `on_chat_start`, `on_chat_resume`, `on_message`; Model Profile выбирается через `set_chat_profiles`. [`app.py`](../../app.py)
- Имеет локальный SQLAlchemy/SQLite data layer, схему для threads, steps, elements и feedback, плюс собственный журнал Revision. [`local_agent_chat/chainlit_data.py`](../../local_agent_chat/chainlit_data.py)
- Загружает файлы из `message.elements` в отдельный Sandbox и прикладывает изменённые агентом файлы как `cl.File(display="side")`. [`app.py`](../../app.py)
- Сейчас ждёт полный `answer` от runtime и только затем отправляет одно сообщение, то есть визуального token streaming нет. [`app.py`](../../app.py), [`local_agent_chat/runtime.py`](../../local_agent_chat/runtime.py)
- Разрешает `*/*`, до 20 файлов и 100 MiB на файл, оставляет audio/MCP выключенными, CoT — `full`, message editing — включённым, `unsafe_allow_html` — выключенным; `allow_origins=["*"]`. [`.chainlit/config.toml`](../../.chainlit/config.toml)

## 1. Архитектура Chainlit

Chainlit — монорепозиторий из Python/FastAPI backend, React frontend, библиотеки `@chainlit/react-client` и встраиваемого Copilot; собранные frontend-ассеты копируются в Python wheel и раздаются backend-сервером. [Официальное описание структуры](https://github.com/Chainlit/chainlit/blob/main/AGENTS.md#architecture), [backend build configuration](https://github.com/Chainlit/chainlit/blob/2.11.1/backend/pyproject.toml#L95-L140).

Основной путь события выглядит так: React client отправляет `client_message` через Socket.IO, backend вызывает зарегистрированный `on_message`, вызов `cl.Message.send()` генерирует обратное socket-событие, а client обновляет React state. [Официальное описание communication flow](https://github.com/Chainlit/chainlit/blob/main/AGENTS.md#communication-flow), [`socket.py`](https://github.com/Chainlit/chainlit/blob/2.11.1/backend/chainlit/socket.py), [`emitter.py`](https://github.com/Chainlit/chainlit/blob/2.11.1/backend/chainlit/emitter.py).

Практически это даёт три уровня UI:

1. **Штатный Web App** — Python API + конфигурация темы/CSS/JS; минимальная стоимость сопровождения. [Web App](https://docs.chainlit.io/deploy/webapp).
2. **Web App + Custom Elements** — локальные JSX-компоненты в `public/elements`, встроенные в message/step; лучший компромисс для форм, карточек и специализированных результатов. [Custom Element](https://docs.chainlit.io/api-reference/elements/custom).
3. **Собственный React frontend** на `@chainlit/react-client` — полный контроль над компоновкой, но собственная реализация отображения и UX. [React overview](https://docs.chainlit.io/deploy/react/overview), [installation](https://docs.chainlit.io/deploy/react/installation-and-setup).

Отдельный вариант — **Copilot**, то есть готовый floating/sidebar widget для встраивания в другой сайт. Он поддерживает те же основные функции, умеет сохранять `threadId` в `localStorage`, вызывать функции host-страницы и принимать системный контекст. [Copilot](https://docs.chainlit.io/deploy/copilot).

## 2. Chat lifecycle и состояние

Новая chat session создаётся при подключении пользователя; Chainlit предоставляет hooks `on_chat_start`, `on_message`, `on_stop`, `on_chat_end` и `on_chat_resume`. Resume возможен только при включённых authentication и persistence. [Chat lifecycle](https://docs.chainlit.io/concepts/chat-lifecycle).

`on_chat_start` привязан к WebSocket connection event, а `on_message` вызывается для каждого нового сообщения UI. [on_chat_start](https://docs.chainlit.io/api-reference/lifecycle-hooks/on-chat-start), [on_message](https://docs.chainlit.io/api-reference/lifecycle-hooks/on-message). В 2.11.1 отдельно исправили повторный dispatch `on_chat_start` при reconnect, что важно для идемпотентной инициализации. [CHANGELOG 2.11.1](https://github.com/Chainlit/chainlit/blob/main/CHANGELOG.md#2111---2026-04-22).

`cl.user_session` хранит данные, изолированные по chat session; зарезервированные ключи включают `id`, `user`, `chat_profile`, `chat_settings`, `env`. [User Session](https://docs.chainlit.io/concepts/user-session). Это оперативное UI/session state, а не замена долговременному agent checkpoint или data layer.

Для проекта:

- Сохранять `model_profile` в metadata thread, как сейчас, правильно: profile должен восстанавливаться независимо от WebSocket session. [`app.py`](../../app.py)
- `on_stop` следует связать с отменой agent task и безопасной фиксацией статуса Turn; нативная кнопка Stop и hook предусмотрены lifecycle. [Chat lifecycle: On Stop](https://docs.chainlit.io/concepts/chat-lifecycle#on-stop).
- Инициализация в `on_chat_start` должна оставаться идемпотентной, а фактической границей данных должен быть `thread_id`, не socket/session id. [Lifecycle](https://docs.chainlit.io/concepts/chat-lifecycle), [Chat History](https://docs.chainlit.io/data-persistence/history).

## 3. Messages, Steps и Elements

### Message

`Message` — пользовательская/assistant-реплика; её можно отправлять, stream-ить, обновлять и удалять. К message прикладываются `Element[]`, `Action[]`, metadata/tags, command и modes. [Message API](https://docs.chainlit.io/api-reference/message).

Рекомендация: один пользовательский Turn должен иметь один основной assistant Message. Статусы и tool activity не следует дописывать в его текст — для этого есть Steps и TaskList.

### Step

`Step` моделирует отдельную операцию и может быть вложен в другой step; он хранит input/output, тип, язык, metadata/tags, error state, время и display-параметры. Декоратор `@cl.step` автоматически связывает вызов функции с step, а `Step` class даёт ручное управление. [Step concept](https://docs.chainlit.io/concepts/step), [Step decorator](https://docs.chainlit.io/api-reference/step-decorator), [Step class](https://docs.chainlit.io/api-reference/step-class).

UI умеет показывать chain of thought в режимах `hidden`, `tool_call`, `full`; это именно отображение steps, и публичный production UI обычно должен выбирать `tool_call`, а не раскрывать внутреннее рассуждение. [UI config: `cot`](https://docs.chainlit.io/backend/config/ui#options).

Рекомендация для Deep Agent:

- `type="tool"`: shell, Python, read/write/list/search и прочие наблюдаемые tool calls;
- короткий `name` + безопасно сокращённый input;
- output только после фильтрации секретов и ограничения размера;
- `is_error=True` для ошибок;
- вложенность для подагентов/составных операций;
- `default_open=False` для шумных шагов, но открывать ошибку или шаг, требующий внимания.

### Elements

Element — контент, прикреплённый к Message или Step. Для обычных Elements — Text, Image, Dataframe (pandas и с 2.11.0 polars), File, PDF viewer, Audio, Video, Plotly, Pyplot и Custom — placement задаётся как `inline`, `side` или `page`; `TaskList` является специальным элементом, отправляется напрямую и показывается в отдельной области рядом с чатом. [Element concept](https://docs.chainlit.io/concepts/element), [TaskList API](https://docs.chainlit.io/api-reference/elements/tasklist), [официальный API index элементов](https://docs.chainlit.io/llms.txt), [2.11.0 changelog](https://github.com/Chainlit/chainlit/blob/main/CHANGELOG.md#2110---2026-04-07).

Подходящее отображение:

| Данные | Элемент | Размещение |
|---|---|---|
| Небольшая иллюстрация/preview | `Image` | `inline` |
| Большой артефакт или изменённый файл | `File`, `PDF`, `Text` | `side` |
| Табличный результат | `Dataframe` | `inline`/`side` |
| График | `Plotly` | `inline` |
| Ход длительной работы | `TaskList` | отдельная область рядом с чатом |
| Diff, approval form, file manifest | `CustomElement` | `inline`/`side` |

Источники API: [Image](https://docs.chainlit.io/api-reference/elements/image), [File](https://docs.chainlit.io/api-reference/elements/file), [PDF](https://docs.chainlit.io/api-reference/elements/pdf), [Dataframe](https://docs.chainlit.io/api-reference/elements/dataframe), [Plotly](https://docs.chainlit.io/api-reference/elements/plotly), [TaskList](https://docs.chainlit.io/api-reference/elements/tasklist), [Custom](https://docs.chainlit.io/api-reference/elements/custom).

## 4. Actions, формы и `Ask*`

`Action` добавляет к сообщению кнопку с `name`, JSON `payload`, label/icon/tooltip; нажатие вызывает соответствующий `@cl.action_callback`, а action можно обновить или удалить. [Action API](https://docs.chainlit.io/api-reference/action).

Подходящие сценарии: «повторить», «показать diff», «скачать всё», «применить», «откатить», «продолжить». Payload нельзя считать доверенным: это ввод клиента, который backend обязан авторизовать и валидировать.

Ask APIs синхронно запрашивают у пользователя строку, файл, выбор action или заполнение Custom Element; пока ответа нет, UI и текущий coroutine заблокированы. [Ask User](https://docs.chainlit.io/advanced-features/ask-user). Доступны:

- `AskUserMessage`: строка с timeout. [API](https://docs.chainlit.io/api-reference/ask/ask-for-input)
- `AskFileMessage`: MIME/extension allowlist, max size/count; API ограничивает `max_files` значением 10. [API](https://docs.chainlit.io/api-reference/ask/ask-for-file)
- `AskActionMessage`: выбор из действий. [API](https://docs.chainlit.io/api-reference/ask/ask-for-action)
- `AskElementMessage`: произвольная consent-gated форма, возвращающая подтверждённые props. [API](https://docs.chainlit.io/api-reference/ask/ask-for-element)

`AskElementMessage` особенно подходит для опасных операций: агент формирует предзаполненную форму, пользователь проверяет/редактирует её и подтверждает, после чего backend выполняет действие с подтверждёнными данными. [Consent-gated flow](https://docs.chainlit.io/advanced-features/ask-user#interactive-consent-gated-forms).

Ограничение UX: Ask блокирует дальнейший chat input. Поэтому обычные развилки лучше делать неблокирующими `Action`, а Ask использовать только когда без ответа невозможно безопасно продолжать Turn.

## 5. Settings, Profiles, Starters, Commands и Modes

`ChatSettings` создаёт динамическую форму из Checkbox, DatePicker, MultiSelect, RadioGroup, Select, Slider, Switch, Tags и TextInput; inputs можно группировать по tabs. `send()` обновляет UI и committed session settings, а появившийся в 2.11.0 `refresh()` меняет форму без фиксации значения. [Chat Settings](https://docs.chainlit.io/api-reference/chat-settings), [input widgets index](https://docs.chainlit.io/llms.txt).

Chat settings можно держать в composer modal или в resizable right sidebar; выбор задаётся `UI.chat_settings_location`, а стартовое состояние — `default_chat_settings_open`. [UI config](https://docs.chainlit.io/backend/config/ui#options).

`ChatProfile` имеет internal name, markdown description, icon, display name, starters и per-profile `config_overrides`; выбранный профиль доступен как `user_session["chat_profile"]`. [Chat Profiles API](https://docs.chainlit.io/api-reference/chat-profiles). Это хороший UI для **неизменяемой конфигурации нового Chat**, как Model Profile в этом проекте, но не для параметров, которые меняются на каждом Turn.

Starters — стартовые prompt-карточки; их можно локализовать, привязать к profile и с 2.10 группировать по категориям. В 2.11.1 callback категорий получил `chat_profile`; если заданы и starters, и categories, categories имеют приоритет. [Starters](https://docs.chainlit.io/concepts/starters), [2.11.1 release](https://github.com/Chainlit/chainlit/releases/tag/2.11.1).

Commands позволяют выбрать named tool/mode возле composer, а выбранное имя приходит как `message.command`. [Commands](https://docs.chainlit.io/concepts/command), [Message API](https://docs.chainlit.io/api-reference/message). Modes, добавленные в 2.9.4, позволяют нескольким picker-ам передавать `{mode_id: option_id}` в `message.modes`; они требуют колонку `steps.modes` в persistence schema. [Modes](https://docs.chainlit.io/concepts/modes), [migration 2.9.4](https://docs.chainlit.io/guides/migration/2.9.4).

Распределение для проекта:

- Profile: модель/endpoint class и capabilities, фиксированные на Chat.
- Settings: verbosity, показывать tool details, формат ответа; хранить только параметры, которые действительно можно безопасно менять.
- Commands: явный режим текущего Turn — например `/plan`, `/code`, `/explain`.
- Starters: «проанализировать файл», «исправить тест», «объяснить проект», «создать артефакт».
- Categories: «Код», «Анализ», «Файлы», адаптированные к Model Profile.

## 6. Authentication, persistence, history и feedback

Chainlit app публичен по умолчанию. Для приватности требуются `CHAINLIT_AUTH_SECRET` и password, OAuth или header callback; identifier каждого пользователя должен быть уникальным, иначе данные могут смешаться. [Authentication overview](https://docs.chainlit.io/authentication/overview).

Header auth предназначен для делегирования проверки reverse proxy, но callback приложения всё равно отвечает за фактическую валидацию header и должен вернуть `User` или `None`. [Header authentication](https://docs.chainlit.io/authentication/header). Текущий callback всегда возвращает одного `local-user`, поэтому безопасность целиком опирается на JupyterHub/proxy boundary. [`app.py`](../../app.py)

По умолчанию chats/elements не сохраняются; persistence подключается data layer. [Persistence overview](https://docs.chainlit.io/data-persistence/overview). Официальный SQLAlchemy layer тестируется с PostgreSQL и поддерживает blob clients для Azure/S3; его schema включает users, threads, steps, elements и feedback. [SQLAlchemy data layer](https://docs.chainlit.io/data-layers/sqlalchemy).

History с поиском, resume и thread list появляется только вместе с persistence **и** authentication; resume обрабатывается `on_chat_resume`. [Chat History](https://docs.chainlit.io/data-persistence/history). Feedback thumbs up/down и текстовый comment появляются при persistence. [Human Feedback](https://docs.chainlit.io/data-persistence/feedback).

Для этого проекта:

- Оставить локальный SQLite допустимо для single-user/single-process deployment; это вывод из локальной архитектуры, а не гарантия Chainlit. [`docs/architecture.md`](../architecture.md)
- При переходе к нескольким JupyterHub users identifier должен выводиться из доверенного Hub identity, а не быть константой. [Auth uniqueness requirement](https://docs.chainlit.io/authentication/overview).
- Custom schema нужно сверять при каждом minor upgrade: в официальную `steps` schema после 2.0 добавлялись `command`, `defaultOpen`, `modes`. [SQLAlchemy migrations](https://docs.chainlit.io/data-layers/sqlalchemy#sqlalchemy-data-layer).
- Feedback уже есть в локальной schema, поэтому сначала достаточно проверить UI round-trip и добавить локальный аналитический запрос; отдельный SaaS для этого не обязателен. [`chainlit_data.py`](../../local_agent_chat/chainlit_data.py), [Feedback](https://docs.chainlit.io/data-persistence/feedback).

## 7. Streaming и ощущение скорости

Chainlit stream-ит и Message, и Step. Базовый protocol: создать пустой message/step, вызывать `stream_token()` для чанков, затем `update()`; интеграции могут предоставить streaming через callback handler. [Streaming](https://docs.chainlit.io/advanced-features/streaming), [Message API](https://docs.chainlit.io/api-reference/message).

Главный UX-дефицит текущего приложения — `Agent.run()` возвращает готовую строку, а `app.py` отправляет её только после завершения. [`runtime.py`](../../local_agent_chat/runtime.py), [`app.py`](../../app.py). Рекомендуемая граница:

```text
agent event stream
  ├─ token(delta)       -> Message.stream_token
  ├─ tool_started       -> Step.send
  ├─ tool_output(delta) -> Step.stream_token / Step.update
  ├─ artifact_changed   -> Element после завершения tool
  ├─ task_status        -> TaskList update
  └─ final              -> Message.update + persistence commit
```

Нужно различать live UI и canonical persistence: частичные токены можно показывать сразу, но Turn становится завершённым только после финального события. При stop/error message и step должны получить явный terminal state, а история не должна выглядеть как успешный полный ответ.

## 8. Файлы, аудио и multimodal

Spontaneous uploads позволяют drag-and-drop/attach к обычному сообщению; backend получает их в `message.elements`. MIME types, max files и max MiB задаются в `[features.spontaneous_file_upload]`. [Multi-Modality](https://docs.chainlit.io/advanced-features/multi-modal), [features config](https://docs.chainlit.io/backend/config/features#file-upload).

Chainlit отображает Image/File/PDF/Audio/Video как элементы, но интерпретация содержимого остаётся задачей приложения/модели. [Elements index](https://docs.chainlit.io/llms.txt). То есть наличие image element не гарантирует vision capability выбранного Model Profile.

Для микрофона нужно включить `[features.audio].enabled=true` и реализовать `on_audio_chunk`; завершение потока обрабатывает `on_audio_end`. С 2.7 наличие hook больше не включает audio автоматически. [Audio config](https://docs.chainlit.io/backend/config/features#audio), [on_audio_chunk](https://docs.chainlit.io/api-reference/lifecycle-hooks/on-audio-chunk), [on_audio_end](https://docs.chainlit.io/api-reference/lifecycle-hooks/on-audio-end).

Рекомендации безопасности и UX:

- заменить `accept=["*/*"]` явным allowlist для поддерживаемых форматов;
- проверять фактический тип/сигнатуру файла server-side, не доверять extension/MIME клиента;
- показывать upload progress, итоговый manifest и причины отказа;
- не включать audio до появления STT/realtime pipeline и политики хранения;
- не дублировать крупные binaries в SQLite — текущая blob/Sandbox архитектура уже следует этому принципу. [`docs/architecture.md`](../architecture.md)

## 9. Custom frontend, Custom Elements, тема, CSS, JS и переводы

### Самый устойчивый слой

Тема построена на CSS variables; `public/theme.json` задаёт light/dark variables и custom fonts. [Theme](https://docs.chainlit.io/customisation/theme). `.chainlit/config.toml` настраивает name, layout, sidebar, default theme, CoT, logos, avatar, header links, alerts и meta tags. [UI config](https://docs.chainlit.io/backend/config/ui).

Custom CSS и JS подключаются конфигурацией `custom_css`/`custom_js`. [Custom CSS](https://docs.chainlit.io/customisation/custom-css), [Custom JS](https://docs.chainlit.io/customisation/custom-js). CSS selectors, привязанные к внутренней DOM-структуре, и JS monkey-patches являются хрупкими при upgrade; предпочтительны documented CSS variables, config и Custom Elements.

Translations живут в `.chainlit/translations/<locale>.json`; locale выбирается по browser language или принудительно через `UI.language`, а `chainlit_<locale>.md` локализует welcome page. [Translation](https://docs.chainlit.io/customisation/translation), [UI language](https://docs.chainlit.io/backend/config/ui#options). В списке встроенных locale нет русского, поэтому нужен `ru-RU.json`; файлы проверяются `chainlit lint-translations`. [Built-in languages и lint](https://docs.chainlit.io/customisation/translation#built-in-languages).

### Custom Elements

`CustomElement` рендерит `public/elements/<Name>.jsx`, получает JSON props, работает в окружении React + Tailwind/shadcn и поддерживает только allowlisted imports. [Custom Element](https://docs.chainlit.io/api-reference/elements/custom). Глобальные API позволяют обновить/удалить element, вызвать Chainlit action и отправить user message. [Custom Element APIs](https://docs.chainlit.io/api-reference/elements/custom#available-apis).

Это лучший механизм для:

- diff viewer с подтверждением;
- карточки Sandbox manifest;
- progress/status dashboard;
- структурированного tool result;
- consent form через `AskElementMessage`;
- предпросмотра набора файлов.

Ограничения: только JSX, props внедряются особым способом, imports ограничены, компонент живёт внутри topology message/step, а не управляет всей страницей. [Правила JSX и imports](https://docs.chainlit.io/api-reference/elements/custom#how-to-write-the-jsx-file).

### Полный build и React client

`UI.custom_build` может указать каталог собственного production build. [UI config: custom build](https://docs.chainlit.io/backend/config/ui#options). Более явный путь — отдельное React приложение с `@chainlit/react-client`: hooks управляют WebSocket session, messages, state/actions/asks/settings, отправкой сообщений/files/audio/actions/stop, authentication и HTTP API. [React usage](https://docs.chainlit.io/deploy/react/usage).

Это оправдано, только если нужны глобальная компоновка и navigation, недоступные штатному frontend. Иначе цена — самостоятельные accessibility, mobile UX, error/reconnect states, history UI, file upload, Ask UI и регулярная синхронизация npm/Python протоколов.

## 10. Copilot, embed и интеграции

Copilot вставляется одним script и `mountChainlitWidget`; режимы — floating или, с 2.11.0, resizable sidebar. Он поддерживает messages, streaming, elements, audio, Ask User, history, profiles и feedback. [Copilot](https://docs.chainlit.io/deploy/copilot), [sidebar mode](https://docs.chainlit.io/deploy/copilot#sidebar-mode).

Widget может вызывать функции host page через `CopilotFunction`/`chainlit-call-fn` и получать context/system messages через `sendChainlitMessage`. [Copilot function calling](https://docs.chainlit.io/deploy/copilot#function-calling), [send a message](https://docs.chainlit.io/deploy/copilot#send-a-message). Для cross-origin embed нужно сузить/настроить `allow_origins`; для разных доменов с auth документация требует `CHAINLIT_COOKIE_SAMESITE=none`. [Copilot security](https://docs.chainlit.io/deploy/copilot#security).

Официальные integration adapters есть для FastAPI, LangChain/LangGraph, OpenAI, Semantic Kernel, Mistral, LiteLLM, LlamaIndex, Embedchain и message-based backends; отдельные delivery surfaces — Teams, Slack и Discord. [Официальный docs index](https://docs.chainlit.io/llms.txt), [deployment platforms](https://docs.chainlit.io/deploy/overview#available-platforms).

Для текущего standalone JupyterHub UI Copilot не нужен. Он становится полезен, если чат должен появиться внутри IDE/notebook portal или другого web-продукта и обмениваться с host контекстом.

## 11. Deployment, security, observability и testing

### Deployment

Chainlit использует WebSocket; при горизонтальном масштабировании документация требует session affinity/sticky sessions и предлагает ограничить transport до `websocket`, если load balancer плохо маршрутизирует long-lived connection. Для subpath используется `--root-path`. [Deployment tips](https://docs.chainlit.io/deploy/overview#tips--tricks).

Это подтверждает текущий запуск через JupyterHub proxy с `APP_ROOT_PATH`. [`README.md`](../../README.md), [`scripts/run.sh`](../../scripts/run.sh). Локальные in-memory backends и per-process locks дополнительно означают, что нынешняя реализация рассчитана на один процесс; для replicas потребуется вынести runtime coordination/state. [`runtime.py`](../../local_agent_chat/runtime.py), [`docs/architecture.md`](../architecture.md).

### Security checklist

- Не оставлять `allow_origins=["*"]` при доступе вне доверенного same-origin proxy; поле непосредственно определяет разрешённые origins app/copilot. [Project config](https://docs.chainlit.io/backend/config/project#options)
- Сохранять `unsafe_allow_html=false`: документация прямо помечает HTML rendering как security risk. [Features config](https://docs.chainlit.io/backend/config/features#other)
- Не сохранять user API keys без необходимости; `persist_user_env` по умолчанию false, а `mask_user_env` влияет только на отображение. [Project config](https://docs.chainlit.io/backend/config/project#options)
- Уникально связывать authenticated user с thread ownership. [Authentication](https://docs.chainlit.io/authentication/overview)
- Не доверять Action/CustomElement payload, filenames, MIME, markdown links или Copilot host events.
- Обновляться минимум выше 2.10.0: в 2.10.1 закрыта уязвимость проверки ownership при WebSocket session restore. [Release 2.10.1](https://github.com/Chainlit/chainlit/releases/tag/2.10.1)
- Учитывать dependency security floor: 2.11.1 требует современные FastAPI/Starlette из-за CVE-2024-47874. [CHANGELOG 2.11.1](https://github.com/Chainlit/chainlit/blob/main/CHANGELOG.md#2111---2026-04-22), [2.11.1 dependencies](https://github.com/Chainlit/chainlit/blob/2.11.1/backend/pyproject.toml#L35-L66)

### Observability

Steps, timestamps, `isError`, metadata/tags, feedback и generation fields дают хорошую продуктовую трассу в data layer. [Step API](https://docs.chainlit.io/api-reference/step-class), [SQLAlchemy schema](https://docs.chainlit.io/data-layers/sqlalchemy), [tags/metadata](https://docs.chainlit.io/data-persistence/tags-metadata).

Однако Chainlit не следует считать полноценной observability platform: в 2.11.1 объявлено, что LiteralAI прекращает работу и будет удалён в будущем. [CHANGELOG 2.11.1](https://github.com/Chainlit/chainlit/blob/main/CHANGELOG.md#2111---2026-04-22). Поэтому рекомендуется писать структурированные app logs/metrics/traces независимо от UI persistence, связывая их `thread_id`, `turn_id`, `step_id`, profile и duration.

### Testing

Официальный debugging path запускает приложение через `run_chainlit(__file__)` из IDE. [Testing & Debugging](https://docs.chainlit.io/advanced-features/test-debug). Сам репозиторий Chainlit имеет backend pytest, frontend unit/Vitest и Cypress E2E suites, что показывает уровень, необходимый при собственном frontend. [Официальные команды тестов](https://github.com/Chainlit/chainlit/blob/main/AGENTS.md#tests).

Для проекта рекомендуются:

1. Unit tests event-to-Message/Step mapping без browser.
2. Integration tests data layer: streaming finalization, feedback, element persistence, resume, Revision.
3. Browser smoke tests: first chat, reconnect, resume, stop, upload/download, edit/revise, mobile viewport.
4. Contract test после каждого Chainlit upgrade для custom SQLite schema и Custom Elements.

## 12. Версии и совместимость на 2026-08-21

| Компонент | Проверенное состояние | Значение для проекта |
|---|---|---|
| Chainlit stable | 2.11.1, release 2026-04-22 | Совпадает с установленным 2.11.1. [Release](https://github.com/Chainlit/chainlit/releases/tag/2.11.1) |
| Python | `>=3.10,<3.14` | Python 3.12 совместим. [`pyproject`](https://github.com/Chainlit/chainlit/blob/2.11.1/backend/pyproject.toml#L1-L35) |
| FastAPI / Starlette | `fastapi>=0.116.1`, `starlette>=0.47.2` | Не следует принудительно понижать transitive dependencies. [`pyproject`](https://github.com/Chainlit/chainlit/blob/2.11.1/backend/pyproject.toml#L35-L66) |
| Socket.IO | `python-socketio>=5.11,<6` | Custom client должен соответствовать protocol bundled react-client. [`pyproject`](https://github.com/Chainlit/chainlit/blob/2.11.1/backend/pyproject.toml#L35-L66) |
| Локальный pin | `chainlit>=2.11,<2.12` | Разумно для стабильности custom data layer. [`pyproject.toml`](../../pyproject.toml) |
| Project status | Community-maintained с 2025-05-01; Chainlit SAS не гарантирует будущие updates | Нужны pin, upgrade tests и возможность замены frontend/backend seam. [Official backend README](https://github.com/Chainlit/chainlit/blob/main/backend/README.md) |

Критичные изменения недавних версий:

- 2.9.4 добавил `steps.modes` и потребовал migration persistence. [Release 2.9.4](https://github.com/Chainlit/chainlit/releases/tag/2.9.4)
- 2.10.1 исправил ownership validation при WebSocket restore. [Release 2.10.1](https://github.com/Chainlit/chainlit/releases/tag/2.10.1)
- 2.11.0 добавил Copilot sidebar, polars Dataframe и новый PDF viewer. [CHANGELOG 2.11.0](https://github.com/Chainlit/chainlit/blob/main/CHANGELOG.md#2110---2026-04-07)
- 2.11.1 исправил reconnect lifecycle и locale fallback, расширил profile-aware starter categories. [CHANGELOG 2.11.1](https://github.com/Chainlit/chainlit/blob/main/CHANGELOG.md#2111---2026-04-22)

Не рекомендуется переходить на `main`/git dependency: стабильный tag 2.11.1 уже покрывает нужные функции, а локальный data layer опирается на internal schema/API.

## 13. Ограничения и trade-offs

### Сильные стороны

- Очень короткий путь от Python agent events до интерактивного web UI. [Message](https://docs.chainlit.io/api-reference/message), [Step](https://docs.chainlit.io/api-reference/step-class)
- Нативные streaming, tool steps, artifacts, forms, history и feedback. [Official docs index](https://docs.chainlit.io/llms.txt)
- Возможность постепенно углублять UI: config → Custom Elements → React client. [Custom Element](https://docs.chainlit.io/api-reference/elements/custom), [React overview](https://docs.chainlit.io/deploy/react/overview)
- Self-hosted Apache-2.0 open source. [Official repository](https://github.com/Chainlit/chainlit)

### Риски и ограничения

- WebSocket affinity и process-local session/runtime усложняют horizontal scaling. [Deployment](https://docs.chainlit.io/deploy/overview#account-for-websockets)
- Ask UX блокирует весь дальнейший ввод до ответа/timeout. [Ask User](https://docs.chainlit.io/advanced-features/ask-user)
- Custom Element ограничен allowlisted runtime/imports и message topology. [Custom Element rules](https://docs.chainlit.io/api-reference/elements/custom#how-to-write-the-jsx-file)
- Custom CSS/JS может зависеть от внутренних DOM details; documented theme variables устойчивее. [Theme](https://docs.chainlit.io/customisation/theme), [custom CSS](https://docs.chainlit.io/customisation/custom-css)
- Full React даёт контроль, но требует заново собрать весь chat UX, несмотря на наличие low-level hooks. [React usage](https://docs.chainlit.io/deploy/react/usage)
- Persistence schema эволюционирует между minor releases; локальная копия требует миграций. [SQLAlchemy data layer](https://docs.chainlit.io/data-layers/sqlalchemy)
- Проект community-maintained, а LiteralAI sunset снижает уверенность в встроенной долгосрочной observability story. [Official README](https://github.com/Chainlit/chainlit/blob/main/backend/README.md), [CHANGELOG 2.11.1](https://github.com/Chainlit/chainlit/blob/main/CHANGELOG.md#2111---2026-04-22)
- Built-in Russian locale отсутствует; локализацию нужно поддерживать самим. [Translations](https://docs.chainlit.io/customisation/translation#built-in-languages)

## 14. Рекомендуемая UI-архитектура для этого репозитория

### Решение

Оставить Chainlit 2.11.x как **UI shell и transport**, а `ChatRuntime` — как независимый domain/application layer. Между ними ввести типизированный поток событий, чтобы Chainlit-specific objects создавались только в adapter layer.

```text
Deep Agent / Sandbox
        │ typed AgentEvent
        ▼
ChatRuntime (Turn transaction, checkpoint, Revision, cancellation)
        │ async event stream
        ▼
Chainlit UI adapter
  Message · Step · TaskList · Element · Action · AskElement
        │ Socket.IO/WebSocket
        ▼
Stock Chainlit React UI + small Custom Elements
```

Такая граница сохраняет нынешние гарантии Revision и позволяет позже заменить только UI adapter на React client, не переписывая agent/runtime.

### Фаза 1 — высокая ценность, низкий риск

1. **Token streaming.** Изменить agent service/runtime contract с `str` на async event stream; финальный текст всё равно сохранять как canonical answer. Использовать `Message.stream_token()`/`update()`. [Streaming API](https://docs.chainlit.io/advanced-features/streaming)
2. **Tool Steps.** Отображать shell/Python/filesystem операции как `Step(type="tool")`, ошибки — явным состоянием; сменить `cot` с `full` на `tool_call`. [Step](https://docs.chainlit.io/api-reference/step-class), [UI cot](https://docs.chainlit.io/backend/config/ui#options)
3. **Stop/cancellation.** Подключить `on_stop` к task cancellation и terminal state. [Lifecycle](https://docs.chainlit.io/concepts/chat-lifecycle#on-stop)
4. **TaskList.** Для длительных задач показывать план/прогресс, не смешивая его с ответом. [TaskList](https://docs.chainlit.io/api-reference/elements/tasklist)
5. **Starters/categories.** Добавить profile-aware сценарии на welcome screen. [Starters](https://docs.chainlit.io/concepts/starters)
6. **Русский UI.** Создать `ru-RU.json`, локализовать watermark и ключевые controls; валидировать `chainlit lint-translations`. [Translation](https://docs.chainlit.io/customisation/translation)
7. **Theme.** Добавить `public/theme.json`; ограничить custom CSS documented tokens. [Theme](https://docs.chainlit.io/customisation/theme)

### Фаза 2 — интерактивные agent workflows

1. `Action` для repeat/show diff/download/apply/rollback. [Action](https://docs.chainlit.io/api-reference/action)
2. `AskActionMessage` для обязательного бинарного подтверждения опасной операции. [Ask Action](https://docs.chainlit.io/api-reference/ask/ask-for-action)
3. `AskElementMessage` + Custom Element для review/edit/confirm структурированного плана операции. [Ask Element](https://docs.chainlit.io/api-reference/ask/ask-for-element)
4. Custom `DiffViewer` и `SandboxManifest`; props должны содержать только serializable presentation data, а backend повторно проверяет permission и текущую revision.
5. ChatSettings только для безопасных UI/response preferences; Model Profile остаётся immutable per Chat. [Chat Settings](https://docs.chainlit.io/api-reference/chat-settings), [Chat Profiles](https://docs.chainlit.io/api-reference/chat-profiles)
6. Включить feedback и покрыть persistence/resume test. [Feedback](https://docs.chainlit.io/data-persistence/feedback)

### Фаза 3 — hardening

1. Сузить `allow_origins`; заменить upload wildcard allowlist-ом. [Project config](https://docs.chainlit.io/backend/config/project), [Features config](https://docs.chainlit.io/backend/config/features)
2. Добавить structured logs/metrics с IDs и durations; не связывать observability roadmap с LiteralAI.
3. Browser E2E для reconnect/resume/revision/stop/upload/custom forms.
4. Документировать и автоматизировать schema diff/migration при каждом minor upgrade.
5. Если появится multi-user deployment, получать verified JupyterHub identity и перейти с process-local coordination на shared primitives.

### Когда переходить на собственный React frontend

Переход оправдан, если подтверждены минимум два из требований:

- постоянный file tree/editor/diff рядом с чатом;
- несколько независимых рабочих панелей и собственная маршрутизация;
- дизайн-система продукта, несовместимая со штатной topology;
- сложная keyboard/accessibility модель;
- встраивание chat state в существующее React приложение глубже, чем позволяет Copilot;
- необходимость независимо версионировать web UI.

До этого Custom Elements дают большую часть интерактивности при существенно меньшей площади сопровождения.

## 15. Приоритетный backlog

| Приоритет | Изменение | Пользовательский эффект | Риск |
|---|---|---|---|
| P0 | Streaming final answer | Немедленная обратная связь | Средний: меняется runtime contract |
| P0 | Tool Steps + Stop | Прозрачность и контроль | Средний: cancellation semantics |
| P0 | Upload allowlist и CORS hardening | Безопасность | Низкий |
| P1 | TaskList progress | Понятность длинных задач | Низкий |
| P1 | Starters/categories | Быстрый старт | Низкий |
| P1 | `ru-RU` + theme | Цельный локализованный UX | Низкий |
| P1 | Feedback round-trip | Измеримый UX | Низкий/средний |
| P2 | Diff/manifest Custom Elements | Богатая работа с файлами | Средний |
| P2 | Consent-gated AskElement | Безопасные действия агента | Средний |
| P3 | Full React frontend | Полный контроль layout | Высокий; только по подтверждённым требованиям |

## Итог

Chainlit 2.11.1 достаточно зрел для удобного интерактивного UI этого проекта без frontend rewrite. Наибольший эффект дадут не CSS-изменения, а правильное отображение domain events: token streaming, tool Steps, TaskList, отмена, Actions и consent-gated forms. Штатный UI следует считать готовой оболочкой, Custom Elements — расширениями для специализированных workflow, а React client — запасным архитектурным выходом, а не стартовой точкой.
