# Кейсы agent UI на Chainlit: промежуточные шаги и удобный интерфейс

Дата исследования: **2026-08-21**. Цель — найти в официальных и зрелых открытых реализациях Chainlit переносимые UX-паттерны для Local Agent Chat, прежде всего для отображения хода работы Deep Agents.

## Резюме

Полностью собственный frontend проекту сейчас не нужен. Chainlit 2.11.1 уже предоставляет необходимые примитивы: потоковые `Message` и `Step`, вложенность шагов, ошибки и статусы, `TaskList`, actions, подтверждения, файлы и произвольные JSX-элементы. [Message API](https://docs.chainlit.io/api-reference/message), [Step API](https://docs.chainlit.io/api-reference/step-class), [Streaming](https://docs.chainlit.io/advanced-features/streaming).

Наиболее удачная модель, повторяющаяся в официальном Cookbook и сторонних интеграциях:

```text
Сообщение пользователя
├─ Выполнение                          Step(type="run")
│  ├─ Чтение файла                    Step(type="tool")
│  ├─ Поиск по проекту                Step(type="tool")
│  ├─ Shell: pytest                   Step(type="tool")
│  └─ Изменение файла                 Step(type="tool")
└─ Финальный ответ агента             streaming Message
```

То есть диалог остаётся коротким, а наблюдаемая деятельность агента раскрывается по требованию. Показывать следует операции, их безопасные входы, результаты, ошибки и созданные артефакты — **не необработанное внутреннее chain-of-thought модели**.

Для текущего проекта рекомендуются два этапа:

1. Быстрый прототип — передать `cl.LangchainCallbackHandler` в `RunnableConfig` Deep Agents, отфильтровать служебные узлы LangGraph и проверить реальную трассу.
2. Целевая реализация — независимый типизированный поток событий агента и отдельный Chainlit-адаптер, управляющий жизненным циклом `Step`/`Message`. Он позволит безопасно сокращать вывод, давать операциям понятные имена и правильно переигрывать историю после редактирования сообщения.

## Метод

Использованы первичные источники: документация и исходный код Chainlit 2.11.1, официальный [Chainlit Cookbook](https://github.com/Chainlit/cookbook/tree/218e4f9a1837a46fcaaf40b3ca3033d71b7fe66e), официальные примеры Microsoft AutoGen и исходный Chainlit-адаптер Langroid. Примеры Cookbook проверены на commit `218e4f9a1837a46fcaaf40b3ca3033d71b7fe66e`; это важно, потому что часть примеров использует устаревшие версии LangGraph или provider API.

Этот документ дополняет общий обзор [`chainlit-interactive-ui.md`](./chainlit-interactive-ui.md): здесь внимание сосредоточено на конкретных реализациях и решениях, которые можно перенести в проект.

## Почему текущий интерфейс выглядит «неактивным»

Сейчас [`AgentService.run()`](../../local_agent_chat/agent_service.py) вызывает `graph.ainvoke(...)`, ждёт готовый результат и возвращает строку. [`ChatRuntime`](../../local_agent_chat/runtime.py) также оперирует только готовым `answer`, а [`app.py`](../../app.py) отправляет один `cl.Message` уже после полного завершения Turn.

Поэтому UI не получает:

- токены финального ответа по мере генерации;
- события начала и завершения tool calls;
- stdout/stderr выполняющейся команды;
- явный running/error/cancelled status;
- связь созданного файла с породившим его шагом.

Настройка `cot = "full"` в [`.chainlit/config.toml`](../../.chainlit/config.toml) сама по себе ничего не добавляет: Chainlit может отобразить только те `Step`, которые создаёт приложение или callback handler.

## Разобранные реализации

| Реализация | Полезный паттерн | Что перенять | Ограничение |
|---|---|---|---|
| [OpenAI Data Analyst](https://github.com/Chainlit/cookbook/blob/218e4f9a1837a46fcaaf40b3ca3033d71b7fe66e/openai-data-analyst/app.py) | Центральный обработчик provider events; один streaming answer; tool calls превращаются в дочерние `Step`; код и логи обновляются в существующем шаге; изображения и файлы прикрепляются как элементы | Архитектуру `agent events → UI adapter`, потоковые input/output, артефакты рядом с породившим их шагом, отмену через `on_stop` | Реализация привязана к API конкретного провайдера |
| [BigQuery](https://github.com/Chainlit/cookbook/blob/218e4f9a1837a46fcaaf40b3ca3033d71b7fe66e/bigquery/app.py) | Корневой `run`, вложенные стадии генерации SQL, выполнения и анализа; SQL stream-ится с подсветкой; ответ идёт отдельно | Небольшое понятное дерево вместо россыпи служебных событий; язык ввода `sql`/`bash`/`python` | Жёстко заданный pipeline проще произвольного агента |
| [LangGraph + Tavily](https://github.com/Chainlit/cookbook/blob/218e4f9a1837a46fcaaf40b3ca3033d71b7fe66e/langgraph-tavily/app.py) | `cl.LangchainCallbackHandler` передаётся в граф; `ChannelRead`, `ChannelWrite`, `RunnableLambda`, `__start__`, `_execute` скрываются | Самый короткий путь к первой полезной трассе Deep Agents; deny/allow list внутренних runs | Код самого примера устарел; фильтр нужно подобрать по фактической трассе Deep Agents |
| [OpenAI Responses advanced](https://github.com/Chainlit/cookbook/blob/218e4f9a1837a46fcaaf40b3ca3033d71b7fe66e/openai-responses-gpt5-functions-streaming-multi-modal-reasoning-super-advanced/app.py) | Большой event router; статусы tool execution; потоковая генерация Python; настройка видимости деталей; MIME-зависимые Image/Dataframe/File | Таксономию событий, переключатель подробности, preview артефактов и running state | Не копировать локальный `subprocess`: он не является безопасной песочницей. Не создавать новый Step на каждую смену статуса |
| [Extended Thinking](https://github.com/Chainlit/cookbook/blob/218e4f9a1837a46fcaaf40b3ca3033d71b7fe66e/extended-thinking-in-the-ui/app.py) | Промежуточный поток отделён от финального `Message` | Разделение «ход работы» и «ответ» | Не выводить raw chain-of-thought; допустимы только reasoning summary или фактические статусы |
| [PyAutoGen](https://github.com/Chainlit/cookbook/blob/218e4f9a1837a46fcaaf40b3ca3033d71b7fe66e/pyautogen/async_app.py) и [Microsoft AutoGen](https://github.com/microsoft/autogen/blob/027ecf0a379bcc1d09956d46d12d44a3ad9cee14/python/samples/agentchat_chainlit/app_team_user_proxy.py) | Стабильный `author` для каждого агента; `AskActionMessage` для Continue/Approve/Reject; пользователь — участник workflow | В будущем — метки подагентов и явное подтверждение опасной операции | Каждое межагентное сообщение как верхнеуровневый chat bubble быстро создаёт шум; подагентов лучше вкладывать в Step |
| [HumanLayer](https://github.com/Chainlit/cookbook/blob/218e4f9a1837a46fcaaf40b3ca3033d71b7fe66e/humanlayer-openai/app.py) | Долгоживущий вложенный шаг «ожидаю подтверждения», затем обновление того же объекта результатом | Один status step для ожидания пользователя, без серии системных сообщений | Внешний сервис проекту не нужен; нативного `AskActionMessage` достаточно |
| [Langroid Chainlit callbacks](https://github.com/langroid/langroid/blob/e0c2a34369d1fad25a709ed80f69c67d51883681/langroid/agent/callbacks/chainlit.py) | Полноценный отдельный adapter; рекурсивное подключение subagents; сохранение `parent_id`; spinner создаётся сразу и обновляется результатом; отдельное отображение ошибок | Архитектурную границу между агентом и UI, стабильную identity событий, вложенность subagents | Не копировать framework-specific state и сломанные старые nested-tool examples |
| [PDF QA](https://github.com/Chainlit/cookbook/blob/218e4f9a1837a46fcaaf40b3ca3033d71b7fe66e/pdf-qa/app.py) | Статус обработки обновляется на месте; источники прикрепляются как `Text` elements | Upload/indexing status и цитаты рядом с ответом | Не использовать отдельные сообщения для каждого короткого статуса |
| [Custom Element PDF selector](https://github.com/Chainlit/cookbook/blob/218e4f9a1837a46fcaaf40b3ca3033d71b7fe66e/custom-element/public/elements/PdfSelector.jsx) | JSX-компонент меняет props и вызывает backend action | В перспективе — карточка diff, file manifest или выбор файла | Для базовых tool steps избыточен; сначала использовать штатные элементы |

## Повторяющиеся удачные UX-паттерны

### 1. Один объект на одну операцию

При старте tool call создаётся `Step` со spinner, затем **тот же Step** получает output, error и время завершения. Серия сообщений «запустил → выполняю → завершил» хуже: она раздувает чат, ломает визуальную связь событий и плохо восстанавливается из истории.

`Step` поддерживает `input`, `output`, `parent_id`, `show_input`, `language`, `default_open`, `auto_collapse`, streaming и update. [Step API](https://docs.chainlit.io/api-reference/step-class), [Step source 2.11.1](https://github.com/Chainlit/chainlit/blob/2.11.1/backend/chainlit/step.py).

Практический вариант:

```python
step = cl.Step(
    name="Shell: pytest",
    type="tool",
    parent_id=run_step.id,
    show_input="bash",
    default_open=True,
    auto_collapse=True,
    icon="terminal",
)
```

`show_input` лучше всегда задавать явно. Перед `send()` вход и затем output необходимо очистить от токенов, переменных окружения и чрезмерно больших фрагментов файлов. Для ошибки шаг следует оставить раскрытым и установить `is_error=True`; перед финальным `update()` стоит отключить `auto_collapse`, иначе frontend может свернуть и ошибочный шаг.

### 2. Финальный ответ отделён от трассы

Один пустой assistant `Message` создаётся в начале Turn, получает токены ответа и финализируется через `update()`. Tool calls и статусы живут в `Step`, поэтому пользователь может читать чистый ответ, не разбирая лог выполнения. Chainlit независимо stream-ит `Message` и `Step`. [Streaming](https://docs.chainlit.io/advanced-features/streaming).

Полагаться на `stream_final_answer=True` у LangChain callback не стоит: он распознаёт финальный ответ эвристически и для графовых агентов может выбрать не тот LLM stream. Надёжнее маршрутизировать финальные LangGraph chunks явно.

### 3. Progressive disclosure

Нормально завершившиеся технические шаги должны сворачиваться. Активный шаг, ошибка, подтверждение или новый значимый артефакт — раскрываться. В основном тексте остаются только итог и короткая сводка изменений.

Для пользовательского режима рекомендуется:

```toml
[UI]
cot = "tool_call"
```

Режимы `hidden`, `tool_call`, `full` документированы в [UI config](https://docs.chainlit.io/backend/config/ui). `full` полезен как диагностический режим разработчика, но обычно показывает слишком много внутренних LangChain/LangGraph runs.

### 4. Понятные действия вместо имён tools

| Событие агента | Название в UI | Ввод | Результат |
|---|---|---|---|
| `read_file` | Чтение файла | относительный путь и диапазон | краткая сводка или число строк |
| `write_file` / `edit_file` | Изменение файла | путь | diff/размер и `File` element |
| `list_files` / `glob` / `grep` | Поиск по файлам | pattern и область | количество совпадений, сокращённый список |
| shell command | Shell: короткая команда | `bash` | exit code, ограниченные stdout/stderr |
| Python command | Python | `python` | stdout/stderr и созданные артефакты |
| subagent | Подзадача: имя | безопасная постановка | короткий итог; дочерние tools вложены внутрь |

Полная команда и сырой вывод доступны в раскрытом шаге; в названии нужна короткая, стабильная и сканируемая формулировка.

### 5. Артефакт рядом с причиной

Изменённый файл, изображение, таблица или PDF должны прикрепляться к соответствующему Step либо к финальному Message. Chainlit поддерживает `File`, `Image`, `Dataframe`, `PDF`, `Plotly` и размещение `inline`, `side`, `page`. [Elements](https://docs.chainlit.io/concepts/element).

При `display="side"` имя элемента следует явно упомянуть в тексте, чтобы ссылка была заметна. Программно открытый sidebar не заменяет persisted element: после resume важный артефакт должен оставаться привязанным к сохранённому Message/Step.

### 6. TaskList — только для настоящего плана

`TaskList` хорошо показывает крупные этапы со статусами `READY`, `RUNNING`, `DONE`, `FAILED`. [TaskList API](https://docs.chainlit.io/api-reference/elements/tasklist). Для Deep Agents его можно синхронизировать с явными задачами `write_todos`: например, «изучить проект → изменить код → запустить тесты».

Не нужно придумывать план из скрытого reasoning и дублировать каждый tool call одновременно в `TaskList` и `Step`.

### 7. Подтверждение — состояние Turn, а не отдельный чат

Если позже появятся операции, требующие разрешения, `AskActionMessage` даёт явные Approve/Reject, а окружающий Step остаётся в статусе ожидания. [AskActionMessage](https://docs.chainlit.io/api-reference/ask/ask-for-action). Для текущей полностью локальной песочницы такой flow не обязателен.

## Варианты интеграции с Deep Agents

### Вариант A: штатный LangChain callback — быстрый прототип

Chainlit уже содержит tracer, который преобразует LangChain runs `agent`, `chain`, `llm`, `retriever`, `tool` в дерево `Step`, хранит связь `run_id`/`parent_run_id`, обновляет input/output, фиксирует время и ошибки. [LangChain integration](https://docs.chainlit.io/integrations/langchain), [callback source](https://github.com/Chainlit/chainlit/blob/2.11.1/backend/chainlit/langchain/callbacks.py).

Схематично:

```python
callback = cl.LangchainCallbackHandler(
    to_ignore=[
        "ChannelRead",
        "ChannelWrite",
        "RunnableLambda",
        "__start__",
        "_execute",
    ]
)
config = RunnableConfig(callbacks=[callback], configurable={...})
```

Плюсы: мало изменений, быстро видно фактические run names Deep Agents. Минусы: входы tools выводятся без проектного sanitizer; названия и детализация зависят от внутренностей LangChain; callback связывает agent service с request-local UI object. Поэтому этот вариант разумен как instrumentation spike и резервный fallback, но не как окончательная граница модулей.

### Вариант B: типизированный event stream — целевая реализация

`AgentService` публикует независимые от Chainlit события, а адаптер в `app.py` переводит их в UI:

```text
turn_started
token(delta)
tool_started(id, parent_id, kind, safe_input)
tool_output(id, delta)
tool_finished(id, safe_output, artifacts)
tool_failed(id, safe_error)
task_changed(...)
turn_finished(answer)
turn_cancelled / turn_failed
```

Преимущества:

- сервис агента не импортирует Chainlit;
- единое место для redaction, truncation и человекочитаемых labels;
- стабильные event IDs не зависят от UI;
- проще тестировать порядок и terminal states;
- можно корректно связать streaming с Revision, checkpoint и sandbox snapshot;
- позднее тот же поток можно отдать другому frontend.

На уровне LangGraph источником может быть `astream()`/`astream_events()`, но адаптер обязан отфильтровать промежуточные LLM chunks, tool-call chunks и служебные graph events по типу и metadata.

### Вариант C: CustomElement или собственный frontend

`CustomElement` оправдан для одного богатого представления — например, diff viewer или manifest рабочей папки. Он получает props и может вызывать backend action. [Custom Element API](https://docs.chainlit.io/api-reference/elements/custom).

Полный custom frontend пока не даёт соразмерной пользы: он потребует заново реализовать streaming, history, resume, steps, elements, Ask flow и совместимость с обновлениями Chainlit.

## Целевая спецификация UI для этого проекта

### Жизненный цикл Turn

1. Сразу создать финальный пустой `Message` и корневой `Step(name="Выполнение", type="run")`.
2. На `tool_started` создать дочерний Step, показать spinner и безопасный вход.
3. Потоковый stdout показывать только для достаточно долгой команды; короткий output установить одним `update()`.
4. На success завершить шаг и свернуть его; на failure отметить `is_error=True`, оставить раскрытым и показать exit code/stderr.
5. Финальные LLM tokens направлять в основной `Message`.
6. После pull из sandbox вычислить изменённые файлы, прикрепить их и перечислить имена в тексте.
7. Только после terminal event и успешной записи Revision считать Turn завершённым.
8. На Stop отменить agent task, завершить активные Steps как cancelled и не изображать частичный ответ успешным.

### Редактирование сообщения и replay

Это обязательная часть согласованности UI с состоянием агента. Когда пользователь редактирует старое сообщение:

1. восстановить agent checkpoint и snapshot песочницы перед исходным Turn;
2. архивировать/удалить прежний assistant Message и **все descendant Steps и elements** исходного и более поздних Turns;
3. запустить отредактированный Turn заново;
4. создать новое дерево Steps и новый streaming result;
5. атомарно заменить ветку runtime history и зафиксировать новую Revision;
6. при ошибке восстановить прежние checkpoint, sandbox snapshot и видимую ветку Chainlit.

Если обновить только текст пользовательского сообщения, старые tool steps будут описывать уже несуществовавшие действия, а файлы и checkpoint разойдутся с визуальной историей. Retry использует ту же семантику branching/replay, а не просто добавляет второй ответ.

Локальная Chainlit SQLite-схема уже содержит `parentId`, `showInput`, `modes`, `defaultOpen` и `autoCollapse`, поэтому для хранения обычных Steps новая миграция не нужна. Но логику архивирования всей ветки нужно проверить интеграционным тестом.

### Политика видимости и безопасности

- По умолчанию: `cot = "tool_call"`.
- Не показывать raw chain-of-thought и скрытые system prompts.
- Всегда очищать API keys, authorization headers, environment и похожие секреты.
- Ограничивать размер input, stdout и stderr; сохранять полный большой результат как downloadable artifact только если это безопасно.
- Показывать относительные sandbox paths, не внутренние host paths.
- Ошибки формулировать как факт операции и безопасный stderr, без дампа всего process environment.
- Один активный Step обновлять на месте; не создавать «кардиограмму» из статусных сообщений.

## Приоритетный план внедрения

### P0 — основной UX

1. Зафиксировать контракт agent events и terminal states.
2. Сделать короткий spike с `LangchainCallbackHandler`, записать реальные имена runs Deep Agents и настроить фильтр шума.
3. Перевести финальный ответ с `ainvoke → string` на явный streaming.
4. Реализовать lifecycle tool Steps, ошибки и cancellation через `on_stop`.
5. Связать replay после edit с удалением/архивированием старого дерева Steps и elements.

### P1 — качество и артефакты

1. Добавить labels/icons, syntax highlighting, redaction и truncation.
2. Прикреплять File/Image/Dataframe по MIME и явно перечислять созданные файлы.
3. Синхронизировать `TaskList` только с явным агентским планом.
4. Переключить основной UI с `cot="full"` на `cot="tool_call"`.

### P2 — дополнительные возможности

1. Настройка «Кратко / подробно» для tool outputs.
2. `AskActionMessage` для потенциально опасных действий, если политика песочницы изменится.
3. Один `CustomElement` для diff/workspace manifest после подтверждения потребности.
4. Starters и follow-up actions для частых сценариев.

## Критерии приёмки будущей реализации

- Первый визуальный status появляется до первого долгого tool call.
- Финальный ответ поступает токенами и остаётся одним Message.
- Для shell/Python видны команда, running state, exit code и ограниченные stdout/stderr.
- Успешные шаги сворачиваются; ошибочный остаётся раскрытым.
- Служебные `ChannelRead/Write`, `RunnableLambda` и checkpoint events не видны.
- Stop действительно отменяет работу и оставляет корректный terminal state.
- После reload/resume дерево Steps, элементы и результат восстанавливаются из SQLite.
- После редактирования старого сообщения нет прежних descendant Steps, ответов или файлов; агент стартует с правильного checkpoint/snapshot.
- Ни Step, ни Message не раскрывает ключи, environment или абсолютные host paths.
- Созданные файлы доступны из истории и визуально связаны с породившим их Turn.

## Итоговая рекомендация

Начать со встроенного `LangchainCallbackHandler` как диагностического прототипа, но архитектуру целевого решения строить как `Deep Agents/LangGraph events → project event contract → Chainlit adapter`. Это перенимает лучшее из официального Data Analyst case и Langroid: быстрый, живой интерфейс без custom frontend, стабильное дерево наблюдаемых действий и контроль над безопасностью, persistence и replay после редактирования сообщения.
