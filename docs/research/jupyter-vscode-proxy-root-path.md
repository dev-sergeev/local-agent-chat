# DELETE с JSON body через JupyterHub → VS Code proxy → Chainlit

Дата исследования: 2026-08-25.

## Краткий вывод

`root_path` настроен для другой задачи и не является причиной HTTP 500. Он сообщает Chainlit, под каким внешним URL-префиксом живут HTML, API и Socket.IO. Он не влияет на допустимые сочетания HTTP-метода и тела запроса внутри промежуточного прокси.

Точная причина наблюдаемого поведения находится в `jupyter-server-proxy 4.5.0`, через который `jupyter_vscode_proxy 0.7` публикует code-server под `/vscode`:

1. Chainlit штатно удаляет диалог запросом `DELETE /project/thread` с JSON-телом `{"threadId": ...}`. Это видно и в [клиенте Chainlit](https://github.com/Chainlit/chainlit/blob/f5f3fa8664a5ff9a50148919004e9317dd446d5e/libs/react-client/src/api/index.tsx#L115-L168), и в вызове [deleteThread](https://github.com/Chainlit/chainlit/blob/f5f3fa8664a5ff9a50148919004e9317dd446d5e/libs/react-client/src/api/index.tsx#L240-L244); серверный endpoint также ожидает модель запроса в теле ([Chainlit 2.11.1](https://github.com/Chainlit/chainlit/blob/dd14df53709c6c1389faa82b406c8bfa8e9b61bc/backend/chainlit/server.py#L1225-L1246)).
2. `jupyter-server-proxy` объявляет обработчики `GET`, `POST`, `PUT`, `DELETE`, `HEAD`, `PATCH` и `OPTIONS`, то есть проблема не в отсутствии метода `delete` ([handlers.py 4.5.0](https://github.com/jupyterhub/jupyter-server-proxy/blob/068b9277db5f6bf26b493feca5fcdaccf63b55da/jupyter_server_proxy/handlers.py#L732-L768)).
3. Прокси берёт исходные `method` и `body`, но создаёт новый Tornado `HTTPRequest` без `allow_nonstandard_methods=True` ([построение запроса](https://github.com/jupyterhub/jupyter-server-proxy/blob/068b9277db5f6bf26b493feca5fcdaccf63b55da/jupyter_server_proxy/handlers.py#L327-L349), [чтение body](https://github.com/jupyterhub/jupyter-server-proxy/blob/068b9277db5f6bf26b493feca5fcdaccf63b55da/jupyter_server_proxy/handlers.py#L357-L420)).
4. Tornado по умолчанию считает тело ожидаемым только у `POST`, `PATCH` и `PUT`. Если у `DELETE` есть тело, `SimpleAsyncHTTPClient` выбрасывает `ValueError: Body must be None for method DELETE (unless allow_nonstandard_methods is true)` ([официальный исходный код Tornado](https://github.com/tornadoweb/tornado/blob/master/tornado/simple_httpclient.py#L403-L419)). Это согласуется с уже зафиксированным разработчиками `jupyter-server-proxy` классом проблем: их issue требует проксировать запросы независимо от содержимого и прямо называет `allow_nonstandard_methods` одним из решений ([issue #329](https://github.com/jupyterhub/jupyter-server-proxy/issues/329)).
5. В buffered path `jupyter-server-proxy` ловит только `HTTPError`, а не этот `ValueError` ([proxy fetch](https://github.com/jupyterhub/jupyter-server-proxy/blob/068b9277db5f6bf26b493feca5fcdaccf63b55da/jupyter_server_proxy/handlers.py#L504-L528)); стандартный Tornado handler превращает необработанное исключение в HTTP 500 ([официальный код Tornado](https://github.com/tornadoweb/tornado/blob/a55abe3e3bf59994f29b2f7084c46341f0d4f6a7/tornado/web.py#L1928-L1948)).
6. `GET` не имеет тела и проходит эту проверку. Поэтому успешный `GET /user/admin/vscode/proxy/8765/...` и HTTP 500 у `DELETE` с JSON body — ожидаемая асимметрия именно этого дефекта.

Локально установлены `jupyter-server-proxy 4.5.0`, `jupyter_vscode_proxy 0.7`, `Tornado 6.5.8` и `Chainlit 2.11.1`. Минимальное локальное воспроизведение на этих пакетах дало ровно указанный `ValueError` при `AsyncHTTPClient.fetch(HTTPRequest(..., method="DELETE", body=b"{}"))` до обращения к upstream.

## Фактическая цепочка запроса

```text
браузер
  → JupyterHub configurable-http-proxy: /user/admin/...
  → Jupyter Server пользователя
  → jupyter-server-proxy named server: /vscode/...
  → code-server: /proxy/8765/...
  → Chainlit: /project/thread
```

`jupyter_vscode_proxy` не является самостоятельным HTTP-прокси. Он регистрирует `code-server` как named server `vscode` для `jupyter-server-proxy`; его конфигурация запускает `code-server --auth none ...` и возвращает обычный server-process config ([исходный код версии 0.7](https://github.com/betatim/vscode-binder/blob/25d70ec4ca208d01a6bcbfd616de1b2b2c7d431c/jupyter_vscode_proxy/__init__.py#L74-L94)). Следовательно, HTTP 500 с оформлением Jupyter Server и отсутствие запроса в логах Chainlit однозначно указывают на первую из двух внутренних прокси-ступеней.

Вторая ступень, code-server, устроена иначе: и `/proxy/:port`, и `/absproxy/:port` зарегистрированы через `router.all`, то есть принимают все обычные HTTP-методы ([официальный route source](https://github.com/coder/code-server/blob/main/src/node/routes/index.ts#L95-L117)); затем запрос передаётся в `http-proxy`. По имеющимся данным code-server не является точкой текущего сбоя.

## Правильное исправление для всех HTTP-методов

Для сохранения текущего публичного URL минимальное и общее исправление — создавать исходящий запрос `jupyter-server-proxy` с `allow_nonstandard_methods=True`:

```python
req = httpclient.HTTPRequest(
    client_uri,
    method=self.request.method,
    body=body,
    allow_nonstandard_methods=True,
    # остальные существующие параметры без изменений
)
```

Это следует внести в `_build_proxy_request`, либо эквивалентно вернуть параметр из переопределённого `proxy_request_options()`. На практике безопаснее распространять небольшой pinned fork/wheel `jupyter-server-proxy 4.5.0` и тест, чем изменять файл внутри `site-packages` вручную при каждом запуске. Патч должен быть отправлен upstream; простое обновление до текущей `4.5.0` проблему не решает — в этой версии флага нет.

Почему это правильнее частного обхода:

- сохраняются метод, JSON body и `Content-Type`, как их отправил Chainlit;
- исправляются все API, использующие допустимое, но нетипичное сочетание method/body, а не только удаление диалога;
- не требуется менять контракт Chainlit, который явно использует JSON body для `DELETE`;
- не смешиваются две независимые задачи: HTTP transport и внешний URL prefix.

Изменение Chainlit на `DELETE` без тела или подмена метода на `POST` — частный workaround, нарушающий его штатный клиент-серверный контракт. Настройка только `root_path`, `/proxy` против `/absproxy` или заголовка `X-Forwarded-Prefix` также не устранит исключение Tornado: оно возникает до отправки запроса следующему upstream.

## Применённая совместимость на уровне приложения

Этот репозиторий не управляет процессом Jupyter Server: исправление его установленного пакета потребовало бы изменения `site-packages` и перезапуска пользовательского сервера, который одновременно держит текущие Jupyter/VS Code-сессии. Поэтому для работающего экземпляра применён обратимый transport workaround, не меняющий контракт внутри Chainlit:

1. [`public/proxy-method-override.js`](../../public/proxy-method-override.js) загружается до основного frontend bundle и преобразует только same-origin `DELETE` с непустым body в `POST`, сохраняя URL, body, content type, credentials и прочие параметры. Исходный метод записывается в узкий заголовок `X-Proxy-Method-Override: DELETE`.
2. [`RestoreProxyMethodMiddleware`](../../local_agent_chat/proxy_prefix.py) принимает только сочетание `POST` + точное значение этого заголовка, удаляет служебный заголовок и восстанавливает `scope["method"] = "DELETE"` до входа в штатный router Chainlit.
3. Для Chainlit запрос остаётся обычным `DELETE /project/thread` с исходным JSON body; его серверный endpoint и data layer не изменялись. Для Jupyter Server Proxy запрос является допустимым `POST` с body и поэтому проходит Tornado.

Перехват обобщён на все same-origin `DELETE` с body, поэтому тем же способом покрыты штатные удаления feedback/element и disconnect MCP, а не только Thread. Если инфраструктурный патч с `allow_nonstandard_methods=True` будет установлен, этот слой можно оставить (он сохраняет серверный контракт) или удалить после повторного полного регрессионного прогона.

Проверка на реальном публичном пути дала развилочный результат: исходный `DELETE` с JSON body вернул Jupyter Server `500` и не удалил тестовый Thread; туннелированный запрос вернул Chainlit `200 {"success": true}`. На основном хранилище специально созданный acceptance Thread после такого запроса исчез из `threads`, `steps`, runtime history, active branches и Sandbox, а повторная загрузка `/project/threads` его не вернула.

## Роль `root_path` и выбор `/proxy`/`/absproxy`

Официальная документация Chainlit говорит использовать `--root-path /prefix`, когда приложение опубликовано на subpath ([deployment overview](https://docs.chainlit.io/deploy/overview#deploying-on-a-subpath), [CLI option](https://docs.chainlit.io/backend/command-line)). Реализация Chainlit монтирует Socket.IO под этим префиксом и строит `APIRouter(prefix=config.run.root_path)` ([server source](https://github.com/Chainlit/chainlit/blob/dd14df53709c6c1389faa82b406c8bfa8e9b61bc/backend/chainlit/server.py#L228-L263)). Поэтому для текущего внешнего адреса значение должно соответствовать всему видимому браузеру префиксу:

```text
/user/admin/vscode/proxy/8765
```

Но обе текущие прокси-ступени в режиме `/proxy` удаляют свой служебный префикс перед upstream:

- `jupyter-server-proxy` документирует, что `/proxy/<port>` переписывается с удалением префикса; `/proxy/absolute/<port>` отключает это поведение ([официальная документация](https://jupyter-server-proxy.readthedocs.io/en/latest/arbitrary-ports-hosts.html));
- code-server также удаляет `/proxy/<port>`, а `/absproxy/<port>` передаёт путь как есть ([официальное руководство](https://github.com/coder/code-server/blob/main/docs/guide.md#stripping-proxyport-from-the-request-path)).

Следствие: `root_path` нужен браузеру и Chainlit для генерации правильных внешних URL, но при strip-proxy Chainlit физически получает `/project/thread`. Поэтому либо приложение должно восстанавливать внешний префикс в ASGI `scope.path` (в этом репозитории это делает `RestoreProxyPrefixMiddleware`), либо нужно согласованно перейти на абсолютный proxy mode. Эти варианты решают маршрутизацию пути, но не дефект `DELETE` body.

Если использовать code-server `/absproxy`, его документация требует передать внешний base path через `--abs-proxy-base-path` при размещении самого code-server под префиксом ([guide](https://github.com/coder/code-server/blob/main/docs/guide.md#prefixing-absproxyport-with-a-path)). Тогда публичный URL и `APP_ROOT_PATH` изменятся на `/user/admin/vscode/absproxy/8765`. Это не является рекомендуемым исправлением текущего 500: запрос всё равно сначала проходит через `jupyter-server-proxy /vscode` и упадёт там без патча.

## Альтернативная архитектура без VS Code proxy

Если есть доступ к конфигурации самого Hub, Chainlit можно зарегистрировать как JupyterHub Service и направить публичный маршрут непосредственно на `http://127.0.0.1:8765`. JupyterHub добавляет service с `url` в основную proxy table под `/services/<name>` и передаёт приложению `JUPYTERHUB_SERVICE_PREFIX` ([официальная документация Services](https://jupyterhub.readthedocs.io/en/latest/reference/services.html)). ConfigurableHTTPProxy — обёртка над `node-http-proxy`; добавленный route проксирует запросы с публичного prefix на target ([официальный CHP README](https://github.com/jupyterhub/configurable-http-proxy)). Это устраняет обе вложенные ступени `/vscode/proxy`.

У такого варианта есть обязательное условие: основной CHP маршрутизирует, но приложение должно самостоятельно проверять JupyterHub OAuth/HubAuth либо находиться за другим проверенным auth middleware. Нельзя считать сам факт маршрута `/services/...` достаточной авторизацией и нельзя оставлять локальный `header_auth_callback`, безусловно принимающий любого посетителя. Для текущей задачи pinned patch `jupyter-server-proxy` сохраняет существующую границу аутентификации и потому имеет меньший радиус изменения.

## Пошаговая проверка после инфраструктурного патча

Проверять следует на реальном публичном URL, с браузерной cookie-сессией, не печатая cookie, токен или заголовок `Authorization` в логи.

1. **Прямой Chainlit.** На `127.0.0.1:8765` выполнить аутентифицированные `GET /project/settings`, создание/получение истории и `DELETE /project/thread` с тестовым `threadId`. Убедиться, что удаление возвращает 200 и запись исчезает.
2. **Через code-server, минуя Jupyter Server, если этот адрес доступен локально.** Повторить те же операции на `/proxy/8765`. Это отдельно доказывает вторую прокси-ступень.
3. **Развилочный тест первой ступени.** Отправить на тот же named-server URL `DELETE` без body к безопасному диагностическому endpoint. Если запрос достигает следующего upstream, а вариант с `{}` даёт 500, причина подтверждена независимо от Chainlit. Не использовать для API путь `/hub/user-redirect/...`: это GET-oriented redirect endpoint, и для `DELETE` у него давно зафиксирован отдельный 405 ([jupyter-server-proxy issue #149](https://github.com/jupyterhub/jupyter-server-proxy/issues/149)).
4. **Через полный публичный путь.** Повторить на `/user/admin/vscode/proxy/8765`. До патча ожидается 500 Jupyter Server; после патча запрос должен появиться в access log Chainlit и вернуть его ответ.
5. **Матрица методов.** Проверить `GET`/`HEAD` без body, `POST`/`PUT`/`PATCH` с JSON body, `DELETE` с JSON body, `OPTIONS`, multipart upload, Socket.IO WebSocket и SSE. Для каждого шага сравнить метод, path, query, `Content-Type`, длину body и status на входе и upstream, не логируя чувствительные значения.
6. **Приёмка интерфейса.** Создать тестовый диалог, обновить страницу и убедиться, что он есть в истории. Нажать удаление, подтвердить действие, дождаться исчезновения строки без error toast, затем обновить страницу и убедиться, что диалог не восстановился.
7. **Регрессии prefix.** Проверить загрузку assets, `/project/settings`, `/auth/header`, `/ws/socket.io`, вложения и переходы после refresh именно под полным префиксом. Отдельно проверить trailing slash у стартового URL.

## Что считать доказательством исправления

Одного успешного `GET` недостаточно. Исправление доказано только когда:

- полный публичный `DELETE` с непустым JSON body доходит до Chainlit;
- Chainlit отвечает 200, а запись действительно удалена из persistence;
- после reload удалённый диалог не появляется снова;
- остальные методы, WebSocket и загрузка файлов продолжают работать;
- в Jupyter Server больше нет `ValueError` про body/method;
- проверка выполнена через тот же `/user/<user>/vscode/proxy/<port>` URL, которым пользуется браузер.
