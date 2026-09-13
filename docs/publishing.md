# Выпуск LocalChat в PyPI

Пакет называется `local-agent-chat`, команда приложения — `localchat`. Версия задаётся в `pyproject.toml`; Git-тег выпуска должен точно совпадать с ней: `v0.1.0` для версии `0.1.0`. Проверяются Python 3.12 и 3.13 на Linux. Учётная запись PyPI и настройка доверенного издателя принадлежат владельцу проекта; GitHub-доступ сам по себе не предоставляет доступ к PyPI.

## Однократная настройка владельцем

1. Создайте отдельные аккаунты на [PyPI](https://pypi.org/account/register/) и [TestPyPI](https://test.pypi.org/account/register/), подтвердите email и включите [двухфакторную аутентификацию](https://pypi.org/help/#twofa). Сохраните recovery codes у себя.
2. В настройках GitHub-репозитория создайте environments `testpypi` и `pypi`, если их ещё нет.
3. На каждом индексе откройте Account settings → Publishing и добавьте pending publisher для нового проекта. Если проект уже принадлежит вам, добавьте publisher в настройках существующего проекта. Имя должно быть доступно для первой публикации.

| Поле | TestPyPI | PyPI |
|---|---|---|
| PyPI Project Name | `local-agent-chat` | `local-agent-chat` |
| Owner | `dev-sergeev` | `dev-sergeev` |
| Repository name | `local-agent-chat` | `local-agent-chat` |
| Workflow name | `publish.yml` | `publish.yml` |
| Environment name | `testpypi` | `pypi` |

Это имя **файла** workflow, а не его отображаемое название. При публикации из форка укажите владельца и репозиторий форка. Для другого имени пакета обновите также ссылки и команды workflow.

Trusted Publishing использует краткоживущую OIDC-авторизацию GitHub; пароль и постоянный API-токен в GitHub Secrets не нужны. [Создание проекта через Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/), [настройка публикации](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## Проверка перед выпуском

```bash
python -m pip install -e '.[test]' build twine
pytest -q
ruff check .
ruff format --check .
python -m build
python -m twine check dist/*
python scripts/check_distribution.py dist/*.whl
```

Собирайте выпуск из чистого checkout, чтобы `dist/` содержал только одну версию. `build` создаёт sdist, а wheel собирает из него: так проверяется и полнота исходного архива. `check_distribution.py` создаёт отдельный venv, устанавливает wheel с обычными зависимостями, запускает приложение вне checkout и проверяет загрузку UI через прокси, вложение и инструмент, пять запросов, правки третьего и первого запросов, SQLite, переустановку и возобновление чата. Локальная модель-имитатор не требует ключей и сетевых LLM-запросов. Скрипт печатает путь к логам и JSON-результату.

Основной CI выполняет эти проверки на обеих версиях Python. Артефакт `localchat-dist` содержит дистрибутивы, `distribution-check-*` — логи проверки установки. Пакет не включает `.env` пользователя, базы, вложения или журналы работы.

## Выпуск

1. Обновите версию и описание изменений, отправьте коммит в GitHub и дождитесь успешного CI.
2. Создайте и отправьте соответствующий тег, например `git tag v0.1.0` и `git push origin v0.1.0`.
3. Для предварительного выпуска используйте версию вида `0.1.0rc1`, тег `v0.1.0rc1` и запустите Actions → **Publish package** → Run workflow на этом теге с `repository=testpypi`. Номера предварительных выпусков тоже должны быть уникальны.
4. Для стабильной версии создайте GitHub Release на новом соответствующем теге. Публикация Release запускает workflow автоматически. Другой вариант — ручной запуск на теге с `repository=pypi`.

Workflow сначала повторяет CI, затем публикует в TestPyPI, скачивает оттуда wheel, сверяет SHA-256 с проверенным артефактом и повторно проверяет установленное приложение. Только после успеха те же дистрибутивы отправляются в PyPI. GitHub prerelease и ручной `repository=testpypi` останавливаются после TestPyPI.

Индексы не позволяют повторно использовать уже загруженные имена файлов. Для нового содержимого увеличьте версию. Если TestPyPI уже успешно пройден, а финальная публикация упала из-за настройки PyPI, исправьте publisher и повторите **только failed jobs** того же запуска: так используется первоначальный артефакт. Не запускайте заново весь выпуск с занятым номером.

## Проверка опубликованной версии пользователем

```bash
pipx install --python python3.12 local-agent-chat
localchat --version
localchat init
localchat run
```

Для ручной проверки предварительной версии скачайте только пакет из TestPyPI (`pip download --no-deps --only-binary=:all: --index-url https://test.pypi.org/simple/ local-agent-chat==0.1.0rc1`), затем установите скачанный wheel через pipx. Его зависимости будут загружены из обычного PyPI, поэтому не требуется смешивать оба индекса через `--extra-index-url`.
