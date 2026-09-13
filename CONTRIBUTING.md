# Contributing

## Локальный запуск

```bash
python -m pip install -e '.[test]' build twine
localchat init --config-dir . --data-dir .local-agent-chat
localchat run --config-dir .
```

Не используйте реальные секреты в тестах. Проверяйте запрет выхода из песочницы на файлах, созданных через `tmp_path`; тест не должен читать настоящие системные или пользовательские файлы.

## Перед PR

```bash
pytest -q
ruff check .
ruff format --check .
python -m compileall -q local_agent_chat
bash -n scripts/run.sh
python -m build
python -m twine check dist/*
python scripts/check_distribution.py dist/*.whl
```

- Добавьте тест для изменённого поведения.
- Сохраняйте ровно четыре инструмента чтения песочницы: `ls`, `read_file`, `glob`, `grep`. Проверяйте суммаризацию, перезапуск и Revision через реальный `create_agent` с управляемой тестовой моделью.
- Обновите README или `docs/architecture.md`, если изменился публичный flow.
- Для нового архитектурного решения добавьте короткий ADR; термины меняйте через `CONTEXT.md`.
- Не коммитьте `.env`, `models.yaml`, `.local-agent-chat/`, SQLite, скриншоты и логи.

Карта файлов и точек настройки есть в README.

Проверка дистрибутива создаёт чистое окружение вне checkout, устанавливает только runtime-зависимости и проверяет UI, загрузку файла, вызов инструмента, правку исторического запроса и восстановление после переустановки через локальную тестовую модель. Ресурсы UI находятся в `local_agent_chat/assets/`; CLI копирует их в отдельный временный каталог при запуске.
