# Contributing

## Локальный запуск

```bash
python -m pip install -e '.[test]'
cp models.example.yaml models.yaml
cp .env.example .env
```

Не используйте реальные секреты в тестах. Shell-команды выполняются на хосте, поэтому разрабатывайте в отдельном контейнере/VM.

## Перед PR

```bash
pytest -q
ruff check .
ruff format --check .
python -m compileall -q local_agent_chat app.py
bash -n scripts/run.sh
```

- Добавьте тест для изменённого поведения.
- Обновите README или `docs/architecture.md`, если изменился публичный flow.
- Для нового архитектурного решения добавьте короткий ADR; термины меняйте через `CONTEXT.md`.
- Не коммитьте `.env`, `models.yaml`, `.local-agent-chat/`, SQLite, скриншоты и логи.

Карта файлов и точек настройки есть в README.
