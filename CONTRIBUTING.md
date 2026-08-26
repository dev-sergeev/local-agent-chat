# Contributing

## Локальный запуск

```bash
python -m pip install -e '.[test]'
cp models.example.yaml models.yaml
cp .env.example .env
```

Не используйте реальные секреты в тестах. Для проверок Host Files создавайте данные через `tmp_path`: тест не должен читать пользовательские или системные файлы только потому, что они доступны процессу.

## Перед PR

```bash
pytest -q
ruff check .
ruff format --check .
python -m compileall -q local_agent_chat app.py
bash -n scripts/run.sh
```

- Добавьте тест для изменённого поведения.
- Сохраняйте единый Agent-файловый интерфейс: `ls`, `read_file`, `glob`, `grep` в обоих режимах; Project Skill не может добавить запись или исполнение.
- Обновите README или `docs/architecture.md`, если изменился публичный flow.
- Для нового архитектурного решения добавьте короткий ADR; термины меняйте через `CONTEXT.md`.
- Не коммитьте `.env`, `models.yaml`, `.local-agent-chat/`, SQLite, скриншоты и логи.

Карта файлов и точек настройки есть в README.
