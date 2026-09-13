# Contributing

## Локальный запуск

```bash
python -m pip install -e '.[test]'
cp models.example.yaml models.yaml
cp .env.example .env
```

Не используйте реальные секреты в тестах. Проверяйте запрет выхода из песочницы на файлах, созданных через `tmp_path`; тест не должен читать настоящие системные или пользовательские файлы.

## Перед PR

```bash
pytest -q
ruff check .
ruff format --check .
python -m compileall -q local_agent_chat app.py
bash -n scripts/run.sh
```

- Добавьте тест для изменённого поведения.
- Сохраняйте ровно четыре инструмента чтения песочницы: `ls`, `read_file`, `glob`, `grep`. Проверяйте суммаризацию, перезапуск и Revision через реальный `create_agent` с управляемой тестовой моделью.
- Обновите README или `docs/architecture.md`, если изменился публичный flow.
- Для нового архитектурного решения добавьте короткий ADR; термины меняйте через `CONTEXT.md`.
- Не коммитьте `.env`, `models.yaml`, `.local-agent-chat/`, SQLite, скриншоты и логи.

Карта файлов и точек настройки есть в README.
