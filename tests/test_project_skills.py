import logging
import re
from pathlib import Path

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.middleware import SkillsMiddleware
from deepagents.middleware.skills import SkillMetadata


def _load_skills(skills_dir: Path) -> dict[str, SkillMetadata]:
    middleware = SkillsMiddleware(
        backend=FilesystemBackend(root_dir="/", virtual_mode=False),
        sources=[skills_dir.resolve().as_posix()],
    )
    update = middleware.before_agent({}, None, {})
    return {skill["name"]: skill for skill in (update or {})["skills_metadata"]}


def test_deep_agents_loads_a_skill_package(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "incident-summary"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\n"
        "name: incident-summary\n"
        "description: Creates incident summaries when the user requests a postmortem.\n"
        "---\n\n"
        "# Incident summary\n",
        encoding="utf-8",
    )

    loaded = _load_skills(skills_dir)

    assert loaded["incident-summary"]["path"] == skill_file.as_posix()
    assert "postmortem" in loaded["incident-summary"]["description"]


def test_every_project_skill_is_discoverable_by_deep_agents(
    caplog: pytest.LogCaptureFixture,
) -> None:
    skills_dir = Path(__file__).resolve().parents[1] / "skills"
    skill_directories = sorted(path for path in skills_dir.iterdir() if path.is_dir())

    for directory in skill_directories:
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", directory.name), (
            f"Skill directory must use ASCII kebab-case: {directory}"
        )

    with caplog.at_level(logging.WARNING, logger="deepagents.middleware.skills"):
        loaded = _load_skills(skills_dir)

    assert set(loaded) == {directory.name for directory in skill_directories}
    for name, metadata in loaded.items():
        assert metadata["description"].strip()
        assert len(metadata["description"]) <= 1024
        assert Path(metadata["path"]) == skills_dir / name / "SKILL.md"
    assert [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ] == []
