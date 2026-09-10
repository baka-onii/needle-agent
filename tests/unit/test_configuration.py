"""One set of editable defaults, strict loading, portable export, and prompt contracts."""

import json
from dataclasses import replace

import pytest

from agent_runtime import AgentConfig
from agent_runtime.config import export_config, init_config, load_config, parse_config
from agent_runtime.models.reasoning import build_system_prompt, build_translator_prompt
from agent_runtime.protocol.intent import check_atomic_action, literal_write_content, write_action
from agent_runtime.protocol.parser import parse_response
from agent_runtime.tools.registry import create_default_registry


def test_init_loads_relative_workspace_and_prompt_files(tmp_path, monkeypatch):
    destination = init_config(tmp_path / "settings" / "needle.toml")
    prompt = destination.parent / "prompts" / "reasoning.md"
    prompt.write_text("Use concise, careful reasoning.")
    monkeypatch.chdir(tmp_path)
    config = load_config(destination, environ={})
    assert config.workspace_root == str(destination.parent)
    assert config.reasoning_prompt == prompt.read_text()
    assert config.llm_max_tokens == 4096 and config.needle_max_tokens == 256
    assert config.confirmation_prompt and config.translator_prompt
    with pytest.raises(ValueError, match="nothing was overwritten"):
        init_config(destination)
    assert prompt.read_text() == "Use concise, careful reasoning."


def test_config_environment_explicit_override_precedence(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('[models]\nllm_model = "from-file"\n[runtime]\nmax_tool_steps = 7\n')
    env = {
        "NEEDLE_CONFIG": str(path),
        "NEEDLE_LLM_MODEL": "from-env",
        "NEEDLE_LLM_API_KEY": "secret",
    }
    assert load_config(environ=env).llm_model == "from-env"
    config = load_config(overrides={"llm_model": "from-cli"}, environ=env)
    assert config.llm_model == "from-cli" and config.max_tool_steps == 7
    assert "secret" not in repr(config) and "secret" not in export_config(config)


def test_workspace_configuration_is_never_implicitly_trusted(tmp_path, monkeypatch):
    (tmp_path / "needle.toml").write_text('[models]\nllm_model = "untrusted"\n')
    monkeypatch.chdir(tmp_path)
    assert load_config(environ={}).llm_model == AgentConfig().llm_model


def test_portable_toml_and_json_round_trip(tmp_path):
    config = replace(
        AgentConfig(), reasoning_prompt='Custom "prompt"\nनमस्ते', llm_api_key="private", max_stalls=5
    )
    document = export_config(config, include_server=False)
    values = parse_config(document)
    assert "private" not in document and "workspace_root" not in document
    assert AgentConfig(**values).reasoning_prompt == config.reasoning_prompt
    assert values["max_stalls"] == 5
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"models": {"mode": "demo"}, "runtime": {"max_tool_steps": 4}}))
    assert load_config(path, environ={}).mode == "demo"


@pytest.mark.parametrize(
    "document",
    [
        "[runtime]\nmax_stallz = 3",
        '[secret]\nkey = "no"',
        '[prompts]\nreasoning_prompt = "inline"\nreasoning_prompt_file = "x.md"',
        '[prompts]\nreasoning_prompt_file = "/etc/passwd"',
    ],
)
def test_strict_keys_and_browser_import_cannot_read_files(document):
    with pytest.raises(ValueError):
        parse_config(document)


@pytest.mark.parametrize(
    "values",
    [
        {"read_only": "false"},
        {"max_stalls": True},
        {"needle_max_tokens": 0},
        {"llm_temperature": float("nan")},
        {"confirmation_prompt": " "},
        {"reasoning_prompt": "\x00"},
        {"default_timezone": "not/a/timezone"},
        {"mode": "unknown"},
    ],
)
def test_invalid_settings_fail_at_load(values):
    with pytest.raises(ValueError):
        load_config(overrides=values, environ={})


def test_both_models_get_external_instructions_and_workspace_data(tmp_path):
    (tmp_path / "src").mkdir()
    config = AgentConfig(
        workspace_root=str(tmp_path),
        reasoning_prompt="CUSTOM REASON",
        translator_prompt="CUSTOM TRANSLATE",
    )
    tools = create_default_registry(config).list()
    reason = build_system_prompt(tools, config)
    translate = build_translator_prompt(config)
    assert "CUSTOM REASON" in reason and "CUSTOM TRANSLATE" in translate
    for prompt in (reason, translate):
        assert str(tmp_path) in prompt and '"src/"' in prompt
        assert '"."' in prompt
    assert "write_file" in reason


def test_snapshot_is_bounded_optional_and_does_not_follow_symlinks(tmp_path):
    (tmp_path / "outside").symlink_to("/etc", target_is_directory=True)
    for i in range(30):
        (tmp_path / (str(i) + "x" * 30)).touch()
    config = AgentConfig(
        workspace_root=str(tmp_path), workspace_listing_chars=128, max_directory_entries=10
    )
    prompt = build_translator_prompt(config)
    assert '"outside/"' not in prompt and "limit reached" in prompt
    disabled = build_translator_prompt(replace(config, include_workspace_listing=False))
    assert "snapshot disabled" in disabled


def test_fenced_payload_may_close_next_to_outer_tag():
    action = write_action("test.txt", "<final>literal data</final>").rstrip()
    parsed = parse_response(f"<tool>{action}</tool>")
    assert parsed.final_answer is None
    assert literal_write_content(parsed.actions[0].instruction) == "<final>literal data</final>"
    check_atomic_action(write_action("a.txt", "Read x and then write y"))


def test_approval_tier_defaults_and_validation(tmp_path):
    from agent_runtime.config import load_config

    config = AgentConfig()
    assert "delete_file" in config.require_approval_for
    assert "run_python" in config.require_approval_for
    assert "write_file" not in config.require_approval_for
    assert "read_file" not in config.require_approval_for
    path = tmp_path / "needle.toml"
    path.write_text('[safety]\nrequire_approval_for = ["delete_file"]\n')
    assert load_config(path, environ={}).require_approval_for == ("delete_file",)
    path.write_text('[safety]\nrequire_approval_for = [""]\n')
    with pytest.raises(ValueError, match="require_approval_for"):
        load_config(path, environ={})


def test_reasoning_prompt_documents_payload_contracts():
    from agent_runtime.models.reasoning import build_system_prompt

    tools = create_default_registry(AgentConfig()).list()
    prompt = build_system_prompt(tools)
    assert "payload: one <content> block fills 'content'" in prompt
    assert "<text-1> fills 'old_text', <text-2> fills 'new_text'" in prompt


def test_reasoning_prompt_guides_temp_file_staging():
    prompt = AgentConfig().reasoning_prompt
    assert "temp file" in prompt.casefold()
    assert "scratch/" in prompt
