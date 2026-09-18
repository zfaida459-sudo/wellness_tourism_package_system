import json
from pathlib import Path

import pytest
import yaml

from scripts.deploy import GitHub, RUNTIME_FILES, prepare_publication


def test_reject_smoke_or_failed_release(tmp_path):
    for filename in ["registry.json", "report.json"]:
        (tmp_path / filename).write_text(json.dumps({"smoke": True, "passed_gate": True}))
    with pytest.raises(ValueError, match="full run"):
        prepare_publication(tmp_path, "owner/project")


def test_stale_branch_is_not_written(monkeypatch):
    client = GitHub("owner/project", "test-token-not-real")
    monkeypatch.setattr(client, "head", lambda branch: "newer-revision")
    with pytest.raises(RuntimeError, match="stale"):
        client.commit_files("main", {"data/manifest.json": "{}"}, "older-revision")


def test_runtime_allowlist_excludes_data_and_training():
    assert all(not path.startswith(("data/", "artifacts/")) for path in RUNTIME_FILES)
    assert "wellness_tourism/train.py" not in RUNTIME_FILES
    assert all(Path(path).exists() for path in RUNTIME_FILES)


def test_workflow_has_gated_publish_and_no_force_push():
    source = Path(".github/workflows/pipeline.yml").read_text()
    workflow = yaml.safe_load(source)
    assert workflow["jobs"]["train"]["needs"] == "checks"
    assert workflow["permissions"]["contents"] == "read"
    assert "--force" not in source
    assert "--publish" in source