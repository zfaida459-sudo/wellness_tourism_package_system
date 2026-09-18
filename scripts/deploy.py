"""Preview or publish a verified GitHub Release and Community Cloud snapshot."""

import argparse
import copy
import json
import os
from pathlib import Path
import re

import requests

from wellness_tourism.data import sha256
from wellness_tourism.predict import load_registered_model

RUNTIME_FILES = [
    "app.py", "requirements.txt", ".streamlit/config.toml", "wellness_tourism/__init__.py",
    "wellness_tourism/schema.py", "wellness_tourism/predict.py",
]
MAIN_OUTPUTS = [
    "data/train.csv", "data/test.csv", "data/manifest.json", "data/splits.json",
    "data_analysis.ipynb",
]


def prepare_publication(run_dir, repository):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
        raise ValueError("Repository must have the form owner/name.")
    run_dir = Path(run_dir).resolve()
    report = json.loads((run_dir / "report.json").read_text())
    registry = json.loads((run_dir / "registry.json").read_text())
    if report["smoke"] or registry["smoke"] or not report["passed_gate"] or not registry["passed_gate"]:
        raise ValueError("Only a full run passing both stage gates may be published.")
    registry = copy.deepcopy(registry)
    registry["repository"] = repository
    assets = []
    for stage, entry in registry["models"].items():
        path = run_dir / entry["asset"]
        if path.parent != run_dir or sha256(path) != entry["sha256"]:
            raise ValueError("Release model checksum mismatch.")
        local_registry = copy.deepcopy(registry)
        local_registry["models"][stage]["path"] = str(path)
        load_registered_model(local_registry, stage)
        entry.pop("path", None)
        entry["url"] = f"https://github.com/{repository}/releases/download/{registry['release']}/{path.name}"
        assets.append(path)
    assets.extend(sorted(run_dir.glob("*.csv")))
    assets.extend(sorted(run_dir.glob("*.png")))
    assets.extend(run_dir / name for name in ["report.json", "manifest.json", "splits.json"])
    registry_text = json.dumps(registry, indent=2) + "\n"
    snapshot = {name: Path(name).read_text(encoding="utf-8") for name in RUNTIME_FILES}
    snapshot["models/registry.json"] = registry_text
    snapshot["README.md"] = (
        "# Visit with Us\n\nValidated Community Cloud deployment.\n\n"
        f"Source: https://github.com/{repository}\n\nModel release: {registry['release']}\n"
    )
    snapshot[".deployment.json"] = json.dumps({"managed_by": "wellness-tourism", "release": registry["release"]})
    main_files = {name: Path(name).read_text(encoding="utf-8") for name in MAIN_OUTPUTS}
    main_files["models/registry.json"] = registry_text
    main_files["reports/latest.json"] = json.dumps(report, indent=2) + "\n"
    return registry, report, assets, snapshot, main_files


class GitHub:
    def __init__(self, repository, token):
        self.repository = repository
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}",
                                     "Accept": "application/vnd.github+json",
                                     "X-GitHub-Api-Version": "2022-11-28"})

    def request(self, method, endpoint, allow_missing=False, **kwargs):
        base_url = f"https://api.github.com/repos/{self.repository}"
        url = f"{base_url}/{endpoint.lstrip('/')}" if endpoint else base_url
        
        response = self.session.request(method, url, timeout=(10, 120), **kwargs)
        if allow_missing and response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json() if response.content else None

    def head(self, branch):
        reference = self.request("GET", f"git/ref/heads/{branch}", allow_missing=True)
        return reference["object"]["sha"] if reference else None

    def commit_files(self, branch, files, expected_head, clean_tree=False):
        if self.head(branch) != expected_head:
            raise RuntimeError(f"{branch} changed; refusing stale publication.")
        parent = self.request("GET", f"git/commits/{expected_head}") if expected_head else None
        tree_input = {"tree": [{"path": path, "mode": "100644", "type": "blob", "content": content}
                               for path, content in sorted(files.items())]}
        if parent and not clean_tree:
            tree_input["base_tree"] = parent["tree"]["sha"]
        tree = self.request("POST", "git/trees", json=tree_input)
        if parent and parent["tree"]["sha"] == tree["sha"]:
            return expected_head
        commit = self.request("POST", "git/commits", json={
            "message": "Publish validated wellness tourism release",
            "tree": tree["sha"], "parents": [expected_head] if expected_head else [],
            "author": {"name": "github-actions[bot]", "email": "41898282+github-actions[bot]@users.noreply.github.com"},
        })
        if expected_head:
            self.request("PATCH", f"git/refs/heads/{branch}", json={"sha": commit["sha"], "force": False})
        else:
            self.request("POST", "git/refs", json={"ref": f"refs/heads/{branch}", "sha": commit["sha"]})
        return commit["sha"]


def publish(run_dir, repository, source_sha):
    registry, report, assets, snapshot, main_files = prepare_publication(run_dir, repository)
    if report["source_revision"] != source_sha:
        raise ValueError("Training source revision differs from the publication source.")
    api = GitHub(repository, os.environ["GH_TOKEN"])
    metadata = api.request("GET", "")
    if metadata["private"] or metadata["default_branch"] != "main":
        raise ValueError("This approved workflow requires a public repository with main as default branch.")
    if api.head("main") != source_sha:
        raise RuntimeError("main advanced after training; rerun the workflow on the new revision.")
    deployment_head = api.head("streamlit")
    if deployment_head:
        marker = api.request("GET", "contents/.deployment.json?ref=streamlit", allow_missing=True)
        if marker is None:
            raise ValueError("Existing streamlit branch is not managed by this project; refusing to replace it.")
    release = api.request("POST", "releases", json={
        "tag_name": registry["release"], "target_commitish": source_sha,
        "name": registry["release"], "draft": True,
        "body": "Validated pre-contact and post-interaction classifiers. See report.json and CV results for metrics and limitations.",
    })
    try:
        upload_url = release["upload_url"].split("{")[0]
        for path in assets:
            with path.open("rb") as stream:
                response = api.session.post(upload_url, params={"name": path.name}, data=stream,
                                            headers={"Content-Type": "application/octet-stream"}, timeout=(10, 180))
            response.raise_for_status()
        response = api.session.post(upload_url, params={"name": "registry.json"},
                                    data=main_files["models/registry.json"].encode(),
                                    headers={"Content-Type": "application/json"}, timeout=(10, 60))
        response.raise_for_status()
        api.request("PATCH", f"releases/{release['id']}", json={"draft": False})
        # Check public download availability before advancing the deployment pointer.
        for entry in registry["models"].values():
            with requests.get(entry["url"], stream=True, timeout=(10, 120)) as response:
                response.raise_for_status()
                import hashlib

                digest = hashlib.sha256()
                for block in response.iter_content(1024 * 1024):
                    digest.update(block)
                if digest.hexdigest() != entry["sha256"]:
                    raise ValueError("Public release checksum verification failed.")
        api.commit_files("main", main_files, source_sha)
        deployed_sha = api.commit_files("streamlit", snapshot, deployment_head, clean_tree=True)
        print(f"Published {release['html_url']}\nDeployment branch revision: {deployed_sha}")
    except Exception:
        print("Publication stopped. Do not advance the app manually; inspect the draft/release and rerun from current main.")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--run-dir")
    parser.add_argument("--source-sha", default=os.environ.get("GITHUB_SHA"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.repo:
        parser.error("Provide --repo owner/name or GITHUB_REPOSITORY.")
    run_dir = arguments.run_dir or json.loads(Path("artifacts/latest.json").read_text())["run_dir"]
    if arguments.publish:
        if not arguments.source_sha:
            parser.error("--source-sha is required for publishing.")
        publish(run_dir, arguments.repo, arguments.source_sha)
    else:
        registry, _, assets, snapshot, main_files = prepare_publication(run_dir, arguments.repo)
        print(json.dumps({"dry_run": True, "repository": arguments.repo, "release": registry["release"],
                          "assets": [path.name for path in assets], "main_updates": list(main_files),
                          "streamlit_files": list(snapshot)}, indent=2))


if __name__ == "__main__":
    main()