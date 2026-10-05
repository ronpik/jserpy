"""scripts/bump_version.py prepares, validates and publishes releases without surprises."""

import argparse
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib", reason="the release script needs Python 3.11 or later")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bump_version", ROOT / "scripts/bump_version.py")
bump_version = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bump_version)
RF = bump_version.ReleaseFlow
REAL_VALIDATE = RF.validate
REAL_CHECK_TOOLS = RF.check_tools

PYPROJECT = """\
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "jserpy"
version = "0.1.3"
description = "A Python library"
requires-python = ">=3.10"
dependencies = [
    "typing-inspect>=0.9.0",
    "numpy>=1.13.0",    # First version to include numpy.generic
]

[tool.example]
version = "9.9.9"

[project.urls]
"Homepage" = "https://github.com/ronpik/jserpy\""""  # Like the real file, without a final newline.


def write_tree(root, version="0.1.3"):
    (root / "pyproject.toml").write_text(
        PYPROJECT.replace('version = "0.1.3"', f'version = "{version}"', 1)
    )
    return root


def git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def project_version(root):
    return tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]


def remote_refs(root):
    """Whole ref names on origin; a substring check would confuse v0.1.4 with v0.1.4.dev0."""
    return {line.split()[1] for line in git(root, "ls-remote", "origin").splitlines()} - {"HEAD"}


@pytest.fixture(scope="module", autouse=True)
def hermetic_git():
    """Keep the developer's git setup (hooks, templates, GIT_DIR) out of the throwaway repos."""
    with pytest.MonkeyPatch.context() as patch:
        for name in list(os.environ):
            if name.startswith("GIT_") and name not in ("GIT_EXEC_PATH",):
                patch.delenv(name)
        patch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
        patch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
        patch.setenv("GIT_CONFIG_NOSYSTEM", "1")
        yield


@pytest.fixture(scope="module")
def repository_template(tmp_path_factory, hermetic_git):
    """What every test repository starts as, built once: main, pushed to a bare origin."""
    root = write_tree(tmp_path_factory.mktemp("repository"))
    git(root, "init", "-b", "main")
    for key, value in (
        ("user.email", "release@example.test"),
        ("user.name", "Release Test"),
        ("commit.gpgsign", "false"),
        ("tag.gpgsign", "false"),
        ("maintenance.auto", "false"),
    ):
        git(root, "config", key, value)
    git(root, "add", ".")
    git(root, "commit", "-m", "Initial")
    remote = root / "remote.git"
    remote.mkdir()
    git(remote, "init", "--bare", "-b", "main")
    (root / ".git/info/exclude").write_text("remote.git/\n")  # The remote lives inside the fixture.
    git(root, "remote", "add", "origin", str(remote))
    git(root, "push", "origin", "main")
    return root


@pytest.fixture
def repository(tmp_path, repository_template):
    """A private copy of the template: the same commit, in its own .git with its own origin."""
    root = write_tree(tmp_path)
    for name in (".git", "remote.git"):
        shutil.copytree(repository_template / name, root / name, symlinks=True)
    git(root, "remote", "set-url", "origin", str(root / "remote.git"))
    return root


@pytest.fixture(autouse=True)
def no_validation(monkeypatch):
    """Validation builds and tests the package and needs uv; its own tests stub the commands."""
    monkeypatch.setattr(RF, "validate", lambda self: None)
    monkeypatch.setattr(RF, "check_tools", lambda self: None)


def flow(root, **options):
    args = argparse.Namespace(
        version="0.1.4",
        bump=None,
        dry_run=False,
        push=False,
        tag=False,
        release=False,
        pr=False,
    )
    for name, value in options.items():
        setattr(args, name, value)
    return bump_version.ReleaseFlow(root, args)


def commit_main(root, message, version=None):
    if version:
        write_tree(root, version)
    git(root, "add", "-A")
    git(root, "commit", "-m", message)
    git(root, "push", "origin", "main")


def merge_release(root, branch):
    """Stand in for the human who merges the release PR into main."""
    git(root, "switch", "main")
    git(root, "merge", "--no-ff", "-m", f"Merge {branch}", branch)
    git(root, "push", "origin", "main")


# --- Versions ----------------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0.1.3", "1.0.0", "0.1.4.dev0", "2.10.3.dev12"])
def test_parse_version_accepts_canonical_forms(value):
    assert str(bump_version.parse_version(value)) == value


@pytest.mark.parametrize(
    "value",
    ["0.1", "1.0.0.0", "01.0.0", "0.1.3rc1", "0.1.3.post1", "0.1.3-dev0", "0.1.3.dev", "v0.1.3"]
    + ["1!0.1.3", "0.1.3+local", "garbage", ""],
)
def test_parse_version_rejects_noncanonical_forms(value):
    with pytest.raises(ValueError):
        bump_version.parse_version(value)


@pytest.mark.parametrize(
    ("current", "bump", "expected"),
    [
        ("0.1.3", "patch", "0.1.4"),
        ("0.1.3", "minor", "0.2.0"),
        ("0.1.3", "major", "1.0.0"),
        ("0.1.3", "patch-dev", "0.1.4.dev0"),
        ("0.1.3", "minor-dev", "0.2.0.dev0"),
        ("0.1.3", "major-dev", "1.0.0.dev0"),
        ("0.1.4.dev0", "dev", "0.1.4.dev1"),
        ("0.1.4.dev9", "dev", "0.1.4.dev10"),
        ("0.1.4.dev3", "final", "0.1.4"),
    ],
)
def test_next_version(current, bump, expected):
    assert bump_version.next_version(current, bump) == expected


@pytest.mark.parametrize(
    ("current", "bump"),
    [
        ("0.1.3", "dev"),  # Stable: there is no series to continue.
        ("0.1.3", "final"),  # Already stable.
        ("0.1.4.dev0", "patch"),  # Ambiguous inside a development series.
        ("0.1.4.dev0", "minor-dev"),
        ("0.1.3", "snapshot"),  # Handled by the flow, not by inference.
        ("0.1.3", None),
    ],
)
def test_next_version_rejects_ambiguous_requests(current, bump):
    with pytest.raises(ValueError):
        bump_version.next_version(current, bump)


def test_snapshot_version_round_trips():
    sha = "46a160c727" + "0" * 30
    version = bump_version.snapshot_version("0.1.3", sha)
    assert version == "0.1.3+g46a160c727"
    assert bump_version.split_snapshot(version) == ("0.1.3", "46a160c727")
    dev = bump_version.snapshot_version("0.1.4.dev2", sha)
    assert bump_version.split_snapshot(dev) == ("0.1.4.dev2", "46a160c727")


@pytest.mark.parametrize(
    "value",
    ["0.1.3", "0.1.3+g46a160c72", "0.1.3+gABCDEF0123", "0.1.3+46a160c727", "0.1+g46a160c727"],
)
def test_split_snapshot_rejects_other_spellings(value):
    assert bump_version.split_snapshot(value) is None


def test_snapshot_version_needs_a_full_sha():
    with pytest.raises(ValueError, match="full commit sha"):
        bump_version.snapshot_version("0.1.3", "46a160c727")


@pytest.mark.parametrize(
    ("url", "slug"),
    [
        ("git@github.com-ronpik:ronpik/jserpy.git", "ronpik/jserpy"),  # An ssh config alias.
        ("git@github.com:ronpik/jserpy.git", "ronpik/jserpy"),
        ("https://github.com/ronpik/jserpy.git", "ronpik/jserpy"),
        ("https://github.com/ronpik/jserpy", "ronpik/jserpy"),
        ("https://github.com/ronpik/jserpy/", "ronpik/jserpy"),
        ("https://user:token@github.com/ronpik/jserpy.git", "ronpik/jserpy"),
        ("ssh://git@github.com/ronpik/jserpy.git", "ronpik/jserpy"),
        ("git@github.com:ronpik/my.dotted.repo.git", "ronpik/my.dotted.repo"),
    ],
)
def test_repo_slug(url, slug):
    assert bump_version.repo_slug(url) == slug


@pytest.mark.parametrize("url", ["/tmp/some/remote.git", "jserpy", ""])
def test_repo_slug_rejects_non_urls(url):
    with pytest.raises(ValueError, match="Cannot tell the GitHub repository"):
        bump_version.repo_slug(url)


# --- Editing pyproject.toml --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bump", "expected"), [("patch", "0.1.4"), ("minor", "0.2.0"), ("major", "1.0.0")]
)
def test_prepare_edits_only_the_project_version(tmp_path, bump, expected):
    write_tree(tmp_path)
    current, target, edits = bump_version.prepare(tmp_path, None, bump)
    assert (current, target) == ("0.1.3", expected)
    assert list(edits) == [tmp_path / "pyproject.toml"]
    text = edits[tmp_path / "pyproject.toml"]
    # Byte for byte the old file, except that one value; the other tables' "version" stays.
    assert text == PYPROJECT.replace('version = "0.1.3"', f'version = "{expected}"', 1)
    assert tomllib.loads(text)["tool"]["example"]["version"] == "9.9.9"
    assert not text.endswith("\n")
    assert project_version(tmp_path) == "0.1.3"  # prepare() writes nothing.


@pytest.mark.parametrize(
    "target", ["0.1.3", "0.1.2", "0.1.3.dev0", "0.1.4rc1", "junk", "0.1.4+g0123456789"]
)
def test_prepare_rejects_invalid_or_nonincreasing_targets(tmp_path, target):
    write_tree(tmp_path)
    with pytest.raises(ValueError):
        bump_version.prepare(tmp_path, target, None)


def test_prepare_accepts_a_snapshot_of_the_declared_version(tmp_path):
    write_tree(tmp_path)
    _, target, edits = bump_version.prepare(tmp_path, "0.1.3+g0123456789", None)
    assert tomllib.loads(edits[tmp_path / "pyproject.toml"])["project"]["version"] == target


def test_prepare_refuses_to_build_on_a_snapshot_version(tmp_path):
    write_tree(tmp_path, "0.1.3+g0123456789")
    with pytest.raises(ValueError, match="snapshot version"):
        bump_version.prepare(tmp_path, None, "patch")


def test_prepare_requires_a_static_version(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\ndynamic = ["version"]\n')
    with pytest.raises(ValueError, match="static"):
        bump_version.prepare(tmp_path, None, "patch")


def test_prepare_continues_a_development_series(tmp_path):
    write_tree(tmp_path, "0.1.4.dev0")
    assert bump_version.prepare(tmp_path, None, "dev")[1] == "0.1.4.dev1"
    assert bump_version.prepare(tmp_path, None, "final")[1] == "0.1.4"
    with pytest.raises(ValueError, match="ambiguous"):
        bump_version.prepare(tmp_path, None, "patch")


# --- Preparing and publishing a release --------------------------------------------------------


def test_dry_run_changes_nothing(repository, monkeypatch, capsys):
    monkeypatch.setattr(
        bump_version.ReleaseFlow, "validate", lambda self: pytest.fail("a dry run validated")
    )
    runner = flow(repository, dry_run=True, tag=True)
    before = git(repository, "show-ref")
    runner.execute()
    assert git(repository, "show-ref") == before
    assert git(repository, "status", "--porcelain") == ""
    assert git(repository, "branch", "--show-current") == "main"
    assert not runner.state_path.exists()
    assert "Target: 0.1.4 (stable release)" in capsys.readouterr().out


def test_prepare_commits_only_the_version_on_a_release_branch(repository):
    base = git(repository, "rev-parse", "HEAD")
    flow(repository).execute()
    assert git(repository, "branch", "--show-current") == "release/v0.1.4"
    assert git(repository, "status", "--porcelain") == ""
    assert git(repository, "log", "-1", "--format=%s") == "Release jserpy 0.1.4"
    assert git(repository, "rev-parse", "HEAD~1") == base
    assert git(repository, "diff", "--name-only", "HEAD~1", "HEAD") == "pyproject.toml"
    assert project_version(repository) == "0.1.4"
    assert "refs/heads/release/v0.1.4" not in remote_refs(repository)  # Nothing is pushed.
    assert "refs/tags/v0.1.4" not in remote_refs(repository)


def test_bump_flag_infers_the_version_from_main(repository):
    flow(repository, version=None, bump="minor").execute()
    assert git(repository, "branch", "--show-current") == "release/v0.2.0"
    assert project_version(repository) == "0.2.0"


def test_push_tag_and_resume_are_idempotent(repository):
    flow(repository).execute()
    commit = git(repository, "rev-parse", "HEAD")
    flow(repository, push=True).execute()
    assert remote_refs(repository) >= {"refs/heads/release/v0.1.4"}
    assert "refs/tags/v0.1.4" not in remote_refs(repository)
    flow(repository, tag=True).execute()
    flow(repository, tag=True).execute()
    assert git(repository, "cat-file", "-t", "v0.1.4") == "tag"
    assert git(repository, "tag", "-l", "-n1", "v0.1.4").split(None, 1)[1] == "jserpy 0.1.4"
    assert git(repository, "rev-parse", "v0.1.4^{commit}") == commit
    assert git(repository, "rev-parse", "HEAD") == commit
    assert {"refs/heads/release/v0.1.4", "refs/tags/v0.1.4"} <= remote_refs(repository)


def test_tag_pushes_branch_and_tag_together(repository):
    flow(repository, tag=True).execute()
    assert {"refs/heads/release/v0.1.4", "refs/tags/v0.1.4"} <= remote_refs(repository)


def test_release_creates_the_github_release_once(repository, monkeypatch):
    calls = []
    monkeypatch.setattr(bump_version.ReleaseFlow, "github_repo", lambda self: "owner/repo")
    monkeypatch.setattr(bump_version.ReleaseFlow, "existing_release", lambda self: None)
    monkeypatch.setattr(
        bump_version.ReleaseFlow,
        "github",
        lambda self, *args: calls.append(args) or subprocess.CompletedProcess(args, 0, "URL", ""),
    )
    flow(repository, release=True).execute()
    assert calls == [
        (
            "release",
            "create",
            "v0.1.4",
            "--verify-tag",
            "--title",
            "jserpy 0.1.4",
            "--generate-notes",
        )
    ]
    assert {"refs/heads/release/v0.1.4", "refs/tags/v0.1.4"} <= remote_refs(repository)
    monkeypatch.setattr(
        bump_version.ReleaseFlow, "existing_release", lambda self: {"html_url": "URL"}
    )
    calls.clear()
    flow(repository, release=True).execute()
    assert calls == []


def test_development_release_is_a_prerelease(repository, monkeypatch):
    calls = []
    monkeypatch.setattr(bump_version.ReleaseFlow, "github_repo", lambda self: "owner/repo")
    monkeypatch.setattr(bump_version.ReleaseFlow, "existing_release", lambda self: None)
    monkeypatch.setattr(
        bump_version.ReleaseFlow,
        "github",
        lambda self, *args: calls.append(args) or subprocess.CompletedProcess(args, 0, "URL", ""),
    )
    flow(repository, version="0.1.4.dev0", release=True).execute()
    assert calls[0][-2:] == ("--prerelease", "--latest=false")


def test_pr_is_created_once(repository, monkeypatch):
    calls = []

    def github(self, *args):
        calls.append(args)
        out = "[]" if args[:2] == ("pr", "list") else "PR URL"
        return subprocess.CompletedProcess(args, 0, out, "")

    monkeypatch.setattr(bump_version.ReleaseFlow, "github_repo", lambda self: "owner/repo")
    monkeypatch.setattr(bump_version.ReleaseFlow, "github", github)
    flow(repository, pr=True).execute()
    created = [call for call in calls if call[:2] == ("pr", "create")]
    assert len(created) == 1
    assert created[0][created[0].index("--title") + 1] == "Release jserpy 0.1.4"
    assert created[0][created[0].index("--base") + 1] == "main"
    assert "refs/heads/release/v0.1.4" in remote_refs(repository)


def test_dirty_main_is_rejected(repository):
    (repository / "unrelated.txt").write_text("work")
    with pytest.raises(ValueError, match="clean worktree"):
        flow(repository).execute()
    git(repository, "add", "unrelated.txt")
    with pytest.raises(ValueError, match="clean worktree"):
        flow(repository).execute()


def test_main_must_match_origin(repository):
    (repository / "new.txt").write_text("work")
    git(repository, "add", "new.txt")
    git(repository, "commit", "-m", "Not pushed")
    with pytest.raises(ValueError, match="main must match origin/main"):
        flow(repository).execute()


def test_fresh_release_must_start_from_main(repository):
    git(repository, "switch", "-c", "feature")
    with pytest.raises(ValueError, match="Start from main"):
        flow(repository).execute()
    with pytest.raises(ValueError, match="Bumps start from main, not from feature"):
        flow(repository, version=None, bump="patch").execute()


def test_a_taken_tag_stops_an_inferred_version(repository):
    """The repository's own history: tag v0.1.4 sits on main while pyproject.toml says 0.1.3."""
    git(repository, "tag", "-a", "v0.1.4", "-m", "Merge pull request #4")
    git(repository, "push", "origin", "v0.1.4")
    before = git(repository, "show-ref")
    with pytest.raises(ValueError, match="already exists.*higher explicit --version"):
        flow(repository, version=None, bump="patch").execute()
    with pytest.raises(ValueError, match="not a release commit of this script"):
        flow(repository, tag=True).execute()  # An explicit version is not allowed to adopt it.
    assert git(repository, "show-ref") == before
    flow(repository, version="0.1.5", tag=True).execute()  # The next free version works.
    assert "refs/tags/v0.1.5" in remote_refs(repository)


def test_validation_failure_resumes(repository, monkeypatch):
    monkeypatch.setattr(
        bump_version.ReleaseFlow,
        "validate",
        lambda self: (_ for _ in ()).throw(ValueError("failed tests")),
    )
    with pytest.raises(ValueError, match="failed tests"):
        flow(repository).execute()
    assert git(repository, "branch", "--show-current") == "release/v0.1.4"
    assert git(repository, "log", "-1", "--format=%s") == "Initial"  # Nothing is committed.
    monkeypatch.setattr(bump_version.ReleaseFlow, "validate", lambda self: None)
    flow(repository).execute()
    assert git(repository, "log", "-1", "--format=%s") == "Release jserpy 0.1.4"
    assert git(repository, "status", "--porcelain") == ""


def test_resume_refuses_user_edits(repository, monkeypatch):
    monkeypatch.setattr(
        bump_version.ReleaseFlow,
        "validate",
        lambda self: (_ for _ in ()).throw(ValueError("failed")),
    )
    with pytest.raises(ValueError):
        flow(repository).execute()
    with (repository / "pyproject.toml").open("a") as handle:
        handle.write("# user edit\n")
    monkeypatch.setattr(bump_version.ReleaseFlow, "validate", lambda self: None)
    with pytest.raises(ValueError, match="changed after release preparation"):
        flow(repository).execute()


def test_validation_that_edits_files_prevents_the_commit(repository, monkeypatch):
    def validate(self):
        (self.root / "stray.txt").write_text("left behind")

    monkeypatch.setattr(bump_version.ReleaseFlow, "validate", validate)
    with pytest.raises(ValueError, match="Validation left unrelated changes"):
        flow(repository).execute()
    assert git(repository, "log", "-1", "--format=%s") == "Initial"


def test_conflicting_tag_on_another_commit_is_rejected(repository):
    flow(repository).execute()
    git(repository, "tag", "-a", "v0.1.4", "main", "-m", "Wrong commit")
    with pytest.raises(ValueError, match="different commits"):
        flow(repository, tag=True).execute()


def test_resume_from_a_remote_only_branch(repository):
    flow(repository, push=True).execute()
    commit = git(repository, "rev-parse", "HEAD")
    git(repository, "switch", "main")
    git(repository, "branch", "-D", "release/v0.1.4")
    flow(repository, tag=True).execute()
    assert git(repository, "rev-parse", "release/v0.1.4") == commit
    assert git(repository, "rev-parse", "v0.1.4^{commit}") == commit
    assert git(repository, "branch", "--show-current") == "main"


def test_a_tampered_release_commit_is_not_reused(repository):
    flow(repository).execute()
    path = repository / "pyproject.toml"
    path.write_text(path.read_text().replace("numpy>=1.13.0", "numpy>=2.0"))
    git(repository, "commit", "-a", "--amend", "--no-edit")
    with pytest.raises(ValueError, match="unexpected edits in pyproject.toml"):
        flow(repository, tag=True).execute()


def test_development_series_from_main(repository):
    flow(repository, version=None, bump="patch-dev", tag=True).execute()
    assert project_version(repository) == "0.1.4.dev0"
    merge_release(repository, "release/v0.1.4.dev0")
    flow(repository, version=None, bump="dev").execute()
    assert project_version(repository) == "0.1.4.dev1"
    merge_release(repository, "release/v0.1.4.dev1")
    flow(repository, version=None, bump="final").execute()
    assert project_version(repository) == "0.1.4"


def test_leaving_a_development_series_is_noted(repository, capsys):
    commit_main(repository, "Development release", "0.1.4.dev0")
    flow(repository, version="0.2.0", dry_run=True).execute()
    assert "leaves the 0.1.4 development series" in capsys.readouterr().out


# --- Snapshots ---------------------------------------------------------------------------------


def feature_branch(root):
    git(root, "switch", "-c", "feature/work")
    (root / "work.txt").write_text("work")
    git(root, "add", "work.txt")
    git(root, "commit", "-m", "Work in progress")
    return git(root, "rev-parse", "HEAD")


def test_snapshot_is_cut_from_a_feature_branch_and_switches_back(repository):
    head = feature_branch(repository)
    version = f"0.1.3+g{head[:10]}"
    flow(repository, version=None, bump="snapshot").execute()
    assert git(repository, "branch", "--show-current") == "feature/work"
    assert git(repository, "rev-parse", "HEAD") == head
    assert project_version(repository) == "0.1.3"
    assert git(repository, "status", "--porcelain") == ""
    release = f"release/v{version}"
    assert git(repository, "log", "-1", "--format=%s", release) == f"Release jserpy {version}"
    assert git(repository, "rev-parse", f"{release}~1") == head
    assert "work.txt" in git(repository, "ls-tree", "--name-only", release)
    assert (
        tomllib.loads(git(repository, "show", f"{release}:pyproject.toml"))["project"]["version"]
        == version
    )


def test_snapshot_tag_is_pushed_alone(repository, capsys):
    head = feature_branch(repository)
    version = f"0.1.3+g{head[:10]}"
    flow(repository, version=None, bump="snapshot", tag=True).execute()
    refs = remote_refs(repository)
    assert f"refs/tags/v{version}" in refs
    assert f"refs/heads/release/v{version}" not in refs
    assert "feature/work" not in " ".join(refs)
    assert git(repository, "branch", "--show-current") == "feature/work"
    assert "nothing is merged back" in capsys.readouterr().out
    flow(repository, version=version, tag=True).execute()  # Resuming is a no-op.
    flow(repository, version=version, push=True).execute()  # --push adds the branch.
    assert f"refs/heads/release/v{version}" in remote_refs(repository)


def test_snapshot_can_be_cut_from_main_itself(repository):
    head = git(repository, "rev-parse", "HEAD")
    flow(repository, version=None, bump="snapshot", tag=True).execute()
    assert f"refs/tags/v0.1.3+g{head[:10]}" in remote_refs(repository)
    assert git(repository, "branch", "--show-current") == "main"
    assert git(repository, "rev-parse", "HEAD") == head


def test_snapshot_from_a_detached_head_switches_back_to_it(repository):
    head = feature_branch(repository)
    git(repository, "switch", "--detach", head)
    flow(repository, version=None, bump="snapshot", tag=True).execute()
    assert git(repository, "branch", "--show-current") == ""
    assert git(repository, "rev-parse", "HEAD") == head
    assert f"refs/tags/v0.1.3+g{head[:10]}" in remote_refs(repository)


def test_snapshot_of_an_older_commit_is_cut_from_a_branch_there(repository):
    base = git(repository, "rev-parse", "HEAD")
    feature_branch(repository)
    git(repository, "switch", "-c", "older", base)
    flow(repository, version=None, bump="snapshot").execute()
    assert git(repository, "rev-parse", f"release/v0.1.3+g{base[:10]}~1") == base


def test_snapshot_needs_a_clean_worktree(repository):
    feature_branch(repository)
    (repository / "work.txt").write_text("changed")
    with pytest.raises(ValueError, match="clean worktree"):
        flow(repository, version=None, bump="snapshot").execute()


def test_snapshot_refuses_pr_and_release(repository):
    feature_branch(repository)
    with pytest.raises(ValueError, match="never merged"):
        flow(repository, version=None, bump="snapshot", pr=True).execute()
    with pytest.raises(ValueError, match="PyPI workflow, which rejects"):
        flow(repository, version=None, bump="snapshot", release=True).execute()
    assert not [r for r in remote_refs(repository) if "release" in r or "tags" in r]


def test_snapshot_refuses_to_start_from_a_release_branch(repository):
    flow(repository).execute()
    with pytest.raises(ValueError, match="starts from the branch to snapshot"):
        flow(repository, version=None, bump="snapshot").execute()


def test_snapshot_version_must_name_head(repository):
    feature_branch(repository)
    with pytest.raises(ValueError, match="is not the snapshot of HEAD"):
        flow(repository, version="0.1.3+g0123456789").execute()
    with pytest.raises(ValueError, match="A snapshot version is"):
        flow(repository, version="0.1.4+local").execute()


def test_a_merged_snapshot_is_never_built_upon(repository):
    commit_main(repository, "Merged by mistake", "0.1.3+g0123456789")
    with pytest.raises(ValueError, match="snapshot version"):
        flow(repository, version="0.1.4").execute()


# --- Validation --------------------------------------------------------------------------------


def fake_build(version, name="jserpy"):
    """A stand-in for `uv build --out-dir DIR`: a wheel and an sdist declaring version."""
    metadata = f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"

    def build(command):
        out = Path(command[command.index("--out-dir") + 1])
        with zipfile.ZipFile(out / f"{name}-{version}-py3-none-any.whl", "w") as wheel:
            wheel.writestr(f"{name}-{version}.dist-info/METADATA", metadata)
        with tarfile.open(out / f"{name}-{version}.tar.gz", "w:gz") as sdist:
            member = tarfile.TarInfo(f"{name}-{version}/PKG-INFO")
            member.size = len(metadata.encode())
            sdist.addfile(member, io.BytesIO(metadata.encode()))

    return build


def validate(root, monkeypatch, build, tests=None):
    """Run the real validate() with `uv` replaced: build() fakes `uv build`, tests() `uv run`."""
    commands = []

    def run(self, *command, check=True, cwd=None):
        commands.append(command)
        if command[:2] == ("uv", "build"):
            build(command)
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:2] == ("uv", "run"):
            return subprocess.CompletedProcess(command, 0, tests() if tests else "1 passed\n", "")
        raise AssertionError(f"unexpected command {command}")

    monkeypatch.setattr(bump_version.ReleaseFlow, "run", run)
    runner = flow(root)
    runner.name = "jserpy"
    REAL_VALIDATE(runner)
    return commands


def test_validate_builds_then_tests_the_built_wheel(repository, monkeypatch):
    commands = validate(repository, monkeypatch, fake_build("0.1.4"))
    assert [command[:2] for command in commands] == [("uv", "build"), ("uv", "run")]
    test = commands[1]
    assert "--no-project" in test and "pytest" in test
    assert test[test.index("--with") + 1].endswith("jserpy-0.1.4-py3-none-any.whl")


def test_validate_accepts_another_spelling_of_the_name(repository, monkeypatch):
    validate(repository, monkeypatch, fake_build("0.1.4", "JSerPy"))


def test_validate_rejects_artifacts_of_another_version(repository, monkeypatch):
    with pytest.raises(ValueError, match="declares version 0.1.3, expected 0.1.4"):
        validate(repository, monkeypatch, fake_build("0.1.3"))


def test_validate_rejects_artifacts_of_another_project(repository, monkeypatch):
    with pytest.raises(ValueError, match="declares the name other"):
        validate(repository, monkeypatch, fake_build("0.1.4", "other"))


def test_validate_stops_at_failing_tests(repository, monkeypatch):
    def failing():
        raise ValueError("pytest failed")

    with pytest.raises(ValueError, match="pytest failed"):
        validate(repository, monkeypatch, fake_build("0.1.4"), tests=failing)


def test_failing_validation_leaves_the_release_uncommitted(repository, monkeypatch):
    def failing(self):
        raise ValueError("uv failed")

    monkeypatch.setattr(bump_version.ReleaseFlow, "validate", failing)
    with pytest.raises(ValueError, match="uv failed"):
        flow(repository).execute()
    assert git(repository, "log", "-1", "--format=%s") == "Initial"
    assert json.loads(next((repository / ".git").glob("jserpy-release-*.json")).read_text())


def test_built_version_reads_both_artifacts(tmp_path):
    fake_build("0.1.4")(["uv", "build", "--out-dir", str(tmp_path)])
    assert bump_version.built_version(tmp_path / "jserpy-0.1.4-py3-none-any.whl") == (
        "jserpy",
        "0.1.4",
    )
    assert bump_version.built_version(tmp_path / "jserpy-0.1.4.tar.gz") == ("jserpy", "0.1.4")


# --- Running programs --------------------------------------------------------------------------


def test_a_failing_command_reports_both_of_its_streams(repository):
    """A failing pytest reports on stdout while uv's chatter fills stderr; the user needs both."""
    script = (
        "import sys; print('FAILED test_x - boom'); "
        "print('Installed 10 packages', file=sys.stderr); sys.exit(1)"
    )
    with pytest.raises(ValueError, match="FAILED test_x - boom") as error:
        flow(repository).run(sys.executable, "-c", script)
    assert "Installed 10 packages" in str(error.value)


def test_a_missing_program_is_named(repository):
    with pytest.raises(ValueError, match="no-such-program-here is required but was not found"):
        flow(repository).run("no-such-program-here")


def test_run_leaves_line_endings_alone(repository):
    out = flow(repository).run(
        sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'a\\r\\nb')"
    )
    assert out.stdout == "a\r\nb"


def test_run_raises_when_a_command_fails(repository):
    with pytest.raises(ValueError, match="failed"):
        flow(repository).git("rev-parse", "no-such-ref")


def test_ref_raises_outside_a_repository(tmp_path):
    with pytest.raises(ValueError):
        flow(tmp_path).ref("refs/heads/main")


def test_run_hides_the_virtualenv(repository, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/nonexistent/venv")
    assert "VIRTUAL_ENV" not in flow(repository).run("env").stdout


@pytest.mark.parametrize(
    ("release", "pr", "dry_run", "needed"),
    [(False, False, False, "uv"), (True, False, False, "uv and gh"), (False, True, True, "gh")],
)
def test_missing_tools_are_reported_before_anything_changes(
    repository, monkeypatch, release, pr, dry_run, needed
):
    monkeypatch.setattr(bump_version.shutil, "which", lambda name: None if name != "git" else "git")
    runner = flow(repository, release=release, pr=pr, dry_run=dry_run, version="0.1.4")
    monkeypatch.setattr(RF, "check_tools", REAL_CHECK_TOOLS)
    before = git(repository, "show-ref")
    with pytest.raises(ValueError, match=f"{needed} must be installed"):
        runner.execute()
    assert git(repository, "show-ref") == before
    assert git(repository, "branch", "--show-current") == "main"


def test_a_dry_run_needs_no_uv(repository, monkeypatch):
    monkeypatch.setattr(bump_version.shutil, "which", lambda name: "x" if name == "git" else None)
    monkeypatch.setattr(RF, "check_tools", REAL_CHECK_TOOLS)
    flow(repository, dry_run=True).execute()


# --- Line endings ------------------------------------------------------------------------------


def test_prepare_preserves_crlf_line_endings(tmp_path):
    crlf = PYPROJECT.replace("\n", "\r\n")
    (tmp_path / "pyproject.toml").write_bytes(crlf.encode())
    _, _, edits = bump_version.prepare(tmp_path, "0.1.4", None)
    assert edits[tmp_path / "pyproject.toml"] == crlf.replace('"0.1.3"', '"0.1.4"', 1)


def test_a_crlf_release_commit_changes_one_line_and_verifies(repository):
    crlf = PYPROJECT.replace("\n", "\r\n").encode()
    (repository / "pyproject.toml").write_bytes(crlf)
    commit_main(repository, "CRLF")
    flow(repository, tag=True).execute()
    assert git(repository, "diff", "--numstat", "HEAD~1", "HEAD") == "1\t1\tpyproject.toml"
    shown = subprocess.run(
        ["git", "show", "HEAD:pyproject.toml"], cwd=repository, capture_output=True
    ).stdout
    assert shown == crlf.replace(b'"0.1.3"', b'"0.1.4"', 1)


# --- Release branches --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "mine"),
    [
        ("release/v0.1.4", True),
        ("release/v0.1.4.dev2", True),
        ("release/v0.1.3+g0123456789", True),
        ("release/0.1/fix-version", False),  # This repository's own earlier branches.
        ("release/0.1/numpy-reqs", False),
        ("release/v0.1", False),
        ("release/v0.1.4-rc1", False),
        ("main", False),
        ("", False),
    ],
)
def test_only_the_scripts_own_branches_count_as_release_branches(name, mine):
    assert bump_version.is_release_branch(name) is mine


def test_a_snapshot_can_be_cut_from_a_legacy_release_branch(repository):
    git(repository, "switch", "-c", "release/0.1/hotfix")
    head = git(repository, "rev-parse", "HEAD")
    flow(repository, version=None, bump="snapshot", tag=True).execute()
    assert git(repository, "branch", "--show-current") == "release/0.1/hotfix"
    assert f"refs/tags/v0.1.3+g{head[:10]}" in remote_refs(repository)


def test_a_snapshot_is_not_reused_from_another_release_branch(repository):
    head = feature_branch(repository)
    version = f"0.1.3+g{head[:10]}"
    flow(repository, version=None, bump="snapshot").execute()
    git(repository, "switch", "-c", "release/v9.9.9")
    with pytest.raises(ValueError, match=r"not a release/v\* branch"):
        flow(repository, version=version, tag=True).execute()


# --- Existing releases -------------------------------------------------------------------------


def test_an_own_release_is_offered_for_resuming(repository):
    flow(repository, tag=True).execute()
    git(repository, "switch", "main")
    with pytest.raises(ValueError, match=r"as a release of this script.*resume it with --version"):
        flow(repository, version=None, bump="patch").execute()


def test_an_already_cut_snapshot_is_offered_for_tagging(repository):
    head = feature_branch(repository)
    flow(repository, version=None, bump="snapshot").execute()
    with pytest.raises(ValueError, match=rf"already cut as 0.1.3\+g{head[:10]}.*--tag"):
        flow(repository, version=None, bump="snapshot").execute()


def test_the_foreign_tag_message_suggests_the_next_version(repository):
    git(repository, "tag", "-a", "v0.1.4", "-m", "Merge pull request #4")
    git(repository, "push", "origin", "v0.1.4")
    with pytest.raises(ValueError, match="for example 0.1.5"):
        flow(repository, version=None, bump="patch").execute()
    commit_main(repository, "Dev", "0.1.4.dev0")
    git(repository, "tag", "-a", "v0.1.4.dev1", "-m", "x")
    git(repository, "push", "origin", "v0.1.4.dev1")
    with pytest.raises(ValueError, match="for example 0.1.4.dev2"):
        flow(repository, version=None, bump="dev").execute()


def test_a_release_commit_of_another_project_is_not_reused(repository):
    flow(repository).execute()
    path = repository / "pyproject.toml"
    path.write_text(path.read_text().replace('name = "jserpy"', 'name = "other"'))
    git(repository, "commit", "-a", "--amend", "-m", "Release other 0.1.4")
    git(repository, "switch", "main")
    with pytest.raises(ValueError, match="is for the project other"):
        flow(repository, tag=True).execute()


def test_a_remote_lightweight_tag_is_refused_even_by_a_dry_run(repository):
    flow(repository, push=True).execute()
    git(repository, "tag", "v0.1.4")
    git(repository, "push", "origin", "v0.1.4")
    git(repository, "tag", "-d", "v0.1.4")
    with pytest.raises(ValueError, match="is a lightweight tag"):
        flow(repository, tag=True, dry_run=True).execute()


def test_lightweight_tag_is_rejected(repository):
    flow(repository).execute()
    git(repository, "tag", "v0.1.4")
    with pytest.raises(ValueError, match="is a lightweight tag"):
        flow(repository, tag=True).execute()


def test_tag_and_branch_are_pushed_atomically(repository):
    hook = repository / "remote.git/hooks/update"
    hook.write_text('#!/bin/sh\ncase "$1" in refs/tags/*) echo no >&2; exit 1;; esac\n')
    hook.chmod(0o755)
    with pytest.raises(ValueError, match="failed"):
        flow(repository, tag=True).execute()
    assert "refs/heads/release/v0.1.4" not in remote_refs(repository)


def test_a_failed_publish_says_the_commit_is_kept_and_resumable(repository):
    hook = repository / "remote.git/hooks/update"
    hook.write_text("#!/bin/sh\necho refused >&2\nexit 1\n")
    hook.chmod(0o755)
    with pytest.raises(ValueError, match=r"stays on release/v0.1.4.*--version 0.1.4") as error:
        flow(repository, tag=True).execute()
    assert "refused" in str(error.value)
    hook.unlink()
    flow(repository, tag=True).execute()  # The same command finishes the job.
    assert {"refs/heads/release/v0.1.4", "refs/tags/v0.1.4"} <= remote_refs(repository)
    assert git(repository, "rev-list", "--count", "main..release/v0.1.4") == "1"


def test_tag_run_pushes_the_branch_when_only_the_tag_is_remote(repository):
    flow(repository).execute()
    git(repository, "tag", "-a", "v0.1.4", "-m", "jserpy 0.1.4")
    git(repository, "push", "origin", "v0.1.4")
    flow(repository, tag=True).execute()
    assert "refs/heads/release/v0.1.4" in remote_refs(repository)


def test_reusing_a_tagged_commit_recreates_the_branch(repository):
    flow(repository, tag=True).execute()
    git(repository, "switch", "main")
    git(repository, "branch", "-D", "release/v0.1.4")
    git(repository, "push", "origin", ":refs/heads/release/v0.1.4")
    flow(repository, tag=True).execute()
    assert "refs/heads/release/v0.1.4" in remote_refs(repository)
    branch = git(repository, "rev-parse", "release/v0.1.4")
    assert branch == git(repository, "rev-parse", "v0.1.4^{commit}")


# --- verify_commit -----------------------------------------------------------------------------


def test_a_release_commit_touching_another_file_is_not_reused(repository):
    flow(repository).execute()
    (repository / "extra.txt").write_text("x")
    git(repository, "add", "extra.txt")
    git(repository, "commit", "--amend", "--no-edit")
    with pytest.raises(ValueError, match="exactly pyproject.toml"):
        flow(repository, tag=True).execute()


def test_a_release_commit_with_another_subject_is_not_reused(repository):
    flow(repository).execute()
    git(repository, "commit", "--amend", "-m", "Something else")
    with pytest.raises(ValueError, match="does not identify a release commit"):
        flow(repository, tag=True).execute()


def test_a_snapshot_commit_must_name_its_parent(repository):
    feature_branch(repository)
    version = "0.1.3+g0123456789"
    git(repository, "switch", "-c", f"release/v{version}")
    write_tree(repository, version)
    git(repository, "commit", "-am", f"Release jserpy {version}")
    git(repository, "switch", "feature/work")
    with pytest.raises(ValueError, match="does not name"):
        flow(repository, version=version, tag=True).execute()


def test_a_commit_rewritten_by_a_hook_is_not_released(repository):
    hook = repository / ".git/hooks/commit-msg"
    hook.write_text('#!/bin/sh\necho Tweaked > "$1"\n')
    hook.chmod(0o755)
    with pytest.raises(ValueError, match="does not identify a release commit"):
        flow(repository).execute()


# --- Resuming ----------------------------------------------------------------------------------


def interrupted(repository, monkeypatch):
    def boom(self):
        raise ValueError("failed")

    monkeypatch.setattr(RF, "validate", boom)
    with pytest.raises(ValueError, match="failed"):
        flow(repository).execute()
    monkeypatch.setattr(RF, "validate", lambda self: None)


def test_a_failed_preparation_says_how_to_resume_or_drop_it(repository, monkeypatch):
    monkeypatch.setattr(
        RF, "validate", lambda self: (_ for _ in ()).throw(ValueError("tests failed"))
    )
    with pytest.raises(ValueError, match="tests failed") as error:
        flow(repository).execute()
    message = str(error.value)
    assert "prepared on release/v0.1.4 but not committed" in message
    assert "re-run with --version 0.1.4" in message
    assert "git branch -D release/v0.1.4" in message and "git switch main" in message
    # The advice works: dropping the preparation by hand leaves a pristine repository.
    journal = next((repository / ".git").glob("jserpy-release-*.json"))
    for command in (
        ("restore", "--staged", "--worktree", "--", "pyproject.toml"),
        ("switch", "main"),
        ("branch", "-D", "release/v0.1.4"),
    ):
        git(repository, *command)
    journal.unlink()
    assert git(repository, "status", "--porcelain") == ""
    monkeypatch.setattr(RF, "validate", lambda self: None)
    flow(repository).execute()


def test_a_snapshot_failure_says_where_to_go_back(repository, monkeypatch):
    feature_branch(repository)
    monkeypatch.setattr(RF, "validate", lambda self: (_ for _ in ()).throw(ValueError("nope")))
    with pytest.raises(ValueError, match="git switch feature/work"):
        flow(repository, version=None, bump="snapshot").execute()


def test_resume_from_the_wrong_branch_is_refused(repository, monkeypatch):
    interrupted(repository, monkeypatch)
    git(repository, "restore", "pyproject.toml")
    git(repository, "switch", "main")
    with pytest.raises(ValueError, match="Resume interrupted preparation from release/v0.1.4"):
        flow(repository).execute()
    assert git(repository, "log", "-1", "--format=%s") == "Initial"


def test_resume_refuses_a_moved_base(repository, monkeypatch):
    interrupted(repository, monkeypatch)
    journal = next((repository / ".git").glob("jserpy-release-*.json"))
    state = json.loads(journal.read_text())
    runner = flow(repository)
    runner.branch = "release/v0.1.4"
    runner.check_partial(state)  # Untouched: fine.
    with pytest.raises(ValueError, match="base changed"):
        runner.check_partial({**state, "base": "0" * 40})


def test_resume_refuses_unrelated_work(repository, monkeypatch):
    interrupted(repository, monkeypatch)
    (repository / "stray.txt").write_text("x")
    with pytest.raises(ValueError, match=r"Unrelated work \(stray.txt\) prevents resuming"):
        flow(repository).execute()


def test_resume_refuses_unexpected_staged_content(repository, monkeypatch):
    interrupted(repository, monkeypatch)
    path = repository / "pyproject.toml"
    edited = path.read_text()
    path.write_text(edited.replace("0.1.4", "0.1.5"))
    git(repository, "add", "pyproject.toml")
    path.write_text(edited)
    with pytest.raises(ValueError, match="Unexpected staged content"):
        flow(repository).execute()


def test_a_dirty_tree_on_another_branch_with_a_leftover_state_is_rejected(repository, monkeypatch):
    interrupted(repository, monkeypatch)
    git(repository, "switch", "main")  # Carries the edit along.
    with pytest.raises(ValueError, match="clean worktree"):
        flow(repository).execute()


def test_a_dirty_tree_after_the_release_commit_is_rejected(repository):
    flow(repository).execute()
    (repository / "stray.txt").write_text("x")
    with pytest.raises(ValueError, match="clean worktree"):
        flow(repository, tag=True).execute()


def test_validation_that_edits_pyproject_prevents_the_commit(repository, monkeypatch):
    monkeypatch.setattr(
        RF, "validate", lambda self: (self.root / "pyproject.toml").write_text("# x\n")
    )
    with pytest.raises(ValueError, match="Validation modified pyproject.toml"):
        flow(repository).execute()


def test_a_corrupt_journal_is_named(repository):
    flow(repository).execute()
    journal = next((repository / ".git").glob("jserpy-release-*.json"))
    journal.write_text("{not json")
    with pytest.raises(ValueError, match="is not a valid journal"):
        flow(repository, tag=True).execute()


def test_a_run_interrupted_after_its_commit_still_switches_a_snapshot_back(repository, monkeypatch):
    head = feature_branch(repository)
    real = RF.verify_commit
    calls = []

    def dying(self, commit):
        calls.append(commit)
        if len(calls) == 1:  # Right after `git commit`, before the journal learns of it.
            raise ValueError("died")
        return real(self, commit)

    monkeypatch.setattr(RF, "verify_commit", dying)
    with pytest.raises(ValueError, match="died"):
        flow(repository, version=None, bump="snapshot").execute()
    assert git(repository, "branch", "--show-current") == f"release/v0.1.3+g{head[:10]}"
    flow(repository, version=f"0.1.3+g{head[:10]}", tag=True).execute()
    assert git(repository, "branch", "--show-current") == "feature/work"
    assert f"refs/tags/v0.1.3+g{head[:10]}" in remote_refs(repository)


def test_a_switch_back_that_fails_is_reported_by_hand(repository, tmp_path_factory, monkeypatch):
    feature_branch(repository)
    checkout = tmp_path_factory.mktemp("elsewhere") / "worktree"

    def hold_the_branch(self):  # Somebody else checks the start branch out meanwhile.
        git(self.root, "worktree", "add", str(checkout), "feature/work")

    monkeypatch.setattr(RF, "validate", hold_the_branch)
    printed = io.StringIO()
    monkeypatch.setattr(sys, "stdout", printed)
    flow(repository, version=None, bump="snapshot").execute()
    assert "Switch back to feature/work by hand" in printed.getvalue()
    assert git(repository, "branch", "--show-current").startswith("release/v0.1.3+g")


def test_a_switch_back_that_only_fails_in_a_hook_is_not_reported(repository, monkeypatch, capsys):
    feature_branch(repository)
    hook = repository / ".git/hooks/post-checkout"
    hook.write_text(
        '#!/bin/sh\n[ "$(git branch --show-current)" = feature/work ] && exit 1\nexit 0\n'
    )
    hook.chmod(0o755)
    flow(repository, version=None, bump="snapshot").execute()
    assert git(repository, "branch", "--show-current") == "feature/work"
    assert "by hand" not in capsys.readouterr().out


# --- Dry runs and fetching ---------------------------------------------------------------------


def test_dry_run_does_not_fetch(repository):
    flow(repository, push=True).execute()
    git(repository, "switch", "main")
    git(repository, "branch", "-D", "release/v0.1.4")
    before = git(repository, "show-ref")
    flow(repository, dry_run=True, tag=True).execute()
    assert git(repository, "show-ref") == before


@pytest.fixture
def stale_clone(repository, tmp_path_factory):
    """A clone made before the release branch exists, so its objects are not local."""
    base = tmp_path_factory.mktemp("clone")
    git(base, "clone", "-q", str(repository / "remote.git"), str(base / "c"))
    clone = base / "c"
    for key, value in (
        ("user.email", "c@x.test"),
        ("user.name", "C"),
        ("commit.gpgsign", "false"),
        ("tag.gpgsign", "false"),
    ):
        git(clone, "config", key, value)
    flow(repository, push=True).execute()
    return clone


def test_resume_in_a_clone_that_has_to_fetch(stale_clone):
    flow(stale_clone, tag=True).execute()
    assert "refs/tags/v0.1.4" in remote_refs(stale_clone)


def test_a_dry_run_in_a_clone_says_it_could_not_verify(stale_clone, capsys):
    flow(stale_clone, dry_run=True).execute()
    out = capsys.readouterr().out
    assert "requires fetching" in out
    assert "(not verified)" in out


def test_a_dry_run_checks_main_against_origin(repository):
    (repository / "new.txt").write_text("w")
    git(repository, "add", "new.txt")
    git(repository, "commit", "-m", "Not pushed")
    with pytest.raises(ValueError, match="main must match origin/main"):
        flow(repository, dry_run=True).execute()


def test_a_dry_run_of_an_interrupted_release_checks_the_files(repository, monkeypatch):
    interrupted(repository, monkeypatch)
    with (repository / "pyproject.toml").open("a") as handle:
        handle.write("# user edit\n")
    with pytest.raises(ValueError, match="changed after release preparation"):
        flow(repository, dry_run=True).execute()


def test_a_dry_run_of_an_interrupted_release_plans_the_resume(repository, monkeypatch, capsys):
    interrupted(repository, monkeypatch)
    flow(repository, dry_run=True).execute()
    assert "Would resume interrupted preparation" in capsys.readouterr().out


def test_dry_run_plan_for_a_snapshot_tag_pushes_no_branch(repository, capsys):
    feature_branch(repository)
    flow(repository, version=None, bump="snapshot", tag=True, dry_run=True).execute()
    out = capsys.readouterr().out
    assert "Branch push: not requested" in out
    assert "Would switch back to feature/work" in out


def test_dry_run_plan_for_a_release_tags(repository, monkeypatch, capsys):
    monkeypatch.setattr(RF, "github_repo", lambda self: "o/r")
    monkeypatch.setattr(RF, "existing_release", lambda self: None)
    flow(repository, release=True, dry_run=True).execute()
    out = capsys.readouterr().out
    assert "Tag: ensure annotated tag and push" in out
    assert "Branch push: requested" in out


def test_a_note_is_not_printed_within_the_same_series(repository, capsys):
    commit_main(repository, "Development release", "0.1.4.dev0")
    flow(repository, version="0.1.4.dev1", dry_run=True).execute()
    assert "development series" not in capsys.readouterr().out


def test_the_closing_hint_does_not_ask_for_a_tag_that_is_already_pushed(repository, capsys):
    feature_branch(repository)
    flow(repository, version=None, bump="snapshot", tag=True).execute()
    capsys.readouterr()
    flow(repository, version=git(repository, "tag", "-l").removeprefix("v")).execute()
    assert "Tag and push it" not in capsys.readouterr().out


# --- GitHub ------------------------------------------------------------------------------------


def fake_gh(tmp_path, monkeypatch, body):
    bin_dir = tmp_path / "ghbin"
    bin_dir.mkdir(exist_ok=True)
    gh = bin_dir / "gh"
    gh.write_text("#!/bin/sh\n" + body)
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def release_flow(repository, version="0.1.4"):
    runner = flow(repository, version=version, release=True)
    runner.repo, runner.tag = "o/r", f"v{version}"
    return runner


def release_json(tag="v0.1.4", draft="false", prerelease="false"):
    body = f'{{"tag_name":"{tag}","draft":{draft},"prerelease":{prerelease},"html_url":"U"}}'
    return f"echo '{body}'\n"


NOT_FOUND = 'case "$2" in repos/*/releases/tags/*) echo "gh: Not Found (HTTP 404)" >&2; exit 1;; '


def test_existing_release_prerelease_mismatch(repository, tmp_path, monkeypatch):
    fake_gh(tmp_path, monkeypatch, release_json(prerelease="true"))
    with pytest.raises(ValueError, match="needs a stable release"):
        release_flow(repository).existing_release()


def test_existing_release_of_a_development_build_must_be_a_prerelease(
    repository, tmp_path, monkeypatch
):
    fake_gh(tmp_path, monkeypatch, release_json(tag="v0.1.4.dev0"))
    with pytest.raises(ValueError, match="needs a prerelease"):
        release_flow(repository, "0.1.4.dev0").existing_release()


def test_existing_release_wrong_tag(repository, tmp_path, monkeypatch):
    fake_gh(tmp_path, monkeypatch, release_json(tag="v0.1.5"))
    with pytest.raises(ValueError, match="unexpected tag"):
        release_flow(repository).existing_release()


def test_existing_release_draft_on_the_tags_endpoint(repository, tmp_path, monkeypatch):
    fake_gh(tmp_path, monkeypatch, release_json(draft="true"))
    with pytest.raises(ValueError, match="is a draft"):
        release_flow(repository).existing_release()


def test_existing_release_draft_found_by_listing(repository, tmp_path, monkeypatch):
    fake_gh(tmp_path, monkeypatch, NOT_FOUND + "*) echo U;; esac\n")
    with pytest.raises(ValueError, match="is a draft"):
        release_flow(repository).existing_release()


def test_existing_release_none_when_404_and_no_draft(repository, tmp_path, monkeypatch):
    fake_gh(tmp_path, monkeypatch, NOT_FOUND + "esac\n")
    assert release_flow(repository).existing_release() is None


def test_existing_release_is_returned(repository, tmp_path, monkeypatch):
    fake_gh(tmp_path, monkeypatch, release_json())
    assert release_flow(repository).existing_release()["html_url"] == "U"


def test_existing_release_fails_closed_on_other_errors(repository, tmp_path, monkeypatch):
    fake_gh(
        tmp_path,
        monkeypatch,
        'case "$2" in repos/*/releases/tags/*) echo "HTTP 500" >&2; exit 1;; esac\n',
    )
    with pytest.raises(ValueError, match="Cannot inspect GitHub release"):
        release_flow(repository).existing_release()
    fake_gh(tmp_path, monkeypatch, NOT_FOUND + '*) echo "HTTP 500" >&2; exit 1;; esac\n')
    with pytest.raises(ValueError, match="Cannot inspect GitHub release"):
        release_flow(repository).existing_release()


def test_github_repo_asks_gh_about_the_remote_slug(repository, tmp_path, monkeypatch):
    git(repository, "remote", "set-url", "origin", "git@github.com-ronpik:ronpik/jserpy.git")
    log = tmp_path / "gh.log"
    body = f'echo "$@" >> {log}\necho \'{{"nameWithOwner":"renamed/jserpy"}}\'\n'
    fake_gh(tmp_path, monkeypatch, body)
    assert flow(repository).github_repo() == "renamed/jserpy"
    assert "repo view ronpik/jserpy" in log.read_text()


@pytest.mark.parametrize(
    ("state", "created"), [("OPEN", False), ("MERGED", False), ("CLOSED", True)]
)
def test_an_existing_pr_is_reused_unless_closed(repository, monkeypatch, state, created):
    calls = []

    def github(self, *args):
        calls.append(args)
        listing = json.dumps([{"state": state, "url": "U"}])
        return subprocess.CompletedProcess(
            args, 0, listing if args[:2] == ("pr", "list") else "PR", ""
        )

    monkeypatch.setattr(RF, "github_repo", lambda self: "o/r")
    monkeypatch.setattr(RF, "github", github)
    flow(repository, pr=True).execute()
    assert any(call[:2] == ("pr", "create") for call in calls) == created
    listing = next(call for call in calls if call[:2] == ("pr", "list"))
    assert listing[listing.index("--state") + 1] == "all"


def test_a_release_pushes_before_it_asks_github_to_publish(repository, monkeypatch):
    seen = []
    monkeypatch.setattr(RF, "github_repo", lambda self: "o/r")
    monkeypatch.setattr(RF, "existing_release", lambda self: None)

    def github(self, *args):
        seen.append(("release", {"refs/tags/v0.1.4"} <= remote_refs(self.root)))
        return subprocess.CompletedProcess(args, 0, "URL", "")

    monkeypatch.setattr(RF, "github", github)
    flow(repository, release=True).execute()
    assert seen == [("release", True)]


# --- Validation, in detail ---------------------------------------------------------------------


def test_validate_tests_on_the_oldest_supported_python(repository, monkeypatch):
    test = validate(repository, monkeypatch, fake_build("0.1.4"))[1]
    assert test[test.index("--python") + 1] == "3.10"
    assert test[-7:] == (
        "--with",
        "pytest",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--import-mode=importlib",
    )


def test_validate_needs_a_wheel_and_an_sdist(repository, monkeypatch):
    def only_wheel(command):
        fake_build("0.1.4")(command)
        next(Path(command[command.index("--out-dir") + 1]).glob("*.tar.gz")).unlink()

    with pytest.raises(ValueError, match="one wheel and one sdist"):
        validate(repository, monkeypatch, only_wheel)


def test_validate_builds_a_clean_export_not_the_working_tree(repository, monkeypatch):
    """Untracked files must not reach the build; the release edit must."""
    (repository / "stray.py").write_text("print('not committed')")
    path = repository / "pyproject.toml"
    path.write_text(path.read_text().replace('"0.1.3"', '"0.1.4"', 1))
    seen = {}

    def run(self, *command, check=True, cwd=None):
        if command[:2] == ("uv", "build"):
            seen["cwd"] = cwd
            seen["files"] = sorted(p.name for p in Path(cwd).iterdir())
            seen["pyproject"] = (Path(cwd) / "pyproject.toml").read_text()
            fake_build("0.1.4")(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(RF, "run", run)
    runner = flow(repository)
    runner.name = "jserpy"
    REAL_VALIDATE(runner)
    assert Path(seen["cwd"]) != repository
    assert "stray.py" not in seen["files"] and "pyproject.toml" in seen["files"]
    assert 'version = "0.1.4"' in seen["pyproject"]
    assert not Path(seen["cwd"]).exists()  # Cleaned up.


def test_validate_runs_pytest_in_the_export_too(repository, monkeypatch):
    cwds = []

    def run(self, *command, check=True, cwd=None):
        cwds.append((command[:2], cwd))
        if command[:2] == ("uv", "build"):
            fake_build("0.1.4")(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(RF, "run", run)
    runner = flow(repository)
    runner.name = "jserpy"
    REAL_VALIDATE(runner)
    assert cwds[0][1] == cwds[1][1] and Path(cwds[0][1]) != repository


@pytest.mark.parametrize(
    ("requires", "oldest"),
    [
        (">=3.10", "3.10"),
        (">=3.9,<4", "3.9"),
        (">=3.10.2", "3.10"),
        ("==3.11.*", "3.11"),
        ("~=3.12", "3.12"),
        (None, f"{sys.version_info.major}.{sys.version_info.minor}"),
        ("<4", f"{sys.version_info.major}.{sys.version_info.minor}"),
    ],
)
def test_oldest_python(requires, oldest):
    assert bump_version.oldest_python(requires) == oldest


# --- prepare, in detail ------------------------------------------------------------------------


def test_prepare_ignores_version_keys_outside_the_project_table(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[tool.before]\nversion = "8.8.8"\n\n' + PYPROJECT)
    _, _, edits = bump_version.prepare(tmp_path, "0.1.4", None)
    text = next(iter(edits.values()))
    assert tomllib.loads(text)["tool"]["before"]["version"] == "8.8.8"
    assert tomllib.loads(text)["project"]["version"] == "0.1.4"


@pytest.mark.parametrize(
    "spelling", ["version = '0.1.3'", '  version = "0.1.3"', '"version" = "0.1.3"']
)
def test_prepare_refuses_version_spellings_it_cannot_rewrite(tmp_path, spelling):
    (tmp_path / "pyproject.toml").write_text(PYPROJECT.replace('version = "0.1.3"', spelling, 1))
    with pytest.raises(ValueError, match=r"exactly one line of the form version"):
        bump_version.prepare(tmp_path, "0.1.4", None)


@pytest.mark.parametrize("spelling", ['version="0.1.3"', 'version = "0.1.3"  # current'])
def test_prepare_rewrites_the_other_common_spellings(tmp_path, spelling):
    (tmp_path / "pyproject.toml").write_text(PYPROJECT.replace('version = "0.1.3"', spelling, 1))
    _, _, edits = bump_version.prepare(tmp_path, "0.1.4", None)
    assert tomllib.loads(next(iter(edits.values())))["project"]["version"] == "0.1.4"


# --- The command line --------------------------------------------------------------------------


def cli(monkeypatch, repository, *argv):
    monkeypatch.setattr(bump_version, "ROOT", repository)
    monkeypatch.setattr(sys, "argv", ["bump_version.py", *argv])
    bump_version.main()


def test_cli_patch_dev(repository, monkeypatch, capsys):
    cli(monkeypatch, repository, "--patch", "--dev", "--dry-run")
    assert "Target: 0.1.4.dev0 (development prerelease)" in capsys.readouterr().out


def test_cli_final_flag(repository, monkeypatch, capsys):
    commit_main(repository, "Development release", "0.1.4.dev0")
    cli(monkeypatch, repository, "--final", "--dry-run")
    assert "Target: 0.1.4 (stable release)" in capsys.readouterr().out


def test_cli_snapshot_flag(repository, monkeypatch, capsys):
    cli(monkeypatch, repository, "--snapshot", "--dry-run")
    assert "(snapshot, git tag only)" in capsys.readouterr().out


def test_cli_release_through_main(repository, monkeypatch, capsys):
    cli(monkeypatch, repository, "--patch", "--tag")
    assert {"refs/heads/release/v0.1.4", "refs/tags/v0.1.4"} <= remote_refs(repository)
    assert "Merge the release branch into main" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ((), "is required"),
        (("--dev", "--final"), "--dev and --final are mutually exclusive"),
        (("--dev", "--snapshot"), "--dev and --snapshot are mutually exclusive"),
        (("--dev", "--version", "0.1.4"), "--version already names the build"),
        (("--patch", "--minor"), "not allowed with"),
        (
            ("--patch", "--rel"),
            "unrecognized arguments",
        ),  # No abbreviations: --rel is not --release.
    ],
)
def test_cli_rejects_bad_requests(repository, monkeypatch, capsys, argv, message):
    with pytest.raises(SystemExit) as exit_info:
        cli(monkeypatch, repository, *argv)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_runtime_errors_are_one_line_without_the_usage_block(repository, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli(monkeypatch, repository, "--dev", "--dry-run")
    assert str(exit_info.value.code).startswith("bump_version.py: error: --dev alone continues")
    assert "usage:" not in capsys.readouterr().err


def run_script(code):
    """Run the script's import section under a doctored interpreter."""
    script = ROOT / "scripts/bump_version.py"
    return subprocess.run(
        [sys.executable, "-c", f"import runpy; {code}; runpy.run_path({str(script)!r})"],
        capture_output=True,
        text=True,
    )


def test_a_missing_packaging_gets_a_pointer_to_uv():
    result = run_script("import sys; sys.modules['packaging'] = None")
    assert result.returncode == 1
    assert "needs the 'packaging' package" in result.stderr and "uv run" in result.stderr
    assert "Traceback" not in result.stderr


def test_an_old_python_gets_a_pointer_to_uv():
    result = run_script("import sys; sys.modules['tomllib'] = None")
    assert result.returncode == 1
    assert "needs Python 3.11 or later" in result.stderr and "Traceback" not in result.stderr


def test_the_examples_in_the_help_name_real_options():
    parser_options = {"--patch", "--dry-run", "--pr", "--version", "--release", "--dev", "--final"}
    parser_options |= {"--snapshot", "--tag"}
    used = set()
    for line in bump_version.EXAMPLES.splitlines():
        if "bump_version.py" in line:
            used |= {word for word in line.split() if word.startswith("--")}
    assert used <= parser_options


# --- Second round: what the real repository and long-running use showed ---------------------------


def test_a_lightweight_tag_is_a_taken_version_not_a_dead_end(repository):
    """This repository's real v0.1.4 is a lightweight tag on main while pyproject says 0.1.3."""
    git(repository, "tag", "v0.1.4")
    git(repository, "push", "origin", "v0.1.4")
    with pytest.raises(ValueError, match="higher explicit --version, for example 0.1.5"):
        flow(repository, version=None, bump="patch", dry_run=True).execute()
    git(repository, "tag", "-d", "v0.1.4")  # Remote only now: a dry run cannot fetch it.
    with pytest.raises(ValueError, match="higher explicit --version, for example 0.1.5"):
        flow(repository, version=None, bump="patch", dry_run=True).execute()
    with pytest.raises(ValueError, match="is a lightweight tag"):
        flow(repository, version="0.1.4", tag=True).execute()  # Explicit: reuse is refused.
    flow(repository, version="0.1.5", tag=True).execute()


def test_an_abandoned_preparation_is_offered_for_resuming_or_dropping(repository, monkeypatch):
    interrupted(repository, monkeypatch)
    git(repository, "restore", "pyproject.toml")
    git(repository, "switch", "main")
    with pytest.raises(
        ValueError, match="already prepared on release/v0.1.4 but not committed"
    ) as e:
        flow(repository, version=None, bump="patch").execute()
    assert "git branch -D release/v0.1.4" in str(e.value)


def test_the_resume_hint_repeats_the_requested_flags(repository, monkeypatch):
    monkeypatch.setattr(RF, "validate", lambda self: (_ for _ in ()).throw(ValueError("nope")))
    monkeypatch.setattr(RF, "github_repo", lambda self: "o/r")
    monkeypatch.setattr(RF, "existing_release", lambda self: None)
    with pytest.raises(ValueError, match=r"re-run with --version 0.1.4 --tag --release to resume"):
        flow(repository, tag=True, release=True).execute()


def test_a_commit_that_fails_verification_is_not_reported_as_uncommitted(repository):
    hook = repository / ".git/hooks/commit-msg"
    hook.write_text('#!/bin/sh\necho Tweaked > "$1"\n')
    hook.chmod(0o755)
    with pytest.raises(
        ValueError, match="A commit was made on release/v0.1.4 but is not a valid"
    ) as e:
        flow(repository).execute()
    assert "not committed" not in str(e.value) and "git branch -D release/v0.1.4" in str(e.value)


def test_the_drop_hint_quotes_paths(repository, monkeypatch):
    monkeypatch.setattr(RF, "validate", lambda self: (_ for _ in ()).throw(ValueError("nope")))
    runner = flow(repository)
    with pytest.raises(ValueError):
        runner.execute()
    runner.state_path = repository / "dir with space" / "journal.json"
    assert "rm '" in runner.drop_hint({})


def test_ctrl_c_during_validation_leaves_a_hint_instead_of_a_traceback(repository, monkeypatch):
    def interrupt(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(RF, "validate", interrupt)
    with pytest.raises(ValueError, match="Interrupted\nRelease 0.1.4 is prepared on"):
        flow(repository).execute()
    monkeypatch.setattr(RF, "validate", lambda self: None)
    flow(repository).execute()  # Resumes.
    assert git(repository, "log", "-1", "--format=%s") == "Release jserpy 0.1.4"


def test_main_reports_ctrl_c_in_one_line(repository, monkeypatch, capsys):
    monkeypatch.setattr(RF, "execute", lambda self: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(SystemExit) as exit_info:
        cli(monkeypatch, repository, "--patch")
    assert exit_info.value.code == "bump_version.py: interrupted"


def test_the_hint_says_when_a_tracked_file_has_to_change(repository, monkeypatch):
    monkeypatch.setattr(RF, "validate", lambda self: (_ for _ in ()).throw(ValueError("nope")))
    with pytest.raises(ValueError, match="If a tracked file has to change, drop the release"):
        flow(repository).execute()


def test_a_second_run_for_the_same_version_is_refused(repository):
    runner = flow(repository)
    runner.version = "0.1.4"
    runner.state_path = repository / ".git" / "jserpy-release-0.1.4.json"
    lock = runner.lock()
    try:
        with pytest.raises(ValueError, match="Another run for 0.1.4 is going"):
            flow(repository).execute()
        assert git(repository, "branch", "--show-current") == "main"
    finally:
        lock.unlink()
    flow(repository).execute()
    assert not list((repository / ".git").glob("jserpy-release-*.lock"))  # Released after a run.


def test_a_lock_of_a_dead_process_is_taken_over(repository):
    runner = flow(repository)
    runner.version = "0.1.4"
    runner.state_path = repository / ".git" / "jserpy-release-0.1.4.json"
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    runner.state_path.with_suffix(".lock").write_text(str(dead.pid))
    flow(repository).execute()


def test_a_dry_run_takes_no_lock(repository):
    runner = flow(repository, dry_run=True)
    runner.execute()
    assert not list((repository / ".git").glob("jserpy-release-*.lock"))


def test_the_lock_is_released_when_the_run_fails(repository, monkeypatch):
    monkeypatch.setattr(RF, "validate", lambda self: (_ for _ in ()).throw(ValueError("nope")))
    with pytest.raises(ValueError):
        flow(repository).execute()
    assert not list((repository / ".git").glob("jserpy-release-*.lock"))


@pytest.mark.parametrize(
    "journal",
    ["not json", "[]", '{"base": "x"}', "\u00ff\u00fe".encode("latin-1").decode("latin-1")],
)
def test_a_journal_of_the_wrong_shape_is_named(repository, journal):
    flow(repository).execute()
    path = next((repository / ".git").glob("jserpy-release-*.json"))
    path.write_bytes(journal.encode("latin-1"))
    with pytest.raises(ValueError, match="is not a valid journal"):
        flow(repository, tag=True).execute()


@pytest.mark.parametrize(
    "header", ["[project]  # the metadata", "[ project ]", "[project]\t", "[ project ] # x"]
)
def test_prepare_accepts_toml_spellings_of_the_project_header(tmp_path, header):
    (tmp_path / "pyproject.toml").write_text(PYPROJECT.replace("[project]", header, 1))
    _, _, edits = bump_version.prepare(tmp_path, "0.1.4", None)
    assert tomllib.loads(next(iter(edits.values())))["project"]["version"] == "0.1.4"


def test_prepare_names_invalid_toml(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project\nname = ")
    with pytest.raises(ValueError, match="not valid TOML"):
        bump_version.prepare(tmp_path, "0.1.4", None)


def test_prepare_explains_that_a_declared_version_cannot_be_released(tmp_path):
    write_tree(tmp_path, "0.1.5")
    with pytest.raises(ValueError, match="already declares 0.1.5.*tag main by hand"):
        bump_version.prepare(tmp_path, "0.1.5", None)
    with pytest.raises(ValueError, match="must be greater than 0.1.5"):
        bump_version.prepare(tmp_path, "0.1.4", None)


def test_an_untagged_declared_version_is_noted_before_a_bump_skips_it(repository, capsys):
    flow(repository, version=None, bump="patch", dry_run=True).execute()
    assert "declares 0.1.3, but no tag v0.1.3 exists" in capsys.readouterr().out
    git(repository, "tag", "-a", "v0.1.3", "-m", "x")
    flow(repository, version=None, bump="patch", dry_run=True).execute()
    assert "no tag" not in capsys.readouterr().out
    git(repository, "tag", "-d", "v0.1.3")
    git(repository, "tag", "-a", "0.1.3", "-m", "legacy spelling without the v")
    flow(repository, version=None, bump="patch", dry_run=True).execute()
    assert "no tag" not in capsys.readouterr().out


def test_no_note_for_an_explicit_version(repository, capsys):
    flow(repository, dry_run=True).execute()
    assert "Note: pyproject.toml declares" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("requires", "oldest"),
    [
        (">3.9", "3.9"),
        (">=3.9,>=3.10", "3.10"),  # Every lower bound applies.
        ("==3.10.*,>=3.9", "3.10"),
        (">=3.8,!=3.8.*", "3.8"),
        ("!=3.10.0", f"{sys.version_info.major}.{sys.version_info.minor}"),
    ],
)
def test_oldest_python_with_several_bounds(requires, oldest):
    assert bump_version.oldest_python(requires) == oldest


def test_a_tracked_symlink_to_an_absolute_path_is_exported(repository, monkeypatch):
    os.symlink("/etc", repository / "docs_link")
    commit_main(repository, "Add a link")
    seen = {}

    def run(self, *command, check=True, cwd=None):
        if command[:2] == ("uv", "build"):
            seen["link"] = os.readlink(Path(cwd) / "docs_link")
            fake_build("0.1.4")(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(RF, "run", run)
    runner = flow(repository)
    runner.name = "jserpy"
    REAL_VALIDATE(runner)
    assert seen["link"] == "/etc"


def test_an_export_that_fails_is_a_clean_error(repository):
    runner = flow(repository)
    with pytest.raises(ValueError, match="Cannot export the tracked tree"):
        runner.export_tree(repository / "pyproject.toml")  # A file is no destination.


def test_a_corrupt_built_artifact_is_a_clean_error(tmp_path):
    (tmp_path / "x-1-py3-none-any.whl").write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="Cannot read the built x-1-py3-none-any.whl"):
        bump_version.built_version(tmp_path / "x-1-py3-none-any.whl")


def test_validate_imports_the_wheel_not_the_exported_sources(repository, monkeypatch):
    test = validate(repository, monkeypatch, fake_build("0.1.4"))[1]
    assert "--import-mode=importlib" in test


def test_the_refusals_name_the_refs_involved(repository):
    flow(repository, push=True).execute()
    git(repository, "tag", "-a", "v0.1.4", "-m", "x", "main")  # Tag and branch differ.
    with pytest.raises(
        ValueError, match=r"release/v0.1.4 and v0.1.4 point to different commits \(.*,"
    ):
        flow(repository, tag=True).execute()


def test_a_release_only_on_the_remote_is_not_mistaken_for_a_foreign_one(stale_clone):
    with pytest.raises(ValueError, match="not fetched here to check; if it is a release of this"):
        flow(stale_clone, version=None, bump="patch", dry_run=True).execute()
