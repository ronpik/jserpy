# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging>=24"]
# ///
"""Prepare, validate and commit a resumable release on release/vX.Y.Z or release/vX.Y.Z.devN.

Stable and development releases start from a clean main that matches origin/main. A snapshot
(--snapshot) cuts X.Y.Z[.devN]+g<10 hex of HEAD> from any clean, non-release branch (or a detached
HEAD); it is published by its git tag alone and never merged back.

Publishing to PyPI is done by the "Upload Python Package" workflow when a GitHub release is
published, so --release is what ships a version; PyPI rejects snapshots (local version labels),
which therefore stop at the tag.

Needs git, uv and (for --release and --pr) an authenticated gh. Run it with uv, which installs the
script's own dependency: uv run scripts/bump_version.py --patch
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

try:
    import tomllib
    from packaging.specifiers import SpecifierSet
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version
except ImportError as error:  # Exercised through a subprocess in the tests.
    if error.name == "tomllib":
        raise SystemExit(
            "bump_version.py needs Python 3.11 or later; run it as "
            "'uv run scripts/bump_version.py ...' so uv picks one"
        ) from None
    raise SystemExit(
        "bump_version.py needs the 'packaging' package; run it as "
        "'uv run scripts/bump_version.py ...' so uv installs it from the script's inline metadata"
    ) from None

ROOT = Path(__file__).resolve().parents[1]
BUMPS = ("patch", "minor", "major", "patch-dev", "minor-dev", "major-dev", "dev", "final")
SNAPSHOT_HASH = 10  # Hex characters of the base commit's full sha carried by a snapshot version.
# A snapshot spells its base version plus a PEP 440 local label: "g" and the hash prefix.
SNAPSHOT = re.compile(rf"(?P<public>[^+]+)\+g(?P<hash>[0-9a-f]{{{SNAPSHOT_HASH}}})")
# The branches this script creates; other release/* branches (release/0.1/fix) are ordinary ones.
RELEASE_BRANCH = re.compile(rf"release/v\d+\.\d+\.\d+(\.dev\d+)?(\+g[0-9a-f]{{{SNAPSHOT_HASH}}})?")
# owner/repo of a git remote URL: https://host/o/r.git, ssh://git@host/o/r.git or git@host:o/r.git.
# The host is not interpreted, since an ~/.ssh/config alias such as github.com-work names none.
REMOTE_URL = re.compile(
    r"^(?:[a-z][a-z0-9+.-]*://)?(?:[^@/\s]+@)?[^:/\s]+[:/](?P<owner>[^/\s]+)/(?P<name>[^/\s]+?)"
    r"(?:\.git)?/?$"
)
EXAMPLES = """\
examples (from main, whose pyproject.toml declares 0.1.3):
  uv run scripts/bump_version.py --patch --dry-run              # 0.1.3 -> 0.1.4, plan only
  uv run scripts/bump_version.py --patch --pr                   # prepare, push and open the PR
  uv run scripts/bump_version.py --version 0.1.4 --release      # tag, push and publish to PyPI
  uv run scripts/bump_version.py --patch --dev --dry-run        # 0.1.3 -> 0.1.4.dev0, plan only
  uv run scripts/bump_version.py --dev --dry-run                # 0.1.4.dev0 -> 0.1.4.dev1
  uv run scripts/bump_version.py --final --dry-run              # 0.1.4.devN -> 0.1.4
examples (from any clean branch or detached HEAD):
  uv run scripts/bump_version.py --snapshot --dry-run           # plan 0.1.3+g<10 hex of HEAD>
  uv run scripts/bump_version.py --snapshot --tag               # cut it, push only its tag
examples (resuming, with the version the earlier run printed):
  uv run scripts/bump_version.py --version 0.1.4 --tag
"""


def parse_version(value: str) -> Version:
    """Accept only the canonical stable X.Y.Z and development X.Y.Z.devN spellings."""
    try:
        parsed = Version(value)
    except InvalidVersion:
        parsed = None
    if (
        parsed is None
        or str(parsed) != value
        or len(parsed.release) != 3
        or parsed.epoch
        or parsed.pre is not None
        or parsed.post is not None
        or parsed.local is not None
    ):
        raise ValueError(f"Expected a canonical X.Y.Z or X.Y.Z.devN version, got {value!r}")
    return parsed


def split_snapshot(value: str) -> tuple[str, str] | None:
    """Split a snapshot X.Y.Z[.devN]+g<hash> into (public version, hash); None otherwise.

    The single parser of the snapshot spelling; snapshot_version is its single builder.
    """
    match = SNAPSHOT.fullmatch(value)
    if match is None:
        return None
    try:
        parse_version(match["public"])
    except ValueError:
        return None
    return match["public"], match["hash"]


def snapshot_version(current: str, sha: str) -> str:
    """Build the snapshot version of commit sha (a full sha) on top of the canonical current."""
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError(f"Expected a full commit sha, got {sha!r}")
    return f"{parse_version(current)}+g{sha[:SNAPSHOT_HASH]}"


def current_version(project: dict) -> str:
    """The version pyproject.toml declares, unless the tree stands on a snapshot release commit."""
    current = project.get("version")
    if not isinstance(current, str):
        raise ValueError("pyproject.toml must declare a static [project] version")
    if split_snapshot(current) is not None:
        raise ValueError(
            f"pyproject.toml declares the snapshot version {current}; a snapshot release commit "
            "is never built upon or merged, so switch to the branch to release from (and revert "
            "the merge that brought it here, if any)"
        )
    return current


def next_version(current: str, bump: str | None) -> str:
    """Infer the target from the current version and a bump mode (one of BUMPS)."""
    parsed = parse_version(current)
    major, minor, patch = parsed.release
    if bump not in BUMPS:
        raise ValueError("Choose a version or a bump level")
    if bump == "dev":
        if not parsed.is_devrelease:
            raise ValueError(
                f"--dev alone continues a development series, but {current} is stable; use "
                f"--patch --dev for {major}.{minor}.{patch + 1}.dev0, --minor --dev for "
                f"{major}.{minor + 1}.0.dev0 or --major --dev for {major + 1}.0.0.dev0"
            )
        return f"{parsed.base_version}.dev{parsed.dev + 1}"
    if bump == "final":
        if not parsed.is_devrelease:
            raise ValueError(
                f"--final promotes a development version, but {current} is already stable"
            )
        return parsed.base_version
    if parsed.is_devrelease:
        flag = "--" + bump.replace("-dev", " --dev")
        raise ValueError(
            f"{flag} is ambiguous from development version {current}; use --dev for "
            f"{parsed.base_version}.dev{parsed.dev + 1}, --final for {parsed.base_version}, "
            "or an explicit --version"
        )
    level, _, suffix = bump.partition("-")
    if level == "major":
        target = f"{major + 1}.0.0"
    elif level == "minor":
        target = f"{major}.{minor + 1}.0"
    else:
        target = f"{major}.{minor}.{patch + 1}"
    return f"{target}.dev0" if suffix == "dev" else target


def read_text(path: Path) -> str:
    """Read a file without translating its line endings (a CRLF pyproject.toml stays CRLF)."""
    return path.read_bytes().decode("utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def is_release_branch(name: str) -> bool:
    """Whether name is a branch this script makes: release/vX.Y.Z, .devN or a snapshot's."""
    return RELEASE_BRANCH.fullmatch(name) is not None


def oldest_python(requires: str | None) -> str:
    """The oldest minor Python that requires-python admits, else the one running this script."""
    # Every lower bound applies, so the highest one is the floor; an upper bound or != names none.
    bounds = [
        Version(spec.version.rstrip(".*")).release[:2]
        for spec in SpecifierSet(requires or "")
        if spec.operator in (">=", ">", "==", "~=")
    ]
    return ".".join(str(part) for part in max(bounds, default=sys.version_info[:2]))


def project_table(text: str) -> dict:
    """The [project] table of a pyproject.toml text."""
    try:
        project = tomllib.loads(text).get("project")
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"pyproject.toml is not valid TOML: {error}") from None
    if not isinstance(project, dict) or not isinstance(project.get("name"), str):
        raise ValueError("pyproject.toml must have a [project] table with a name")
    return project


def repo_slug(url: str) -> str:
    """The owner/repo that a git remote URL names, whatever host alias it goes through."""
    match = REMOTE_URL.match(url.strip())
    if match is None:
        raise ValueError(f"Cannot tell the GitHub repository from the remote URL {url!r}")
    return f"{match['owner']}/{match['name']}"


def built_version(path: Path) -> tuple[str, str]:
    """(name, version) that a built wheel or sdist declares in its metadata."""
    try:
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as wheel:
                names = [
                    n for n in wheel.namelist() if n.count("/") == 1 and n.endswith("/METADATA")
                ]
                if len(names) != 1:
                    raise ValueError(f"{path.name} does not hold exactly one METADATA file")
                text = wheel.read(names[0]).decode("utf-8")
        else:
            with tarfile.open(path) as sdist:
                names = [
                    n for n in sdist.getnames() if n.count("/") == 1 and n.endswith("/PKG-INFO")
                ]
                if len(names) != 1:
                    raise ValueError(f"{path.name} does not hold exactly one PKG-INFO file")
                text = sdist.extractfile(names[0]).read().decode("utf-8")
    except (zipfile.BadZipFile, tarfile.TarError, OSError) as error:
        raise ValueError(f"Cannot read the built {path.name}: {error}") from None
    metadata = Parser().parsestr(text)
    return metadata["Name"], metadata["Version"]


def prepare(root: Path, target: str | None, bump: str | None) -> tuple[str, str, dict[Path, str]]:
    """Validate all inputs and prepare edits before writing any files."""
    project_path = root / "pyproject.toml"
    project_text = read_text(project_path)
    project = project_table(project_text)
    current = current_version(project)
    if target is None:
        target = next_version(current, bump)
    snapshot = split_snapshot(target)
    if snapshot is not None:
        if snapshot[0] != current:
            raise ValueError(
                f"Snapshot {target} is built on version {snapshot[0]}, but pyproject.toml "
                f"declares {current}"
            )
    elif parse_version(target) == parse_version(current):
        raise ValueError(
            f"pyproject.toml already declares {target}; this script commits the bump itself, so it "
            "cannot release a version that is already declared (tag main by hand for that) - "
            "pass a higher version"
        )
    elif parse_version(target) < parse_version(current):
        raise ValueError(f"New version {target} must be greater than {current}")

    # Limit the replacement to the [project] table; never replace dependency versions.
    header = r"^\[[ \t]*project[ \t]*\][ \t]*(?:#[^\n]*)?\r?\n"  # Spaces and a comment are TOML.
    project_pattern = re.compile(rf"(?ms)({header})(.*?)(?=^\[|\Z)")
    version_pattern = re.compile(r'(?m)^(version\s*=\s*)"[^"]+"')
    project_match = project_pattern.search(project_text)
    if project_match is None:
        raise ValueError("Missing [project] table")
    updated_table, count = version_pattern.subn(
        lambda match: f'{match[1]}"{target}"', project_match[2]
    )
    if count != 1:
        raise ValueError(
            'Expected exactly one line of the form version = "X.Y.Z" in the [project] table '
            f"of pyproject.toml, found {count}; the script rewrites only that spelling"
        )
    updated_project = (
        project_text[: project_match.start(2)]
        + updated_table
        + project_text[project_match.end(2) :]
    )
    return current, target, {project_path: updated_project}


class ReleaseFlow:
    """Reconcile local and remote release state without rewriting published history."""

    files = ("pyproject.toml",)

    def __init__(self, root, args):
        self.root, self.args = root, args
        self.version = args.version
        self.remote = "origin"
        self.main = "main"
        self.current = None  # The version pyproject.toml declares, once a bump inferred from it.

    @property
    def snapshot(self):
        """(public version, hash) of a snapshot target; None for a numbered version."""
        return split_snapshot(self.version)

    @property
    def public(self):
        """The target without its snapshot label."""
        snapshot = self.snapshot
        return snapshot[0] if snapshot else self.version

    @property
    def is_dev(self):
        """A development build X.Y.Z.devN, published as a GitHub prerelease (a snapshot is not)."""
        return self.snapshot is None and parse_version(self.version).is_devrelease

    @property
    def subject(self):
        """The release commit's message, which verify_commit recognises it by."""
        return f"Release {self.name} {self.version}"

    def note_series(self, current):
        """Warn when the target leaves a development series that --final would have promoted."""
        before = parse_version(current)
        if before.is_devrelease and before.base_version != parse_version(self.public).base_version:
            print(
                f"Note: {self.version} leaves the {before.base_version} development series "
                f"({current}); --final would have promoted it to {before.base_version}."
            )

    def run(self, *command, check=True, cwd=None):
        # Nested uv commands target the project environment, never the env uv gave this script.
        env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
        try:
            done = subprocess.run(command, cwd=cwd or self.root, env=env, capture_output=True)
        except FileNotFoundError:
            raise ValueError(f"{command[0]} is required but was not found on PATH") from None
        # Decode by hand: text mode would turn a CRLF file's lines into LF ones.
        result = subprocess.CompletedProcess(
            command,
            done.returncode,
            done.stdout.decode("utf-8", "replace"),
            done.stderr.decode("utf-8", "replace"),
        )
        if check and result.returncode:
            # Both streams: a failing pytest reports on stdout while uv chats on stderr.
            detail = "\n".join(
                part.rstrip() for part in (result.stdout, result.stderr) if part.strip()
            )
            raise ValueError(f"{' '.join(command)} failed:\n{detail}")
        return result

    def git(self, *args):
        return self.run("git", *args).stdout.rstrip("\n")

    def ref(self, ref):
        result = self.run("git", "show-ref", "--verify", "--quiet", ref, check=False)
        if result.returncode not in (0, 1):
            raise ValueError(result.stderr)
        return self.git("rev-parse", ref) if result.returncode == 0 else None

    def remote_refs(self):
        lines = self.git(
            "ls-remote",
            self.remote,
            f"refs/heads/{self.main}",
            self.branch_ref,
            self.tag_ref,
            self.tag_ref + "^{}",
        )
        return dict((ref, sha) for sha, ref in (line.split() for line in lines.splitlines()))

    def blob(self, commit, path):
        return self.run("git", "show", f"{commit}:{path}").stdout

    def check_tools(self):
        """Fail before anything changes when a program the run needs is missing."""
        needed = ["git"] + ([] if self.args.dry_run else ["uv"])
        needed += ["gh"] if self.args.release or self.args.pr else []
        missing = [tool for tool in needed if shutil.which(tool) is None]
        if missing:
            raise ValueError(f"{' and '.join(missing)} must be installed and on PATH")

    def verify_commit(self, commit):
        project = project_table(self.blob(commit, "pyproject.toml"))
        if project["name"] != self.name:
            raise ValueError(f"Existing release commit is for the project {project['name']}")
        if project.get("version") != self.version:
            raise ValueError("Existing release commit has a different package version")
        if self.git("log", "-1", "--format=%s", commit) != self.subject:
            raise ValueError("Existing branch/tag does not identify a release commit")
        changed = set(
            self.git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()
        )
        if changed != set(self.files):
            raise ValueError("Release commit must change exactly pyproject.toml")
        parents = self.git("rev-list", "--parents", "-n", "1", commit).split()
        if len(parents) != 2:
            raise ValueError("Expected a release commit with exactly one parent")
        base = project_table(self.blob(parents[1], "pyproject.toml")).get("version")
        snapshot = self.snapshot
        if snapshot is not None:
            if base != snapshot[0]:
                raise ValueError("Snapshot commit is not built on its parent's version")
            if not parents[1].startswith(snapshot[1]):
                raise ValueError("Snapshot version does not name the release commit's parent")
        elif parse_version(base) >= parse_version(self.version):
            raise ValueError("Release commit does not increase its parent's version")

        # Reconstruct the expected release from the parent, detecting any other edit.
        with tempfile.TemporaryDirectory(prefix="jserpy-release-verify-") as directory:
            root = Path(directory)
            for name in self.files:
                write_text(root / name, self.blob(parents[1], name))
            _, _, expected = prepare(root, self.version, None)
            for path, content in expected.items():
                if self.blob(commit, path.name) != content:
                    raise ValueError(f"Existing release commit has unexpected edits in {path.name}")

    def export_tree(self, destination):
        """The tracked files at HEAD plus the release edits: the tree CI would check out."""
        archived = subprocess.run(
            ["git", "archive", "--format=tar", "HEAD"], cwd=self.root, capture_output=True
        )
        if archived.returncode:
            raise ValueError(f"git archive failed:\n{archived.stderr.decode(errors='replace')}")
        try:
            with tarfile.open(fileobj=io.BytesIO(archived.stdout)) as tree:
                # The tar filter (Python 3.12 and 3.11.4) keeps symlinks as a checkout would, while
                # still refusing members outside the destination; older Pythons have no filters.
                tree.extractall(
                    destination, **({"filter": "tar"} if hasattr(tarfile, "tar_filter") else {})
                )
        except (tarfile.TarError, OSError) as error:
            raise ValueError(f"Cannot export the tracked tree of HEAD: {error}") from None
        for name in self.files:
            write_text(destination / name, read_text(self.root / name))

    def validate(self):
        """Build the release from a clean export and test the built wheel, never the sources."""
        project = project_table(read_text(self.root / "pyproject.toml"))
        # The oldest supported Python, so that newer syntax or APIs cannot slip into a release.
        python = oldest_python(project.get("requires-python"))
        with tempfile.TemporaryDirectory(prefix="jserpy-release-") as scratch:
            tree, dist = Path(scratch) / "tree", Path(scratch) / "dist"
            tree.mkdir()
            dist.mkdir()
            self.export_tree(tree)
            print("Running: uv build", flush=True)
            self.run("uv", "build", "--out-dir", str(dist), cwd=tree)  # The sdist, then its wheel.
            artifacts = sorted(dist.glob("*.whl")) + sorted(dist.glob("*.tar.gz"))
            if len(artifacts) != 2:
                raise ValueError(f"Expected one wheel and one sdist, built {len(artifacts)} files")
            for artifact in artifacts:
                name, version = built_version(artifact)
                if canonicalize_name(name) != canonicalize_name(self.name):
                    raise ValueError(f"{artifact.name} declares the name {name}, not {self.name}")
                if Version(version) != Version(self.version):
                    raise ValueError(
                        f"{artifact.name} declares version {version}, expected {self.version}"
                    )
            wheel = next(dist.glob("*.whl"))
            command = [
                "uv",
                "run",
                "--no-project",
                "--python",
                python,
                "--with",
                str(wheel),
                "--with",
                "pytest",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--import-mode=importlib",  # Never put the exported sources before the wheel.
            ]
            print("Running:", " ".join(command), flush=True)
            print(self.run(*command, cwd=tree).stdout, end="", flush=True)

    def save(self, state):
        temporary = self.state_path.with_suffix(".tmp")
        write_text(temporary, json.dumps(state))
        temporary.replace(self.state_path)

    def check_partial(self, state):
        if self.git("branch", "--show-current") != self.branch:
            raise ValueError(f"Resume interrupted preparation from {self.branch}")
        if self.git("rev-parse", "HEAD") != state["base"]:
            raise ValueError("Interrupted release base changed; refusing to overwrite work")
        # Only the original or exact generated bytes are accepted, including the index.
        status = self.git("status", "--porcelain", "--untracked-files=all")
        allowed = set(self.files)
        for line in status.splitlines():
            if line[3:] not in allowed:
                raise ValueError(
                    f"Unrelated work ({line[3:]}) prevents resuming release preparation"
                )
        for name in self.files:
            if read_text(self.root / name) not in (state["original"][name], state["edits"][name]):
                raise ValueError(
                    f"{name} changed after release preparation; refusing to overwrite it"
                )
            index = self.blob("", name)  # The staged content: ":name".
            if index not in (state["original"][name], state["edits"][name]):
                raise ValueError(f"Unexpected staged content in {name}")

    def flags(self):
        """The publishing flags of this run, to repeat when it has to be resumed."""
        asked = (("--push", self.args.push), ("--tag", self.args.tag))
        asked += (("--release", self.args.release), ("--pr", self.args.pr))
        return " ".join(flag for flag, on in asked if on)

    def resume_command(self):
        return f"--version {self.version} {self.flags()}".strip()

    def drop_hint(self, state):
        """The commands that drop an unfinished preparation; the script deletes no work itself."""
        back = self.return_to(state) if self.snapshot is not None else None
        leave = ["git", "switch", *(back[0] if back else [self.main])]
        return " && ".join(
            [
                f"git restore --staged --worktree -- {' '.join(self.files)}",
                shlex.join(leave),
                f"git branch -D {shlex.quote(self.branch)}",
                f"rm {shlex.quote(str(self.state_path))}",
            ]
        )

    def unfinished(self, state, action):
        """Run action; if it fails the release is left half prepared, so say what to do next."""
        try:
            return action()
        except (ValueError, KeyboardInterrupt) as error:
            reason = str(error) or "Interrupted"
            drop = self.drop_hint(state)
            if self.git("rev-parse", "HEAD") != state["base"]:  # The commit itself was made.
                advice = (
                    f"A commit was made on {self.branch} but is not a valid release commit "
                    f"(git show HEAD). Drop it with:\n  {drop}"
                )
            else:
                advice = (
                    f"Release {self.version} is prepared on {self.branch} but not committed. If "
                    f"the cause was outside the repository (network, uv, an interrupted run), "
                    f"re-run with {self.resume_command()} to resume it. If a tracked file has to "
                    f"change, drop the release with:\n  {drop}\nfix {self.main} and start again."
                )
            raise ValueError(f"{reason}\n{advice}") from None

    def lock(self):
        """Keep a second run for this version out; a lock whose process is gone is taken over."""
        path = self.state_path.with_suffix(".lock")
        for _ in range(2):
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    pid = int(read_text(path).strip())
                    os.kill(pid, 0)
                except ProcessLookupError:
                    path.unlink(missing_ok=True)  # Its run died without cleaning up.
                    continue
                except (OSError, ValueError):
                    pid = None
                raise ValueError(
                    f"Another run for {self.version} is going (process {pid or 'unknown'}); if "
                    f"none is, delete {path}"
                ) from None
            with os.fdopen(descriptor, "w") as handle:
                handle.write(str(os.getpid()))
            return path
        raise ValueError(f"Cannot take the lock {path}")

    def prepare_commit(self, state):
        self.check_partial(state)
        allowed = set(self.files)
        for name in self.files:
            write_text(self.root / name, state["edits"][name])
        self.validate()
        # Validation must not introduce or stage unrelated work.
        for line in self.git("status", "--porcelain", "--untracked-files=all").splitlines():
            if line[3:] not in allowed:
                raise ValueError("Validation left unrelated changes; release not committed")
        for name in self.files:
            if read_text(self.root / name) != state["edits"][name]:
                raise ValueError(f"Validation modified {name}; release not committed")
        self.git("add", "--", *self.files)
        self.git("commit", "-m", self.subject)
        commit = self.git("rev-parse", "HEAD")
        self.verify_commit(commit)
        state["commit"] = commit
        self.save(state)
        return commit

    def github(self, *args):
        return self.run("gh", *args, "--repo", self.repo)

    def github_repo(self):
        """owner/repo of the remote, as GitHub names it (a renamed repository resolves)."""
        slug = repo_slug(self.git("remote", "get-url", self.remote))
        view = self.run("gh", "repo", "view", slug, "--json", "nameWithOwner")
        return json.loads(view.stdout)["nameWithOwner"]

    def existing_release(self):
        draft = (
            f"GitHub release for {self.tag} is a draft; publish or delete it on GitHub, "
            "the script never edits releases"
        )
        result = self.run("gh", "api", f"repos/{self.repo}/releases/tags/{self.tag}", check=False)
        if result.returncode:
            if "HTTP 404" not in result.stderr:
                raise ValueError(f"Cannot inspect GitHub release: {result.stderr.strip()}")
            # The tags endpoint serves published releases only; a draft for the tag answers 404.
            drafts = self.run(
                "gh",
                "api",
                f"repos/{self.repo}/releases?per_page=100",
                "--paginate",
                "--jq",
                f'.[] | select(.draft and .tag_name == "{self.tag}") | .html_url',
                check=False,
            )
            if drafts.returncode and "HTTP 404" not in drafts.stderr:
                raise ValueError(f"Cannot inspect GitHub release: {drafts.stderr.strip()}")
            if drafts.stdout.strip():
                raise ValueError(draft)
            return None
        release = json.loads(result.stdout)
        if release["tag_name"] != self.tag:
            raise ValueError("Existing release has unexpected tag")
        if release["draft"]:  # Defensive: the tags endpoint is documented to omit drafts.
            raise ValueError(draft)
        if release["prerelease"] != self.is_dev:
            found = "a prerelease" if release["prerelease"] else "a stable release"
            wanted = "a prerelease" if self.is_dev else "a stable release"
            raise ValueError(
                f"GitHub release for {self.tag} is {found} but {self.version} needs {wanted}; "
                "fix it on GitHub, the script never edits releases"
            )
        return release

    def head_snapshot(self):
        """The snapshot version of HEAD, from the version its own pyproject.toml declares."""
        project = project_table(self.blob("HEAD", "pyproject.toml"))
        return snapshot_version(current_version(project), self.git("rev-parse", "HEAD"))

    def return_to(self, state):
        """Where a snapshot run ends: the branch it was cut from, or else the commit it was cut on.

        A branch that is gone or is the release branch itself is not returned to; nothing else
        moves HEAD, so the run then stays on the release branch.
        """
        start = state.get("start")
        if start is None:
            return None
        if not start:  # Cut from a detached HEAD.
            return ["--detach", state["base"]], f"{state['base'][:SNAPSHOT_HASH]} (detached)"
        if start != self.branch and self.ref(f"refs/heads/{start}"):
            return [start], start
        return None

    def switch_back(self, state):
        """Leave the release branch of a snapshot for where it was cut; a failure costs no more."""
        back = self.return_to(state)
        if back is None or self.git("branch", "--show-current") != self.branch:
            return
        try:
            self.git("switch", *back[0])  # Work goes on where the snapshot was cut.
        except ValueError as error:
            # The commit is journaled and publishing names it explicitly, so a branch git cannot
            # check out here (held by another worktree, a hook) only costs the switch.
            print(str(error).rstrip())
            now = self.git("branch", "--show-current")
            landed = now == back[0][0] or (
                not now and self.git("rev-parse", "HEAD") == state["base"]
            )
            if not landed:
                print(f"Switch back to {back[1]} by hand")

    def check_fresh_start(self, current_branch, remote):
        """A fresh release starts from main at origin/main; a snapshot from its base commit."""
        snapshot = self.snapshot
        if snapshot is not None:
            expected = self.head_snapshot()
            if expected != self.version:
                public, commit_hash = split_snapshot(expected)
                if commit_hash == snapshot[1]:
                    raise ValueError(
                        f"Snapshot {self.version} is built on version {snapshot[0]}, but HEAD's "
                        f"pyproject.toml declares {public}; use --snapshot or --version {expected}"
                    )
                raise ValueError(
                    f"{self.version} is not the snapshot of HEAD, {expected} is; use --snapshot "
                    f"for it, or cut {self.version} from a branch at commit {snapshot[1]}"
                )
            return
        if current_branch != self.main:
            raise ValueError(f"Fresh releases must start from {self.main}")
        # A merged snapshot stops here, before "update main" could suggest pushing it. A file
        # that cannot be read stays prepare()'s to report, after the remote check as before.
        try:
            project = project_table(read_text(self.root / "pyproject.toml"))
            merged = split_snapshot(project["version"]) is not None
        except (OSError, KeyError, TypeError, ValueError):
            merged = False
        if merged:
            current_version(project)
        if self.git("rev-parse", "HEAD") != remote.get(f"refs/heads/{self.main}"):
            raise ValueError(f"{self.main} must match {self.remote}/{self.main}; update it first")

    def taken(self, commit, incomplete, state):
        """Why an inferred version cannot be used: its release branch or tag already exists."""
        exists = f"{self.branch} or {self.tag} already exists"
        if incomplete:
            return (
                f"Release {self.version} is already prepared on {self.branch} but not committed; "
                f"resume it with {self.resume_command()}, or drop it with:\n  "
                f"{self.drop_hint(state)}"
            )
        if self.snapshot is not None:
            return (
                f"{exists}: HEAD is already cut as {self.version}; publish it with "
                f"--version {self.version} --tag"
            )
        parsed = parse_version(self.version)
        later = (
            f"{parsed.base_version}.dev{parsed.dev + 1}"
            if parsed.is_devrelease
            else f"{parsed.major}.{parsed.minor}.{parsed.micro + 1}"
        )
        resume = f"resume it with --version {self.version}"
        other = f"pass a higher explicit --version, for example {later}"
        if self.run("git", "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode:
            return (  # A dry run does not fetch, so the commit cannot be checked here.
                f"{exists}, and its commit is not fetched here to check; if it is a release of "
                f"this script, {resume}, otherwise {other}"
            )
        try:
            self.verify_commit(commit)
        except ValueError:
            return (
                f"{exists}, but not as a release this script can resume, so the version "
                f"{self.version} inferred from pyproject.toml is taken (a tag ahead of the version "
                f"its commit declares?); {other}"
            )
        return (
            f"{exists} as a release of this script, so the version {self.version} inferred from "
            f"pyproject.toml is taken; {resume}, or {other}"
        )

    def lightweight(self, sha):
        return (
            f"Tag {self.tag} at {sha[:SNAPSHOT_HASH]} is a lightweight tag and this script reuses "
            "annotated ones only; delete it if it is a leftover, or pass another --version"
        )

    def note_untagged(self):
        """Warn when pyproject.toml declares a version nobody tagged: a bump would skip it."""
        current = self.current
        tags = [f"v{current}", current]
        listed = self.git("ls-remote", "--tags", self.remote, *(f"refs/tags/{t}" for t in tags))
        if not listed.strip() and not any(self.ref(f"refs/tags/{t}") for t in tags):
            print(
                f"Note: pyproject.toml declares {current}, but no tag v{current} exists, so that "
                f"version looks unreleased and {self.version} skips it. This script commits the "
                f"bump itself; to publish {current} as it is, tag {self.main} by hand."
            )

    def resolve_version(self, current_branch):
        """Settle self.version from --version, a bump level or the snapshot of HEAD."""
        if self.version is None and self.args.bump == "snapshot":
            if is_release_branch(current_branch):
                raise ValueError(
                    f"--snapshot starts from the branch to snapshot, not from {current_branch}; "
                    "use explicit --version to resume the release on it"
                )
            self.version = self.head_snapshot()
        elif self.version is None:
            if current_branch != self.main:
                here = current_branch or "a detached HEAD"
                raise ValueError(
                    f"Bumps start from {self.main}, not from {here}; switch to {self.main}, or "
                    "pass an explicit --version to resume a release"
                )
            self.current, self.version, _ = prepare(self.root, None, self.args.bump)
        if self.snapshot is None:
            if "+" in self.version:
                # The only local label the script accepts; name its exact spelling.
                raise ValueError(
                    f"A snapshot version is X.Y.Z[.devN]+g<{SNAPSHOT_HASH} lower-case hex of its "
                    f"commit>, got {self.version!r}"
                )
            parse_version(self.version)
        elif self.args.pr:
            raise ValueError(
                "--pr opens a release PR to main, but a snapshot is never merged; drop --pr"
            )
        elif self.args.release:
            raise ValueError(
                "--release publishes through the PyPI workflow, which rejects a snapshot's local "
                "version label; use --tag to publish the snapshot as a git tag"
            )

    def execute(self):
        current_branch = self.git("branch", "--show-current")
        self.check_tools()
        self.resolve_version(current_branch)
        self.branch = f"release/v{self.version}"
        self.tag = f"v{self.version}"
        self.branch_ref = f"refs/heads/{self.branch}"
        self.tag_ref = f"refs/tags/{self.tag}"
        if self.snapshot is not None:
            # Cut from any branch (or detached HEAD) but a release branch and reused from anywhere
            # but another one; an interrupted preparation resumes on its own release branch.
            if is_release_branch(current_branch) and current_branch != self.branch:
                raise ValueError(
                    f"Cut {self.version} from the branch to snapshot or reuse it from any branch "
                    f"that is not a release/v* branch; resume its preparation from {self.branch}"
                )
        elif current_branch not in (self.main, self.branch):
            raise ValueError(f"Start from {self.main} or resume from {self.branch}")
        self.state_path = Path(
            self.git("rev-parse", "--git-path", f"jserpy-release-{self.version}.json")
        )
        if not self.state_path.is_absolute():
            self.state_path = self.root / self.state_path
        lock = None if self.args.dry_run else self.lock()
        try:
            self.proceed(current_branch)
        finally:
            if lock:
                lock.unlink(missing_ok=True)

    def journal(self):
        """The saved state of an earlier, interrupted run of this version, if there is one."""
        if not self.state_path.exists():
            return None
        try:
            state = json.loads(read_text(self.state_path))
            if not isinstance(state, dict) or not {"base", "original", "edits"} <= state.keys():
                raise ValueError
        except ValueError:  # Not JSON, not UTF-8 (both are ValueErrors) or not a journal.
            raise ValueError(
                f"{self.state_path} is not a valid journal; delete it if no release of "
                f"{self.version} is being prepared"
            ) from None
        return state

    def proceed(self, current_branch):
        """Prepare, verify and publish the release, resuming whatever an earlier run left."""
        state = self.journal()
        dirty = self.git("status", "--porcelain", "--untracked-files=all")
        if dirty and not (state and not state.get("commit") and current_branch == self.branch):
            raise ValueError(
                "A clean worktree is required (no staged, unstaged or untracked files)"
            )
        self.name = project_table(read_text(self.root / "pyproject.toml"))["name"]
        remote = self.remote_refs()  # Fail closed on network/authentication errors.
        if self.current is not None:
            self.note_untagged()
        local_branch = self.ref(self.branch_ref)
        local_tag = self.ref(self.tag_ref)
        # Dry runs read remote state but never fetch or change refs.
        if not self.args.dry_run:
            for ref in (self.branch_ref, self.tag_ref):
                if remote.get(ref) and not self.ref(ref):
                    self.git("fetch", "--no-tags", self.remote, f"{ref}:{ref}")
            local_branch = self.ref(self.branch_ref)
            local_tag = self.ref(self.tag_ref)
        remote_tag_commit = remote.get(self.tag_ref + "^{}", remote.get(self.tag_ref))
        tag_commit = self.git("rev-parse", f"{self.tag_ref}^{{commit}}") if local_tag else None
        candidates = [
            sha
            for sha in (local_branch, remote.get(self.branch_ref), tag_commit, remote_tag_commit)
            if sha
        ]
        incomplete = (
            state
            and not state.get("commit")
            and local_branch == state["base"]
            and not local_tag
            and not remote.get(self.branch_ref)
        )
        if self.args.bump and candidates:
            # Whatever the existing state is, an inferred version cannot be used; say why first.
            raise ValueError(self.taken(candidates[0], incomplete, state))
        if local_branch and remote.get(self.branch_ref) and local_branch != remote[self.branch_ref]:
            raise ValueError(
                f"Local {self.branch} ({local_branch[:SNAPSHOT_HASH]}) and {self.remote}'s "
                f"({remote[self.branch_ref][:SNAPSHOT_HASH]}) differ; reconcile them by hand, "
                "published history is never rewritten"
            )
        if local_tag and remote.get(self.tag_ref) and local_tag != remote[self.tag_ref]:
            raise ValueError(
                f"Local and remote tag {self.tag} disagree ({local_tag[:SNAPSHOT_HASH]} and "
                f"{remote[self.tag_ref][:SNAPSHOT_HASH]}); tags are never moved"
            )
        if remote.get(self.tag_ref) and self.tag_ref + "^{}" not in remote:
            raise ValueError(self.lightweight(remote[self.tag_ref]))  # ls-remote peels only these.
        if len(set(candidates)) > 1:
            raise ValueError(
                f"{self.branch} and {self.tag} point to different commits "
                f"({', '.join(sorted({sha[:SNAPSHOT_HASH] for sha in candidates}))})"
            )
        commit = candidates[0] if candidates and not incomplete else None
        if self.args.release or self.args.pr:
            self.repo = self.github_repo()
        existing = self.existing_release() if self.args.release else None
        verified = True
        if commit:
            available = self.run("git", "cat-file", "-e", f"{commit}^{{commit}}", check=False)
            verified = available.returncode == 0
            if not verified and self.args.dry_run:
                print("Remote release commit requires fetching before it can be validated")
            else:
                try:
                    self.verify_commit(commit)
                except ValueError as error:
                    raise ValueError(
                        f"{self.branch} or {self.tag} exists at {commit[:SNAPSHOT_HASH]} but is "
                        f"not a release commit of this script: {error}. Inspect it and delete the "
                        "branch or tag if it is a leftover, or pass another --version"
                    ) from None
            if local_tag and self.git("cat-file", "-t", self.tag_ref) != "tag":
                raise ValueError(self.lightweight(local_tag))
        if self.args.dry_run:
            self.plan(current_branch, remote, state, commit, incomplete, existing, verified)
            return
        if not commit:
            if not incomplete:
                self.check_fresh_start(current_branch, remote)
                current, _, edits = prepare(self.root, self.version, None)
                self.note_series(current)
                state = {
                    "base": self.git("rev-parse", "HEAD"),
                    "original": {name: read_text(self.root / name) for name in self.files},
                    "edits": {path.name: value for path, value in edits.items()},
                }
                if self.snapshot is not None:
                    state["start"] = current_branch  # Where work goes on once committed.
                self.save(state)
                self.git("switch", "-c", self.branch)
            commit = self.unfinished(state, lambda: self.prepare_commit(state))
            if self.snapshot is not None:
                self.switch_back(state)
        else:
            if not local_branch:
                self.git("branch", self.branch, commit)
            if state and not state.get("commit"):
                # An earlier run died after its commit but before journaling it.
                state["commit"] = commit
                self.save(state)
                if self.snapshot is not None:
                    self.switch_back(state)
        try:
            self.publish(commit, remote, existing)
        except ValueError as error:
            raise ValueError(
                f"{error}\nThe release commit {commit[:SNAPSHOT_HASH]} stays on {self.branch}; "
                f"fix the cause and re-run with {self.resume_command()} to go on"
            ) from None
        self.conclude(commit, remote)

    def plan(self, current_branch, remote, state, commit, incomplete, existing, verified):
        """What a dry run reports, once every check that could still reject the target passed."""
        kind = "stable release"
        if self.snapshot is not None:
            kind = "snapshot, git tag only"
        elif self.is_dev:
            kind = "development prerelease"
        target = f"Target: {self.version} ({kind})"
        if not commit:
            if incomplete:
                self.unfinished(state, lambda: self.check_partial(state))
                print(target)
                print("Would resume interrupted preparation and validate before committing")
                if self.snapshot is not None and (back := self.return_to(state)):
                    print(f"Would switch back to {back[1]} once committed")
            else:
                self.check_fresh_start(current_branch, remote)
                current, _, _ = prepare(self.root, self.version, None)
                print(target)
                print(f"Would create {self.branch}, prepare files, validate and commit")
                if self.snapshot is not None:
                    print(f"Would switch back to {current_branch or 'the detached HEAD'}")
                self.note_series(current)
        else:
            print(target)
            print(f"Would reuse release commit {commit}" + ("" if verified else " (not verified)"))
        # --tag and --release carry the branch along, except for a snapshot.
        pushing = self.args.push or self.args.pr
        if self.snapshot is None:
            pushing = pushing or self.args.tag or self.args.release
        print(f"Branch push: {'requested' if pushing else 'not requested'}")
        tagging = self.args.tag or self.args.release
        print(f"Tag: {'ensure annotated tag and push' if tagging else 'not requested'}")
        if self.args.release:
            print(
                "Release already exists"
                if existing
                else "Would create GitHub "
                + ("prerelease (--prerelease --latest=false)" if self.is_dev else "Release")
                + ", which the PyPI workflow publishes"
            )
        if self.args.pr:
            print("Would ensure a release PR to main exists")

    def publish(self, commit, remote, existing):
        """Tag, push, release and open the PR that were asked for; every step is idempotent."""
        if self.args.tag or self.args.release:
            if not self.ref(self.tag_ref):
                self.git("tag", "-a", self.tag, commit, "-m", f"{self.name} {self.version}")
            if self.snapshot is not None and not self.args.push:
                # A snapshot publishes its tag alone: the release commit is reachable from it,
                # and the remote collects no release branch per snapshot. --push adds the branch.
                if remote.get(self.tag_ref) != self.ref(self.tag_ref):
                    self.git("push", self.remote, f"{self.tag_ref}:{self.tag_ref}")
            elif remote.get(self.branch_ref) != commit or remote.get(self.tag_ref) != self.ref(
                self.tag_ref
            ):
                self.git(
                    "push",
                    "--atomic",
                    self.remote,
                    f"{self.branch_ref}:{self.branch_ref}",
                    f"{self.tag_ref}:{self.tag_ref}",
                )
        elif self.args.push or self.args.pr:
            if remote.get(self.branch_ref) != commit:
                self.git("push", self.remote, f"{self.branch_ref}:{self.branch_ref}")
        if self.args.release:
            if existing:
                print(f"Release already exists: {existing['html_url']}")
            else:
                print(
                    self.github(
                        "release",
                        "create",
                        self.tag,
                        "--verify-tag",
                        "--title",
                        f"{self.name} {self.version}",
                        "--generate-notes",
                        *(("--prerelease", "--latest=false") if self.is_dev else ()),
                    ).stdout.strip()
                )
                print(
                    f"The PyPI workflow publishes {self.name} {self.version} from this release: "
                    f"https://github.com/{self.repo}/actions"
                )
        if self.args.pr:
            prs = json.loads(
                self.github(
                    "pr",
                    "list",
                    "--head",
                    self.branch,
                    "--base",
                    self.main,
                    "--state",
                    "all",
                    "--json",
                    "state,url",
                ).stdout
            )
            usable = [pr for pr in prs if pr["state"] in ("OPEN", "MERGED")]
            if usable:
                print(f"Release PR already exists: {usable[0]['url']}")
            else:
                print(
                    self.github(
                        "pr",
                        "create",
                        "--head",
                        self.branch,
                        "--base",
                        self.main,
                        "--title",
                        self.subject,
                        "--body",
                        f"Bump the package version to {self.version}. Validated by building the "
                        "sdist and wheel, checking the version they declare and running the tests "
                        "against the built wheel.",
                    ).stdout.strip()
                )

    def conclude(self, commit, remote):
        """Say what is left to do."""
        print(f"Release commit: {commit}\nBranch: {self.branch}")
        if self.snapshot is not None and self.args.tag:
            try:
                slug = repo_slug(self.git("remote", "get-url", self.remote))
                install = (
                    f'Install it with: pip install "git+https://github.com/{slug}.git@{self.tag}"; '
                )
            except ValueError:  # A remote that is no GitHub URL has no install line to print.
                install = ""
            print(f"{install}nothing is merged back, and {self.branch} can be deleted.")
        elif self.snapshot is not None:
            if not remote.get(self.tag_ref):
                print(f"Tag and push it with --version {self.version} --tag.")
        elif self.is_dev:
            parsed = parse_version(self.version)
            print(
                f"Merge {self.branch} into {self.main} through a PR before the next build; --dev "
                f"then prepares {parsed.base_version}.dev{parsed.dev + 1} and --final prepares "
                f"{parsed.base_version}."
            )
        else:
            print(
                f"Merge the release branch into {self.main} through a PR before preparing the "
                "next release."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,  # A prefix such as --r must not quietly mean --release.
    )
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument(
        "--version",
        metavar="X.Y.Z[.devN][+gHASH]",
        help="Explicit stable, development or snapshot version; required for resuming",
    )
    for level in ("patch", "minor", "major"):
        choice.add_argument(
            f"--{level}",
            dest="bump",
            action="store_const",
            const=level,
            help=f"Next {level} release from the stable version on main; with --dev, its .dev0",
        )
    choice.add_argument(
        "--final",
        dest="bump",
        action="store_const",
        const="final",
        help="Promote the X.Y.Z.devN on main to stable X.Y.Z",
    )
    choice.add_argument(
        "--snapshot",
        dest="bump",
        action="store_const",
        const="snapshot",
        help="Development snapshot of HEAD from any clean non-release branch: its version plus "
        "+g<10 hex of HEAD>, published by its git tag alone and never merged back",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Development build: X.Y.Z.dev0 with a level, or the next .devN on its own",
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect state without mutations")
    parser.add_argument("--push", action="store_true", help="Push the release branch")
    parser.add_argument(
        "--tag",
        action="store_true",
        help="Create annotated tag and push branch/tag (a snapshot pushes its tag alone)",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Tag, push and create the GitHub release, which the PyPI workflow then publishes "
        "(a prerelease for X.Y.Z.devN; refused for a snapshot)",
    )
    parser.add_argument(
        "--pr",
        action="store_true",
        help="Push and create a release PR; never merge (refused for a snapshot)",
    )
    args = parser.parse_args()
    if args.dev:
        if args.version is not None:
            parser.error("--version already names the build; drop --dev")
        if args.bump == "final":
            parser.error("--dev and --final are mutually exclusive")
        if args.bump == "snapshot":
            parser.error("--dev and --snapshot are mutually exclusive")
        args.bump = f"{args.bump}-dev" if args.bump else "dev"
    elif args.version is None and args.bump is None:
        parser.error(
            "one of the arguments --version --patch --minor --major --final --dev --snapshot "
            "is required"
        )
    try:
        ReleaseFlow(ROOT, args).execute()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"{parser.prog}: error: {error}") from None
    except KeyboardInterrupt:
        raise SystemExit(f"{parser.prog}: interrupted") from None


if __name__ == "__main__":
    main()
