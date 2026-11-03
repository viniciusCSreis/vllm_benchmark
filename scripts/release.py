"""Bump the package version, push its tag, and draft a GitHub release."""

import argparse
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def run(*args):
    print(f"+ {shlex.join(args)}", flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version", nargs="?", help="Defaults to the next patch version."
    )
    args = parser.parse_args()

    if git_output("status", "--porcelain"):
        parser.error("Commit or stash your changes before releasing.")
    if git_output("branch", "--show-current") != "main":
        parser.error("Switch to main before releasing.")

    repo = git_output("remote", "get-url", "origin")
    run("gh", "auth", "status")
    run("git", "fetch", "origin", "--tags")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"], cwd=ROOT
    ).returncode:
        parser.error("Pull the latest changes from origin/main before releasing.")

    version_file = ROOT / "vllm_bench" / "__init__.py"
    source = version_file.read_text()
    current = re.search(r'^__version__ = "([^"]+)"$', source, re.MULTILINE).group(1)
    versions = [tuple(map(int, current.split(".")))]
    for tag in git_output("tag", "--list", "v*").splitlines():
        if re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
            versions.append(tuple(map(int, tag[1:].split("."))))
    latest = max(versions)
    version = (
        args.version.removeprefix("v")
        if args.version
        else f"{latest[0]}.{latest[1]}.{latest[2] + 1}"
    )
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        parser.error("Use a version in X.Y.Z format, such as 0.2.0.")
    if tuple(map(int, version.split("."))) <= latest:
        parser.error(f"Version must be newer than {'.'.join(map(str, latest))}.")
    tag = f"v{version}"

    print(f"Releasing {tag} (package version: {current}).", flush=True)
    version_file.write_text(
        source.replace(f'__version__ = "{current}"', f'__version__ = "{version}"', 1)
    )
    run("git", "add", "vllm_bench/__init__.py")
    run("git", "commit", "-m", f"Release {tag}")
    run("git", "tag", "-a", tag, "-m", f"Release {tag}")
    run("git", "push", "--atomic", "origin", "HEAD:refs/heads/main", f"refs/tags/{tag}")
    run(
        "gh",
        "release",
        "create",
        tag,
        "--repo",
        repo,
        "--draft",
        "--verify-tag",
        "--generate-notes",
        "--title",
        tag,
    )
    print("Draft created. Wait for CI to pass, then publish it on GitHub.")


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
