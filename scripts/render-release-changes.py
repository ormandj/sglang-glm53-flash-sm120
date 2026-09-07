#!/usr/bin/env python3
"""Extract one exact stable release's changes for repository release bodies."""

import argparse
import re
from pathlib import Path


def release_changes(changelog: str, tag: str) -> str:
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
        raise ValueError("release tag must be complete stable SemVer")
    lines = changelog.splitlines()
    heading = re.compile(r"^## " + re.escape(tag) + r"(?:\s|$)")
    starts = [index for index, line in enumerate(lines) if heading.match(line)]
    if len(starts) != 1:
        raise ValueError(f"expected exactly one changelog section for {tag}")
    start = starts[0] + 1
    end = next((index for index in range(start, len(lines)) if lines[index].startswith("## ")), len(lines))
    changes = "\n".join(lines[start:end]).strip()
    if not changes or not re.search(r"(?m)^- \S", changes):
        raise ValueError(f"{tag} must contain release-change bullets")
    if re.search(
        r"(?:[a-z0-9-]+\.)?home\.[a-z0-9.-]+|registry\.internal\.example|/Users/[^/]+/|~/git/[^/]+/|"
        r"qualified internal digest|promote without rebuilding|"
        r"primary project repository|MEMORY-CHECK:|ordered-marker oracle",
        changes,
        re.IGNORECASE,
    ):
        raise ValueError(f"{tag} contains internal housekeeping; write public release notes")
    return changes + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changelog", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    try:
        print(release_changes(args.changelog.read_text(), args.tag), end="")
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
