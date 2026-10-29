#!/usr/bin/env python3
"""Fail the wheel job unless both GIL and free-threaded tags were built.

cibuildwheel can skip a selector that does not match an available
interpreter. This check looks at the wheelhouse itself so a job that
only produced GIL-enabled wheels does not pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

# GIL-enabled CPython 3.11–3.15, plus free-threaded 3.14 and 3.15.
# Experimental cp313t is not a cibuildwheel 4 target.
REQUIRED_TAGS = (
    "cp311",
    "cp312",
    "cp313",
    "cp314",
    "cp314t",
    "cp315",
    "cp315t",
)


def python_tag(wheel_name: str) -> str | None:
    stem = wheel_name[:-4] if wheel_name.endswith(".whl") else wheel_name
    for part in stem.split("-"):
        if part.startswith("cp3"):
            return part
    return None


def main(argv: list[str]) -> int:
    wheelhouse = Path(argv[1] if len(argv) > 1 else "wheelhouse")
    wheels = sorted(wheelhouse.glob("*.whl"))
    found: set[str] = set()
    print(f"Wheels in {wheelhouse}:")
    if not wheels:
        print("  (none)")
    for wheel in wheels:
        tag = python_tag(wheel.name)
        if tag is not None:
            found.add(tag)
        print(f"  {wheel.name}")

    missing = [tag for tag in REQUIRED_TAGS if tag not in found]
    if missing:
        print(
            "Missing CPython wheel tags: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1

    print("GIL and free-threaded wheel tags present: " + ", ".join(REQUIRED_TAGS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
