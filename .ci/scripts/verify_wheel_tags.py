#!/usr/bin/env python3
"""Fail the wheel job unless both GIL and free-threaded tags were built.

cibuildwheel can skip a selector that does not match an available
interpreter. This check looks at the wheelhouse itself so a job that
only produced GIL-enabled wheels does not pass.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# GIL-enabled ABI tags cp311–cp315, plus free-threaded ABI tags cp314t and
# cp315t. Wheel names look like kvxopt-1.2.3-cp314-cp314t-....whl: the
# interpreter tag stays cp314 and the ABI tag carries the "t".
REQUIRED_TAGS = (
    "cp311",
    "cp312",
    "cp313",
    "cp314",
    "cp314t",
    "cp315",
    "cp315t",
)
_ABI_TAG = re.compile(r"^cp3\d+t?$")


def abi_tag(wheel_name: str) -> str | None:
    """Return the ABI tag, which is the last cp3xx component in the name.

    Free-threaded wheels are tagged cp314-cp314t, so the interpreter tag
    alone cannot tell them apart from the GIL-enabled cp314-cp314 wheel.
    """
    stem = wheel_name[:-4] if wheel_name.endswith(".whl") else wheel_name
    tags = [part for part in stem.split("-") if _ABI_TAG.fullmatch(part)]
    return tags[-1] if tags else None


def main(argv: list[str]) -> int:
    wheelhouse = Path(argv[1] if len(argv) > 1 else "wheelhouse")
    wheels = sorted(wheelhouse.glob("*.whl"))
    found: set[str] = set()
    print(f"Wheels in {wheelhouse}:", flush=True)
    if not wheels:
        print("  (none)", flush=True)
    for wheel in wheels:
        tag = abi_tag(wheel.name)
        if tag is not None:
            found.add(tag)
        print(f"  {wheel.name}", flush=True)

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
