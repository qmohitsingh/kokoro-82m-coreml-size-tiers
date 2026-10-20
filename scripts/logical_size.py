"""Print the logical byte size of a file or directory.

Uses file sizes, not APFS allocated blocks, so compression and clones do not
change the number. This is the size recorded for a Release .app.
"""

import os
import sys
from pathlib import Path


def logical_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            total += (Path(dirpath) / name).stat().st_size
    return total


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    total = logical_bytes(root)
    print(f"{total}\t{total / 1_000_000:.1f} MB\t{root}")
    if root.is_dir() and len(sys.argv) > 2 and sys.argv[2] == "--breakdown":
        children = sorted(root.iterdir(), key=lambda path: logical_bytes(path), reverse=True)
        for child in children[:40]:
            size = logical_bytes(child)
            print(f"{size}\t{size / 1_000_000:.1f} MB\t{child.name}")


if __name__ == "__main__":
    main()
