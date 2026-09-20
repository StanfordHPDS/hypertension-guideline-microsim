"""Reset tracked-file mtimes to the time of their last touching commit.

git does not preserve mtimes across checkout, so files end up stamped with
whatever sub-second time git happened to write them. Make then sees
prerequisites and targets out of build order and flags spurious rebuilds.
This script restores each tracked file's mtime to the second-precision
timestamp of the most recent non-merge commit that touched it, which is
what Make needs to reason about the DAG correctly.

Dirty files (anything with staged or working-tree changes) are skipped so
an in-progress edit is never overwritten.

We also bump pipeline-output files to a floor derived from the latest
commit that touched any output tree. A "re-run pipeline" commit only
records files whose contents changed; outputs that were regenerated
byte-identically are absent from the commit and would otherwise revert
to their previous (older) commit time, looking stale against sources
that were touched in the rerun's parent commits. The floor encodes the
implicit "every output was regenerated as of this commit" fact that git
cannot record directly.

This is a minimal stdlib-only stand-in for `git-restore-mtime` from the
`git-tools` package.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

COMMIT_SENTINEL = "__COMMIT__ "

OUTPUT_PREFIXES: tuple[str, ...] = (
    "data_and_models/afc_outputs/",
    "data_and_models/afc_models/",
    "data_and_models/nhanes_data_cohorts/",
    "data_and_models/nhanes_inputs/",
    "figures/standard/",
    "figures/social/",
    "results/",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _tracked_files() -> set[str]:
    return {p for p in _git("ls-files", "-z").split("\0") if p}


def _dirty_files() -> set[str]:
    # Renames/copies emit two NUL-separated entries: 'R  new\0old'. Both
    # paths count as dirty so we never stomp on either side.
    parts = [p for p in _git("status", "--porcelain=v1", "-z").split("\0") if p]
    dirty: set[str] = set()
    i = 0
    while i < len(parts):
        entry = parts[i]
        dirty.add(entry[3:])
        if entry[:1] in {"R", "C"} and i + 1 < len(parts):
            i += 1
            dirty.add(parts[i])
        i += 1
    return dirty


def _walk_commits():
    out = _git(
        "log",
        "--reverse",
        "--no-merges",
        "--name-only",
        f"--format={COMMIT_SENTINEL}%ct",
    )
    ts: int | None = None
    for line in out.splitlines():
        if line.startswith(COMMIT_SENTINEL):
            ts = int(line[len(COMMIT_SENTINEL) :])
        elif line and ts is not None:
            yield ts, line


def _latest_output_commit_ts() -> int:
    paths = [p.rstrip("/") for p in OUTPUT_PREFIXES]
    out = _git("log", "-1", "--no-merges", "--format=%ct", "--", *paths).strip()
    return int(out) if out else 0


def main() -> int:
    os.chdir(_git("rev-parse", "--show-toplevel").strip())

    tracked = _tracked_files()
    dirty = _dirty_files()
    now = time.time()

    touched: dict[str, int] = {}
    skipped: set[str] = set()
    for ts, path in _walk_commits():
        if path not in tracked:
            continue
        if path in dirty:
            skipped.add(path)
            continue
        try:
            os.utime(path, (now, ts))
        except FileNotFoundError:
            continue
        touched[path] = ts

    floor = _latest_output_commit_ts()
    floored = 0
    for path, own_ts in touched.items():
        if own_ts >= floor:
            continue
        if not any(path.startswith(p) for p in OUTPUT_PREFIXES):
            continue
        try:
            os.utime(path, (now, floor))
        except FileNotFoundError:
            continue
        floored += 1

    print(
        f"restored mtimes on {len(touched)} files, "
        f"skipped {len(skipped)} dirty, "
        f"floored {floored} outputs to last pipeline run"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
