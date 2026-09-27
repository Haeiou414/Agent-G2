#!/usr/bin/env bash
# Install ALFWorld while limiting Fast Downward's otherwise unbounded make -j.
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=autodl_env.sh
source "$ROOT_DIR/reproduction/autodl_env.sh"

ENV_NAME=${AGENT_G2_CONDA_ENV:-agent-g2}
PYTHON="$CONDA_ENVS_PATH/$ENV_NAME/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Conda environment is missing: $PYTHON" >&2
    exit 2
fi
export PATH="$(dirname "$PYTHON"):$PATH"
export AGENT_G2_BUILD_JOBS=${AGENT_G2_BUILD_JOBS:-1}

VERSION=20.6.4
SHA256=ceeed05692ac6a4023de196ca3f8cfeffe80d23070ca8fda5991b7b46f8cb752
URL=https://files.pythonhosted.org/packages/34/1a/3a9b5bda34534f5513f24e047a1c99aa9d30a1c2ecf8b9eec54b9f740617/fast_downward_textworld-20.6.4.tar.gz
BUILD_ROOT="$AGENT_G2_DATA_ROOT/build"
ARCHIVE="$BUILD_ROOT/fast_downward_textworld-$VERSION.tar.gz"
SOURCE_DIR="$BUILD_ROOT/fast_downward_textworld-$VERSION"
mkdir -p "$BUILD_ROOT"

if ! "$PYTHON" -c 'import importlib.metadata as m; assert m.version("fast-downward-textworld") == "20.6.4"' >/dev/null 2>&1; then
    if [[ ! -f "$ARCHIVE" ]]; then
        curl --fail --location --retry 3 "$URL" --output "$ARCHIVE"
    fi
    printf '%s  %s\n' "$SHA256" "$ARCHIVE" | sha256sum --check --status
    if [[ ! -d "$SOURCE_DIR" ]]; then
        tar -xzf "$ARCHIVE" -C "$BUILD_ROOT"
    fi
    if ! grep -q AGENT_G2_BUILD_JOBS "$SOURCE_DIR/build.py"; then
        patch -d "$SOURCE_DIR" -p1 < "$ROOT_DIR/reproduction/patches/fast_downward_limited_jobs.patch"
    fi
    "$PYTHON" -m pip install --no-build-isolation --no-deps "$SOURCE_DIR"
fi

"$PYTHON" -m pip install alfworld==0.4.2

# The upstream downloader copies these files only after its optional visual
# detector download, which can fail even when all text-game archives succeeded.
ALFWORLD_PACKAGE_DATA=$("$PYTHON" -c 'from pathlib import Path; import alfworld; print(Path(alfworld.__file__).parent / "data")')
for name in alfred.pddl alfred.twl2; do
    if [[ ! -f "$ALFWORLD_DATA/logic/$name" ]]; then
        install -D "$ALFWORLD_PACKAGE_DATA/$name" "$ALFWORLD_DATA/logic/$name"
    fi
done
