#!/usr/bin/env bash
# Puts the headers of bgfx, bx and bimg and the shader includes of bgfx in <prefix>/include, at the revisions pinned
# below. Enough for `make test` to type-check the runtime against the real bgfx API; it does not build shaderc
# (see build-bgfx.sh for that). Used by `make deps` and by .github/actions/bgfx-headers.
#
#   fetch-bgfx-headers.sh <prefix>
set -euo pipefail

prefix=${1:?usage: fetch-bgfx-headers.sh <prefix>}

# The revisions that bgfx.cmake pins together (bump them together).
BGFX_REV=fe9bbdec66110f8626b9251a4947fc336ac3f62c
BX_REV=0e5bbca6878000f6371c65c53fe06f6f0ae87ca9
BIMG_REV=9bfbd6493b48a15238f2d1374a718ec026524515

if [ -f "$prefix/lib/libbgfx.a" ]; then
    echo "$prefix already holds a full bgfx build (build-bgfx.sh): its headers match its libraries, nothing to fetch" >&2
    exit 0
fi

fetch() { # fetch <repository> <revision> <directory>
    git init -q "$3"
    git -C "$3" fetch -q --depth 1 "https://github.com/bkaradzic/$1.git" "$2"
    git -C "$3" checkout -q FETCH_HEAD
}
work="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/bgfx-sources"
rm -rf "$work"
fetch bgfx "$BGFX_REV" "$work/bgfx"
fetch bx "$BX_REV" "$work/bx"
fetch bimg "$BIMG_REV" "$work/bimg"

include="$prefix/include"
mkdir -p "$include"
rm -rf "$include/bgfx" "$include/bx" "$include/bimg"
cp -r "$work/bgfx/include/bgfx" "$include/bgfx"
cp "$work"/bgfx/src/*.sh "$include/bgfx/"
cp -r "$work/bx/include/bx" "$include/bx"
cp -r "$work/bimg/include/bimg" "$include/bimg"
rm -rf "$work"
echo "bgfx headers installed in $include"
