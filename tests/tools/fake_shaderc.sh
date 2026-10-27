#!/usr/bin/env bash
# Fake shaderc, used to test the --shaderc / --embed plumbing without a GPU or bgfx.
#
# Writes a minimal C array (when --bin2c is given) or a dummy binary, and logs the arguments it received in
# <output>.args so that tests can check how the tool invoked it (quoting, -i, --platform, ...).
# FAKE_SHADERC_FAIL=1 simulates a compilation failure (exit code 1).
args=("$@")
out=""
name=""
while [ $# -gt 0 ]; do
    case "$1" in
        -o) out="$2"; shift 2 ;;
        --bin2c) name="$2"; shift 2 ;;
        *) shift ;;
    esac
done
if [ -n "${FAKE_SHADERC_FAIL:-}" ]; then
    echo "fake shaderc: simulated failure" >&2
    exit 1
fi
if [ -n "$name" ]; then
    printf 'static const uint8_t %s[4] = { 0x43, 0x53, 0x48, 0x0c };\n' "$name" > "$out"
else
    printf 'CSH\f' > "$out"
fi
printf '%s\n' "${args[*]}" > "$out.args"
