#!/usr/bin/env bash
# One-line install: detect OS/arch (+ EP override) and fetch the matching
# GitHub release artifact.
#   curl -fsSL .../install.sh | bash
# Env: T3RAG_EP=cpu|cuda|coreml|openvino (default cpu),
#      T3RAG_VERSION=vX.Y.Z (default: latest),
#      T3RAG_DIR (default ~/.local/bin)
set -euo pipefail

REPO="horus14-wq/t3ragrust"
EP="${T3RAG_EP:-cpu}"
DIR="${T3RAG_DIR:-$HOME/.local/bin}"

os="$(uname -s)"
arch="$(uname -m)"
case "$os" in
  Linux)  case "$arch" in x86_64) target="x86_64-unknown-linux-gnu" ;; *) echo "unsupported arch: $arch" >&2; exit 1 ;; esac ;;
  Darwin) case "$arch" in arm64) target="aarch64-apple-darwin" ;; x86_64) target="x86_64-apple-darwin" ;; *) echo "unsupported arch: $arch" >&2; exit 1 ;; esac ;;
  *) echo "unsupported OS: $os" >&2; exit 1 ;;
esac

# EP sanity: coreml only exists on macOS; cuda/openvino on linux+windows.
if [ "$EP" = "coreml" ] && [ "$os" != "Darwin" ]; then
  echo "ep=coreml only ships for macOS — falling back to cpu" >&2; EP=cpu
fi

ver="${T3RAG_VERSION:-$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | grep -m1 '"tag_name"' | cut -d'"' -f4)}"
[ -n "$ver" ] || { echo "could not resolve latest release" >&2; exit 1; }

name="t3rag-${ver}-${EP}-${target}"
url="https://github.com/$REPO/releases/download/${ver}/${name}.tar.gz"

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
echo "downloading $url"
curl -fsSL "$url" -o "$tmp/pkg.tar.gz"
curl -fsSL "$url.sha256" -o "$tmp/pkg.sha256"
# Accepts both `HASH  file` and hash-only .sha256 files. Probe sha256sum
# (GNU coreutils) first — minimal/musl images may lack perl's shasum.
(cd "$tmp" && echo "$(cut -d' ' -f1 pkg.sha256)  pkg.tar.gz" \
  | if command -v sha256sum >/dev/null; then sha256sum -c -; else shasum -a 256 -c -; fi) \
  || { echo "checksum mismatch" >&2; exit 1; }
tar -xzf "$tmp/pkg.tar.gz" -C "$tmp"
mkdir -p "$DIR"
# Install everything in the package dir — EP flavors also ship the
# provider libs (libonnxruntime_providers_*.so / *.dll) the binary
# dlopens from its own directory ($ORIGIN rpath).
install -m 0755 "$tmp/$name/"* "$DIR/"
echo "installed $DIR/t3rag ($EP, $target)"
"$DIR/t3rag" doctor || true
