# install.ps1 — Windows counterpart to install.sh (GH#2).
#
#   irm https://raw.githubusercontent.com/horus14-wq/t3ragrust/main/install.ps1 | iex
#
# Env overrides:
#   $env:T3RAG_EP       = cpu | cuda | directml | openvino   (default: cpu)
#   $env:T3RAG_VERSION  = vX.Y.Z                             (default: latest release)
#   $env:T3RAG_DIR      = install dir                        (default: ~\.local\bin)
$ErrorActionPreference = 'Stop'

$Repo   = 'horus14-wq/t3ragrust'
$Ep     = if ($env:T3RAG_EP)   { $env:T3RAG_EP }   else { 'cpu' }
$Dir    = if ($env:T3RAG_DIR)  { $env:T3RAG_DIR }  else { Join-Path $HOME '.local\bin' }
$Target = 'x86_64-pc-windows-msvc'

if ($env:T3RAG_VERSION) {
    $ver = $env:T3RAG_VERSION
} else {
    $rel = Invoke-RestMethod -UseBasicParsing "https://api.github.com/repos/$Repo/releases/latest"
    $ver = $rel.tag_name
}
if (-not $ver) { throw 'could not resolve latest release' }

$name = "t3rag-$ver-$Ep-$Target"
$base = "https://github.com/$Repo/releases/download/$ver"
$tmp  = Join-Path ([IO.Path]::GetTempPath()) ([IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    $zip = Join-Path $tmp "$name.zip"
    Write-Host "downloading $base/$name.zip"
    Invoke-WebRequest -UseBasicParsing "$base/$name.zip" -OutFile $zip
    # GH#6: .Content can arrive as [byte[]] for these assets — decode to
    # UTF-8 first, then take the 64-hex digest rather than a token split
    # (byte[0] of a hash starting 'e' is ASCII 101 → bogus "want 101").
    $raw = (Invoke-WebRequest -UseBasicParsing "$base/$name.zip.sha256").Content
    if ($raw -is [byte[]]) { $raw = [Text.Encoding]::UTF8.GetString($raw) }
    $want = ([regex]::Match(($raw -as [string]).Trim(), '[A-Fa-f0-9]{64}')).Value.ToLower()
    if (-not $want) { throw "could not parse sha256 artifact" }
    $got  = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
    if ($want -ne $got) { throw "checksum mismatch (want $want, got $got)" }

    Expand-Archive $zip -DestinationPath $tmp
    # Older zips nest the payload under pkg\ or dist\ — locate the dir
    # holding t3rag.exe and copy its whole contents (exe + provider
    # DLLs, which EP flavors ship alongside the binary).
    $pkgDir = Get-ChildItem -Recurse -Directory $tmp |
        Where-Object { Test-Path (Join-Path $_.FullName 't3rag.exe') } |
        Select-Object -First 1
    if (-not $pkgDir) { throw 'no t3rag.exe found in package' }
    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    Copy-Item (Join-Path $pkgDir.FullName '*') -Destination $Dir -Force
    Write-Host "installed $Dir\t3rag.exe ($Ep, $Target)"
    & (Join-Path $Dir 't3rag.exe') doctor
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
