# Windows build helper for the CUDA fast path.
#
#   .\build.ps1                      # CUDA, architecture detected from the local GPU
#   .\build.ps1 -Architectures 90    # force an SM (e.g. 90 for H100)
#   .\build.ps1 -CpuOnly             # CPU-only build (fast path disabled)
#
# CMake must see MSVC's environment (rc.exe/mt.exe), so configure+build run from
# a temporary .cmd that first calls vcvars64.bat.

param(
    [switch]$CpuOnly,
    [string]$Architectures = "auto",
    [string]$BuildDir = "build",
    [string]$VcVars = ""
)

$ErrorActionPreference = "Stop"
$root = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $root

if (-not $VcVars) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($install) { $VcVars = Join-Path $install "VC\Auxiliary\Build\vcvars64.bat" }
    }
}
if (-not $VcVars) {
    # vswhere does not list every channel (e.g. VS 18 BuildTools); scan the usual roots.
    foreach ($base in @("${env:ProgramFiles(x86)}\Microsoft Visual Studio", "$env:ProgramFiles\Microsoft Visual Studio")) {
        if (-not (Test-Path -LiteralPath $base)) { continue }
        $found = Get-ChildItem -Path $base -Filter vcvars64.bat -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match 'VC\\Auxiliary\\Build' } |
            Select-Object -First 1 -ExpandProperty FullName
        if ($found) { $VcVars = $found; break }
    }
}
if (-not $VcVars -or -not (Test-Path -LiteralPath $VcVars)) {
    throw "vcvars64.bat not found; pass -VcVars <path>"
}

if (-not $CpuOnly) {
    # A binary built for the wrong SM fails at run time with "no kernel image is
    # available" only when the GPU backend is first used. Detect the local card.
    if ($Architectures -eq "auto") {
        $cap = $null
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            $cap = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
        }
        if ($cap -and $cap -match '^\s*(\d+)\.(\d+)\s*$') {
            $Architectures = "$($matches[1])$($matches[2])"
            Write-Host "CUDA architecture: $Architectures (detected)"
        } else {
            $Architectures = "89"
            Write-Host "CUDA architecture: 89 (default; pass -Architectures to override)"
        }
    }
    $defines = "-DCMAKE_BUILD_TYPE=Release -DS2_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$Architectures -DGGML_CUDA_GRAPHS=ON"
} else {
    $defines = "-DCMAKE_BUILD_TYPE=Release -DS2_CUDA=OFF"
}

$bat = Join-Path $env:TEMP "s2-build-$PID.cmd"
@"
@echo off
call "$VcVars" >nul
cd /d "$root"
cmake -B $BuildDir -G Ninja $defines
if errorlevel 1 exit /b 1
cmake --build $BuildDir --parallel
"@ | Set-Content -LiteralPath $bat -Encoding ascii

try {
    & cmd /c $bat
    if ($LASTEXITCODE -ne 0) { throw "s2.cpp build failed" }
} finally {
    Remove-Item -LiteralPath $bat -Force -ErrorAction SilentlyContinue
}

# s2.exe loads ggml*.dll from its own directory; Ninja puts them in build\bin.
$bin = Join-Path $root "$BuildDir\bin"
$out = Join-Path $root $BuildDir
if (Test-Path -LiteralPath $bin) {
    Copy-Item -Path (Join-Path $bin "*.dll") -Destination $out -Force
}

# The CUDA runtime pieces and the MSVC runtime are not produced by this build but
# are needed at run time. Copy them next to s2.exe so the folder is self-contained;
# without this the exe dies on a machine that has only the NVIDIA driver.
function Copy-RuntimeDll([string[]]$dirs, [string]$pattern) {
    foreach ($dir in $dirs) {
        if (-not $dir -or -not (Test-Path -LiteralPath $dir)) { continue }
        $files = Get-ChildItem -Path (Join-Path $dir $pattern) -File -ErrorAction SilentlyContinue
        if ($files) {
            foreach ($f in $files) {
                Copy-Item -LiteralPath $f.FullName -Destination $out -Force
                Write-Host "  + $($f.Name)"
            }
            return $true
        }
    }
    return $false
}

if (-not $CpuOnly) {
    $cudaDirs = @()
    if ($env:CUDA_PATH) {
        $cudaDirs += (Join-Path $env:CUDA_PATH "bin\x64")
        $cudaDirs += (Join-Path $env:CUDA_PATH "bin")
    }
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc) { $cudaDirs += (Split-Path $nvcc.Source) }
    $cudaDirs += (Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' -Directory -ErrorAction SilentlyContinue |
                  Sort-Object Name -Descending |
                  ForEach-Object { Join-Path $_.FullName 'bin\x64' })

    $okCublas   = Copy-RuntimeDll $cudaDirs 'cublas64_*.dll'
    $okCublasLt = Copy-RuntimeDll $cudaDirs 'cublasLt64_*.dll'
    if (-not $okCublas) {
        Write-Warning "cublas64_*.dll not found; s2.exe will need the CUDA toolkit on PATH."
    }
    if (-not $okCublasLt) {
        Write-Warning "cublasLt64_*.dll not found; s2.exe will need the CUDA toolkit on PATH."
    }
}

$vcRoot = Split-Path (Split-Path (Split-Path $VcVars))   # ...\VC
$redistRoot = Join-Path $vcRoot 'Redist\MSVC'
$crt = Get-ChildItem -Path $redistRoot -Directory -ErrorAction SilentlyContinue |
       Sort-Object Name -Descending |
       ForEach-Object { Join-Path $_.FullName 'x64\Microsoft.VC143.CRT' } |
       Where-Object { Test-Path -LiteralPath $_ } |
       Select-Object -First 1
if ($crt) {
    Copy-Item -Path (Join-Path $crt '*.dll') -Destination $out -Force
} else {
    Write-Warning "MSVC runtime redistributable not found; target machines need the VC++ redistributable."
}

# Smoke test with a PATH that only has the build folder and Windows itself: this is
# what a copied build looks like on a clean machine.
$exe = Join-Path $out 's2.exe'
$systemPath = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\Wbem"
$savedPath = $env:PATH
try {
    $env:PATH = "$out;$systemPath"
    $version = & $exe --version 2>&1
    $code = $LASTEXITCODE
} finally {
    $env:PATH = $savedPath
}
if ($code -ne 0) {
    throw "s2.exe does not start with a minimal PATH (exit $code). Missing DLL next to it; run dumpbin /dependents on s2.exe and its ggml DLLs."
}
Write-Host "s2 built: $exe ($($version -join ' '))"
