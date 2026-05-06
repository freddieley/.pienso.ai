param(
    [string]$DataRoot = ".",
    [string]$Pattern = "*.txt",
    [string]$ListPath = "corpora_list.txt",
    [string]$ModelPath = "model_fused.bin",
    [string]$EvalFile = "conversation_train.txt",
    [int]$Cycles = 5,
    [int]$PassesPerCycle = 1,
    [int]$Slots = 16,
    [int]$CtxCount = 32768,
    [int]$MinBytes = 256
)

$ErrorActionPreference = "Stop"

function Write-Header([string]$msg) {
    Write-Host "`n=== $msg ===" -ForegroundColor Cyan
}

function Build-List {
    param([string]$Root, [string]$Glob, [string]$OutFile, [int]$MinSize)

    $files = Get-ChildItem -Path $Root -Recurse -File -Filter $Glob |
        Where-Object { $_.Length -ge $MinSize } |
        Where-Object { $_.FullName -notmatch "\\(\.git|node_modules|bin|obj|\.vs)\\" } |
        Select-Object -ExpandProperty FullName

    if (-not $files -or $files.Count -eq 0) {
        throw "No files found under '$Root' matching '$Glob'."
    }

    "# auto-generated corpus list" | Set-Content -Path $OutFile -Encoding ASCII
    $files | Add-Content -Path $OutFile -Encoding ASCII
    return $files.Count
}

Write-Header "Compiling full_lm"
gcc -O2 -std=c11 -Wall -Wextra full_lm.c -o full_lm.exe
if ($LASTEXITCODE -ne 0) {
    throw "Compile failed."
}

$count = Build-List -Root $DataRoot -Glob $Pattern -OutFile $ListPath -MinSize $MinBytes
Write-Host "Corpus files indexed: $count"

for ($c = 1; $c -le $Cycles; $c++) {
    Write-Header "Cycle $c / $Cycles"

    $count = Build-List -Root $DataRoot -Glob $Pattern -OutFile $ListPath -MinSize $MinBytes
    Write-Host "Indexed files: $count"

    if (Test-Path $ModelPath) {
        .\full_lm.exe resume-list $ModelPath $ListPath $PassesPerCycle
    } else {
        .\full_lm.exe train-list $ListPath $ModelPath $PassesPerCycle $Slots $CtxCount
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed in cycle $c."
    }

    .\full_lm.exe eval $ModelPath $EvalFile 0
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed in cycle $c."
    }
}

Write-Header "Done"
Write-Host "Model: $ModelPath"
Write-Host "List:  $ListPath"
