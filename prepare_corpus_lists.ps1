param(
    [string]$DataRoot = ".",
    [string]$Pattern = "*.txt",
    [string]$TrainList = "train_list.txt",
    [string]$EvalList = "eval_list.txt",
    [double]$EvalFraction = 0.1,
    [int]$Seed = 42,
    [int]$MinBytes = 256
)

$ErrorActionPreference = "Stop"

if ($EvalFraction -le 0.0 -or $EvalFraction -ge 1.0) {
    throw "EvalFraction must be in (0,1)."
}

$files = Get-ChildItem -Path $DataRoot -Recurse -File -Filter $Pattern |
    Where-Object { $_.Length -ge $MinBytes } |
    Where-Object { $_.FullName -notmatch "\\(\.git|node_modules|bin|obj|\.vs)\\" } |
    Select-Object -ExpandProperty FullName

if (-not $files -or $files.Count -lt 2) {
    throw "Need at least 2 files for a train/eval split."
}

$rand = New-Object System.Random($Seed)
$shuffled = $files | Sort-Object { $rand.Next() }

$evalCount = [Math]::Max(1, [int]([Math]::Round($shuffled.Count * $EvalFraction)))
if ($evalCount -ge $shuffled.Count) { $evalCount = $shuffled.Count - 1 }

$eval = $shuffled | Select-Object -First $evalCount
$train = $shuffled | Select-Object -Skip $evalCount

"# auto-generated train list" | Set-Content -Path $TrainList -Encoding ASCII
$train | Add-Content -Path $TrainList -Encoding ASCII

"# auto-generated eval list" | Set-Content -Path $EvalList -Encoding ASCII
$eval | Add-Content -Path $EvalList -Encoding ASCII

Write-Host "train files: $($train.Count)"
Write-Host "eval files:  $($eval.Count)"
Write-Host "train list:  $TrainList"
Write-Host "eval list:   $EvalList"
