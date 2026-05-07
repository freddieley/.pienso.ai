param(
    [string]$Exe = ".\full_lm.exe",
    [string]$Model = "smoke.bin",
    [string]$Patch = "quick_patches.txt",
    [string]$Cases = "fast_cases.tsv",
    [int]$MaxLen = 80,
    [int]$MinConf = 220,
    [string]$Abstain = "?"
)

if (-not (Test-Path $Cases)) {
    Write-Host "cases file missing: $Cases"
    exit 1
}

$total = 0
$exact = 0
$abstained = 0

Get-Content $Cases | ForEach-Object {
    if ($_ -match '^\s*$' -or $_.StartsWith('#')) { return }
    $parts = $_ -split "`t", 2
    if ($parts.Count -lt 2) { return }
    $prompt = $parts[0]
    $expected = $parts[1]

    $out = & $Exe gen-fast $Model $Patch $prompt $MaxLen $MinConf $Abstain
    $total++
    if ($out -eq ($prompt + $expected)) { $exact++ }
    if ($out.EndsWith($Abstain)) { $abstained++ }
}

Write-Host "total=$total"
Write-Host "exact=$exact"
Write-Host ("exact_rate={0:P2}" -f ($(if ($total -gt 0) { $exact / $total } else { 0 })))
Write-Host "abstained=$abstained"
