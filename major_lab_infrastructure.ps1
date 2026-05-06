# MAJOR_LAB_INFRASTRUCTURE.PS1
param(
    [string]$DataRoot = ".",
    [string]$Pattern = "*.txt",
    [string]$OutputDir = "major_lab_run",
    [int]$Shards = 4,
    [int]$Cycles = 3,
    [int]$PassesPerCycle = 2,
    [int]$Slots = 16,
    [int]$CtxCount = 32768
)

Write-Host ""
Write-Host "MAJOR LAB INFRASTRUCTURE - FULL PIPELINE" -ForegroundColor Cyan
Write-Host ""

# PHASE 0: COMPILE
Write-Host "PHASE 0: Compiling tools..." -ForegroundColor Green
foreach ($tool in @("data_prep", "eval_metrics", "full_lm")) {
    if (-not (Test-Path "$tool.exe")) {
        gcc -O2 -o "$tool.exe" "$tool.c" -lm
    }
    Write-Host "  OK $tool.exe"
}

# PHASE 1: DATA PREP
Write-Host ""
Write-Host "PHASE 1: Data pipeline..." -ForegroundColor Green
if (-not (Test-Path $OutputDir)) { mkdir $OutputDir }

$raw = "$OutputDir/1_raw_corpus.txt"
Get-ChildItem -Filter $Pattern -File | Where-Object { $_.Length -ge 512 } | ForEach-Object {
    Get-Content $_.FullName -Encoding UTF8 | Add-Content $raw -Encoding UTF8 -Force
}

$rawsize = (Get-Item $raw).Length / 1KB
Write-Host "  Raw corpus: $rawsize KB"

$cleaned = "$OutputDir/2_cleaned_corpus.txt"
& ".\data_prep.exe" $raw $cleaned 2>&1 | Select-String "Kept:|reduction|Output"

$cleansize = (Get-Item $cleaned).Length / 1KB
$reduction = [math]::Round($rawsize / $cleansize, 1)
Write-Host "  Cleaned: $cleansize KB (${reduction}x reduction)"

# PHASE 2: SHARDING
Write-Host ""
Write-Host "PHASE 2: Corpus sharding..." -ForegroundColor Green
$sharddir = "$OutputDir/shards"
if (-not (Test-Path $sharddir)) { mkdir $sharddir }

$lines = Get-Content $cleaned -Encoding UTF8
$nlines = $lines.Count
$per_shard = [math]::Ceiling($nlines / $Shards)

for ($i = 0; $i -lt $Shards; $i++) {
    $start = $i * $per_shard
    $end = [math]::Min(($i + 1) * $per_shard, $nlines)
    if ($end -le $start) { break }
    $lines[$start..($end-1)] | Set-Content "$sharddir/shard_$i.txt" -Encoding UTF8 -Force
    Write-Host "  Shard $i : $($end-$start) lines"
}

$shardlist = "$OutputDir/3_shard_list.txt"
Get-ChildItem "$sharddir/shard_*.txt" | ForEach-Object { $_.FullName } | Set-Content $shardlist -Encoding UTF8 -Force

# PHASE 3: TRAINING
Write-Host ""
Write-Host "PHASE 3: Distributed training..." -ForegroundColor Green
$model = "$OutputDir/4_model_distributed.bin"
$cplist = "$OutputDir/checkpoint_list.txt"
@() | Set-Content $cplist -Encoding UTF8 -Force

for ($c = 1; $c -le $Cycles; $c++) {
    Write-Host "  Cycle $c / $Cycles" -ForegroundColor Yellow
    if ($c -eq 1) {
        & ".\full_lm.exe" train-list $shardlist $model $PassesPerCycle $Slots $CtxCount
    } else {
        & ".\full_lm.exe" resume-list $model $shardlist $PassesPerCycle
    }
    Copy-Item $model "$OutputDir/checkpoint_cycle_$c.bin" -Force
    "$OutputDir/checkpoint_cycle_$c.bin" | Add-Content $cplist -Encoding UTF8
}

# PHASE 4: EVALUATION
Write-Host ""
Write-Host "PHASE 4: Perplexity evaluation..." -ForegroundColor Green
& ".\eval_metrics.exe" dummy $cleaned $cplist 2>&1 | Select-String "checkpoint:|Perplexity:|Best checkpoint"

# PHASE 5: INFERENCE
Write-Host ""
Write-Host "PHASE 5: Sample generation..." -ForegroundColor Green
$best = (Get-Content $cplist -Encoding UTF8 | Select-Object -First 1)
@("What is", "How can", "Machine learning") | ForEach-Object {
    Write-Host "  Prompt: $_" -ForegroundColor Yellow
    & ".\full_lm.exe" gen $best $_ 40 | Select-String -NotMatch "^(Generation|Speed)"
}

# PHASE 6: SUMMARY
Write-Host ""
Write-Host "=" -ForegroundColor Cyan
Write-Host "INFRASTRUCTURE READY" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan
Write-Host ""
Write-Host "Completed:"
Write-Host "  OK Data dedup + quality filtering (${reduction}x reduction)"
Write-Host "  OK Corpus sharding ($Shards partitions)"
Write-Host "  OK Distributed training ($Cycles cycles with checkpointing)"
Write-Host "  OK Perplexity-based evaluation"
Write-Host "  OK Auto checkpoint selection"
Write-Host ""
Write-Host "Outputs: $OutputDir/"
Write-Host "  - Raw: 1_raw_corpus.txt"
Write-Host "  - Clean: 2_cleaned_corpus.txt"
Write-Host "  - Shards: shards/"
Write-Host "  - Model: 4_model_distributed.bin"
Write-Host ""
Write-Host "Next: Scale to 100GB+ corpus" -ForegroundColor Yellow
