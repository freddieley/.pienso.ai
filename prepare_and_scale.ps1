# PREPARE_AND_SCALE.PS1 - Infrastructure for major lab scale
# 
# Pipeline:
#   1. Data preparation (dedup, quality filtering)
#   2. Corpus sharding for distributed training  
#   3. Multi-pass training with batch accumulation
#   4. Distributed evaluation and checkpointing
#
# Goal: Scale from 5KB → 1MB (200x) via data cleaning + distributed infrastructure

param(
    [string]$DataRoot = ".",
    [string]$Pattern = "*.txt",
    [int]$Slots = 16,
    [int]$CtxCount = 32768,
    [int]$ShardCount = 4,  # Parallel shards for distributed training
    [int]$PassesPerShard = 2,
    [int]$Cycles = 2,
    [string]$OutputDir = "prepared"
)

# Step 1: Compile tools
Write-Host "=== COMPILING DATA PIPELINE ===" -ForegroundColor Green

if (!(Test-Path "data_prep.exe" -PathType Leaf)) {
    Write-Host "Building data_prep.c..." -ForegroundColor Yellow
    gcc -O2 -o data_prep.exe data_prep.c
    if ($LASTEXITCODE -ne 0) {
        Write-Error "data_prep.c compilation failed"
        exit 1
    }
}

if (!(Test-Path "full_lm.exe" -PathType Leaf)) {
    Write-Host "Building full_lm.c..." -ForegroundColor Yellow
    gcc -O2 -o full_lm.exe full_lm.c
    if ($LASTEXITCODE -ne 0) {
        Write-Error "full_lm.c compilation failed"
        exit 1
    }
}

# Step 2: Create output directories
if (!(Test-Path $OutputDir -PathType Container)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

Write-Host "Output directory: $OutputDir" -ForegroundColor Cyan

# Step 3: Collect raw corpus files
Write-Host "`n=== COLLECTING RAW CORPUS ===" -ForegroundColor Green
$RawCorpusPath = "$OutputDir/raw_corpus.txt"
$raw_count = 0

# Combine all matching txt files into single raw corpus
Get-ChildItem -Path $DataRoot -Filter $Pattern -File | ForEach-Object {
    $filesize = $_.Length / 1KB
    if ($filesize -ge 1) {  # Skip tiny files
        Write-Host "  Adding: $($_.Name) ($filesize KB)" -ForegroundColor Gray
        Get-Content $_.FullName -Encoding UTF8 -ErrorAction SilentlyContinue | Add-Content $RawCorpusPath -Encoding UTF8 -Force
        $raw_count += @(Get-Content $_.FullName -Encoding UTF8 -ErrorAction SilentlyContinue).Length
    }
}

$raw_bytes = (Get-Item $RawCorpusPath -ErrorAction SilentlyContinue).Length / 1KB
Write-Host "Raw corpus: $raw_bytes KB ($raw_count lines)" -ForegroundColor Cyan

# Step 4: Data preparation (dedup + quality filtering)
Write-Host "`n=== DATA PREPARATION (Dedup + Quality Filtering) ===" -ForegroundColor Green

$CleanedCorpusPath = "$OutputDir/cleaned_corpus.txt"
Write-Host "Deduplicating and filtering: $RawCorpusPath → $CleanedCorpusPath" -ForegroundColor Yellow

& ".\data_prep.exe" $RawCorpusPath $CleanedCorpusPath

if ($LASTEXITCODE -ne 0) {
    Write-Error "Data prep failed"
    exit 1
}

$cleaned_bytes = (Get-Item $CleanedCorpusPath -ErrorAction SilentlyContinue).Length / 1KB
$reduction_ratio = $raw_bytes > 0 ? [math]::Round($raw_bytes / $cleaned_bytes, 2) : 1.0
Write-Host "After cleaning: $cleaned_bytes KB (reduction: ${reduction_ratio}x)" -ForegroundColor Cyan

# Step 5: Shard cleaned corpus for distributed training
Write-Host "`n=== CORPUS SHARDING FOR DISTRIBUTED TRAINING ===" -ForegroundColor Green

$lines = @(Get-Content $CleanedCorpusPath -Encoding UTF8 -ErrorAction SilentlyContinue)
$total_lines = $lines.Count
$lines_per_shard = [math]::Ceiling($total_lines / $ShardCount)

Write-Host "Sharding $total_lines lines into $ShardCount shards (~$lines_per_shard lines each)" -ForegroundColor Yellow

$ShardDir = "$OutputDir/shards"
if (!(Test-Path $ShardDir -PathType Container)) {
    New-Item -ItemType Directory -Path $ShardDir -Force | Out-Null
}

for ($i = 0; $i -lt $ShardCount; $i++) {
    $start = $i * $lines_per_shard
    $end = [math]::Min(($i + 1) * $lines_per_shard, $total_lines)
    $shard_file = "$ShardDir/shard_$($i).txt"
    $shard_lines = $lines[$start..($end-1)]
    $shard_lines | Set-Content -Path $shard_file -Encoding UTF8 -Force
    $shard_size = (Get-Item $shard_file).Length / 1KB
    Write-Host "  Shard $i: $($end-$start) lines ($shard_size KB)" -ForegroundColor Gray
}

# Create shard list for full_lm
$ShardListPath = "$OutputDir/shard_list.txt"
Get-ChildItem "$ShardDir/shard_*.txt" -File | ForEach-Object {
    Add-Content $ShardListPath $_.FullName -Encoding UTF8 -Force
}

# Step 6: Distributed training loop
Write-Host "`n=== DISTRIBUTED TRAINING (Multi-Shard + Multi-Cycle) ===" -ForegroundColor Green

$BaseModelPath = "$OutputDir/model_distributed.bin"
$EvalCorpusPath = $CleanedCorpusPath  # Use cleaned corpus for eval too (can split further later)

for ($cycle = 1; $cycle -le $Cycles; $cycle++) {
    Write-Host "`n--- Cycle $cycle / $Cycles ---" -ForegroundColor Cyan
    
    if ($cycle -eq 1) {
        # First cycle: train fresh on all shards
        Write-Host "Training fresh model on shard list (all $ShardCount shards)..." -ForegroundColor Yellow
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        
        & ".\full_lm.exe" train-list $ShardListPath $BaseModelPath $PassesPerShard $Slots $CtxCount 2>&1 | Tee-Object -Variable train_output
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Training failed on cycle $cycle"
            exit 1
        }
    } else {
        # Subsequent cycles: resume training on all shards
        Write-Host "Resuming model training on shard list (pass $cycle)..." -ForegroundColor Yellow
        & ".\full_lm.exe" resume-list $BaseModelPath $ShardListPath $PassesPerShard 2>&1 | Tee-Object -Variable train_output
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Resume training failed on cycle $cycle"
            exit 1
        }
    }
    
    # Evaluation on cleaned corpus
    Write-Host "Evaluating on cleaned corpus (frozen weights)..." -ForegroundColor Yellow
    & ".\full_lm.exe" eval $BaseModelPath $CleanedCorpusPath 0 2>&1 | Tee-Object -Variable eval_output
    
    if ($eval_output -match "Accuracy: ([\d.]+)%") {
        $accuracy = $Matches[1]
        Write-Host "Cycle $cycle accuracy: $accuracy%" -ForegroundColor Green
    }
    
    # Save checkpoint with cycle number
    $checkpoint = "$OutputDir/checkpoint_cycle_$cycle.bin"
    Copy-Item $BaseModelPath $checkpoint -Force
    Write-Host "Checkpoint saved: $checkpoint" -ForegroundColor Cyan
}

# Step 7: Final evaluation and summary
Write-Host "`n=== FINAL EVALUATION ===" -ForegroundColor Green

$FinalModel = $BaseModelPath
Write-Host "Final model: $FinalModel" -ForegroundColor Yellow
Write-Host "Test size: $cleaned_bytes KB" -ForegroundColor Cyan
Write-Host "Distributed shards: $ShardCount" -ForegroundColor Cyan
Write-Host "Training cycles: $Cycles" -ForegroundColor Cyan

# Generate inference example
Write-Host "`n=== SAMPLE GENERATION ===" -ForegroundColor Green
$Prompts = @(
    "What is",
    "How can",
    "When should",
    "Why is"
)

$Prompts | ForEach-Object {
    Write-Host "Prompt: '$_'" -ForegroundColor Yellow
    & ".\full_lm.exe" gen $FinalModel $_ 50 2>&1 | ForEach-Object {
        if ($_ -notmatch "^(Generation|Speed|Tokens)") {
            Write-Host "  $_" -ForegroundColor White
        }
    }
}

Write-Host "`n=== INFRASTRUCTURE SUMMARY ===" -ForegroundColor Green
Write-Host "✓ Data pipeline: dedup + quality filtering (${reduction_ratio}x reduction)" -ForegroundColor Green
Write-Host "✓ Corpus sharding: $ShardCount parallel training shards" -ForegroundColor Green
Write-Host "✓ Distributed training: multi-cycle with checkpointing" -ForegroundColor Green
Write-Host "✓ Cleaned corpus: $CleanedCorpusPath" -ForegroundColor Green
Write-Host "✓ Final model: $FinalModel" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. Scale corpus to 100MB+ (add more training data)" -ForegroundColor Gray
Write-Host "  2. Add perplexity-based checkpoint selection" -ForegroundColor Gray
Write-Host "  3. Implement GPU-parallel sharding" -ForegroundColor Gray
Write-Host "  4. Add safety/toxicity filtering" -ForegroundColor Gray
Write-Host "  5. Deploy inference API with batching" -ForegroundColor Gray
