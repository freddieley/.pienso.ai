# scale_train.ps1 — Parallel data-parallel training + merge
# Trains each shard as a separate process simultaneously, then merges all shard models.
# Usage: .\scale_train.ps1 [-Corpus wiki_corpus_clean.txt] [-Shards 8] [-Cycles 5] ...
param(
    [string]$Corpus        = "wiki_corpus_clean.txt",   # cleaned input corpus
    [string]$OutDir        = "scale_run",               # output directory
    [string]$ModelName     = "model_merged.bin",        # final merged model name
    [int]$Shards           = 8,                         # parallel shard count
    [int]$Cycles           = 3,                         # full pipeline cycles
    [int]$PassesPerShard   = 16,                        # passes per shard per cycle
    [uint32]$Slots         = 32,                        # hash slots per ctx bucket
    [uint32]$CtxCount      = 65536,                     # context table size
    [string]$Exe           = ".\full_lm.exe",
    [string]$DataPrep      = ".\data_prep.exe",
    [int]$MaxJobs          = 0                          # 0 = auto (CPU core count)
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ── helpers ─────────────────────────────────────────────────────────────────
function Banner($msg) {
    $line = "=" * 60
    Write-Host $line
    Write-Host "  $msg"
    Write-Host $line
}

function Die($msg) {
    Write-Host "FATAL: $msg" -ForegroundColor Red
    exit 1
}

# ── compile ──────────────────────────────────────────────────────────────────
Banner "COMPILE"
$needCompile = @()
if (-not (Test-Path $Exe))      { $needCompile += "full_lm.c" }
if (-not (Test-Path $DataPrep)) { $needCompile += "data_prep.c" }

if ("full_lm.c" -in $needCompile) {
    Write-Host "Compiling full_lm.c ..."
    $r = Start-Process gcc -ArgumentList @("-O2","-o",$Exe,"full_lm.c") -Wait -PassThru -NoNewWindow
    if ($r.ExitCode -ne 0) { Die "full_lm.c compile failed" }
    Write-Host "  -> $Exe OK"
}
if ("data_prep.c" -in $needCompile) {
    Write-Host "Compiling data_prep.c ..."
    $r = Start-Process gcc -ArgumentList @("-O2","-o",$DataPrep,"data_prep.c") -Wait -PassThru -NoNewWindow
    if ($r.ExitCode -ne 0) { Die "data_prep.c compile failed" }
    Write-Host "  -> $DataPrep OK"
} else {
    Write-Host "Binaries already compiled."
}

# ── prepare output dir ───────────────────────────────────────────────────────
Banner "SETUP"
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
# Resolve all dirs to absolute paths immediately
$OutDir    = (Resolve-Path $OutDir).Path
$shardsDir = Join-Path $OutDir "shards"
if (-not (Test-Path $shardsDir)) { New-Item -ItemType Directory -Path $shardsDir | Out-Null }
$modelsDir = Join-Path $OutDir "shard_models"
if (-not (Test-Path $modelsDir)) { New-Item -ItemType Directory -Path $modelsDir | Out-Null }
$Exe      = (Resolve-Path $Exe).Path
$Corpus   = (Resolve-Path $Corpus).Path
if (-not (Test-Path $DataPrep)) {
    $DataPrep = Join-Path (Split-Path $Exe) "data_prep.exe"
}
if (Test-Path $DataPrep) { $DataPrep = (Resolve-Path $DataPrep).Path }

if (-not (Test-Path $Corpus)) { Die "Corpus not found: $Corpus" }

# Auto-detect parallelism
if ($MaxJobs -le 0) {
    $MaxJobs = [Environment]::ProcessorCount
    if ($MaxJobs -gt $Shards) { $MaxJobs = $Shards }
}
Write-Host "Corpus    : $Corpus"
Write-Host "OutDir    : $OutDir"
Write-Host "Shards    : $Shards"
Write-Host "Cycles    : $Cycles"
Write-Host "MaxJobs   : $MaxJobs  (CPU cores)"
Write-Host "Slots     : $Slots   CtxCount: $CtxCount"

# ── shard the corpus (streaming — handles multi-GB files without RAM overflow) ─
Banner "SHARD CORPUS"
$corpusSz = (Get-Item $Corpus).Length
Write-Host ("Corpus size: {0:F2} MB" -f ($corpusSz / 1MB))
Write-Host "Counting lines (streaming)..."

# Pass 1: count lines
$totalLines = 0
$cr = [System.IO.StreamReader]::new($Corpus, [System.Text.Encoding]::UTF8)
while ($cr.ReadLine() -ne $null) { $totalLines++ }
$cr.Close()

$perShard = [math]::Ceiling($totalLines / $Shards)
Write-Host ("Total lines: {0}  -> ~{1} lines/shard" -f $totalLines, $perShard)

# Pre-create shard paths
$shardPaths = @()
for ($s = 0; $s -lt $Shards; $s++) {
    $shardPaths += Join-Path $shardsDir ("shard_{0:D2}.txt" -f $s)
}

# Pass 2: stream-write shards
Write-Host "Writing shards (streaming)..."
$shardWriters = @()
for ($s = 0; $s -lt $Shards; $s++) {
    $shardWriters += [System.IO.StreamWriter]::new($shardPaths[$s], $false, [System.Text.Encoding]::UTF8)
}
$cr      = [System.IO.StreamReader]::new($Corpus, [System.Text.Encoding]::UTF8)
$lineNum = 0
$ln      = $cr.ReadLine()
while ($ln -ne $null) {
    $si = [math]::Min([int]($lineNum / $perShard), $Shards - 1)
    $shardWriters[$si].WriteLine($ln)
    $lineNum++
    $ln = $cr.ReadLine()
}
$cr.Close()
$shardWriters | ForEach-Object { $_.Flush(); $_.Close() }

$shardPaths = $shardPaths | ForEach-Object { (Resolve-Path $_).Path }
$actualShards = $Shards
for ($s = 0; $s -lt $Shards; $s++) {
    $sz = if (Test-Path $shardPaths[$s]) { (Get-Item $shardPaths[$s]).Length } else { 0 }
    Write-Host ("  shard {0:D2}: ~{1} lines -> {2:F1} MB" -f $s, $perShard, ($sz / 1MB))
}
Write-Host "Actual shards created: $actualShards"

# ── training cycles ──────────────────────────────────────────────────────────
$mergedModel = Join-Path $OutDir $ModelName
$cycleCheckpoints = @()

# Stable shard model paths (no cycle suffix): resume updates in-place
$stableShardModels = 0..($actualShards-1) | ForEach-Object {
    Join-Path $modelsDir ("shard_{0:D2}.bin" -f $_)
}

for ($cycle = 1; $cycle -le $Cycles; $cycle++) {
    Banner "CYCLE $cycle / $Cycles"

    $shardModels = $stableShardModels  # always the same stable paths

    # ── parallel shard training (batch by MaxJobs) ───────────────────────────
    $batches = [math]::Ceiling($actualShards / $MaxJobs)
    $done    = 0

    for ($batch = 0; $batch -lt $batches; $batch++) {
        $batchStart = $batch * $MaxJobs
        $batchEnd   = [math]::Min($batchStart + $MaxJobs - 1, $actualShards - 1)
        $batchSize  = $batchEnd - $batchStart + 1

        Write-Host ("  Batch {0}: shards {1}-{2} (parallel x{3})" -f `
            ($batch+1), $batchStart, $batchEnd, $batchSize)

        $jobs = @()
        for ($si = $batchStart; $si -le $batchEnd; $si++) {
            $shardPath  = $shardPaths[$si]
            $shardModel = $stableShardModels[$si]  # stable path

            $exeAbs     = $Exe   # already absolute

            $argList = if ($cycle -eq 1) {
                # First cycle: fresh train → creates shard model
                @("train", $shardPath, $shardModel, $PassesPerShard, $Slots, $CtxCount)
            } elseif (Test-Path $shardModel) {
                # Subsequent cycles: resume in-place
                @("resume", $shardModel, $shardPath, $PassesPerShard)
            } else {
                # Fallback: retrain from scratch
                @("train", $shardPath, $shardModel, $PassesPerShard, $Slots, $CtxCount)
            }

            Write-Host ("    -> Job: shard {0:D2}  {1}" -f $si, ($argList[0]))

            $job = Start-Job -ScriptBlock {
                param($exe, $args_arr)
                $r = & $exe @args_arr 2>&1
                [PSCustomObject]@{ Output = $r -join "`n"; ExitCode = $LASTEXITCODE }
            } -ArgumentList $exeAbs, $argList

            $jobs += @{ Job = $job; Shard = $si }
        }

        # Wait for all jobs in this batch
        Write-Host "  Waiting for $($jobs.Count) parallel training jobs..."
        foreach ($j in $jobs) {
            $result = Receive-Job -Job $j.Job -Wait
            Remove-Job -Job $j.Job
            if ($result.ExitCode -ne 0) {
                Write-Host ("  WARNING: shard {0} failed" -f $j.Shard) -ForegroundColor Yellow
                Write-Host $result.Output
            } else {
                $done++
                Write-Host ("  OK shard {0:D2}" -f $j.Shard)
            }
        }
    }

    Write-Host ("  Trained {0}/{1} shards in cycle {2}" -f $done, $actualShards, $cycle)

    # ── merge all shard models ────────────────────────────────────────────────
    Write-Host ""
    Write-Host "  Merging $($shardModels.Count) shard models..."

    $existingModels = $shardModels | Where-Object { Test-Path $_ }
    if ($existingModels.Count -eq 0) { Die "No shard models to merge" }

    $cycleOut = Join-Path $OutDir ("checkpoint_cycle_{0}.bin" -f $cycle)

    if ($existingModels.Count -eq 1) {
        Copy-Item $existingModels[0] $cycleOut
    } else {
        $base      = $existingModels[0]
        $shardArgs = $existingModels[1..($existingModels.Count-1)]
        $mergeArgs = @("merge", $base) + $shardArgs + @($cycleOut)
        Write-Host ("  merge " + $existingModels.Count + " shards -> " + (Split-Path $cycleOut -Leaf))
        & $Exe @mergeArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  WARNING: merge failed, copying first shard" -ForegroundColor Yellow
            Copy-Item $existingModels[0] $cycleOut
        }
    }

    $cycleCheckpoints += $cycleOut
    Write-Host ("  Checkpoint -> $cycleOut")

    # ── quick eval on merged model ────────────────────────────────────────────
    Write-Host ""
    Write-Host "  Quick eval (frozen) on full corpus..."
    $evalOut = & $Exe eval $cycleOut $Corpus 0 2>&1
    $evalLine = ($evalOut | Select-String "accuracy:|acc=") | Select-Object -First 1
    if ($evalLine) { Write-Host ("  EVAL: " + $evalLine.Line) }
    else           { Write-Host "  EVAL: (no accuracy line found)" }
}

# ── finalize: pick best checkpoint ───────────────────────────────────────────
Banner "SELECT BEST CHECKPOINT"
$bestScore = -1
$bestCkpt  = $null

foreach ($ckpt in $cycleCheckpoints) {
    if (-not (Test-Path $ckpt)) { continue }
    $out = & $Exe eval $ckpt $Corpus 0 2>&1
    # Parse accuracy from "accuracy: 21.85%" or "acc=0.2185"
    $acc = 0.0
    if ($out -match "accuracy:\s*([\d.]+)%") { $acc = [double]$Matches[1] / 100 }
    elseif ($out -match "acc=([\d.]+)")       { $acc = [double]$Matches[1] }
    Write-Host ("  {0,-40}  acc={1:F4}" -f (Split-Path $ckpt -Leaf), $acc)
    if ($acc -gt $bestScore) { $bestScore = $acc; $bestCkpt = $ckpt }
}

if ($bestCkpt) {
    Copy-Item $bestCkpt $mergedModel -Force
    Write-Host ""
    Write-Host ("Best: $(Split-Path $bestCkpt -Leaf)  acc=$bestScore")
    Write-Host ("Final model -> $mergedModel")
} else {
    # Fallback: last checkpoint
    if ($cycleCheckpoints.Count -gt 0) {
        Copy-Item $cycleCheckpoints[-1] $mergedModel -Force
        Write-Host "Final model (last checkpoint) -> $mergedModel"
    }
}

# ── sample generation ─────────────────────────────────────────────────────────
Banner "GENERATE SAMPLES"
$prompts = @(
    "The definition of intelligence is",
    "Neural networks learn by",
    "In mathematics, a function is"
)
foreach ($p in $prompts) {
    Write-Host ""
    Write-Host "Prompt: $p"
    $out = & $Exe gen $mergedModel $p 200 2>&1
    Write-Host ($out -join " ")
}

Banner "DONE"
$finalKb = [math]::Round((Get-Item $mergedModel).Length / 1024)
Write-Host "Corpus        : $Corpus"
Write-Host "Shards used   : $actualShards"
Write-Host "Cycles        : $Cycles"
Write-Host "Final model   : $mergedModel  ($finalKb KB)"
