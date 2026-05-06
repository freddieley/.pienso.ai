# build_corpus.ps1 — Massive corpus builder targeting 5B+ byte tokens
# Sources:
#   1. Project Gutenberg — download full plain-text books in parallel workers
#   2. Wikipedia — batched full-article extracts via MediaWiki API (random pages)
#   3. Combine + optional data_prep.exe clean pass
#
# Usage examples:
#   .\build_corpus.ps1                               # defaults: 10K books + 100K wiki articles
#   .\build_corpus.ps1 -GutenbergTarget 15000        # more books
#   .\build_corpus.ps1 -SkipGutenberg                # wiki only
#   .\build_corpus.ps1 -SkipWikipedia                # gutenberg only
#   .\build_corpus.ps1 -Parallel 12 -DelayMs 80      # faster (be polite to servers)
#   .\build_corpus.ps1 -Resume                       # continue interrupted run

param(
    [string]$OutFile        = "big_corpus.txt",   # final combined output
    [string]$WorkDir        = "corpus_work",      # temp dir for worker files + state
    [string]$DataPrep       = "data_prep.exe",    # dedup/filter binary (optional)
    [int]$GutenbergTarget   = 10000,              # Gutenberg books to collect
    [int]$WikiBatches       = 5000,               # Wikipedia request batches (20 articles each = 100K articles)
    [int]$Parallel          = 8,                  # parallel download workers
    [int]$MaxBookKB         = 2048,               # max KB per book (0 = unlimited)
    [int]$DelayMs           = 120,                # inter-request delay per worker (ms)
    [switch]$SkipGutenberg,                       # skip Gutenberg phase
    [switch]$SkipWikipedia,                       # skip Wikipedia phase
    [switch]$SkipClean,                           # skip data_prep.exe clean pass
    [switch]$Resume                               # continue from interrupted run
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

# ── helpers ───────────────────────────────────────────────────────────────────
function Banner([string]$msg) {
    Write-Host ""
    Write-Host ("=" * 60)
    Write-Host "  $msg"
    Write-Host ("=" * 60)
}

function Die([string]$msg) {
    Write-Host "FATAL: $msg" -ForegroundColor Red
    exit 1
}

function BytesFmt([long]$b) {
    if ($b -ge 1GB) { return ("{0:F2} GB" -f ($b / 1GB)) }
    if ($b -ge 1MB) { return ("{0:F1} MB" -f ($b / 1MB)) }
    return ("{0:F0} KB" -f ($b / 1KB))
}

# ── setup work directory ───────────────────────────────────────────────────────
if (-not (Test-Path $WorkDir)) { New-Item -ItemType Directory -Path $WorkDir | Out-Null }
$WorkDir = (Resolve-Path $WorkDir).Path
Write-Host "Work dir : $WorkDir"
Write-Host "Output   : $OutFile"
Write-Host "Workers  : $Parallel"

# ── Project Gutenberg phase ────────────────────────────────────────────────────
if (-not $SkipGutenberg) {
    Banner "GUTENBERG: Download catalog"

    $catalogPath = Join-Path $WorkDir "pg_catalog.csv"

    if ($Resume -and (Test-Path $catalogPath) -and (Get-Item $catalogPath).Length -gt 100000) {
        Write-Host "  Reusing existing catalog: $catalogPath"
    } else {
        Write-Host "  Fetching catalog from gutenberg.org (~7 MB)..."
        try {
            Invoke-WebRequest -Uri "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv" `
                        -UseBasicParsing -TimeoutSec 120 -OutFile $catalogPath `
                        -Headers @{ "User-Agent" = "pienso-corpus-builder/2.0 (educational, bulk-text-research)" }
            Write-Host ("  Catalog saved: {0}" -f (BytesFmt (Get-Item $catalogPath).Length))
        }
        catch { Die "Failed to download Gutenberg catalog: $_" }
    }

    # Parse catalog — columns: Text#, Type, Issued, Title, Language, Authors, ...
    Write-Host "  Parsing catalog for English plain-text books..."
    $catalog = Import-Csv $catalogPath
    $englishIds = @()
    foreach ($row in $catalog) {
        if ($row.Type -eq 'Text' -and ($row.Language -match '\bEnglish\b' -or $row.Language -eq 'en')) {
            $id = $row.'Text#'
            if ($id -match '^\d+$') { $englishIds += [int]$id }
        }
    }
    Write-Host ("  English text books found: {0}" -f $englishIds.Count)

    # Shuffle and limit
    $rng = [System.Random]::new(42)
    $shuffled = $englishIds | Sort-Object { $rng.Next() }
    $selected = $shuffled | Select-Object -First $GutenbergTarget
    Write-Host ("  Selected {0} book IDs for download" -f $selected.Count)

    Banner "GUTENBERG: Parallel download ($Parallel workers)"

    # Divide IDs into $Parallel slices — each worker handles its slice sequentially
    $sliceSize = [math]::Ceiling($selected.Count / $Parallel)
    $maxBytes  = if ($MaxBookKB -gt 0) { $MaxBookKB * 1024 } else { [long]::MaxValue }

    $workerScript = {
        param([int[]]$ids, [string]$outFile, [string]$doneFile, [long]$maxBytes, [int]$delayMs)

        $done = @{}
        if (Test-Path $doneFile) {
            Get-Content $doneFile | ForEach-Object { $done["$_"] = 1 }
        }

        $ok = 0; $fail = 0

        foreach ($id in $ids) {
            if ($done.ContainsKey("$id")) { $ok++; continue }

            # Try primary URL then two fallbacks
            $urls = @(
                "https://www.gutenberg.org/cache/epub/$id/pg$id.txt",
                "https://gutenberg.org/files/$id/$id-0.txt",
                "https://gutenberg.org/files/$id/$id.txt"
            )

            $got = $false
            foreach ($url in $urls) {
                try {
                    $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 45 `
                                -Headers @{ "User-Agent" = "pienso-corpus-builder/2.0 (educational)" }

                    # Content is auto-decoded by PS 5.1 from response charset
                    $text = $resp.Content

                    # Strip Gutenberg header/footer boilerplate
                    $startMark = "*** START OF"
                    $endMark   = "*** END OF"
                    $si = $text.IndexOf($startMark)
                    $ei = $text.IndexOf($endMark)
                    if ($si -gt 0) {
                        $si = $text.IndexOf("`n", $si) + 1
                        if ($ei -gt $si) { $text = $text.Substring($si, $ei - $si) }
                        else             { $text = $text.Substring($si) }
                    }

                    # Cap size
                    if ($text.Length -gt $maxBytes) { $text = $text.Substring(0, [int]$maxBytes) }

                    # Remove non-printable control chars (keep LF, CR, TAB)
                    $text = [System.Text.RegularExpressions.Regex]::Replace(
                        $text, "[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "")

                    $text = $text.Trim()
                    if ($text.Length -lt 1000) { break }  # too short = boilerplate only

                    $block = "=== Gutenberg:$id ===`n$text`n`n"

                    # Mutex-free append — each worker writes to its own file
                    [System.IO.File]::AppendAllText($outFile, $block,
                        [System.Text.Encoding]::UTF8)

                    "$id" | Add-Content -Path $doneFile -Encoding UTF8
                    $ok++
                    $got = $true
                    break
                }
                catch { continue }
            }

            if (-not $got) { $fail++ }

            if ($delayMs -gt 0) { Start-Sleep -Milliseconds $delayMs }
        }

        [PSCustomObject]@{ OK = $ok; Fail = $fail }
    }

    # Launch worker jobs
    $jobs = @()
    for ($w = 0; $w -lt $Parallel; $w++) {
        $start      = $w * $sliceSize
        $end        = [math]::Min($start + $sliceSize - 1, $selected.Count - 1)
        if ($start -gt $selected.Count - 1) { break }
        $slice      = [int[]]($selected[$start..$end])
        $workerOut  = Join-Path $WorkDir ("gutenberg_w{0:D2}.txt" -f $w)
        $workerDone = Join-Path $WorkDir ("gutenberg_w{0:D2}.done" -f $w)
        Write-Host ("  Worker {0}: {1} books -> {2}" -f $w, $slice.Count, (Split-Path $workerOut -Leaf))
        $jobs += Start-Job -ScriptBlock $workerScript `
                    -ArgumentList $slice, $workerOut, $workerDone, $maxBytes, $DelayMs
    }

    # Progress monitor — poll every 30 s while workers run
    Write-Host "  Workers launched. Polling progress every 30 s..."
    $allDone = $false
    while (-not $allDone) {
        Start-Sleep -Seconds 30
        $running = @($jobs | Where-Object { $_.State -eq 'Running' })
        $totalBytes = [long]0
        for ($w = 0; $w -lt $Parallel; $w++) {
            $f = Join-Path $WorkDir ("gutenberg_w{0:D2}.txt" -f $w)
            if (Test-Path $f) { $totalBytes += (Get-Item $f).Length }
        }
        Write-Host ("  [{0}] {1} workers running | downloaded: {2}" -f `
            (Get-Date -Format "HH:mm:ss"), $running.Count, (BytesFmt $totalBytes))
        if ($running.Count -eq 0) { $allDone = $true }
    }

    # Collect results
    $totalOK = 0; $totalFail = 0
    foreach ($j in $jobs) {
        $r = Receive-Job -Job $j -Wait
        Remove-Job -Job $j
        $totalOK   += $r.OK
        $totalFail += $r.Fail
    }
    Write-Host ("  Gutenberg complete: {0} OK, {1} failed" -f $totalOK, $totalFail)

    # Combine worker files into OutFile
    Banner "GUTENBERG: Combining worker files"
    for ($w = 0; $w -lt $Parallel; $w++) {
        $f = Join-Path $WorkDir ("gutenberg_w{0:D2}.txt" -f $w)
        if (Test-Path $f) {
            $sz = (Get-Item $f).Length
            Write-Host ("  Appending worker {0}: {1}" -f $w, (BytesFmt $sz))
            $reader = [System.IO.StreamReader]::new($f, [System.Text.Encoding]::UTF8)
            $writer = [System.IO.StreamWriter]::new($OutFile, $true, [System.Text.Encoding]::UTF8)
            $writer.AutoFlush = $false
            $buf = New-Object char[] 65536
            while (($read = $reader.Read($buf, 0, $buf.Length)) -gt 0) {
                $writer.Write($buf, 0, $read)
            }
            $reader.Close()
            $writer.Flush()
            $writer.Close()
        }
    }
    $outSz = if (Test-Path $OutFile) { (Get-Item $OutFile).Length } else { 0 }
    Write-Host ("  Combined size so far: {0}" -f (BytesFmt $outSz))
}

# ── Wikipedia phase ────────────────────────────────────────────────────────────
if (-not $SkipWikipedia) {
    Banner "WIKIPEDIA: Full article extracts ($WikiBatches batches x 20 articles)"

    $wikiDone = Join-Path $WorkDir "wikipedia.done"
    $wikiOut  = Join-Path $WorkDir "wikipedia_raw.txt"
    $batchesDone = 0
    if ($Resume -and (Test-Path $wikiDone)) {
        $batchesDone = [int](Get-Content $wikiDone)
        Write-Host ("  Resuming from batch {0}" -f $batchesDone)
    }

    $apiBase = "https://en.wikipedia.org/w/api.php"
    $wikiWriter = [System.IO.StreamWriter]::new($wikiOut, $true, [System.Text.Encoding]::UTF8)
    $wikiWriter.AutoFlush = $false

    $totalWikiChars = 0
    $totalWikiArticles = 0

    for ($b = $batchesDone; $b -lt $WikiBatches; $b++) {
        $uri = ($apiBase +
            "?action=query&format=json&utf8=1" +
            "&prop=extracts&explaintext=1&exsectionformat=plain&exlimit=max" +
            "&generator=random&grnnamespace=0&grnlimit=20")
        try {
            $resp = Invoke-WebRequest -Uri $uri -UseBasicParsing -TimeoutSec 30 `
                        -Headers @{ "User-Agent" = "pienso-corpus-builder/2.0 (educational)" }
            $json = $resp.Content | ConvertFrom-Json

            $pages = $json.query.pages.PSObject.Properties.Value
            foreach ($page in $pages) {
                # Safe property check — some pages lack extract (redirects, stubs)
                $extractProp = $page.PSObject.Properties['extract']
                if (-not $extractProp) { continue }
                $extract = $extractProp.Value
                if (-not $extract -or $extract.Length -lt 50) { continue }
                # Remove section header lines (lines like "== History ==")
                $clean = ($extract -split "`n" | Where-Object { $_ -notmatch '^={2,}' }) -join "`n"
                $clean = $clean.Trim()
                if ($clean.Length -lt 50) { continue }
                $block = "=== Wikipedia: $($page.title) ===`n$clean`n`n"
                $wikiWriter.Write($block)
                $totalWikiChars += $block.Length
                $totalWikiArticles++
            }

            # Save progress every 50 batches
            if ($b % 50 -eq 0) {
                $wikiWriter.Flush()
                Set-Content -Path $wikiDone -Value "$b" -Encoding UTF8
                if ($b -gt 0) {
                    Write-Host ("  [{0:D5}/{1}] articles:{2}  chars:{3}" -f `
                        $b, $WikiBatches, $totalWikiArticles, (BytesFmt $totalWikiChars))
                }
            }
        }
        catch {
            Write-Host ("  WARNING: batch {0} failed: {1}" -f $b, $_)
        }

        if ($DelayMs -gt 0) { Start-Sleep -Milliseconds $DelayMs }
    }
    $wikiWriter.Flush()
    $wikiWriter.Close()
    Set-Content -Path $wikiDone -Value "$WikiBatches" -Encoding UTF8
    Write-Host ("  Wikipedia complete: {0} articles, {1}" -f $totalWikiArticles, (BytesFmt $totalWikiChars))

    # Append wiki file to main output
    Banner "WIKIPEDIA: Appending to output"
    $wikiSz = if (Test-Path $wikiOut) { (Get-Item $wikiOut).Length } else { 0 }
    Write-Host ("  Wiki file size: {0}" -f (BytesFmt $wikiSz))
    if ($wikiSz -gt 0) {
        $reader = [System.IO.StreamReader]::new($wikiOut, [System.Text.Encoding]::UTF8)
        $writer = [System.IO.StreamWriter]::new($OutFile, $true, [System.Text.Encoding]::UTF8)
        $writer.AutoFlush = $false
        $buf = New-Object char[] 65536
        while (($read = $reader.Read($buf, 0, $buf.Length)) -gt 0) {
            $writer.Write($buf, 0, $read)
        }
        $reader.Close()
        $writer.Flush()
        $writer.Close()
    }
}

# ── final stats & optional clean pass ─────────────────────────────────────────
Banner "DONE"
if (Test-Path $OutFile) {
    $finalSz = (Get-Item $OutFile).Length
    Write-Host ("Raw corpus size : {0}" -f (BytesFmt $finalSz))
    Write-Host ("Approx tokens   : {0:F1} M byte-tokens" -f ($finalSz / 1e6))
    Write-Host ("Target          : 5,000 M byte-tokens")
    $pct = [math]::Round($finalSz / 50000000, 1)
    Write-Host ("Progress        : {0}% of 5B target" -f $pct)
} else {
    Write-Host "WARNING: Output file not found."
}

if (-not $SkipClean -and (Test-Path $DataPrep)) {
    Banner "CLEAN PASS (data_prep.exe)"
    $cleanFile = [System.IO.Path]::ChangeExtension($OutFile, $null) + "_clean.txt"
    Write-Host "  Running: $DataPrep $OutFile $cleanFile"
    & $DataPrep $OutFile $cleanFile
    if ($LASTEXITCODE -eq 0 -and (Test-Path $cleanFile)) {
        $cleanSz = (Get-Item $cleanFile).Length
        Write-Host ("  Clean corpus: {0}" -f (BytesFmt $cleanSz))
        Write-Host ("  Reduction   : {0:F1}%" -f (100 * (1 - $cleanSz / (Get-Item $OutFile).Length)))
    }
} elseif (-not $SkipClean) {
    Write-Host "NOTE: data_prep.exe not found -- skipping dedup/filter."
    $cleanHint = $OutFile -replace '\.txt$', '_clean.txt'
    Write-Host ("      Run manually: .\data_prep.exe $OutFile $cleanHint")
}

Write-Host ""
Write-Host "Next step: train on the corpus"
Write-Host ("  Byte-level: .\scale_train.ps1 -Corpus $OutFile -OutDir scale_5b -Shards 32 -Cycles 5")
Write-Host ("  Word-level: .\word_lm.exe train $OutFile word_5b.wlm 8")
