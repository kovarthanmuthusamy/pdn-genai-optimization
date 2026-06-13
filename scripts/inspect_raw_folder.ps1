# Inspect multifreq Raw folder layout (run in PowerShell, paste full output to chat)
# Usage:  powershell -ExecutionPolicy Bypass -File "C:\...\gan\scripts\inspect_raw_folder.ps1"
# Or:     cd C:\Users\muthusamy\Desktop   then paste the block below

$Raw = "C:\Users\muthusamy\Desktop\Raw"
if (-not (Test-Path $Raw)) {
    Write-Host "ERROR: Folder not found: $Raw"
    exit 1
}

Write-Host "========== RAW ROOT =========="
Write-Host "Path: $Raw"
Get-ChildItem $Raw | Select-Object Mode, Name, @{N='SizeMB';E={if($_.PSIsContainer){''}else{[math]::Round($_.Length/1MB,2)}}} | Format-Table -AutoSize

Write-Host "`n========== HEATMAP FOLDERS =========="
$hm = Get-ChildItem $Raw -Directory -Filter "heatmap_*MHz" | Sort-Object Name
$hm | ForEach-Object { Write-Host "  $($_.Name)" }
if (-not $hm) { Write-Host "  (none found)" }

Write-Host "`n========== OTHER KEY FOLDERS =========="
@("imp", "Imp", "decap_combinations", "decap", "Decap") | ForEach-Object {
    $p = Join-Path $Raw $_
    if (Test-Path $p) { Write-Host "  OK  $_" } else { Write-Host "  --  $_" }
}

Write-Host "`n========== DECAP CSV =========="
$decapDir = Join-Path $Raw "decap_combinations"
if (Test-Path $decapDir) {
    Get-ChildItem $decapDir -Filter "*.csv" | ForEach-Object { Write-Host "  $($_.FullName)" }
} else {
    Get-ChildItem $Raw -Filter "*.csv" -File | Select-Object -First 5 | ForEach-Object { Write-Host "  $($_.Name)" }
}

Write-Host "`n========== COUNTS (may take ~30s) =========="
$impDir = if (Test-Path (Join-Path $Raw "imp")) { Join-Path $Raw "imp" } else { Join-Path $Raw "Imp" }
$piImp = if (Test-Path $impDir) { (Get-ChildItem $impDir -Directory -Filter "PI-*").Count } else { 0 }
Write-Host "  PI folders under imp/: $piImp"

foreach ($h in $hm | Select-Object -First 3) {
    $n = (Get-ChildItem $h.FullName -Directory -Filter "PI-*").Count
    Write-Host "  PI folders under $($h.Name): $n"
}

Write-Host "`n========== SAMPLE PI (first heatmap folder) =========="
$firstHm = $hm | Select-Object -First 1
if ($firstHm) {
    $pi = Get-ChildItem $firstHm.FullName -Directory -Filter "PI-*" | Sort-Object Name | Select-Object -First 1
    if ($pi) {
        Write-Host "  PI dir: $($pi.FullName)"
        $pg = Join-Path $pi.FullName "Power_GND"
        if (Test-Path $pg) {
            Write-Host "  Power_GND contents:"
            Get-ChildItem $pg -File | Select-Object -First 8 | ForEach-Object { Write-Host "    $($_.Name)" }
        } else {
            Write-Host "  (no Power_GND subfolder)"
            Get-ChildItem $pi.FullName -Recurse -Filter "*.map" -File | Select-Object -First 5 | ForEach-Object {
                Write-Host "    map: $($_.FullName)"
            }
        }
    }
}

Write-Host "`n========== SAMPLE IMPEDANCE (same PI name if possible) =========="
if ($pi -and (Test-Path $impDir)) {
    $impPi = Join-Path $impDir $pi.Name
    if (Test-Path $impPi) {
        Write-Host "  imp PI: $impPi"
        $pg = Join-Path $impPi "Power_GND"
        if (Test-Path $pg) {
            Get-ChildItem $pg -Filter "*.csv" -File | Select-Object -First 6 | ForEach-Object { Write-Host "    $($_.Name)" }
        } else {
            Get-ChildItem $impPi -Recurse -Filter "*IC1*.csv" -File | Select-Object -First 3 | ForEach-Object {
                Write-Host "    $($_.Name)"
            }
        }
    } else {
        Write-Host "  (no matching imp/$($pi.Name))"
        $any = Get-ChildItem $impDir -Directory -Filter "PI-*" | Select-Object -First 1
        if ($any) { Write-Host "  example imp PI: $($any.FullName)" }
    }
}

Write-Host "`n========== *_true SUBFOLDERS? =========="
$trueDirs = Get-ChildItem $Raw -Directory -Filter "*_true"
if ($trueDirs) { $trueDirs | ForEach-Object { Write-Host "  $($_.Name)" } }
else { Write-Host "  (none — flat layout at Raw root)" }

Write-Host "`n========== DONE — paste everything above into chat =========="
