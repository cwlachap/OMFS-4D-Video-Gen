# Test VHAP with conda env (no Streamlit). Uses existing vhap_data.
# Run from project root: .\test_vhap_conda.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = "c:\Users\lacha\Desktop\OMFS 4d Video Gen"
$CondaPython = "C:\Users\lacha\miniconda3\envs\VHAP\python.exe"
$VHAP_ROOT = "$ProjectRoot\vhap_repo"
$DATA_ROOT = "$ProjectRoot\vhap_data\monocular"
$OUTPUT_ROOT = "$ProjectRoot\vhap_output_conda"   # separate from vhap_output
$EXPORT_ROOT = "$ProjectRoot\vhap_export_conda"

# Your existing sequence (folder name under vhap_data/monocular)
$SEQUENCE = "input_video"

# ----- Step 1: Track (FLAME fitting) -----
$TrackOut = "$OUTPUT_ROOT\${SEQUENCE}_whiteBg_staticOffset"
Write-Host "`n=== 1. VHAP Track (conda) ===`n" -ForegroundColor Cyan
Write-Host "  Data:  $DATA_ROOT" -ForegroundColor Gray
Write-Host "  Seq:   $SEQUENCE" -ForegroundColor Gray
Write-Host "  Out:   $TrackOut`n" -ForegroundColor Gray

& $CondaPython "$VHAP_ROOT\vhap\track.py" `
  --data.root-folder $DATA_ROOT `
  --exp.output-folder $TrackOut `
  --data.sequence $SEQUENCE `
  --data.background-color white `
  --data.landmark-source face-alignment `
  --data.landmark-detector-njobs 1

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "`nTrack done.`n" -ForegroundColor Green

# ----- Step 2: Export -----
$ExportOut = "$EXPORT_ROOT\${SEQUENCE}_conda"
Write-Host "=== 2. VHAP Export (conda) ===`n" -ForegroundColor Cyan
Write-Host "  Src:   $TrackOut" -ForegroundColor Gray
Write-Host "  Out:   $ExportOut`n" -ForegroundColor Gray

& $CondaPython "$VHAP_ROOT\vhap\export_as_nerf_dataset.py" `
  --src-folder $TrackOut `
  --tgt-folder $ExportOut `
  --background-color white `
  --no-create-mask-from-mesh

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "`nExport done.`n" -ForegroundColor Green

Write-Host "Conda test finished. Outputs:" -ForegroundColor Green
Write-Host "  Track:  $TrackOut"
Write-Host "  Export: $ExportOut"
Write-Host "`nNext: compare with vhap_output / vhap_export, or run conversion + train on this export.`n"
