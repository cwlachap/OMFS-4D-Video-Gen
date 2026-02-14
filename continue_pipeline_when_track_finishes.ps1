param(
    [string]$TrackTerminalLog = "C:\Users\lacha\.cursor\projects\c-Users-lacha-Desktop-OMFS-4d-Video-Gen\terminals\533092.txt",
    [int]$PollSeconds = 60,
    [int]$Iterations = 50000,
    [double]$LefortMm = 0,
    [double]$BssoMm = 0,
    [int]$RenderIteration = -1,
    [string]$RigMode = "flame_only",
    [string]$CanonicalHeadAsset = "",
    [string]$DeformationMap = "",
    [switch]$EvalStrict,
    [string]$DeterministicIndices = "0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,272,292"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = "c:\Users\lacha\Desktop\OMFS 4d Video Gen"
$VenvPython = "$ProjectRoot\venv\Scripts\python.exe"
$DataConda = "$ProjectRoot\02_Visual_Engine\data_conda"
$ModelConda = "$ProjectRoot\02_Visual_Engine\output\model_conda"
$FinalVideo = "$ProjectRoot\final_prediction_conda.mp4"
$StrictExportDir = "$ProjectRoot\02_Visual_Engine\output\model_conda\eval_strict\deterministic_frames"
$StrictReportDir = "$ProjectRoot\02_Visual_Engine\output\model_conda\eval_strict\reports"

function Get-ExitCodeFromLog {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return $null
    }
    $content = Get-Content -Path $Path -Raw -ErrorAction SilentlyContinue
    if ([string]::IsNullOrWhiteSpace($content)) {
        return $null
    }
    $m = [regex]::Match($content, "exit_code:\s*(\d+)")
    if ($m.Success) {
        return [int]$m.Groups[1].Value
    }
    return $null
}

if (-not (Test-Path $VenvPython)) {
    throw "Missing venv python at $VenvPython"
}
if (-not (Test-Path $DataConda)) {
    throw "Missing converted dataset at $DataConda. Wait for current run to finish export+convert first."
}

Write-Host "Watching track terminal log: $TrackTerminalLog" -ForegroundColor Cyan
Write-Host "Will continue automatically with train + render when tracking job exits cleanly." -ForegroundColor Cyan

while ($true) {
    $exitCode = Get-ExitCodeFromLog -Path $TrackTerminalLog
    if ($null -eq $exitCode) {
        Write-Host ("[{0}] Track job still running; checking again in {1}s..." -f (Get-Date), $PollSeconds) -ForegroundColor Gray
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    if ($exitCode -ne 0) {
        throw "Track pipeline ended with exit_code=$exitCode. Not starting downstream stages."
    }
    break
}

Write-Host "Track pipeline succeeded. Starting downstream train + render..." -ForegroundColor Green

& $VenvPython "$ProjectRoot\02_Visual_Engine\train_ghost.py" `
  --data_dir $DataConda `
  --output_dir $ModelConda `
  --iterations $Iterations
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$renderArgs = @(
    "--model_path", $ModelConda,
    "--data_dir", $DataConda,
    "--output", $FinalVideo,
    "--lefort_mm", $LefortMm,
    "--bsso_mm", $BssoMm,
    "--rig_mode", $RigMode
)
if ($RenderIteration -gt 0) {
    $renderArgs += "--iteration", [string]$RenderIteration
}
if ($CanonicalHeadAsset -ne "") {
    $renderArgs += "--canonical_head_asset", $CanonicalHeadAsset
}
if ($DeformationMap -ne "") {
    $renderArgs += "--deformation_map", $DeformationMap
}
if ($EvalStrict) {
    if (-not (Test-Path $StrictExportDir)) { New-Item -ItemType Directory -Path $StrictExportDir -Force | Out-Null }
    $renderArgs += "--export_frames_dir", $StrictExportDir
    if ($DeterministicIndices -ne "") {
        $renderArgs += "--deterministic_indices", $DeterministicIndices
    }
}

& $VenvPython "$ProjectRoot\02_Visual_Engine\render_surgery.py" @renderArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($EvalStrict) {
    if (-not (Test-Path $StrictReportDir)) { New-Item -ItemType Directory -Path $StrictReportDir -Force | Out-Null }
    & $VenvPython "$ProjectRoot\02_Visual_Engine\validation_reporting.py" `
      --model_path $ModelConda `
      --deterministic_frames_dir $StrictExportDir `
      --output_dir $StrictReportDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "Auto-continue pipeline complete." -ForegroundColor Green
Write-Host "Model: $ModelConda"
Write-Host "Video: $FinalVideo"
