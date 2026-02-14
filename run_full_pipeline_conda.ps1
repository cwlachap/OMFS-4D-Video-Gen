# Full pipeline with VHAP on conda, then convert/train/render on venv.
# Run from project root: .\run_full_pipeline_conda.ps1 -Sequence input_video
# Or with a new video: .\run_full_pipeline_conda.ps1 -Video path\to\video.mp4
#
# Tracking quality: face clearly visible (eyes open), clean background.
# Landmark detection runs on black background by default (better for detectors). If still bad,
# delete landmark2d and try -LandmarkDetectionBg original. -FaceDetector sfd; auto-retry if <40 points.
# If mesh is a tiny blob, try -InitialFocal 1.0. Use -LandmarkSource star to try STAR (if installed).
#
# Speed (avoid 3h texture grind): use downsampled images like Streamlit does:
#   - Re-run preprocessing with "Frame resolution" 512 so images_2 is filled, then this script auto-uses it; or
#   - Pass -Downsample 2 if you already have images_2; or
#   - Pass -Fast for landmark-only tracking (no texture opt, fast but coarser).
#   -MaxFrames 30 = quick test on 30 frames.

param(
    [string]$Sequence = "input_video",
    [string]$Video = "",
    [int]$MaxFrames = 0,
    [string]$FrameSubset = "",
    [int]$Downsample = 0,
    [switch]$Fast,
    [switch]$FullQualityFaster,
    [switch]$CloseGpuClutter,
    [int]$IntervalMedia = 500,
    [int]$IntervalScalar = 100,
    [double]$StepScale = 1.0,
    [double]$EpochScale = 1.0,
    [double]$InitialFocal = 0,
    [string]$LandmarkSource = "face-alignment",
    [string]$FaceDetector = "sfd",
    [string]$LandmarkDetectionBg = "black",
    [int]$Iterations = 5000,
    [double]$LefortMm = 0,
    [double]$BssoMm = 0,
    [int]$RenderIteration = -1,
    [string]$RigMode = "flame_only",
    [string]$CanonicalHeadAsset = "",
    [string]$DeformationMap = "",
    [switch]$EvalStrict,
    [string]$DeterministicIndices = "",
    [switch]$EnableHeadRecon,
    [string]$CaptureRoot = "",
    [switch]$SkipTrain,
    [switch]$SkipRender
)

$ErrorActionPreference = "Stop"
$ProjectRoot = "c:\Users\lacha\Desktop\OMFS 4d Video Gen"
$CondaPython = "C:\Users\lacha\miniconda3\envs\VHAP\python.exe"
$VenvPython = "$ProjectRoot\venv\Scripts\python.exe"
$VHAP_ROOT = "$ProjectRoot\vhap_repo"
$DATA_ROOT = "$ProjectRoot\vhap_data\monocular"
$OUTPUT_TRACK = "$ProjectRoot\vhap_output_conda\${Sequence}_whiteBg_staticOffset"
$EXPORT_ROOT = "$ProjectRoot\vhap_export_conda\${Sequence}_conda"
$DATA_CONDA = "$ProjectRoot\02_Visual_Engine\data_conda"
$MODEL_CONDA = "$ProjectRoot\02_Visual_Engine\output\model_conda"
$FINAL_VIDEO = "$ProjectRoot\final_prediction_conda.mp4"
$STRICT_EXPORT_DIR = "$ProjectRoot\02_Visual_Engine\output\model_conda\eval_strict\deterministic_frames"
$STRICT_REPORT_DIR = "$ProjectRoot\02_Visual_Engine\output\model_conda\eval_strict\reports"

# Ensure conda env binaries (e.g. ninja) are on PATH even when using absolute python path.
$CondaEnvRoot = Split-Path -Parent $CondaPython
$env:PATH = "$CondaEnvRoot;$CondaEnvRoot\Scripts;$CondaEnvRoot\Library\bin;$env:PATH"

function Step($name) {
    Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
    Write-Host "  $name" -ForegroundColor Cyan
    Write-Host ("=" * 60) + "`n" -ForegroundColor Cyan
}

function Stop-GpuClutter {
    $targets = @(
        "chrome",
        "msedge",
        "steamwebhelper",
        "XboxPcApp",
        "XboxPcTray",
        "Copilot",
        "NVIDIA Overlay"
    )
    foreach ($name in $targets) {
        try {
            Get-Process -Name $name -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        } catch {}
    }
}

# When -Video is provided, use its stem as sequence name
if ($Video -ne "") {
    $Sequence = [System.IO.Path]::GetFileNameWithoutExtension($Video)
    $OUTPUT_TRACK = "$ProjectRoot\vhap_output_conda\${Sequence}_whiteBg_staticOffset"
    $EXPORT_ROOT = "$ProjectRoot\vhap_export_conda\${Sequence}_conda"
}

if ($FullQualityFaster) {
    # Keep photometric pipeline enabled while trimming expensive overhead.
    if ($IntervalMedia -eq 500) { $IntervalMedia = 2000 }
    if ($IntervalScalar -eq 100) { $IntervalScalar = 200 }
    if ([Math]::Abs($StepScale - 1.0) -lt 1e-9) { $StepScale = 0.85 }
    if ([Math]::Abs($EpochScale - 1.0) -lt 1e-9) { $EpochScale = 0.85 }
}

$StepScale = [Math]::Max(0.1, [Math]::Min(1.0, $StepScale))
$EpochScale = [Math]::Max(0.1, [Math]::Min(1.0, $EpochScale))

if ($CloseGpuClutter) {
    Write-Host "Closing known GPU-clutter apps before tracking..." -ForegroundColor Yellow
    Stop-GpuClutter
}

# ----- Optional: Preprocess (frame extract + matting) with conda -----
if ($Video -ne "") {
    Step "0. VHAP Preprocess (conda): extract frames + matting"
    $videoPath = (Resolve-Path $Video).Path
    $targetVideo = "$DATA_ROOT\$Sequence.mp4"
    if (-not (Test-Path $DATA_ROOT)) { New-Item -ItemType Directory -Path $DATA_ROOT -Force | Out-Null }
    if (-not (Test-Path $targetVideo)) {
        Copy-Item $videoPath $targetVideo -Force
        Write-Host "Copied video to $targetVideo"
    }
    $preprocessArgs = @(
        "--input", $targetVideo,
        "--matting-method", "robust_video_matting"
    )
    if ($Downsample -eq 2) {
        # Ensure images_2 exists for low-resolution VHAP tracking.
        $preprocessArgs += "--downsample_scales", "2"
    }
    & $CondaPython "$VHAP_ROOT\vhap\preprocess_video.py" @preprocessArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "Preprocess done.`n" -ForegroundColor Green
} else {
    if (-not (Test-Path "$DATA_ROOT\$Sequence\images")) {
        Write-Host "Error: No video provided and no existing data at $DATA_ROOT\$Sequence\images. Use -Video path\to\video.mp4 or -Sequence <name> with existing data." -ForegroundColor Red
        exit 1
    }
    Write-Host "Using existing sequence: $Sequence`n" -ForegroundColor Gray
}

# ----- 1. VHAP Track (conda) -----
# Use downsampled images (images_2) when present - same as Streamlit "Frame resolution 512" = much faster texture stages
$seqPath = "$DATA_ROOT\$Sequence"
$nFull = 0
$nDown2 = 0
if (Test-Path "$seqPath\images") { $nFull = (Get-ChildItem "$seqPath\images" -Filter "*.jpg").Count }
if (Test-Path "$seqPath\images_2") { $nDown2 = (Get-ChildItem "$seqPath\images_2" -Filter "*.jpg").Count }
$useDownsample = $false
if ($Downsample -eq 2) {
    $useDownsample = $true
    Write-Host "Using images_2 (downsampled) - Downsample 2.`n" -ForegroundColor Yellow
} elseif ($Downsample -eq 0 -and $nDown2 -ge [Math]::Max(1, $nFull * 0.5)) {
    $useDownsample = $true
    Write-Host "Using images_2 (downsampled) for faster tracking - same as Streamlit 'Frame resolution 512'.`n" -ForegroundColor Yellow
}
if ($Fast) {
    Write-Host "Fast mode: landmark-only tracking (no texture optimization).`n" -ForegroundColor Yellow
}
# VHAP resolves asset/flame/* relative to cwd; run from VHAP repo root
Step "1. VHAP Track (conda)"
$trackArgs = @(
    "--data.root-folder", $DATA_ROOT,
    "--exp.output-folder", $OUTPUT_TRACK,
    "--data.sequence", $Sequence,
    "--data.background-color", "white",
    "--data.landmark-source", $LandmarkSource,
    "--data.face-detector", $FaceDetector,
    "--data.landmark-detection-background", $LandmarkDetectionBg,
    "--data.landmark-detector-njobs", "1",
    "--log.interval-media", [string]$IntervalMedia,
    "--log.interval-scalar", [string]$IntervalScalar
)
if ($useDownsample) {
    $trackArgs += "--data.n-downsample-rgb", "2"
}
if ($Fast) {
    $trackArgs += "--exp.no-photometric"
}
if ($InitialFocal -gt 0) {
    $trackArgs += "--model.initial-focal-length", [string]$InitialFocal
    Write-Host "Using initial focal length: $InitialFocal`n" -ForegroundColor Yellow
}
if ($MaxFrames -gt 0) {
    $trackArgs += "--data.subset", "tn$MaxFrames"
    Write-Host "Using subset: first $MaxFrames frames (quick test). Omit -MaxFrames for full run.`n" -ForegroundColor Yellow
} elseif ($FrameSubset -ne "") {
    $trackArgs += "--data.subset", $FrameSubset
    Write-Host "Using custom subset: $FrameSubset`n" -ForegroundColor Yellow
}

if ($StepScale -lt 1.0 -or $EpochScale -lt 1.0) {
    $lmkInitRigidSteps = [Math]::Max(50, [int](500 * $StepScale))
    $lmkInitAllSteps = [Math]::Max(50, [int](500 * $StepScale))
    $lmkSeqSteps = [Math]::Max(10, [int](50 * $StepScale))
    $lmkGlobalEpochs = [Math]::Max(5, [int](30 * $EpochScale))
    $rgbInitTextureSteps = [Math]::Max(80, [int](500 * $StepScale))
    $rgbInitAllSteps = [Math]::Max(80, [int](500 * $StepScale))
    $rgbInitOffsetSteps = [Math]::Max(80, [int](500 * $StepScale))
    $rgbSeqSteps = [Math]::Max(10, [int](50 * $StepScale))
    $rgbGlobalEpochs = [Math]::Max(5, [int](30 * $EpochScale))

    $trackArgs += @(
        "--pipeline.lmk-init-rigid.num-steps", [string]$lmkInitRigidSteps,
        "--pipeline.lmk-init-all.num-steps", [string]$lmkInitAllSteps,
        "--pipeline.lmk-sequential-tracking.num-steps", [string]$lmkSeqSteps,
        "--pipeline.lmk-global-tracking.num-epochs", [string]$lmkGlobalEpochs,
        "--pipeline.rgb-init-texture.num-steps", [string]$rgbInitTextureSteps,
        "--pipeline.rgb-init-all.num-steps", [string]$rgbInitAllSteps,
        "--pipeline.rgb-init-offset.num-steps", [string]$rgbInitOffsetSteps,
        "--pipeline.rgb-sequential-tracking.num-steps", [string]$rgbSeqSteps,
        "--pipeline.rgb-global-tracking.num-epochs", [string]$rgbGlobalEpochs
    )

    Write-Host ("Using scaled tracking stages: StepScale={0}, EpochScale={1}" -f $StepScale, $EpochScale) -ForegroundColor Yellow
}
Push-Location $VHAP_ROOT
try {
    & $CondaPython "vhap\track.py" @trackArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally { Pop-Location }
Write-Host "Track done.`n" -ForegroundColor Green

# ----- 2. VHAP Export (conda) -----
Step "2. VHAP Export (conda)"
Push-Location $VHAP_ROOT
try {
    & $CondaPython "vhap\export_as_nerf_dataset.py" `
      --src-folder $OUTPUT_TRACK `
      --tgt-folder $EXPORT_ROOT `
      --background-color white `
      --no-create-mask-from-mesh
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally { Pop-Location }
Write-Host "Export done.`n" -ForegroundColor Green

# ----- 3. Convert to GaussianAvatars format (venv) -----
Step "3. Convert to GaussianAvatars format (venv)"
& $VenvPython "$ProjectRoot\02_Visual_Engine\preprocess_video.py" --convert-only --vhap_export_dir $EXPORT_ROOT --output_dir $DATA_CONDA
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Convert done.`n" -ForegroundColor Green

# ----- 3.5 Optional: build canonical full-head scaffold from multi-sequence captures -----
if ($EnableHeadRecon) {
    Step "3.5 Build canonical full-head scaffold (venv)"
    if ($CaptureRoot -eq "") { $CaptureRoot = "$ProjectRoot\vhap_export_conda" }
    & $VenvPython "$ProjectRoot\02_Visual_Engine\head_recon\ingest_sequences.py" --capture_root $CaptureRoot --output_dir "$ProjectRoot\02_Visual_Engine\output\head_recon"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $VenvPython "$ProjectRoot\02_Visual_Engine\head_recon\register_sequences.py" --manifest "$ProjectRoot\02_Visual_Engine\output\head_recon\sequence_manifest.json" --output_dir "$ProjectRoot\02_Visual_Engine\output\head_recon"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $VenvPython "$ProjectRoot\02_Visual_Engine\head_recon\build_canonical_head.py" --registration "$ProjectRoot\02_Visual_Engine\output\head_recon\registration.json" --output_dir "$ProjectRoot\02_Visual_Engine\output\head_recon"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "Head recon scaffold done.`n" -ForegroundColor Green
}

# ----- 4. Train (venv) -----
if (-not $SkipTrain) {
    Step "4. Train GaussianAvatars (venv)"
    & $VenvPython "$ProjectRoot\02_Visual_Engine\train_ghost.py" --data_dir $DATA_CONDA --output_dir $MODEL_CONDA --iterations $Iterations
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "Train done.`n" -ForegroundColor Green
} else {
    Write-Host "Skipping train (--SkipTrain).`n" -ForegroundColor Gray
}

# ----- 5. Render (venv) -----
if (-not $SkipRender) {
    Step "5. Render (venv)"
    $renderArgs = @(
        "--model_path", $MODEL_CONDA,
        "--data_dir", $DATA_CONDA,
        "--output", $FINAL_VIDEO,
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
        if (-not (Test-Path $STRICT_EXPORT_DIR)) { New-Item -ItemType Directory -Path $STRICT_EXPORT_DIR -Force | Out-Null }
        $renderArgs += "--export_frames_dir", $STRICT_EXPORT_DIR
        if ($DeterministicIndices -ne "") {
            $renderArgs += "--deterministic_indices", $DeterministicIndices
        }
    }
    & $VenvPython "$ProjectRoot\02_Visual_Engine\render_surgery.py" @renderArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "Render done.`n" -ForegroundColor Green

    if ($EvalStrict) {
        Step "6. Strict deterministic validation report (venv)"
        if (-not (Test-Path $STRICT_REPORT_DIR)) { New-Item -ItemType Directory -Path $STRICT_REPORT_DIR -Force | Out-Null }
        & $VenvPython "$ProjectRoot\02_Visual_Engine\validation_reporting.py" `
          --model_path $MODEL_CONDA `
          --deterministic_frames_dir $STRICT_EXPORT_DIR `
          --output_dir $STRICT_REPORT_DIR
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Write-Host "Strict validation report done.`n" -ForegroundColor Green
    }
} else {
    Write-Host "Skipping render (--SkipRender).`n" -ForegroundColor Gray
}

Write-Host "Full pipeline finished." -ForegroundColor Green
Write-Host "  Data:   $DATA_CONDA"
Write-Host "  Model:  $MODEL_CONDA"
if (-not $SkipRender) { Write-Host "  Video:  $FINAL_VIDEO" }
Write-Host ""
