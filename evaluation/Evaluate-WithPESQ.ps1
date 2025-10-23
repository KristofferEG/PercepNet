# PowerShell wrapper to run PercepNet evaluation with PESQ in WSL
# Usage: .\Evaluate-WithPESQ.ps1 -Enhanced "output.wav" -Clean "clean.wav" -Noisy "noisy.wav"

param(
    [Parameter(Mandatory=$true)]
    [string]$Enhanced,
    
    [Parameter(Mandatory=$false)]
    [string]$Clean = "",
    
    [Parameter(Mandatory=$false)]
    [string]$Noisy = "",
    
    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "../evaluation_results",
    
    [Parameter(Mandatory=$false)]
    [switch]$SavePlots,
    
    [Parameter(Mandatory=$false)]
    [string]$Format = "wav",
    
    [Parameter(Mandatory=$false)]
    [int]$SampleRate = 16000
)

# Convert Windows paths to WSL paths
function Convert-ToWSLPath {
    param([string]$WindowsPath)
    
    $absPath = Resolve-Path $WindowsPath -ErrorAction SilentlyContinue
    if (-not $absPath) {
        $absPath = $WindowsPath
    }
    
    # Convert C:\Users\... to /mnt/c/Users/...
    $wslPath = $absPath -replace '\\', '/'
    $wslPath = $wslPath -replace '^([A-Z]):', { "/mnt/$($_.Groups[1].Value.ToLower())" }
    
    return $wslPath
}

Write-Host "=== PercepNet Evaluation with PESQ (WSL) ===" -ForegroundColor Cyan
Write-Host ""

# Convert paths
$enhancedWSL = Convert-ToWSLPath $Enhanced
$outputDirWSL = Convert-ToWSLPath $OutputDir

# Build command
$cmd = "source ~/pesq_env/bin/activate && "
$cmd += "cd '/mnt/c/Users/kegustavussen/OneDrive - GN Store Nord/Documents/GitHub/PercepNet/evaluation' && "
$cmd += "python3 evaluate.py --enhanced '$enhancedWSL' --sr $SampleRate --format $Format --output-dir '$outputDirWSL'"

# Add optional parameters
if ($Clean) {
    $cleanWSL = Convert-ToWSLPath $Clean
    $cmd += " --clean '$cleanWSL'"
    Write-Host "Clean reference: $Clean" -ForegroundColor Green
}

if ($Noisy) {
    $noisyWSL = Convert-ToWSLPath $Noisy
    $cmd += " --noisy '$noisyWSL'"
    Write-Host "Noisy input: $Noisy" -ForegroundColor Green
}

if ($SavePlots) {
    $cmd += " --save-plots"
}

Write-Host "Enhanced audio: $Enhanced" -ForegroundColor Green
Write-Host "Output directory: $OutputDir" -ForegroundColor Green
Write-Host ""

# Run evaluation
wsl bash -c "$cmd"

Write-Host ""
Write-Host "=== Evaluation Complete ===" -ForegroundColor Cyan
Write-Host "Results saved to: $OutputDir\results.txt" -ForegroundColor Yellow
