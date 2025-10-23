# PercepNet Perceptual Quality Evaluation

Focused on perceptually-relevant metrics: **DNSMOS (SIG, BAK, OVR)**, **PESQ**, and **STOI**.

**Sample Rate**: PercepNet native processing at **48 kHz** (input can be 16kHz, upsampled internally).

**Current Model**: 100,000 training steps checkpoint

**Directory Structure**:
- `evaluation/` - All evaluation scripts
- `evaluation_results/` - All evaluation outputs (metrics, plots)
- `deployment_outputs/` - Model inference outputs (PCM/WAV files)
- `training_outputs/` - Training checkpoints and logs

## Quick Start

### Windows (with WSL for PESQ support)

```powershell
cd evaluation

# Full evaluation with PESQ (requires WSL setup)
.\Evaluate-WithPESQ.ps1 -Enhanced ..\deployment_outputs\output.wav -Clean reference.wav -SavePlots

# Or direct command:
wsl bash -c "source ~/pesq_env/bin/activate && cd '/mnt/c/Users/kegustavussen/OneDrive - GN Store Nord/Documents/GitHub/PercepNet/evaluation' && python3 evaluate.py --enhanced ../deployment_outputs/output.wav --clean clean.wav --save-plots"
```

### Windows (DNSMOS + STOI only, no PESQ)

```powershell
cd evaluation
python evaluate.py --enhanced ..\deployment_outputs\output.wav --save-plots
```

### Linux / Container

```bash
cd evaluation
python evaluate.py --enhanced ../deployment_outputs/output.wav --clean reference.wav --save-plots
```

Results will be saved to `../evaluation_results/` by default.

## Metrics

**Non-intrusive** (no clean reference needed):
- **DNSMOS P.835**:
  - SIG: Signal quality (1.0-5.0)
  - BAK: Background noise quality (1.0-5.0)
  - OVR: Overall perceptual quality (1.0-5.0)

**Intrusive** (requires clean reference):
- **PESQ**: Perceptual quality (1.0-4.5, higher is better)
- **STOI**: Speech intelligibility (0.0-1.0, higher is better)

## DNSMOS Setup

Download ONNX models from [Microsoft DNS Challenge](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS):
- `sig_bak_ovr.onnx` (primary model - takes raw audio)
- `model_v8.onnx` (P808 model - optional)

Place in `PercepNet/DNSMOS/` directory. ✅ **Already set up!**

## Installation

### Windows (DNSMOS + STOI)
```bash
pip install numpy librosa soundfile matplotlib onnxruntime pystoi
```

### Windows with PESQ (requires WSL)
```bash
# One-time setup in WSL:
wsl bash -c "cd ~ && python3 -m venv pesq_env && source pesq_env/bin/activate && pip install numpy librosa soundfile matplotlib onnxruntime pystoi pesq"
```

### Linux / Container
```bash
pip install numpy librosa soundfile matplotlib onnxruntime pystoi pesq
```

**Note**: PESQ requires C++ compilation (works in WSL/Linux, needs Visual Studio Build Tools on Windows)

## HPC Usage (AI-LAB)

```bash
# Edit file paths in evaluate_percepnet.sh
sbatch evaluate_percepnet.sh
```
