# PercepNet - Speech Enhancement with Perceptual Quality Metrics

Deep learning-based speech enhancement using RNN architecture, with comprehensive perceptual quality evaluation.

## 📁 Directory Structure

```
PercepNet/
├── evaluation/              # Evaluation scripts (DNSMOS, PESQ, STOI)
│   ├── evaluate.py         # Main evaluation script
│   ├── Evaluate-WithPESQ.ps1  # Windows PowerShell wrapper
│   └── README.md           # Evaluation documentation
│
├── evaluation_results/      # All evaluation outputs
│   ├── eval_*/            # Evaluation run results
│   └── */results.txt      # Metric scores and analysis
│
├── deployment_outputs/      # Model inference outputs
│   ├── *.pcm             # Raw audio output (48kHz)
│   └── *.wav             # Converted WAV files
│
├── training_outputs/        # Training artifacts
│   ├── training.h5       # Training data
│   └── *.err/*.out       # Training logs
│
├── src/                    # C++ source code
│   ├── denoise.cpp       # Core denoising algorithm
│   ├── nnet.cpp          # Neural network inference
│   └── main.cpp          # Entry point
│
├── training_set_sept12_500h/  # Training data and checkpoints
│   └── exp_*/checkpoint-*.pkl # Model checkpoints
│
├── bin/                    # Compiled binaries
│   └── src/percepNet_run # Main executable
│
└── DNSMOS/                # DNSMOS evaluation models
    ├── sig_bak_ovr.onnx
    └── model_v8.onnx
```

## 🚀 Quick Start

### Inference (Speech Enhancement)

```bash
# Using WSL on Windows
wsl bash -c "cd '/mnt/c/path/to/PercepNet' && ./bin/src/percepNet_run input.pcm output.pcm"

# Output will be 48kHz PCM format
```

### Evaluation

```powershell
cd evaluation

# Full evaluation with PESQ (WSL)
.\Evaluate-WithPESQ.ps1 -Enhanced ..\deployment_outputs\output.wav -Clean reference.wav -SavePlots

# DNSMOS + STOI only (Windows)
python evaluate.py --enhanced ..\deployment_outputs\output.wav --save-plots
```

Results saved to `evaluation_results/`

## 📊 Evaluation Metrics

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| **DNSMOS SIG** | Non-intrusive | 1.0-5.0 | Signal quality |
| **DNSMOS BAK** | Non-intrusive | 1.0-5.0 | Background quality |
| **DNSMOS OVR** | Non-intrusive | 1.0-5.0 | Overall quality |
| **PESQ** | Intrusive* | 1.0-4.5 | Perceptual quality |
| **STOI** | Intrusive* | 0.0-1.0 | Speech intelligibility |

*Requires clean reference audio

## 🎯 Current Model

- **Checkpoint**: 100,000 training steps
- **Architecture**: GRU-based RNN with ERB frequency bands
- **Sample Rate**: 48kHz native processing
- **Frame Size**: 480 samples (10ms @ 48kHz)
- **Weights**: Embedded in `nnet_data.c` (370MB)

## ⚙️ Key Files

- `evaluate.py` - Main evaluation script
- `percepNet_run` - Compiled inference binary
- `nnet_data.c` - Model weights (generated from checkpoint)
- `checkpoint-100000steps.pkl` - PyTorch model checkpoint

## 📝 Audio Format Notes

- **Input**: 16kHz or 48kHz PCM (16-bit signed integer)
- **Output**: 48kHz PCM (16-bit signed integer)
- **Processing**: Model upsamples 16kHz to 48kHz internally
- **Conversion**: Use `soundfile` to convert PCM ↔ WAV

## 🔧 Setup

See `evaluation/README.md` for detailed installation instructions.

**Quick install:**
```bash
# Windows (DNSMOS + STOI)
pip install numpy librosa soundfile matplotlib onnxruntime pystoi

# WSL (adds PESQ support)
wsl bash -c "python3 -m venv ~/pesq_env && source ~/pesq_env/bin/activate && pip install numpy librosa soundfile matplotlib onnxruntime pystoi pesq"
```

## 📈 Performance (100k checkpoint)

Test file: `sampledata/speech/speech.pcm` (60s @ 48kHz)

```
DNSMOS P.835:
  SIG (Signal):     2.25
  BAK (Background): 3.15
  OVR (Overall):    1.93
```

Improvement over 20k checkpoint: +10-12% across all metrics

## 🏗️ Building from Source

```bash
# Rebuild with new checkpoint (WSL/Linux)
python dump_percepnet.py checkpoint.pkl nnet_data.c
cd bin && make clean && make -j8
```

## 📚 References

- DNSMOS: [Microsoft DNS Challenge](https://github.com/microsoft/DNS-Challenge)
- Model architecture based on RNNoise principles
- Perceptual evaluation using ITU-T P.835 framework
