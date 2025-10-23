"""
PercepNet Perceptual Quality Evaluation
Focuses on perceptually-relevant metrics: DNSMOS and PESQ
"""

import os
import argparse
import numpy as np
import soundfile as sf
import librosa as lr
import matplotlib.pyplot as plt
import logging

# Optional imports
try:
    import onnxruntime as ort
    import numpy.polynomial.polynomial as poly
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False

try:
    from pystoi import stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# DNSMOS coefficients
COEFS_SIG = np.array([9.651228012789436761e-01, 6.592637550310214145e-01, 7.572372955623894730e-02])
COEFS_BAK = np.array([-3.733460011101781717e+00, 2.700114234092929166e+00, -1.721332907340922813e-01])
COEFS_OVR = np.array([8.924546794696789354e-01, 6.609981731940616223e-01, 7.600269530243179694e-02])


def load_audio(filepath, sr=None, format='wav'):
    """Load audio file (WAV or PCM)."""
    if format == 'pcm':
        audio = np.fromfile(filepath, dtype=np.int16).astype(np.float32) / 32768.0
        if sr is None:
            sr = 16000  # PercepNet native sample rate
        log.info(f"Loaded PCM: {filepath} ({len(audio)} samples, {len(audio)/sr:.2f}s @ {sr}Hz)")
    else:
        audio, sr = sf.read(filepath)
        log.info(f"Loaded WAV: {filepath} ({len(audio)} samples, {len(audio)/sr:.2f}s @ {sr}Hz)")
    
    return audio, sr


def audio_melspec(audio, n_mels=120, frame_size=320, hop_length=160, sr=16000):
    """Compute mel-spectrogram for DNSMOS P808 model."""
    mel_spec = lr.feature.melspectrogram(
        y=audio, sr=sr, n_fft=frame_size+1, 
        hop_length=hop_length, n_mels=n_mels
    )
    mel_spec = (lr.power_to_db(mel_spec, ref=np.max) + 40) / 40
    return mel_spec.T.astype('float32')


def get_polyfit_val(sig, bak, ovr, is_personalized=False):
    """Apply polynomial mapping to raw DNSMOS scores."""
    if is_personalized:
        p_ovr = np.poly1d([-0.00533021,  0.005101,  1.18058466, -0.11236046])
        p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
        p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611,  0.96883132])
    else:
        p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
        p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439])
        p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])
    
    return p_sig(sig), p_bak(bak), p_ovr(ovr)


def calculate_dnsmos(audio, primary_model, p808_model, sr, input_length=9.01, personalized=False):
    """Calculate DNSMOS P.835 metrics (non-intrusive perceptual quality)."""
    if len(audio) < 2 * sr:
        log.warning("Audio too short for DNSMOS (need >2 seconds)")
        return None, None, None
    
    len_samples = int(input_length * sr)
    while len(audio) < len_samples:
        audio = np.append(audio, audio)
    
    num_hops = int(np.floor(len(audio) / sr) - input_length) + 1
    hop_len_samples = sr
    
    predicted_mos_sig_seg = []
    predicted_mos_bak_seg = []
    predicted_mos_ovr_seg = []
    
    for idx in range(num_hops):
        audio_seg = audio[int(idx * hop_len_samples): int((idx + input_length) * hop_len_samples)]
        if len(audio_seg) < len_samples:
            continue
        
        # Primary model: raw audio input
        input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
        oi = {'input_1': input_features}
        mos_sig_raw, mos_bak_raw, mos_ovr_raw = primary_model.run(None, oi)[0][0]
        
        # Apply polynomial fit
        mos_sig, mos_bak, mos_ovr = get_polyfit_val(
            mos_sig_raw, mos_bak_raw, mos_ovr_raw, personalized
        )
        
        predicted_mos_sig_seg.append(mos_sig)
        predicted_mos_bak_seg.append(mos_bak)
        predicted_mos_ovr_seg.append(mos_ovr)
    
    if not predicted_mos_sig_seg:
        return None, None, None
    
    return np.mean(predicted_mos_sig_seg), np.mean(predicted_mos_bak_seg), np.mean(predicted_mos_ovr_seg)


def calculate_pesq(reference, enhanced, sr):
    """Calculate PESQ (Perceptual Evaluation of Speech Quality)."""
    if not HAS_PESQ:
        return None
    
    # Resample to 16kHz for PESQ if needed
    if sr != 16000:
        log.info(f"Resampling from {sr}Hz to 16000Hz for PESQ...")
        reference = lr.resample(reference, orig_sr=sr, target_sr=16000)
        enhanced = lr.resample(enhanced, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Ensure same length
    min_len = min(len(reference), len(enhanced))
    reference = reference[:min_len]
    enhanced = enhanced[:min_len]
    
    return pesq(sr, reference, enhanced, 'wb')


def calculate_stoi(reference, enhanced, sr):
    """Calculate STOI (Short-Time Objective Intelligibility)."""
    if not HAS_STOI:
        return None
    
    # STOI works with 10kHz or 16kHz (extended STOI supports other rates)
    target_sr = 16000 if sr != 10000 else 10000
    
    if sr != target_sr:
        log.info(f"Resampling from {sr}Hz to {target_sr}Hz for STOI...")
        reference = lr.resample(reference, orig_sr=sr, target_sr=target_sr)
        enhanced = lr.resample(enhanced, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # Ensure same length
    min_len = min(len(reference), len(enhanced))
    reference = reference[:min_len]
    enhanced = enhanced[:min_len]
    
    # Calculate STOI using pystoi (extended=True allows any sample rate >= 10kHz)
    return stoi(reference, enhanced, sr, extended=True)


def create_spectrogram_comparison(noisy, enhanced, sr, output_path):
    """Create before/after spectrogram comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Noisy spectrogram
    D_noisy = lr.amplitude_to_db(np.abs(lr.stft(noisy)), ref=np.max)
    img1 = lr.display.specshow(D_noisy, sr=sr, x_axis='time', y_axis='hz', ax=axes[0], cmap='magma')
    axes[0].set_title('Before Enhancement (Noisy Input)')
    axes[0].set_ylabel('Frequency (Hz)')
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
    
    # Enhanced spectrogram
    D_enhanced = lr.amplitude_to_db(np.abs(lr.stft(enhanced)), ref=np.max)
    img2 = lr.display.specshow(D_enhanced, sr=sr, x_axis='time', y_axis='hz', ax=axes[1], cmap='magma')
    axes[1].set_title('After Enhancement (PercepNet Output)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved spectrogram comparison: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate PercepNet perceptual quality (DNSMOS + PESQ)'
    )
    parser.add_argument('--enhanced', required=True, help='Enhanced/processed audio file')
    parser.add_argument('--noisy', default=None, help='Noisy input (for comparison plots)')
    parser.add_argument('--clean', default=None, help='Clean reference (for PESQ)')
    parser.add_argument('--format', default='wav', choices=['wav', 'pcm'], 
                       help='Audio format (default: wav)')
    parser.add_argument('--sr', type=int, default=16000, 
                       help='Sample rate for PCM files (default: 16000 - PercepNet native rate)')
    parser.add_argument('--output-dir', default='../evaluation_results',
                       help='Output directory')
    parser.add_argument('--primary-model', type=str, default='../DNSMOS/sig_bak_ovr.onnx',
                       help='Path to DNSMOS primary model (sig_bak_ovr.onnx)')
    parser.add_argument('--p808-model', type=str, default='../DNSMOS/model_v8.onnx',
                       help='Path to DNSMOS P808 model (model_v8.onnx)')
    parser.add_argument('--personalized-mos', action='store_true',
                       help='Use personalized MOS scoring')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save before/after spectrograms')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load enhanced audio
    log.info("Loading enhanced audio...")
    enhanced, sr = load_audio(args.enhanced, sr=args.sr, format=args.format)
    
    results = {
        'enhanced_file': os.path.basename(args.enhanced),
        'sample_rate': sr,
        'duration_seconds': len(enhanced) / sr
    }
    
    # === DNSMOS (Non-intrusive) ===
    log.info("\n=== DNSMOS P.835 Evaluation ===")
    if not HAS_ONNXRUNTIME:
        log.error("ONNX Runtime not available. Install with: pip install onnxruntime")
    elif not os.path.exists(args.primary_model) or not os.path.exists(args.p808_model):
        log.error(f"DNSMOS models not found:")
        log.error(f"  Primary: {args.primary_model}")
        log.error(f"  P808: {args.p808_model}")
        log.error("Download from: https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS")
    else:
        try:
            # Resample to 16kHz for DNSMOS if needed
            enhanced_16k = enhanced if sr == 16000 else lr.resample(enhanced, orig_sr=sr, target_sr=16000)
            
            primary_model = ort.InferenceSession(args.primary_model)
            p808_model = ort.InferenceSession(args.p808_model)
            
            dnsmos_sig, dnsmos_bak, dnsmos_ovr = calculate_dnsmos(
                enhanced_16k, primary_model, p808_model, 16000, 
                personalized=args.personalized_mos
            )
            
            if dnsmos_sig is not None:
                results['DNSMOS_SIG'] = dnsmos_sig
                results['DNSMOS_BAK'] = dnsmos_bak
                results['DNSMOS_OVR'] = dnsmos_ovr
                log.info(f"  SIG (Signal quality):     {dnsmos_sig:.2f}")
                log.info(f"  BAK (Background quality): {dnsmos_bak:.2f}")
                log.info(f"  OVR (Overall quality):    {dnsmos_ovr:.2f}")
        except Exception as e:
            log.error(f"Error computing DNSMOS: {e}")
            import traceback
            log.error(traceback.format_exc())
    
    # === Intrusive Metrics (Require clean reference) ===
    if args.clean:
        log.info("\n=== Intrusive Metrics (with clean reference) ===")
        try:
            clean, sr_clean = load_audio(args.clean, sr=args.sr, format=args.format)
            
            # Resample if needed
            if sr_clean != sr:
                enhanced_resamp = lr.resample(enhanced, orig_sr=sr, target_sr=sr_clean)
            else:
                enhanced_resamp = enhanced
            
            # PESQ
            if HAS_PESQ:
                pesq_score = calculate_pesq(clean, enhanced_resamp, sr_clean)
                if pesq_score is not None:
                    results['PESQ'] = pesq_score
                    log.info(f"  PESQ score: {pesq_score:.3f}")
            else:
                log.warning("  PESQ not available. Install: pip install pesq")
                log.warning("  (Requires Visual Studio Build Tools on Windows)")
            
            # STOI
            if HAS_STOI:
                stoi_score = calculate_stoi(clean, enhanced_resamp, sr_clean)
                if stoi_score is not None:
                    results['STOI'] = stoi_score
                    log.info(f"  STOI score: {stoi_score:.4f}")
            else:
                log.warning("  STOI not available. Install: pip install torchmetrics")
                
        except Exception as e:
            log.error(f"Error computing intrusive metrics: {e}")
    else:
        log.info("\nSkipping intrusive metrics (no clean reference provided)")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write("PercepNet Perceptual Quality Evaluation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Enhanced file: {results['enhanced_file']}\n")
        f.write(f"Duration: {results['duration_seconds']:.2f}s @ {sr}Hz\n")
        f.write("\n")
        
        if 'DNSMOS_OVR' in results:
            f.write("DNSMOS P.835 (Non-intrusive):\n")
            f.write(f"  SIG (Signal):     {results['DNSMOS_SIG']:.2f}\n")
            f.write(f"  BAK (Background): {results['DNSMOS_BAK']:.2f}\n")
            f.write(f"  OVR (Overall):    {results['DNSMOS_OVR']:.2f}\n")
            f.write("\n")
        
        if 'PESQ' in results or 'STOI' in results:
            f.write("Intrusive Metrics (with clean reference):\n")
            if 'PESQ' in results:
                f.write(f"  PESQ: {results['PESQ']:.3f}\n")
            if 'STOI' in results:
                f.write(f"  STOI: {results['STOI']:.4f}\n")
    
    log.info(f"\nResults saved to: {results_file}")
    
    # Create comparison plots
    if args.save_plots and args.noisy:
        log.info("\nGenerating before/after comparison...")
        noisy, sr_noisy = load_audio(args.noisy, sr=args.sr, format=args.format)
        
        if sr_noisy != sr:
            noisy = lr.resample(noisy, orig_sr=sr_noisy, target_sr=sr)
        
        min_len = min(len(noisy), len(enhanced))
        noisy = noisy[:min_len]
        enhanced_trim = enhanced[:min_len]
        
        plot_file = os.path.join(args.output_dir, 'before_after_comparison.png')
        create_spectrogram_comparison(noisy, enhanced_trim, sr, plot_file)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PERCEPTUAL QUALITY SUMMARY")
    print("=" * 60)
    print(f"File: {args.enhanced}")
    print(f"Duration: {results['duration_seconds']:.2f}s @ {sr}Hz")
    print("-" * 60)
    
    if 'DNSMOS_OVR' in results:
        print("DNSMOS P.835 (1.0-5.0, higher is better):")
        print(f"  Signal Quality (SIG):     {results['DNSMOS_SIG']:.2f}")
        print(f"  Background Quality (BAK): {results['DNSMOS_BAK']:.2f}")
        print(f"  Overall Quality (OVR):    {results['DNSMOS_OVR']:.2f}")
    else:
        print("DNSMOS: Not computed (models not available)")
    
    if 'PESQ' in results or 'STOI' in results:
        print("\nIntrusive Metrics (with clean reference):")
        if 'PESQ' in results:
            print(f"  PESQ (1.0-4.5, higher=better): {results['PESQ']:.3f}")
        if 'STOI' in results:
            print(f"  STOI (0.0-1.0, higher=better): {results['STOI']:.4f}")
    else:
        print("\nIntrusive Metrics: Not computed (no clean reference or dependencies not available)")
    
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
