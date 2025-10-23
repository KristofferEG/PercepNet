import os
import re
import logging
import numpy as np
import librosa as lr
import librosa.display as lrd
import soundfile as sf
import matplotlib.pyplot as plt
from array import array
from pesq import pesq
import onnxruntime as ort
import numpy.polynomial.polynomial as poly

# Optional imports for ML framework integration
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    
try:
    from ml_training_framework.evaluators.base import BaseEvaluator
    HAS_ML_FRAMEWORK = True
except ImportError:
    HAS_ML_FRAMEWORK = False
    # Create dummy base class
    class BaseEvaluator:
        def __init__(self, output_path, evaluator_callbacks):
            pass

log = logging.getLogger(__name__)

# Coefficients for polynomial fitting
COEFS_SIG = np.array([9.651228012789436761e-01, 6.592637550310214145e-01, 7.572372955623894730e-02])
COEFS_BAK = np.array([-3.733460011101781717e+00, 2.700114234092929166e+00, -1.721332907340922813e-01])
COEFS_OVR = np.array([8.924546794696789354e-01, 6.609981731940616223e-01, 7.600269530243179694e-02])

def audio_logpowspec(audio, nfft=320, hop_length=160, sr=16000):
    powspec = (np.abs(lr.core.stft(audio, n_fft=nfft, hop_length=hop_length))) ** 2
    logpowspec = np.log10(np.maximum(powspec, 10 ** (-12)))
    return logpowspec.T

def dnsmos(audio, session_sig, session_bak_ovr, sr, input_length=9):
    if len(audio) < 2 * sr:
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
        input_features = np.array(audio_logpowspec(audio=audio_seg, sr=sr)).astype('float32')[np.newaxis, :, :]
        onnx_inputs_sig = {inp.name: input_features for inp in session_sig.get_inputs()}
        mos_sig = poly.polyval(session_sig.run(None, onnx_inputs_sig), COEFS_SIG)
        onnx_inputs_bak_ovr = {inp.name: input_features for inp in session_bak_ovr.get_inputs()}
        mos_bak_ovr = session_bak_ovr.run(None, onnx_inputs_bak_ovr)
        mos_bak = poly.polyval(mos_bak_ovr[0][0][1], COEFS_BAK)
        mos_ovr = poly.polyval(mos_bak_ovr[0][0][2], COEFS_OVR)
        predicted_mos_sig_seg.append(mos_sig)
        predicted_mos_bak_seg.append(mos_bak)
        predicted_mos_ovr_seg.append(mos_ovr)
    return np.mean(predicted_mos_sig_seg), np.mean(predicted_mos_bak_seg), np.mean(predicted_mos_ovr_seg)

def si_sdr(clean, enhanced, eps=1e-9):
    scaled = clean * np.sum(enhanced * clean) / (np.sum(clean ** 2) + eps)
    return 10 * np.log10(np.sum(scaled ** 2) / (np.sum((scaled - enhanced) ** 2) + eps))

def get_spectrogram(x):
    y, _ = lr.magphase(lr.stft(x, n_fft=512, hop_length=256, win_length=512))
    db = lr.amplitude_to_db(y)
    return db

def create_figure(noisy, clean, pred, sr):
    titles = ['Noisy', 'Clean', 'Enhanced']
    noisy_spec = get_spectrogram(noisy)
    clean_spec = get_spectrogram(clean)
    pred_spec = get_spectrogram(pred)
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for i, spec in enumerate([noisy_spec, clean_spec, pred_spec]):
        lrd.specshow(spec, cmap='magma', y_axis='hz', sr=sr, ax=ax[i])
        ax[i].set_title(titles[i])
    fig.tight_layout()
    return fig


class PercepNetEvaluator(BaseEvaluator):
    def __init__(self, output_path, evaluator_callbacks, output_filename_pattern, output_format, store_spectrograms, metrics_list, sr, eps, sig_model_path, bak_ovr_model_path, input_length):
        BaseEvaluator.__init__(self, output_path=output_path, evaluator_callbacks=evaluator_callbacks)
        self.output_filename_pattern = output_filename_pattern
        self.output_format = output_format
        self.store_spectrograms = store_spectrograms
        self.metrics_list = metrics_list
        self.sr = sr
        self.eps = eps
        self.input_length = input_length
        self.sig_model_path = sig_model_path
        self.bak_ovr_model_path = bak_ovr_model_path

    def _setup(self, mode):
        os.makedirs(self.output_path, exist_ok=True)
        # create spectrogram output directory if needed
        if self.store_spectrograms:
            self.specs_output_path = os.path.join(self.output_path, 'specs')
            os.makedirs(self.specs_output_path, exist_ok=True)
        if mode == 'evaluate':
            if 'dnsmos_p835' in self.metrics_list:
                self.session_sig = ort.InferenceSession(self.sig_model_path)
                self.session_bak_ovr = ort.InferenceSession(self.bak_ovr_model_path)
    
    def _infer_step(self, model, dl_data, use_tflite):
        clean_batch, noisy_batch, _, metadata_batch = dl_data
        # process data
        pred_batch = model(noisy_batch, use_tflite=use_tflite)
        # convert to numpy arrays
        if HAS_TENSORFLOW and tf.is_tensor(clean_batch):
            clean_batch = clean_batch.numpy()
        if HAS_TENSORFLOW and tf.is_tensor(noisy_batch):
            noisy_batch = noisy_batch.numpy()
        if HAS_TENSORFLOW and tf.is_tensor(pred_batch):
            pred_batch = pred_batch.numpy()
        # loop through batch elements
        results = []
        for clean_data, noisy_data, pred_data, metadata in zip(clean_batch, noisy_batch, pred_batch, metadata_batch):
            # store predictions
            filename = os.path.basename(metadata['noisy_filepath'])
            output_filename = re.sub(self.output_filename_pattern, r'pred_\1', filename, 0) + '.' + self.output_format
            output_path = os.path.join(self.output_path, output_filename)
            if self.output_format == 'wav':
                sf.write(output_path, pred_data, self.sr)
            elif self.output_format == 'f32':
                with open(output_path, 'wb') as f:
                    float_array = array('f', pred_data.flatten())
                    float_array.tofile(f)
            elif self.output_format == 'pcm':
                # Save as 16-bit PCM (common for PercepNet)
                with open(output_path, 'wb') as f:
                    int_array = array('h', (pred_data * 32767).astype(np.int16))
                    int_array.tofile(f)
            # compute and store spectrogram
            if self.store_spectrograms:
                fig = create_figure(noisy_data, clean_data, pred_data, self.sr)
                spec_filename = re.sub(self.output_filename_pattern, r'spec_\1', filename, 0) + '.png'
                spec_output_path = os.path.join(self.specs_output_path, spec_filename)
                fig.savefig(spec_output_path)
                plt.close(fig)
            results.append({'path': output_path, 'data': pred_data, 'sr': self.sr})
        return results

    def _evaluate_step(self, dl_data, mode):
        clean_batch, _, pred_batch, metadata_batch = dl_data
        results = []
        for clean_data, pred_data, metadata in zip(clean_batch, pred_batch, metadata_batch):
            filename = os.path.basename(metadata['pred_filepath'])
            sr = metadata['sr'] 
            clean_data = clean_data.numpy()
            pred_data = pred_data.numpy()
            metrics = { 'filename': filename }
            # Beyond Wideband metrics 
            if 'si_sdr' in self.metrics_list and mode == 'evaluate':
                sisdr_pred = si_sdr(clean_data, pred_data, eps=self.eps)
                metrics['sisdr'] = sisdr_pred
            # WB metrics (requires conversion from sr>16 kHz to 16 kHz)
            if sr != 16000:
                clean_data = lr.resample(clean_data, orig_sr=sr, target_sr=16000)
                pred_data = lr.resample(pred_data, orig_sr=sr, target_sr=16000)
                sr = 16000
            if 'pesq' in self.metrics_list and mode == 'evaluate':
                pesq_pred = pesq(sr, clean_data, pred_data, 'wb')
                metrics['pesq'] = pesq_pred
            if 'dnsmos_p835' in self.metrics_list or mode == 'evaluate_nonintrusive':
                dnsmos_pred_SIG, dnsmos_pred_BAK, dnsmos_pred_OVR = dnsmos(pred_data, self.session_sig, self.session_bak_ovr, sr, self.input_length)
                metrics['dnsmos_SIG'] = dnsmos_pred_SIG
                metrics['dnsmos_BAK'] = dnsmos_pred_BAK
                metrics['dnsmos_OVR'] = dnsmos_pred_OVR
            results.append(metrics)
        return results

    def evaluate_audio_files(self, clean_path, enhanced_path, noisy_path=None):
        """
        Evaluate audio files directly without ML framework.
        
        Args:
            clean_path: Path to clean/reference audio file
            enhanced_path: Path to enhanced/processed audio file
            noisy_path: Optional path to noisy audio file (for spectrograms)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Load audio files
        clean_data, sr_clean = sf.read(clean_path)
        enhanced_data, sr_enhanced = sf.read(enhanced_path)
        
        # Ensure sample rates match
        if sr_clean != sr_enhanced:
            enhanced_data = lr.resample(enhanced_data, orig_sr=sr_enhanced, target_sr=sr_clean)
            sr_enhanced = sr_clean
        
        sr = sr_clean
        
        # Initialize metrics dictionary
        metrics = {
            'clean_file': os.path.basename(clean_path),
            'enhanced_file': os.path.basename(enhanced_path),
            'sample_rate': sr
        }
        
        # Compute SI-SDR if requested
        if 'si_sdr' in self.metrics_list:
            sisdr = si_sdr(clean_data, enhanced_data, eps=self.eps)
            metrics['sisdr'] = sisdr
            log.info(f"SI-SDR: {sisdr:.2f} dB")
        
        # Convert to 16kHz for PESQ and DNSMOS if needed
        clean_16k = clean_data
        enhanced_16k = enhanced_data
        if sr != 16000:
            clean_16k = lr.resample(clean_data, orig_sr=sr, target_sr=16000)
            enhanced_16k = lr.resample(enhanced_data, orig_sr=sr, target_sr=16000)
        
        # Compute PESQ if requested
        if 'pesq' in self.metrics_list:
            pesq_score = pesq(16000, clean_16k, enhanced_16k, 'wb')
            metrics['pesq'] = pesq_score
            log.info(f"PESQ: {pesq_score:.2f}")
        
        # Compute DNSMOS if requested
        if 'dnsmos_p835' in self.metrics_list:
            if not hasattr(self, 'session_sig'):
                log.warning("DNSMOS models not loaded. Call _setup('evaluate') first.")
            else:
                dnsmos_SIG, dnsmos_BAK, dnsmos_OVR = dnsmos(
                    enhanced_16k, self.session_sig, self.session_bak_ovr, 16000, self.input_length
                )
                metrics['dnsmos_SIG'] = dnsmos_SIG
                metrics['dnsmos_BAK'] = dnsmos_BAK
                metrics['dnsmos_OVR'] = dnsmos_OVR
                log.info(f"DNSMOS - SIG: {dnsmos_SIG:.2f}, BAK: {dnsmos_BAK:.2f}, OVR: {dnsmos_OVR:.2f}")
        
        # Generate spectrogram if requested and noisy audio is provided
        if self.store_spectrograms and noisy_path is not None:
            noisy_data, sr_noisy = sf.read(noisy_path)
            if sr_noisy != sr:
                noisy_data = lr.resample(noisy_data, orig_sr=sr_noisy, target_sr=sr)
            
            fig = create_figure(noisy_data, clean_data, enhanced_data, sr)
            spec_filename = os.path.splitext(os.path.basename(enhanced_path))[0] + '_spec.png'
            spec_path = os.path.join(self.specs_output_path, spec_filename)
            fig.savefig(spec_path)
            plt.close(fig)
            log.info(f"Spectrogram saved to: {spec_path}")
        
        return metrics
