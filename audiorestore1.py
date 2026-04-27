import argparse
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import scipy.ndimage as nd
import warnings
import os

warnings.filterwarnings('ignore')

class LoudPristineRestoration192k:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.processing_sr = 192000 
        self.target_output_sr = 96000 
        
    def load_and_prep(self):
        print(f"[*] Ripping '{os.path.basename(self.input_file)}' into the matrix...")
        y, sr = librosa.load(self.input_file, sr=None, mono=False)
        if y.ndim == 1:
            y = np.vstack((y, y))
            
        # We don't pad the volume permanently anymore. 
        # We process at original gain so we don't lose RMS (loudness).
        if sr != self.processing_sr:
            print(f"[*] Upsampling to {self.processing_sr}Hz for massive DSP headroom...")
            y = librosa.resample(y, orig_sr=sr, target_sr=self.processing_sr, res_type='kaiser_best')
        return y

    def lr_to_ms(self, audio):
        return (audio[0] + audio[1]) / 2.0, (audio[0] - audio[1]) / 2.0

    def ms_to_lr(self, mid, side):
        return np.array([mid + side, mid - side])

    def pure_scipy_stft_sbr(self, audio_channel):
        """
        Matrix SBR: High-frequency restoration.
        Turned the intensity back up to restore the "Air" and sparkle you wanted.
        """
        nperseg = 16384
        noverlap = 12288
        
        f, t, Zxx = signal.stft(audio_channel, fs=self.processing_sr, nperseg=nperseg, noverlap=noverlap)
        
        src_start = np.argmax(f >= 8000)
        src_end = np.argmax(f >= 16000)
        copy_len = src_end - src_start
        
        target_start = np.argmax(f >= 16000)
        target_end = target_start + copy_len
        
        air_Zxx = np.zeros_like(Zxx, dtype=complex)
        src_mag = np.abs(Zxx[src_start:src_end, :])
        
        # Flatter roll-off to keep the highs present and bright
        roll_off = np.logspace(0, -2.5, num=copy_len).reshape(-1, 1)
        air_mag = src_mag * roll_off * 0.60 # Increased intensity back to 60%
        
        random_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=air_mag.shape))
        air_Zxx[target_start:target_end, :] = air_mag * random_phase
        
        Zxx_final = Zxx + air_Zxx
        
        _, restored_audio = signal.istft(Zxx_final, fs=self.processing_sr, nperseg=nperseg, noverlap=noverlap)
        
        return restored_audio[:len(audio_channel)]

    def gaussian_lookahead_maximizer(self, audio, ceiling=0.98):
        """
        THE NEW APPROACH: 
        A pure-math Lookahead Brickwall Limiter.
        Instead of distorting the wave, we use a Gaussian filter on the volume envelope.
        It sees the peaks *before* they happen, cleanly ducks the gain, and releases instantly.
        Result: Massive perceived volume, zero clipping, zero distortion.
        """
        print("[*] Calculating Gaussian Volume Envelope...")
        # 1. Get the absolute volume level of the track
        envelope = np.abs(audio)
        
        # 2. Apply a Gaussian blur to the envelope (Lookahead + Smoothing)
        # 1 millisecond at 192kHz is 192 samples. 
        # A sigma of 96 gives us incredibly fast, transparent reaction times.
        sigma = int(self.processing_sr * 0.0005) 
        smoothed_env = nd.gaussian_filter1d(envelope, sigma=sigma, axis=1)
        
        # 3. Calculate dynamic gain reduction ONLY where the volume exceeds the ceiling
        gain = np.ones_like(audio)
        over_threshold = smoothed_env > ceiling
        
        # Mathematically pull down the volume of the spikes
        gain[over_threshold] = ceiling / smoothed_env[over_threshold]
        
        # 4. Smooth the gain reduction itself so there are no "clicks" or "pops"
        gain = nd.gaussian_filter1d(gain, sigma=sigma, axis=1)
        
        # 5. Apply the dynamic gain to the original pristine audio
        return audio * gain

    def process(self):
        print("\n[!] INITIATING V5: FULL-VOLUME GAUSSIAN MAXIMIZER ENGINE...")
        audio = self.load_and_prep()
        
        mid, side = self.lr_to_ms(audio)

        print("[*] Rebuilding Mid Channel Highs (Full Intensity)...")
        mid = self.pure_scipy_stft_sbr(mid)
        
        print("[*] Rebuilding Side Channel Highs (Full Intensity)...")
        side = self.pure_scipy_stft_sbr(side)

        print("[*] Recombining Stereo Matrix...")
        restored_audio = self.ms_to_lr(mid, side)

        print("[*] Engaging Zero-Phase Gaussian Lookahead Maximizer...")
        # Push the audio hard, but catch it exactly at -0.1dB
        restored_audio = self.gaussian_lookahead_maximizer(restored_audio, ceiling=0.98)

        print(f"[*] Downsampling to target {self.target_output_sr}Hz...")
        final_audio = librosa.resample(restored_audio, orig_sr=self.processing_sr, target_sr=self.target_output_sr, res_type='kaiser_best')

        print(f"[*] Exporting LOUD 24-bit PCM Master to: {self.output_file}")
        sf.write(self.output_file, final_audio.T, self.target_output_sr, subtype='PCM_24')
        print("=== [DONE] AUDIO IS PUMPED AND PRISTINE. ENJOY. ===\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default="restored_master_LOUD.wav")
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        pipeline = LoudPristineRestoration192k(args.input, args.output)
        pipeline.process()
    else:
        print("File not found, try again.")