import parselmouth
import numpy as np
import soundfile as sf
import pandas as pd
from scipy import stats, signal
import warnings
import os

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

warnings.filterwarnings("ignore")


class VoiceFeatureExtractor:
    """Извлечение акустических признаков для диагностики ХОБЛ"""

    def __init__(self, sampling_rate: int = 16000):
        self.sampling_rate = sampling_rate
        self.pitch_floor = 75
        self.pitch_ceiling = 300
        self.feature_columns = []
        self.patient_groups = []

    def load_audio(self, path: str) -> tuple:
        """Загрузка аудиофайла (без создания временных файлов)"""
        try:
            data, sr = sf.read(path)
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            return parselmouth.Sound(data, sr), data, sr
        except Exception:
            return None, None, None

    def extract_features(self, sound, audio_values=None, sr=None, audio_path=None):
        """Извлечение признаков (БЕЗ Jitter/Shimmer — они не работают для ХОБЛ)"""
        features = {}
        
        if sound is None or sound.xmax < 0.3:
            features["valid"] = False
            return features
        
        try:
            # 1. PITCH признаки
            pitch = None
            pitch_values = []
            voiced_frames = []
            
            try:
                pitch = sound.to_pitch(pitch_floor=self.pitch_floor, pitch_ceiling=self.pitch_ceiling)
                pitch_values = pitch.selected_array["frequency"]
                voiced_frames = pitch_values[pitch_values > 0]
                
                if len(voiced_frames) > 0:
                    features["mean_pitch"] = float(np.mean(voiced_frames))
                    features["std_pitch"] = float(np.std(voiced_frames))
                    features["min_pitch"] = float(np.min(voiced_frames))
                    features["max_pitch"] = float(np.max(voiced_frames))
                    features["pitch_range"] = float(np.max(voiced_frames) - np.min(voiced_frames))
                else:
                    features["mean_pitch"] = features["std_pitch"] = features["min_pitch"] = 0.0
                    features["max_pitch"] = features["pitch_range"] = 0.0
            except Exception:
                features["mean_pitch"] = features["std_pitch"] = features["min_pitch"] = 0.0
                features["max_pitch"] = features["pitch_range"] = 0.0
            
            # 2. HNR
            try:
                hnr = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 4.5)
                hnr_values = hnr.values
                valid_hnr = hnr_values[(hnr_values >= 0) & (hnr_values <= 50)]
                if len(valid_hnr) > 10:
                    features["hnr_mean"] = float(np.mean(valid_hnr))
                    features["hnr_std"] = float(np.std(valid_hnr))
                else:
                    features["hnr_mean"] = 17.0
                    features["hnr_std"] = 3.0
            except Exception:
                features["hnr_mean"] = 17.0
                features["hnr_std"] = 3.0
            
            # 3. Voicing Ratio
            try:
                if pitch is None:
                    pitch = sound.to_pitch(pitch_floor=self.pitch_floor, pitch_ceiling=self.pitch_ceiling)
                total_frames = pitch.get_number_of_frames()
                voiced = len(pitch.selected_array["frequency"][pitch.selected_array["frequency"] > 0])
                features["voicing_ratio"] = float((voiced / total_frames * 100)) if total_frames > 0 else 0.0
            except Exception:
                features["voicing_ratio"] = 0.0
            
            # 4. Formants
            try:
                formant = sound.to_formant_burg()
                times = np.linspace(0.1, sound.xmax - 0.1, 10)
                f1_values, f2_values, f3_values = [], [], []
                
                for t in times:
                    try:
                        f1 = formant.get_value_at_time(1, t)
                        f2 = formant.get_value_at_time(2, t)
                        f3 = formant.get_value_at_time(3, t)
                        if 0 < f1 < 4000: f1_values.append(f1)
                        if 0 < f2 < 4000: f2_values.append(f2)
                        if 0 < f3 < 4000: f3_values.append(f3)
                    except:
                        continue
                
                if len(f1_values) > 1:
                    features["formant_f1_mean"] = float(np.mean(f1_values))
                    features["formant_f1_cv"] = float(np.std(f1_values) / np.mean(f1_values)) if np.mean(f1_values) > 0 else 0.0
                else:
                    features["formant_f1_mean"] = features["formant_f1_cv"] = 500.0
                    
                if len(f2_values) > 1:
                    features["formant_f2_mean"] = float(np.mean(f2_values))
                    features["formant_f2_cv"] = float(np.std(f2_values) / np.mean(f2_values)) if np.mean(f2_values) > 0 else 0.0
                else:
                    features["formant_f2_mean"] = features["formant_f2_cv"] = 1500.0
                    
                if len(f3_values) > 1:
                    features["formant_f3_mean"] = float(np.mean(f3_values))
                    features["formant_f3_cv"] = float(np.std(f3_values) / np.mean(f3_values)) if np.mean(f3_values) > 0 else 0.0
                else:
                    features["formant_f3_mean"] = features["formant_f3_cv"] = 2500.0
            except Exception:
                features["formant_f1_mean"] = features["formant_f1_cv"] = 500.0
                features["formant_f2_mean"] = features["formant_f2_cv"] = 1500.0
                features["formant_f3_mean"] = features["formant_f3_cv"] = 2500.0
            
            # 5. eGeMAPS признаки (через librosa — без временных файлов!)
            if audio_values is not None and sr is not None and LIBROSA_AVAILABLE:
                try:
                    values_norm = audio_values / np.max(np.abs(audio_values)) if np.max(np.abs(audio_values)) > 0 else audio_values
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(values_norm)), ref=np.max)
                    freqs = librosa.fft_frequencies(sr=sr)
                    
                    f0_mean = np.mean(voiced_frames) if len(voiced_frames) > 0 else 100.0
                    
                    # MFCC
                    mfccs = librosa.feature.mfcc(y=values_norm, sr=sr, n_mfcc=4)
                    features["egemaps_22_mfcc1_sma3_amean"] = float(np.mean(mfccs[0]))
                    features["egemaps_24_mfcc2_sma3_amean"] = float(np.mean(mfccs[1]))
                    features["egemaps_28_mfcc4_sma3_amean"] = float(np.mean(mfccs[3]))
                    
                    # H1-A3
                    h1_idx = np.argmin(np.abs(freqs - f0_mean)) if len(voiced_frames) > 0 else 0
                    a3_idx = np.argmin(np.abs(freqs - 2500))
                    if h1_idx < len(D) and a3_idx < len(D):
                        h1 = D[h1_idx, :].mean()
                        a3 = D[a3_idx, :].mean()
                        h1_a3 = h1 - a3
                        features["egemaps_38_logRelF0-H1-A3_sma3nz_amean"] = float(h1_a3)
                        features["egemaps_39_logRelF0-H1-A3_sma3nz_stddevNorm"] = float(np.std([h1_a3]))
                    else:
                        features["egemaps_38_logRelF0-H1-A3_sma3nz_amean"] = 0.0
                        features["egemaps_39_logRelF0-H1-A3_sma3nz_stddevNorm"] = 0.0
                    
                    # F1 amplitude
                    f1_band = D[np.argmin(np.abs(freqs - 500)), :].mean()
                    f0_band = D[np.argmin(np.abs(freqs - f0_mean)), :].mean() if f0_mean > 0 else 0
                    features["egemaps_44_F1amplitudeLogRelF0_sma3nz_amean"] = float(f1_band)
                    features["egemaps_45_F1amplitudeLogRelF0_sma3nz_stddevNorm"] = float(np.abs(f1_band - f0_band)) if f0_band != 0 else 0.0
                    
                    # F2
                    f2_band = D[np.argmin(np.abs(freqs - 1500)), :].mean()
                    features["egemaps_48_F2bandwidth_sma3nz_amean"] = float(np.std(D[np.argmin(np.abs(freqs - 1500)), :]))
                    features["egemaps_50_F2amplitudeLogRelF0_sma3nz_amean"] = float(f2_band)
                    features["egemaps_51_F2amplitudeLogRelF0_sma3nz_stddevNorm"] = float(np.abs(f2_band - f0_band)) if f0_band != 0 else 0.0
                    
                    # F3
                    f3_band = D[np.argmin(np.abs(freqs - 2500)), :].mean()
                    features["egemaps_56_F3amplitudeLogRelF0_sma3nz_amean"] = float(f3_band)
                    features["egemaps_57_F3amplitudeLogRelF0_sma3nz_stddevNorm"] = float(np.abs(f3_band - f0_band)) if f0_band != 0 else 0.0
                    
                    # Spectral slope
                    idx_500 = np.argmin(np.abs(freqs - 500))
                    idx_1500 = np.argmin(np.abs(freqs - 1500))
                    if idx_1500 > idx_500:
                        slope = np.polyfit(freqs[idx_500:idx_1500], D[idx_500:idx_1500].mean(axis=1), 1)[0]
                        features["egemaps_65_slopeV500-1500_sma3nz_stddevNorm"] = float(np.abs(slope))
                    else:
                        features["egemaps_65_slopeV500-1500_sma3nz_stddevNorm"] = 0.0
                    
                    # Alpha ratio
                    features["egemaps_58_alphaRatioV_sma3nz_amean"] = float(np.mean(librosa.feature.spectral_rolloff(y=values_norm, sr=sr)))
                    
                    # Sound level
                    features["egemaps_87_equivalentSoundLevel_dBp"] = float(librosa.power_to_db(np.mean(values_norm**2)))
                    
                except Exception:
                    # Все eGeMAPS = 0 при ошибке
                    egemaps_list = [
                        "egemaps_22_mfcc1_sma3_amean", "egemaps_24_mfcc2_sma3_amean", "egemaps_28_mfcc4_sma3_amean",
                        "egemaps_38_logRelF0-H1-A3_sma3nz_amean", "egemaps_39_logRelF0-H1-A3_sma3nz_stddevNorm",
                        "egemaps_44_F1amplitudeLogRelF0_sma3nz_amean", "egemaps_45_F1amplitudeLogRelF0_sma3nz_stddevNorm",
                        "egemaps_48_F2bandwidth_sma3nz_amean", "egemaps_50_F2amplitudeLogRelF0_sma3nz_amean",
                        "egemaps_51_F2amplitudeLogRelF0_sma3nz_stddevNorm", "egemaps_56_F3amplitudeLogRelF0_sma3nz_amean",
                        "egemaps_57_F3amplitudeLogRelF0_sma3nz_stddevNorm", "egemaps_58_alphaRatioV_sma3nz_amean",
                        "egemaps_65_slopeV500-1500_sma3nz_stddevNorm", "egemaps_87_equivalentSoundLevel_dBp"
                    ]
                    for feat in egemaps_list:
                        if feat not in features:
                            features[feat] = 0.0
            else:
                egemaps_list = [
                    "egemaps_22_mfcc1_sma3_amean", "egemaps_24_mfcc2_sma3_amean", "egemaps_28_mfcc4_sma3_amean",
                    "egemaps_38_logRelF0-H1-A3_sma3nz_amean", "egemaps_39_logRelF0-H1-A3_sma3nz_stddevNorm",
                    "egemaps_44_F1amplitudeLogRelF0_sma3nz_amean", "egemaps_45_F1amplitudeLogRelF0_sma3nz_stddevNorm",
                    "egemaps_48_F2bandwidth_sma3nz_amean", "egemaps_50_F2amplitudeLogRelF0_sma3nz_amean",
                    "egemaps_51_F2amplitudeLogRelF0_sma3nz_stddevNorm", "egemaps_56_F3amplitudeLogRelF0_sma3nz_amean",
                    "egemaps_57_F3amplitudeLogRelF0_sma3nz_stddevNorm", "egemaps_58_alphaRatioV_sma3nz_amean",
                    "egemaps_65_slopeV500-1500_sma3nz_stddevNorm", "egemaps_87_equivalentSoundLevel_dBp"
                ]
                for feat in egemaps_list:
                    if feat not in features:
                        features[feat] = 0.0
            
            # 6. Spectral признаки
            if audio_values is not None and sr is not None and LIBROSA_AVAILABLE:
                try:
                    if len(audio_values) > 0:
                        values_norm = audio_values / np.max(np.abs(audio_values)) if np.max(np.abs(audio_values)) > 0 else audio_values
                        features["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=values_norm, sr=sr)))
                        features["spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=values_norm, sr=sr)))
                        features["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(values_norm)))
                    else:
                        features["spectral_centroid"] = features["spectral_rolloff"] = features["zero_crossing_rate"] = 0.0
                except Exception:
                    features["spectral_centroid"] = features["spectral_rolloff"] = features["zero_crossing_rate"] = 0.0
            else:
                features["spectral_centroid"] = features["spectral_rolloff"] = features["zero_crossing_rate"] = 0.0
            
            # 7. Duration
            features["total_duration"] = float(sound.xmax)
            
            features["valid"] = True
            
        except Exception:
            features["valid"] = False
        
        return features

    def prepare_dataset(self, data_dir):
        """Подготовка датасета"""
        X, y, valid_files = [], [], []
        self.patient_groups = []
        folder_mapping = {'copd_before': 1, 'copd_after': 0}
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

        for folder_name, label in folder_mapping.items():
            folder_path = os.path.join(data_dir, folder_name)
            if not os.path.exists(folder_path):
                continue

            files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in audio_extensions]

            for filename in files:
                file_path = os.path.join(folder_path, filename)
                patient_id = os.path.splitext(filename)[0].rsplit('-', 1)[0]
                
                sound, audio_values, sr = self.load_audio(file_path)

                if sound is None:
                    continue

                feats = self.extract_features(sound, audio_values, sr, file_path)
                
                if feats and feats.get("valid"):
                    for k, v in feats.items():
                        if isinstance(v, float) and np.isnan(v):
                            feats[k] = 0.0
                    X.append(feats)
                    y.append(label)
                    valid_files.append(file_path)
                    self.patient_groups.append(patient_id)

        if len(X) == 0:
            return np.array([]), np.array([])

        df = pd.DataFrame(X)
        self.feature_columns = list(df.columns)
        
        return df.values, np.array(y)