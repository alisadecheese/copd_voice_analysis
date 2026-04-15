import parselmouth
import numpy as np
import soundfile as sf
import pandas as pd
from scipy import stats, signal
import warnings
import os
import tempfile

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import opensmile
    OPENSMILE_AVAILABLE = True
except ImportError:
    OPENSMILE_AVAILABLE = False
    print("Warning: openSMILE not installed - eGeMAPS will be unavailable")

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
        
        if OPENSMILE_AVAILABLE:
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )

    def load_audio(self, path: str) -> tuple:
        """Загрузка аудиофайла"""
        try:
            if PYDUB_AVAILABLE:
                try:
                    trimmed_path = self.trim_silence(path)
                    data, sr = sf.read(trimmed_path)
                except Exception:
                    data, sr = sf.read(path)
            else:
                data, sr = sf.read(path)
                
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            return parselmouth.Sound(data, sr), data, sr
        except Exception:
            return None, None, None

    def trim_silence(self, audio_path, output_path=None):
        """Удаление тишины в начале и конце файла"""
        try:
            sound = AudioSegment.from_file(audio_path)
            non_silent = detect_nonsilent(sound, min_silence_len=200, silence_thresh=-40)
            
            if non_silent:
                trimmed = sound[non_silent[0][0]:non_silent[-1][1]]
                if output_path:
                    trimmed.export(output_path, format='wav')
                    return output_path
                else:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    trimmed.export(temp_file.name, format='wav')
                    return temp_file.name
        except Exception:
            pass
        
        return audio_path

    def extract_egemaps_features(self, audio_path):
        """Извлечение eGeMAPS признаков через openSMILE (88 параметров)"""
        features = {}
        
        if not OPENSMILE_AVAILABLE or not audio_path:
            return features
        
        try:
            y = self.smile.process_file(audio_path)
            
            for i, col in enumerate(y.columns):
                features[f"egemaps_{i:02d}_{col}"] = float(y[col].values[0])
                
        except Exception as e:
            print(f"  openSMILE error: {e}")
        
        return features

    def extract_features(self, sound, audio_values=None, sr=None, audio_path=None):
        """Извлечение акустических признаков (28 признаков)"""
        features = {}
        
        if sound is None or sound.xmax < 0.3:
            features["valid"] = False
            return features
        
        try:
            # 1. PITCH признаки (5 признаков)
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
                    for k in ["mean_pitch", "std_pitch", "min_pitch", "max_pitch", "pitch_range"]:
                        features[k] = 0.0
            except Exception:
                for k in ["mean_pitch", "std_pitch", "min_pitch", "max_pitch", "pitch_range"]:
                    features[k] = 0.0
            
            # 2. Jitter признаки (5 признаков)
            point_process = None
            try:
                point_process = parselmouth.praat.call(
                    sound, 
                    "To PointProcess (periodic, cc)", 
                    self.pitch_floor, 
                    self.pitch_ceiling
                )
            except Exception:
                pass

            if point_process is not None:
                jitter_methods = [
                    ("jitter_local", "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
                    ("jitter_abs", "Get jitter (absolute)", 0, 0, 0.0001, 0.02, 1.3),
                    ("jitter_rap", "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3),
                    ("jitter_ppq5", "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3),
                    ("jitter_ddp", "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3),
                ]
                for param_name, method_name, *args in jitter_methods:
                    try:
                        value = parselmouth.praat.call(point_process, method_name, *args)
                        if value is not None and not np.isnan(value) and 0 <= value < 1:
                            features[param_name] = float(value)
                        else:
                            features[param_name] = 0.0
                    except Exception:
                        features[param_name] = 0.0
            else:
                for param_name in ["jitter_local", "jitter_abs", "jitter_rap", "jitter_ppq5", "jitter_ddp"]:
                    features[param_name] = 0.0

            # 3. Shimmer признаки (5 признаков)
            if point_process is not None:
                shimmer_methods = [
                    ("shimmer_local", "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 0.001),
                    ("shimmer_apq3", "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 0.001),
                    ("shimmer_apq5", "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 0.001),
                    ("shimmer_apq11", "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 0.001),
                    ("shimmer_dda", "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 0.001),
                ]
                for param_name, method_name, *args in shimmer_methods:
                    try:
                        value = parselmouth.praat.call([sound, point_process], method_name, *args)
                        if value is not None and not np.isnan(value) and 0 <= value < 1:
                            features[param_name] = float(value)
                        else:
                            features[param_name] = 0.0
                    except Exception:
                        features[param_name] = 0.0
            else:
                for param_name in ["shimmer_local", "shimmer_apq3", "shimmer_apq5", "shimmer_apq11", "shimmer_dda"]:
                    features[param_name] = 0.0
            
            # 4. HNR (2 признака)
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
            
            # 5. Voicing Ratio (1 признак)
            try:
                pitch = sound.to_pitch(pitch_floor=self.pitch_floor, pitch_ceiling=self.pitch_ceiling)
                total_frames = pitch.get_number_of_frames()
                voiced = len(pitch.selected_array["frequency"][pitch.selected_array["frequency"] > 0])
                features["voicing_ratio"] = float((voiced / total_frames * 100)) if total_frames > 0 else 0.0
            except Exception:
                features["voicing_ratio"] = 0.0
            
            # 6. Formants (6 признаков)
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
            
            # 7. LGDV признаки (4 признака)
            if audio_values is not None and sr is not None and LIBROSA_AVAILABLE:
                try:
                    rms = librosa.feature.rms(y=audio_values)[0]
                    silence_thresh = np.mean(rms) * 0.1
                    silence_frames = np.where(rms < silence_thresh)[0]
                    
                    if len(silence_frames) > 1:
                        silence_durations = np.diff(silence_frames) / sr
                        total_pause = np.sum(silence_durations)
                        
                        features["lgdv_pause_percent"] = float((total_pause / sound.xmax) * 100) if sound.xmax > 0 else 0.0
                        features["lgdv_pause_rate"] = float(len(silence_durations) / sound.xmax) if sound.xmax > 0 else 0.0
                    else:
                        features["lgdv_pause_percent"] = 0.0
                        features["lgdv_pause_rate"] = 0.0
                    
                    mean_rms = np.mean(rms)
                    features["lgdv_loudness_cv"] = float(np.std(rms) / mean_rms) if mean_rms > 0 else 0.0
                    
                    spectral_flux = librosa.onset.onset_strength(y=audio_values, sr=sr)
                    mean_flux = np.mean(spectral_flux)
                    features["lgdv_spectral_flux_cv"] = float(np.std(spectral_flux) / mean_flux) if mean_flux > 0 else 0.0
                except Exception:
                    for k in ["lgdv_pause_percent", "lgdv_pause_rate", "lgdv_loudness_cv", "lgdv_spectral_flux_cv"]:
                        features[k] = 0.0
            else:
                for k in ["lgdv_pause_percent", "lgdv_pause_rate", "lgdv_loudness_cv", "lgdv_spectral_flux_cv"]:
                    features[k] = 0.0
            
            # 8. eGeMAPS признаки через openSMILE (все 88, потом отфильтруем)
            if audio_path:
                egemaps = self.extract_egemaps_features(audio_path)
                features.update(egemaps)
            
            # 9. Spectral признаки
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
            
            # 10. Duration
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