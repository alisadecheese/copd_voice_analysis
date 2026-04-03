import parselmouth
import numpy as np
import soundfile as sf
import pandas as pd
from scipy import stats
from typing import Dict, Optional, List
import warnings
import os

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa не установлена. MFCC признаки будут недоступны.")

warnings.filterwarnings("ignore")


class VoiceFeatureExtractor:
    """Извлечение акустических признаков (включая eGeMAPS-подобные)"""

    def __init__(self, sampling_rate: int = 16000):
        self.sampling_rate = sampling_rate
        self.pitch_floor = 75
        self.pitch_ceiling = 300
        self.feature_columns = []
        self.patient_groups = []

    def load_audio(self, path: str) -> Optional[parselmouth.Sound]:
        """Загрузка аудиофайла"""
        try:
            data, sr = sf.read(path)
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            return parselmouth.Sound(data, sr)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {path}: {e}")
            return None

    def extract_features(self, sound: parselmouth.Sound) -> Optional[Dict]:
        """Извлечение всех признаков (акустические + спектральные)"""
        features = {}

        try:
            if sound.xmax < 0.3:
                features["valid"] = False
                return features

            # === 1. PITCH признаки (основные) ===
            try:
                pitch = sound.to_pitch(
                    pitch_floor=self.pitch_floor,
                    pitch_ceiling=self.pitch_ceiling
                )
                pitch_values = pitch.selected_array["frequency"]
                voiced_frames = pitch_values[pitch_values > 0]

                if len(voiced_frames) > 0:
                    features["mean_pitch"] = float(np.mean(voiced_frames))
                    features["std_pitch"] = float(np.std(voiced_frames))
                    features["min_pitch"] = float(np.min(voiced_frames))
                    features["max_pitch"] = float(np.max(voiced_frames))
                    features["pitch_range"] = float(np.max(voiced_frames) - np.min(voiced_frames))
                    features["median_pitch"] = float(np.median(voiced_frames))
                    features["pitch_25_percentile"] = float(np.percentile(voiced_frames, 25))
                    features["pitch_75_percentile"] = float(np.percentile(voiced_frames, 75))
                else:
                    for k in ["mean_pitch", "std_pitch", "min_pitch", "max_pitch", 
                              "pitch_range", "median_pitch", "pitch_25_percentile", "pitch_75_percentile"]:
                        features[k] = 0.0
            except Exception as e:
                print(f"  Pitch ошибка: {e}")
                for k in ["mean_pitch", "std_pitch", "min_pitch", "max_pitch", 
                          "pitch_range", "median_pitch", "pitch_25_percentile", "pitch_75_percentile"]:
                    features[k] = 0.0

            # === 2. Jitter признаки (стабильность частоты) ===
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

            # === 3. Shimmer признаки (стабильность амплитуды) ===
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

            # === 4. HNR (Harmonics-to-Noise Ratio) ===
            try:
                hnr = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 4.5)
                hnr_values = hnr.values
                valid_hnr = hnr_values[hnr_values > -10]
                features["hnr_mean"] = float(np.mean(valid_hnr)) if len(valid_hnr) > 0 else 0.0
                features["hnr_std"] = float(np.std(valid_hnr)) if len(valid_hnr) > 1 else 0.0
                features["hnr_min"] = float(np.min(valid_hnr)) if len(valid_hnr) > 0 else 0.0
                features["hnr_max"] = float(np.max(valid_hnr)) if len(valid_hnr) > 0 else 0.0
            except Exception:
                for k in ["hnr_mean", "hnr_std", "hnr_min", "hnr_max"]:
                    features[k] = 0.0

            # === 5. Voicing Ratio ===
            try:
                pitch = sound.to_pitch(pitch_floor=self.pitch_floor, pitch_ceiling=self.pitch_ceiling)
                total_frames = pitch.get_number_of_frames()
                pitch_values = pitch.selected_array["frequency"]
                voiced_frames = len(pitch_values[pitch_values > 0])
                features["voicing_ratio"] = float((voiced_frames / total_frames * 100)) if total_frames > 0 else 0.0
            except Exception:
                features["voicing_ratio"] = 0.0

            # === 6. MFCC (если доступна librosa) ===
            if LIBROSA_AVAILABLE:
                try:
                    values = sound.values.T[0]
                    sr = int(sound.sampling_frequency)
                    
                    # MFCC (13 коэффициентов)
                    mfccs = librosa.feature.mfcc(y=values, sr=sr, n_mfcc=13)
                    for i in range(13):
                        features[f"mfcc_{i+1}"] = float(np.mean(mfccs[i]))
                    
                    # Spectral features
                    spectral_centroid = librosa.feature.spectral_centroid(y=values, sr=sr)
                    features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
                    features["spectral_centroid_std"] = float(np.std(spectral_centroid))
                    
                    spectral_rolloff = librosa.feature.spectral_rolloff(y=values, sr=sr)
                    features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
                    
                    spectral_flux = librosa.onset.onset_strength(y=values, sr=sr)
                    features["spectral_flux_mean"] = float(np.mean(spectral_flux))
                    
                    zcr = librosa.feature.zero_crossing_rate(values)
                    features["zero_crossing_rate_mean"] = float(np.mean(zcr))
                    
                except Exception as e:
                    print(f"  MFCC ошибка: {e}")
                    for i in range(13):
                        features[f"mfcc_{i+1}"] = 0.0
                    for k in ["spectral_centroid_mean", "spectral_centroid_std", 
                              "spectral_rolloff_mean", "spectral_flux_mean", "zero_crossing_rate_mean"]:
                        features[k] = 0.0
            else:
                for i in range(13):
                    features[f"mfcc_{i+1}"] = 0.0
                for k in ["spectral_centroid_mean", "spectral_centroid_std", 
                          "spectral_rolloff_mean", "spectral_flux_mean", "zero_crossing_rate_mean"]:
                    features[k] = 0.0

            # === 7. Energy features ===
            try:
                values = sound.values.T[0]
                if LIBROSA_AVAILABLE:
                    rms = librosa.feature.rms(y=values)
                    features["rms_energy_mean"] = float(np.mean(rms))
                    features["rms_energy_std"] = float(np.std(rms))
                else:
                    features["rms_energy_mean"] = float(np.mean(np.abs(values)))
                    features["rms_energy_std"] = float(np.std(np.abs(values)))
            except Exception:
                features["rms_energy_mean"] = 0.0
                features["rms_energy_std"] = 0.0

            # === 8. Duration features ===
            features["total_duration"] = float(sound.xmax)
            
            features["valid"] = True

        except Exception as e:
            print(f"⚠️ Общая ошибка: {e}")
            features["valid"] = False

        return features

    def _bowley_skew(self, data: np.ndarray) -> float:
        """Расчет коэффициента асимметрии Bowley"""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        median = np.median(data)
        if (q3 - q1) != 0:
            return (q3 + q1 - 2 * median) / (q3 - q1)
        return 0.0

    def prepare_dataset(self, data_dir):
        """Подготовка датасета"""
        X = []
        y = []
        valid_files = []
        self.patient_groups = []

        folder_mapping = {
            'copd_before': 1,
            'copd_after': 0
        }

        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

        print(f"📂 Сканирование папки: {data_dir}")

        for folder_name, label in folder_mapping.items():
            folder_path = os.path.join(data_dir, folder_name)

            if not os.path.exists(folder_path):
                print(f"  ⚠️ Папка не найдена: {folder_path}")
                continue

            files = os.listdir(folder_path)
            audio_files = [f for f in files if os.path.splitext(f)[1].lower() in audio_extensions]

            print(f"  • Найдено файлов в {folder_name}: {len(audio_files)}")

            for filename in audio_files:
                file_path = os.path.join(folder_path, filename)

                name_without_ext = os.path.splitext(filename)[0]
                patient_id = name_without_ext.rsplit('-', 1)[0]

                sound = self.load_audio(file_path)

                if sound is None:
                    print(f"    ⚠️ Пропущен файл (не загружен): {filename}")
                    continue

                feats = self.extract_features(sound)

                if feats and feats.get("valid"):
                    for k, v in feats.items():
                        if isinstance(v, float) and np.isnan(v):
                            feats[k] = 0.0

                    X.append(feats)
                    y.append(label)
                    valid_files.append(file_path)
                    self.patient_groups.append(patient_id)
                else:
                    print(f"    ⚠️ Пропущен файл (ошибка извлечения): {filename}")

        if len(X) == 0:
            print("❌ Не удалось извлечь признаки ни из одного файла!")
            return np.array([]), np.array([])

        df = pd.DataFrame(X)
        self.feature_columns = [c for c in df.columns if c not in ['filename', 'label', 'valid']]

        unique_patients = len(set(self.patient_groups))
        patients_both = len([pid for pid in set(self.patient_groups) if self.patient_groups.count(pid) > 1])

        print(f"✅ Всего обработано файлов: {len(X)}")
        print(f"   - Уникальных пациентов: {unique_patients}")
        print(f"   - Пациентов с парными записями: {patients_both}")
        print(f"   - Класс 1 (До лечения): {y.count(1)}")
        print(f"   - Класс 0 (После лечения): {y.count(0)}")
        print(f"   - Признаков: {len(self.feature_columns)}")

        return df.values, np.array(y)