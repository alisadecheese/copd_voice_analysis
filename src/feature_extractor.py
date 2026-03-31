# src/feature_extractor.py

import parselmouth
import numpy as np
import soundfile as sf
import pandas as pd
from scipy import stats
from typing import Dict, Optional
import warnings
import os

warnings.filterwarnings("ignore")


class VoiceFeatureExtractor:
    """Извлечение акустических признаков через Praat (parselmouth)"""

    def __init__(self, sampling_rate: int = 48000):
        self.sampling_rate = sampling_rate
        self.pitch_floor = 75  # Гц
        self.pitch_ceiling = 300  # Гц
        self.feature_columns = []
        self.patient_groups = []  # 🔴 ИНИЦИАЛИЗАЦИЯ СПИСКА ID ПАЦИЕНТОВ

    def load_audio(self, path: str) -> Optional[parselmouth.Sound]:
        """Загрузка аудиофайла с обработкой ошибок"""
        try:
            data, sr = sf.read(path)
            if len(data.shape) > 1:
                data = data.mean(axis=1)  # Конвертация в моно
            return parselmouth.Sound(data, sr)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {path}: {e}")
            return None

    def extract_features(self, sound: parselmouth.Sound) -> Optional[Dict]:
        """Извлечение всех акустических признаков"""
        features = {}

        try:
            # === Pitch (Частота основного тона) ===
            pitch = sound.to_pitch(
                pitch_floor=self.pitch_floor, pitch_ceiling=self.pitch_ceiling
            )
            pitch_values = pitch.selected_array["frequency"]
            pitch_values = pitch_values[pitch_values > 0]

            if len(pitch_values) > 0:
                features["mean_pitch"] = float(np.mean(pitch_values))
                features["min_pitch"] = float(np.min(pitch_values))
                features["max_pitch"] = float(np.max(pitch_values))
                features["std_pitch"] = float(np.std(pitch_values))
                features["pitch_range"] = float(np.max(pitch_values) - np.min(pitch_values))
                features["bowley_skew"] = float(self._bowley_skew(pitch_values))
                features["kurtosis"] = float(stats.kurtosis(pitch_values))
            else:
                for k in ["mean_pitch", "min_pitch", "max_pitch", "std_pitch", "pitch_range", "bowley_skew", "kurtosis"]:
                    features[k] = 0.0

            # === Jitter (Нестабильность частоты) ===
            jitter_params = {
                "jitter_local": ("Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
                "jitter_abs": ("Get jitter (absolute)", 0, 0, 0.0001, 0.02, 1.3),
                "jitter_rap": ("Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3),
                "jitter_ppq5": ("Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3),
                "jitter_ddp": ("Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3),
            }
            for param_name, args in jitter_params.items():
                try:
                    features[param_name] = float(parselmouth.praat.call(sound, *args))
                except Exception:
                    features[param_name] = 0.0

            # === Shimmer (Нестабильность амплитуды) ===
            shimmer_params = {
                "shimmer_local": ("Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3),
                "shimmer_apq3": ("Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3),
                "shimmer_apq5": ("Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3),
                "shimmer_apq11": ("Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3),
                "shimmer_dda": ("Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3),
            }
            for param_name, args in shimmer_params.items():
                try:
                    features[param_name] = float(parselmouth.praat.call(sound, *args))
                except Exception:
                    features[param_name] = 0.0

            # === HNR (Отношение гармоника/шум) ===
            try:
                hnr = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 4.5)
                hnr_values = hnr.values
                valid_hnr = hnr_values[hnr_values > -10]
                features["hnr_mean"] = float(np.mean(valid_hnr)) if len(valid_hnr) > 0 else 0.0
            except Exception:
                features["hnr_mean"] = 0.0

            # === Voicing Ratio ===
            total_frames = pitch.get_number_of_frames()
            voiced_frames = len(pitch_values)
            features["voicing_ratio"] = float((voiced_frames / total_frames * 100)) if total_frames > 0 else 0.0

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
        """Автоматически сканирует подпапки data_dir и создает датасет."""
        X = []
        y = []
        valid_files = []
        
        # 🔴 СБРОС СПИСКА ID ПАЦИЕНТОВ (важно для повторных запусков)
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
                
                # 🔴 ИЗВЛЕЧЕНИЕ ID ПАЦИЕНТА ИЗ ИМЕНИ ФАЙЛА
                # Пример: "01-5001-1.wav" → ID "01-5001"
                name_without_ext = os.path.splitext(filename)[0]  # "01-5001-1"
                patient_id = name_without_ext.rsplit('-', 1)[0]   # "01-5001"
                
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
                    self.patient_groups.append(patient_id)  # 🔴 СОХРАНЯЕМ ID ПАЦИЕНТА
                else:
                    print(f"    ⚠️ Пропущен файл (ошибка извлечения): {filename}")
        
        if len(X) == 0:
            print("❌ Не удалось извлечь признаки ни из одного файла!")
            return np.array([]), np.array([])
        
        df = pd.DataFrame(X)
        self.feature_columns = [c for c in df.columns if c not in ['filename', 'label', 'valid']]
        
        # 🔴 СТАТИСТИКА ПО ПАЦИЕНТАМ
        unique_patients = len(set(self.patient_groups))
        patients_both = len([pid for pid in set(self.patient_groups) if self.patient_groups.count(pid) > 1])
        patients_only_before = len([pid for pid in set(self.patient_groups) if self.patient_groups.count(pid) == 1 and y[self.patient_groups.index(pid)] == 1])
        
        print(f"✅ Всего обработано файлов: {len(X)}")
        print(f"   - Уникальных пациентов: {unique_patients}")
        print(f"   - Пациентов с парными записями (До и После): {patients_both}")
        print(f"   - Пациентов только с записью До: {patients_only_before}")
        print(f"   - Класс 1 (До лечения): {y.count(1)}")
        print(f"   - Класс 0 (После лечения): {y.count(0)}")
        
        return df.values, np.array(y)