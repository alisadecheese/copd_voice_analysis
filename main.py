# main.py

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats

# Добавляем папку src в путь поиска
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.feature_extractor import VoiceFeatureExtractor
from src.ml_pipeline import COPDClassifierMultiModel


# ==============================================================================
# ФУНКЦИЯ 1: Проверка на дубликаты (чтобы не тестировать на обученных)
# ==============================================================================
def check_for_duplicates(new_folder, train_folders):
    """Проверяет, нет ли файлов из new_folder в обучающих папках"""
    print("\n" + "="*50)
    print("🔍 ПРОВЕРКА НА ДУБЛИКАТЫ")
    print("="*50)
    
    if not os.path.exists(new_folder):
        print(f"  ⚠️ Папка {new_folder} не найдена.")
        return True
    
    new_files = set(os.listdir(new_folder))
    duplicates = []
    
    for train_folder in train_folders:
        if os.path.exists(train_folder):
            train_files = set(os.listdir(train_folder))
            common = new_files.intersection(train_files)
            if common:
                duplicates.extend(common)
                print(f"  ⚠️ Найдены дубликаты в {train_folder}:")
                for f in common:
                    print(f"     - {f}")
    
    if not duplicates:
        print("  ✅ Дубликатов не найдено. Все пациенты новые.")
    
    return len(duplicates) == 0


# ==============================================================================
# ФУНКЦИЯ 2: Создание таблицы статистики (Excel)
# ==============================================================================
def create_statistics_table(X, y, groups, feature_names, output_file="speech_statistics.xlsx"):
    """Создает Excel-таблицу со статистикой признаков"""
    
    # Проверка и исправление количества колонок
    n_features_data = X.shape[1]
    n_features_names = len(feature_names)
    
    if n_features_data != n_features_names:
        print(f"⚠️ Предупреждение: Несовпадение количества признаков!")
        print(f"   В данных столбцов: {n_features_data}")
        print(f"   В списке имен: {n_features_names}")
        print(f"   ✅ Автоматическая корректировка имен...")
        
        if n_features_names < n_features_data:
            missing = n_features_data - n_features_names
            for i in range(missing):
                feature_names.append(f"feature_{n_features_names + i}")
        elif n_features_names > n_features_data:
            feature_names = feature_names[:n_features_data]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['group'] = ['До лечения' if label == 1 else 'После лечения' for label in y]
    df['patient_id'] = groups
    
    statistics = []
    
    for feature in feature_names:
        group_before = df[df['group'] == 'До лечения'][feature]
        group_after = df[df['group'] == 'После лечения'][feature]
        
        # Пациенты с парными записями
        patients_count = df.groupby('patient_id').size()
        patients_both_ids = patients_count[patients_count == 2].index.tolist()
        
        patients_both = df[df['patient_id'].isin(patients_both_ids)]
        before_paired = patients_both[patients_both['group'] == 'До лечения'][feature].values
        after_paired = patients_both[patients_both['group'] == 'После лечения'][feature].values
        
        # Статистика "До"
        median_before = group_before.median()
        mean_before = group_before.mean()
        std_before = group_before.std()
        n_before = len(group_before)
        
        if n_before > 1:
            ci_before = stats.t.interval(0.95, n_before-1, loc=median_before, scale=stats.sem(group_before))
        else:
            ci_before = (np.nan, np.nan)
        
        # Статистика "После"
        median_after = group_after.median()
        mean_after = group_after.mean()
        std_after = group_after.std()
        n_after = len(group_after)
        
        if n_after > 1:
            ci_after = stats.t.interval(0.95, n_after-1, loc=median_after, scale=stats.sem(group_after))
        else:
            ci_after = (np.nan, np.nan)
        
        # Парный t-тест
        if len(before_paired) == len(after_paired) and len(before_paired) > 1:
            try:
                _, p_value_paired = stats.ttest_rel(before_paired, after_paired)
            except:
                p_value_paired = float('nan')
        else:
            p_value_paired = float('nan')
        
        # Независимый t-тест
        if n_before > 1 and n_after > 1:
            try:
                _, p_value_indep = stats.ttest_ind(group_before, group_after)
            except:
                p_value_indep = float('nan')
        else:
            p_value_indep = float('nan')
        
        # Эффект Коэна d
        if n_before > 1 and n_after > 1 and std_before > 0 and std_after > 0:
            pooled_std = np.sqrt(((n_before - 1) * std_before**2 + (n_after - 1) * std_after**2) / (n_before + n_after - 2))
            if pooled_std > 0:
                cohens_d = (mean_before - mean_after) / pooled_std
            else:
                cohens_d = 0.0
        else:
            cohens_d = float('nan')
        
        statistics.append({
            'Признак': feature,
            'До лечения (n=110) - Медиана': median_before,
            'До лечения (n=110) - Среднее': mean_before,
            'До лечения (n=110) - Стд. отклонение': std_before,
            'До лечения (n=110) - 95% ДИ': f"[{ci_before[0]:.3f}, {ci_before[1]:.3f}]" if not np.isnan(ci_before[0]) else "N/A",
            'После лечения (n=60) - Медиана': median_after,
            'После лечения (n=60) - Среднее': mean_after,
            'После лечения (n=60) - Стд. отклонение': std_after,
            'После лечения (n=60) - 95% ДИ': f"[{ci_after[0]:.3f}, {ci_after[1]:.3f}]" if not np.isnan(ci_after[0]) else "N/A",
            'p-value (парный t-тест, n=60)': p_value_paired,
            'p-value (независимый t-тест, n=110/60)': p_value_indep,
            'Значимо (парный)': 'Да' if p_value_paired < 0.05 else 'Нет',
            'Значимо (независимый)': 'Да' if p_value_indep < 0.05 else 'Нет',
            'Размер эффекта (Cohen d)': cohens_d
        })
    
    stats_df = pd.DataFrame(statistics)
    
    try:
        stats_df.to_excel(output_file, index=False, float_format="%.4f")
        print(f"💾 Таблица статистики сохранена в {output_file}")
    except Exception as e:
        csv_file = output_file.replace('.xlsx', '.csv')
        stats_df.to_csv(csv_file, index=False, sep=';', decimal=',')
        print(f"⚠️ Не удалось сохранить в Excel. Сохранено в {csv_file}")
        print(f"   Ошибка: {e}")
    
    print("\n" + "="*150)
    print("📊 ТАБЛИЦА СТАТИСТИКИ ПРИЗНАКОВ")
    print("="*150)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(stats_df.to_string(index=False))
    print("="*150)
    
    return stats_df


# ==============================================================================
# ФУНКЦИЯ 3: Предсказание для новых файлов
# ==============================================================================
def predict_new_files(classifier, analyzer, folder_path):
    """Предсказание для новых файлов (которые не участвовали в обучении)"""
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Папка {folder_path} не найдена. Пропуск тестирования.")
        return
    
    print("\n" + "="*50)
    print("🧪 ПРЕДСКАЗАНИЕ ДЛЯ НОВЫХ ПАЦИЕНТОВ")
    print("="*50)
    
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in audio_extensions]
    
    if not files:
        print(f"⚠️ В папке {folder_path} не найдено аудиофайлов.")
        return
    
    for filename in sorted(files):
        file_path = os.path.join(folder_path, filename)
        sound = analyzer.load_audio(file_path)
        
        if sound:
            feats = analyzer.extract_features(sound)
            if feats and feats.get("valid"):
                for k, v in feats.items():
                    if isinstance(v, float) and np.isnan(v):
                        feats[k] = 0.0
                
                X_new = [[feats.get(col, 0.0) for col in classifier.feature_names]]
                
                pred = classifier.predict(X_new, use_best=True)[0]
                proba = classifier.predict_proba(X_new, use_best=True)[0][1]
                
                status = "🔴 ХОБЛ (До лечения)" if pred == 1 else "🟢 Здоров (После лечения)"
                print(f"  {filename}: {status} (вероятность: {proba:.2%})")
            else:
                print(f"  {filename}: ❌ Ошибка извлечения признаков")
        else:
            print(f"  {filename}: ❌ Ошибка загрузки файла")


# ==============================================================================
# ГЛАВНАЯ ПРОГРАММА
# ==============================================================================
if __name__ == "__main__":
    print("🚀 Запуск системы анализа ХОБЛ (Multi-Model)...")
    
    analyzer = VoiceFeatureExtractor()
    DATA_DIR = "data"
    
    print("📂 Подготовка датасета...")
    X, y = analyzer.prepare_dataset(DATA_DIR)
    
    if len(X) > 0:
        # 🔴 ПОЛУЧАЕМ ID ПАЦИЕНТОВ
        groups = analyzer.patient_groups
        
        print("\n" + "="*50)
        classifier = COPDClassifierMultiModel()
        
        # 🔴 ПЕРЕДАЕМ groups В train()
        classifier.train(X, y, groups=groups, feature_names=analyzer.feature_columns, n_splits=5)
        
        # 3. Сравнение моделей
        print("\n" + "="*50)
        print("📊 СРАВНЕНИЕ МОДЕЛЕЙ")
        print("="*50)
        comparison_df = classifier.get_model_comparison_table()
        print(comparison_df.to_string(index=False))
        
        # 4. Важность признаков
        print("\n" + "="*50)
        print(f"🔬 ВАЖНОСТЬ ПРИЗНАКОВ ({classifier.best_model_name})")
        print("="*50)
        importance = classifier.get_feature_importance()
        if importance:
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for feat, imp in sorted_imp[:5]:
                print(f"  {feat}: {imp:.4f}")
        
        # 5. Сохранение модели
        classifier.save()
        
        # 6. Тест предсказания на обучающих данных (только для отладки)
        print("\n" + "="*50)
        print("🧪 ТЕСТ ПРЕДСКАЗАНИЯ (на обучающих данных)")
        print("="*50)
        if len(X) > 0:
            pred = classifier.predict(X[:1], use_best=True)[0]
            proba = classifier.predict_proba(X[:1], use_best=True)[0][1]
            print(f"  Первый файл: Предсказание={pred[0] if hasattr(pred, '__len__') else pred}, Вероятность={proba[0][1] if hasattr(proba, '__len__') else proba:.2%}")
            print("  ⚠️ Внимание: Это предсказание на обучающих данных (может быть завышено)")
        
        # 7. Создание таблицы статистики
        print("\n" + "="*50)
        print("📈 СОЗДАНИЕ ТАБЛИЦЫ СТАТИСТИКИ")
        print("="*50)
        create_statistics_table(X, y, groups, analyzer.feature_columns, output_file="speech_statistics.xlsx")
        
        # 8. Проверка на дубликаты и тест на новых файлах
        is_clean = check_for_duplicates("data/new_patients", ["data/copd_before", "data/copd_after"])
        if is_clean:
            predict_new_files(classifier, analyzer, "data/new_patients")
        
        print("\n✅ Готово!")
    else:
        print("❌ Нет данных для обучения.")