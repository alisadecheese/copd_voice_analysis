import sys
import os
import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.feature_extractor import VoiceFeatureExtractor
from src.ml_pipeline import COPDClassifierMultiModel


def check_group_differences(X, y, feature_names):
    """Проверка: различаются ли группы статистически"""
    print("\n" + "="*60)
    print("🔍 ПРОВЕРКА РАЗЛИЧИЙ МЕЖДУ ГРУППАМИ")
    print("="*60)
    
    # 🔴 ПРОВЕРКА И ИСПРАВЛЕНИЕ КОЛИЧЕСТВА КОЛОНОК
    n_features_data = X.shape[1]
    n_features_names = len(feature_names)
    
    if n_features_names != n_features_data:
        print(f"⚠️ Корректировка количества признаков: {n_features_names} → {n_features_data}")
        if n_features_names < n_features_data:
            for i in range(n_features_data - n_features_names):
                feature_names.append(f"feature_{n_features_names + i}")
        elif n_features_names > n_features_data:
            feature_names = feature_names[:n_features_data]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['group'] = y
    
    group_0 = df[df['group'] == 0]
    group_1 = df[df['group'] == 1]
    
    significant_features = []
    
    for feat in feature_names:
        g0 = group_0[feat].values
        g1 = group_1[feat].values
        
        if len(g0) > 1 and len(g1) > 1:
            try:
                _, p_value = stats.ttest_ind(g0, g1)
                if p_value < 0.05:
                    significant_features.append((feat, p_value))
            except:
                pass
    
    print(f"Всего признаков: {len(feature_names)}")
    print(f"Значимых различий (p < 0.05): {len(significant_features)}")
    
    if significant_features:
        print("\nТоп-5 значимых признаков:")
        for feat, p in sorted(significant_features, key=lambda x: x[1])[:5]:
            print(f"  {feat}: p = {p:.4f}")
    else:
        print("\n⚠️ НЕТ СТАТИСТИЧЕСКИ ЗНАЧИМЫХ РАЗЛИЧИЙ!")
        print("   Это может означать, что группы акустически неразличимы.")
    
    print("="*60)
    
    return significant_features


def create_statistics_table(X, y, groups, feature_names, output_file="speech_statistics.xlsx"):
    """Создание таблицы статистики"""
    
    df = pd.DataFrame(X, columns=feature_names)
    df['group'] = ['До лечения' if label == 1 else 'После лечения' for label in y]
    df['patient_id'] = groups

    statistics = []
    n_before = len(df[df['group'] == 'До лечения'])
    n_after = len(df[df['group'] == 'После лечения'])

    for feature in feature_names:
        group_before = df[df['group'] == 'До лечения'][feature]
        group_after = df[df['group'] == 'После лечения'][feature]

        median_before = group_before.median()
        if len(group_before) > 1:
            ci_before = stats.t.interval(0.95, len(group_before)-1, loc=median_before, scale=stats.sem(group_before))
            ci_before_str = f"[{ci_before[0]:.2f}; {ci_before[1]:.2f}]"
        else:
            ci_before_str = "[N/A; N/A]"

        median_after = group_after.median()
        if len(group_after) > 1:
            ci_after = stats.t.interval(0.95, len(group_after)-1, loc=median_after, scale=stats.sem(group_after))
            ci_after_str = f"[{ci_after[0]:.2f}; {ci_after[1]:.2f}]"
        else:
            ci_after_str = "[N/A; N/A]"

        if len(group_before) > 1 and len(group_after) > 1:
            try:
                _, p_value = stats.ttest_ind(group_before, group_after)
            except:
                p_value = float('nan')
        else:
            p_value = float('nan')

        value_before = f"{median_before:.2f} {ci_before_str}"
        value_after = f"{median_after:.2f} {ci_after_str}"

        statistics.append({
            'Показатель': feature,
            f'1 (n={n_before})': value_before,
            f'2 (n={n_after})': value_after,
            'p-value': f"{p_value:.3f}" if not np.isnan(p_value) else "N/A"
        })

    stats_df = pd.DataFrame(statistics)

    try:
        stats_df.to_excel(output_file, index=False)
        print(f"💾 Таблица сохранена в {output_file}")
    except Exception as e:
        csv_file = output_file.replace('.xlsx', '.csv')
        stats_df.to_csv(csv_file, index=False, sep=';', decimal=',')
        print(f"⚠️ Сохранено в {csv_file}")

    print("\n" + "="*100)
    print("📊 ТАБЛИЦА СТАТИСТИКИ")
    print("="*100)
    print(stats_df.to_string(index=False))
    print("="*100)

    return stats_df


def predict_new_files(classifier, analyzer, folder_path):
    """Предсказание для новых пациентов"""
    if not os.path.exists(folder_path):
        print(f"⚠️ Папка {folder_path} не найдена.")
        return

    print("\n" + "="*50)
    print("🧪 ПРЕДСКАЗАНИЕ ДЛЯ НОВЫХ ПАЦИЕНТОВ")
    print("="*50)

    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in audio_extensions]

    if not files:
        print(f"⚠️ В папке не найдено аудиофайлов.")
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


if __name__ == "__main__":
    print("🚀 Запуск системы анализа ХОБЛ (Multi-Model)...")
    print("   На основе обзора литературы [10-13]")
    print("="*60)

    analyzer = VoiceFeatureExtractor()
    DATA_DIR = "data"

    print("📂 Подготовка датасета...")
    X, y = analyzer.prepare_dataset(DATA_DIR)

    if len(X) > 0:
        groups = analyzer.patient_groups

        # 🔍 ПРОВЕРКА РАЗЛИЧИЙ МЕЖДУ ГРУППАМИ
        significant = check_group_differences(X, y, analyzer.feature_columns)

        # Фильтрация признаков
        print("\n🔧 ФИЛЬТРАЦИЯ ПРИЗНАКОВ")
        print("="*60)

        non_zero_var = []
        for i, col in enumerate(analyzer.feature_columns):
            col_data = X[:, i]
            unique_vals = len(np.unique(col_data))
            variance = np.var(col_data)

            # Оставляем если > 3 уникальных значений и дисперсия > 0.001
            if unique_vals > 3 and variance > 0.001 and np.sum(col_data != 0) > len(col_data) * 0.3:
                non_zero_var.append(i)
            else:
                print(f"  ⚠️ Исключён: {col} (уникальных={unique_vals}, дисперсия={variance:.6f})")

        X_filtered = X[:, non_zero_var]
        feature_names_filtered = [analyzer.feature_columns[i] for i in non_zero_var]

        print(f"\n✅ Осталось признаков: {len(feature_names_filtered)} из {len(analyzer.feature_columns)}")
        print("="*60)

        print("\n" + "="*50)
        classifier = COPDClassifierMultiModel()

        # Обучение (groups=None т.к. мало парных записей)
        classifier.train(X_filtered, y, groups=None, feature_names=feature_names_filtered, n_splits=5)

        # Сравнение моделей
        print("\n" + "="*50)
        print("📊 СРАВНЕНИЕ МОДЕЛЕЙ")
        print("="*50)
        comparison_df = classifier.get_model_comparison_table()
        print(comparison_df.to_string(index=False))

        # Важность признаков
        print("\n" + "="*50)
        print(f"🔬 ВАЖНОСТЬ ПРИЗНАКОВ ({classifier.best_model_name})")
        print("="*50)
        importance = classifier.get_feature_importance()
        if importance:
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            print("Топ-10 признаков:")
            for feat, imp in sorted_imp[:10]:
                print(f"  {feat}: {imp:.4f}")

        # Сохранение
        classifier.save()

        # Тест предсказания
        print("\n" + "="*50)
        print("🧪 ТЕСТ ПРЕДСКАЗАНИЯ")
        print("="*50)
        if len(X_filtered) > 0:
            pred = classifier.predict(X_filtered[:1], use_best=True)
            proba = classifier.predict_proba(X_filtered[:1], use_best=True)
            pred_val = pred[0] if hasattr(pred, '__len__') else pred
            proba_val = proba[0][1] if hasattr(proba, '__len__') else proba
            print(f"  Предсказание: {pred_val}")
            print(f"  Вероятность ХОБЛ: {proba_val:.2%}")

        # Таблица статистики
        print("\n" + "="*50)
        print("📈 ТАБЛИЦА СТАТИСТИКИ")
        print("="*50)
        create_statistics_table(X_filtered, y, groups, feature_names_filtered, output_file="speech_statistics.xlsx")

        # Предсказание для новых файлов
        predict_new_files(classifier, analyzer, "data/new_patients")

        print("\n✅ Готово!")
    else:
        print("❌ Нет данных для обучения.")