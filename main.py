import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.feature_extractor import VoiceFeatureExtractor
from src.ml_pipeline import COPDClassifierMultiModel


def create_statistics_table(X, y, groups, feature_names, output_file="speech_statistics.xlsx"):
    """
    Создание таблицы статистики.
    
    Как рассчитывается:
    -------------------
    1. Для каждого признака вычисляются медианы в двух группах (До/После)
    2. 95% доверительный интервал (CI) через t-распределение:
       CI = median ± t(0.95, n-1) * SEM, где SEM = std / sqrt(n)
    3. p-value через двусторонний t-тест Стьюдента (непарный, с поправкой Велча)
    4. Если p < 0.05 — различие статистически значимо (*)
    """
    
    n_features_data = X.shape[1]
    if len(feature_names) < n_features_data:
        feature_names += [f"feature_{len(feature_names)+i}" for i in range(n_features_data - len(feature_names))]
    elif len(feature_names) > n_features_data:
        feature_names = feature_names[:n_features_data]

    df = pd.DataFrame(X, columns=feature_names)
    df['group'] = ['До лечения' if label == 1 else 'После лечения' for label in y]
    
    if len(groups) == len(df):
        df['patient_id'] = groups
    else:
        df['patient_id'] = [f"patient_{i}" for i in range(len(df))]

    statistics = []
    n_before = len(df[df['group'] == 'До лечения'])
    n_after = len(df[df['group'] == 'После лечения'])

    for feature in feature_names:
        g0 = df[df['group'] == 'До лечения'][feature]
        g1 = df[df['group'] == 'После лечения'][feature]
        
        med0, med1 = g0.median(), g1.median()
        
        # 95% доверительный интервал
        try:
            if len(g0) > 1 and g0.std() > 0.0001:
                ci0_low, ci0_high = stats.t.interval(0.95, len(g0)-1, loc=med0, scale=stats.sem(g0))
                ci0 = f"[{ci0_low:.2f}; {ci0_high:.2f}]"
            else:
                ci0 = "[N/A; N/A]"
        except Exception:
            ci0 = "[N/A; N/A]"
        
        try:
            if len(g1) > 1 and g1.std() > 0.0001:
                ci1_low, ci1_high = stats.t.interval(0.95, len(g1)-1, loc=med1, scale=stats.sem(g1))
                ci1 = f"[{ci1_low:.2f}; {ci1_high:.2f}]"
            else:
                ci1 = "[N/A; N/A]"
        except Exception:
            ci1 = "[N/A; N/A]"

        # p-value (t-тест)
        p_str = "N/A"
        try:
            if len(g0) > 1 and len(g1) > 1:
                g0_clean = g0.replace([np.inf, -np.inf], np.nan).dropna()
                g1_clean = g1.replace([np.inf, -np.inf], np.nan).dropna()
                if len(g0_clean) > 1 and len(g1_clean) > 1:
                    _, p_val = stats.ttest_ind(g0_clean, g1_clean, equal_var=False)
                    p_str = f"{p_val:.4f}"
        except Exception:
            p_str = "N/A"

        statistics.append({
            'Показатель': feature,
            f'1 (n={n_before})': f"{med0:.2f} {ci0}",
            f'2 (n={n_after})': f"{med1:.2f} {ci1}",
            'p-value': p_str
        })

    stats_df = pd.DataFrame(statistics)
    stats_df.to_excel(output_file, index=False)
    
    print("\n" + "="*100)
    print("ТАБЛИЦА СТАТИСТИКИ")
    print("="*100)
    print(stats_df.to_string(index=False))
    print("="*100)
    
    return stats_df


def predict_new_files(classifier, analyzer, folder_path, threshold=0.5):
    """
    Предсказание для НОВЫХ пациентов (не из обучающей выборки).
    
    Класс определяется по ПОСЛЕДНЕЙ цифре в имени файла:
    - окончание `-1.wav` = ХОБЛ (До лечения)
    - окончание `-2.wav` = Здоров (После лечения)
    """
    if not os.path.exists(folder_path):
        print(f"  Папка {folder_path} не найдена")
        return
    
    print("\nПРЕДСКАЗАНИЕ ДЛЯ НОВЫХ ПАЦИЕНТОВ")
    print(f"  Порог классификации: {threshold:.2f}")
    print()
    
    correct_count = 0
    total_count = 0
    
    for filename in sorted([f for f in os.listdir(folder_path) if f.endswith('.wav')]):
        file_path = os.path.join(folder_path, filename)
        sound, audio_values, sr = analyzer.load_audio(file_path)
        
        if sound:
            feats = analyzer.extract_features(sound, audio_values, sr, file_path)
            
            if feats and feats.get("valid"):
                for k, v in feats.items():
                    if isinstance(v, float) and np.isnan(v):
                        feats[k] = 0.0
                
                X_new = []
                for col in classifier.feature_names:
                    if col in feats:
                        X_new.append(feats[col])
                    else:
                        X_new.append(0.0)
                
                X_new_array = np.array([X_new])
                proba = classifier.predict_proba(X_new_array)[0][1]
                pred = 1 if proba >= threshold else 0
                
                # Определение класса по последней цифре
                if filename.endswith('-1.wav') or filename.endswith('-1.mp3'):
                    true_label = 1  # ХОБЛ (До лечения)
                    true_status = "ХОБЛ"
                elif filename.endswith('-2.wav') or filename.endswith('-2.mp3'):
                    true_label = 0  # Здоров (После лечения)
                    true_status = "Здоров"
                else:
                    true_label = None
                    true_status = "Неизвестно"
                
                pred_status = "ХОБЛ" if pred == 1 else "Здоров"
                
                # Оценка правильности
                if true_label is not None:
                    total_count += 1
                    if pred == true_label:
                        correct_count += 1
                        mark = "✅"
                    else:
                        mark = "❌"
                else:
                    mark = "  "
                
                print(f"  {mark} {filename}: {pred_status} ({proba:.2%}) [Ожид: {true_status}]")
    
    if total_count > 0:
        accuracy = correct_count / total_count * 100
        print(f"\n  Итого: {correct_count}/{total_count} правильно ({accuracy:.1f}%)")
        
        if accuracy >= 75:
            print("  ✅ Отличный результат!")
        elif accuracy >= 60:
            print("  ⚠️ Удовлетворительный результат")
        else:
            print("  ❌ Требуется улучшение модели")


def train_pipeline():
    """Основной пайплайн обучения"""
    
    print("Загрузка данных...")
    analyzer = VoiceFeatureExtractor()
    X, y = analyzer.prepare_dataset("data")
    
    if len(X) == 0:
        print("Нет данных для обучения")
        return
    
    feature_names = analyzer.feature_columns
    groups = analyzer.patient_groups
    print(f"  Образцов: {len(X)}, Признаков: {len(feature_names)}")
    print(f"  Класс 0 (После): {np.sum(y==0)}, Класс 1 (До): {np.sum(y==1)}")
    
    # Фильтрация признаков
    print("\nФильтрация признаков...")
    valid_features = []
    for i, name in enumerate(feature_names):
        if np.std(X[:, i]) > 0.0001:
            valid_features.append(name)
    
    X_clean = X[:, [feature_names.index(f) for f in valid_features]]
    feature_names_clean = valid_features
    print(f"  Осталось признаков: {len(feature_names_clean)}")
    
    # Разделение на train/test
    print("\nРазделение на train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Обучение: {len(X_train)}, Тест: {len(X_test)}")
    
    # Обучение модели
    print("\nОбучение модели...")
    from src.ml_pipeline import COPDClassifierMultiModel
    
    classifier = COPDClassifierMultiModel(class_weight='balanced')
    classifier.train(X_train, y_train, groups=None, feature_names=feature_names_clean, n_splits=5)
    
    classifier.feature_names = feature_names_clean
    
    # 🔴 ТАБЛИЦА СРАВНЕНИЯ МОДЕЛЕЙ (универсальная)
    print("\n" + "="*100)
    print("СРАВНЕНИЕ МОДЕЛЕЙ (кросс-валидация, 5 сплитов)")
    print("="*100)
    
    for name, result in classifier.model_results.items():
        print(f"\n  Модель: {name}")
        print(f"    F1 Median:  {result.get('median_f1', result.get('f1_median', 'N/A'))}")
        print(f"    F1 Std:     {result.get('f1_std', result.get('std_f1', 'N/A'))}")
        print(f"    CI 95%:     [{result.get('ci_lower', 'N/A')}; {result.get('ci_upper', 'N/A')}]")
        print(f"    Accuracy:   {result.get('accuracy', 'N/A')}")
    
    print("\n" + "="*100)
    print(f"  Лучшая модель: {classifier.best_model_name}")
    print("="*100)
    
    # Подбор порога
    print("\nПодбор оптимального порога...")
    y_proba = classifier.predict_proba(X_test)[:, 1]
    
    best_threshold = 0.5
    best_f1 = 0
    for thresh in np.arange(0.30, 0.70, 0.01):
        y_pred_thresh = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred_thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"  Оптимальный порог: {best_threshold:.2f}")
    print(f"  F1 при этом пороге: {best_f1:.4f}")
    
    y_pred = (y_proba >= best_threshold).astype(int)
    
    print(f"\n  F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    copd_pred = np.sum(y_pred == 1)
    healthy_pred = np.sum(y_pred == 0)
    print(f"\n  Предсказано ХОБЛ: {copd_pred} ({copd_pred/len(y_pred)*100:.1f}%)")
    print(f"  Предсказано Здоров: {healthy_pred} ({healthy_pred/len(y_pred)*100:.1f}%)")
    
    print("\nОтчёт по классам:")
    print(classification_report(y_test, y_pred, target_names=['После', 'До']))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nМатрица ошибок:")
    print("  Предсказано: Здоров/ХОБЛ")
    print(f"  Реально Здоров:  [{cm[0,0]:3d} / {cm[0,1]:3d}]")
    print(f"  Реально ХОБЛ:    [{cm[1,0]:3d} / {cm[1,1]:3d}]")
    
    # Статистика по признакам
    print("\nТоп-5 значимых признаков:")
    stats_data = []
    for i, name in enumerate(feature_names_clean):
        group1 = X_clean[y == 1, i]
        group2 = X_clean[y == 0, i]
        
        if len(group1) > 1 and len(group2) > 1:
            try:
                _, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                stats_data.append({
                    'feature': name,
                    'p_value': p_val,
                    'mean_before': np.mean(group1),
                    'mean_after': np.mean(group2)
                })
            except Exception:
                pass
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        stats_df = stats_df.sort_values('p_value')
        
        for idx, row in stats_df.head(5).iterrows():
            stars = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
            print(f"  {row['feature']}: p={row['p_value']:.4f} {stars}")
    
    print("\nВажность признаков (RandomForest):")
    imp = classifier.get_feature_importance()
    if imp:
        for feat, val in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {feat}: {val:.4f}")
    
    print("\nСоздание таблицы статистики...")
    create_statistics_table(X_clean, y, groups, feature_names_clean)
    
    classifier.save()
    
    # Предсказание для новых файлов
    predict_new_files(classifier, analyzer, "data/new_patients", threshold=best_threshold)
    
    print("\nОбучение завершено")


if __name__ == "__main__":
    print("Запуск системы анализа ХОБЛ")
    print("="*60)
    
    train_pipeline()
    
    print("="*60)
    print("Готово")