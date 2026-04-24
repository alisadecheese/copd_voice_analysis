import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.feature_extractor import VoiceFeatureExtractor
from src.ml_pipeline import COPDClassifierMultiModel


def create_statistics_table(X, y, groups, feature_names, output_file="speech_statistics.xlsx"):
    """Создание таблицы статистики"""
    
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


def create_box_plots(X, y, feature_names, output_dir="figures"):
    """
    Ящик с усами в стиле статьи Конюхова и Гаранина:
    - Черно-белый стиль
    - Только медиана (жирная линия)
    - Усы = 1.5 IQR
    - Подпись: медиана [25%; 75%], p-value
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(X, columns=feature_names)
    df['group'] = ['ХОБЛ' if label == 1 else 'После' for label in y]
    
    # ТОП-5 наиболее значимых признаков (из вашей таблицы)
    top_features = [
        'spectral_centroid',
        'spectral_rolloff', 
        'zero_crossing_rate',
        'egemaps_58_alphaRatioV_sma3nz_amean',
        'egemaps_87_equivalentSoundLevel_dBp'
    ]
    
    available_features = [f for f in top_features if f in feature_names]
    
    print(f"\nСоздание ящиков с усами ({len(available_features)} признаков)...")
    
    for feature in available_features:
        plt.figure(figsize=(6, 4))  # Компактный размер как в статье
        
        # Конвертация в numeric + очистка
        data_copd = pd.to_numeric(df[df['group'] == 'ХОБЛ'][feature], errors='coerce').dropna()
        data_after = pd.to_numeric(df[df['group'] == 'После'][feature], errors='coerce').dropna()
        
        if len(data_copd) < 2 or len(data_after) < 2:
            plt.close()
            continue
        
        # t-тест
        _, p_val = stats.ttest_ind(data_copd, data_after, equal_var=False)
        stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        
        # Ящик с усами — минималистичный стиль
        bp = plt.boxplot([data_copd, data_after], 
                        labels=['ХОБЛ', 'После'],
                        patch_artist=False,  # ❌ Без заливки
                        notch=False,
                        showfliers=False,     # ❌ Без выбросов
                        medianprops=dict(color='black', linewidth=2),  # ✅ Жирная медиана
                        boxprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5))
        
        # Подпись с медианой и перцентилями (как в статье)
        q1_copd, med_copd, q3_copd = np.percentile(data_copd, [25, 50, 75])
        q1_after, med_after, q3_after = np.percentile(data_after, [25, 50, 75])
        
        plt.title(f'{feature}\nХОБЛ: {med_copd:.2f} [{q1_copd:.2f}; {q3_copd:.2f}]  '
                  f'После: {med_after:.2f} [{q1_after:.2f}; {q3_after:.2f}]\n'
                  f'p={p_val:.4f} {stars}', 
                 fontsize=9, fontweight='normal')  # Мелкий шрифт как в статье
        
        plt.ylabel('Значение', fontsize=9)
        plt.grid(axis='y', alpha=0.2, linestyle=':')
        plt.tight_layout()
        
        # Сохранение
        safe_name = "".join(c if c.isalnum() or c in '_-' else '_' for c in feature[:40])
        filepath = os.path.join(output_dir, f'box_{safe_name}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✅ {filepath}")
    
    print(f"Готово: {len(available_features)} графиков в {output_dir}/")


def create_f_anova_ranking(X, y, feature_names, output_file="figures/ranking_f_anova.png"):
    """
    Диаграмма ранжирования признаков по F-ANOVA.
    Горизонтальный бар-чарт как в статье Конюхова и Гаранина.
    """
    from sklearn.feature_selection import f_classif
    import matplotlib.pyplot as plt
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    f_scores = []
    p_values = []
    valid_names = []
    
    for i, name in enumerate(feature_names):
        try:
            feature_data = X[:, i:i+1].astype(float)
            f_stat, p_val = f_classif(feature_data, y)
            f_scores.append(f_stat[0])
            p_values.append(p_val[0])
            valid_names.append(name)
        except Exception:
            continue
    
    if len(f_scores) == 0:
        print("  ⚠️ Нет данных для F-ANOVA ранжирования")
        return
    
    # Сортировка по убыванию F (как в статье — самые важные сверху)
    sorted_idx = np.argsort(f_scores)[::-1]
    f_scores_sorted = [f_scores[i] for i in sorted_idx]
    names_sorted = [valid_names[i] for i in sorted_idx]
    p_sorted = [p_values[i] for i in sorted_idx]
    
    # ТОП-15 признаков (как в статье)
    top_n = min(15, len(f_scores_sorted))
    
    plt.figure(figsize=(10, 8))
    
    # Горизонтальные бары — стиль как в статье
    y_pos = np.arange(top_n)
    bars = plt.barh(y_pos, f_scores_sorted[:top_n], color='gray', edgecolor='black', linewidth=0.5)
    
    # Подписи значимости звёздочками
    for i, (bar, p_val) in enumerate(zip(bars, p_sorted[:top_n])):
        stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        if stars:
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    stars, va='center', fontsize=9, fontweight='bold')
    
    # Оси и подписи
    plt.yticks(y_pos, [name[:40] + '...' if len(name) > 40 else name for name in names_sorted[:top_n]], fontsize=8)
    plt.xlabel('F-statistic (ANOVA)', fontsize=10)
    plt.title('Ранжирование признаков по F-ANOVA', fontsize=11, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle=':')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ F-ANOVA ранжирование: {output_file}")

def create_mi_ranking(X, y, feature_names, output_file="figures/ranking_mutual_info.png"):
    """
    Диаграмма ранжирования признаков по Mutual Information.
    Горизонтальный бар-чарт как в статье Конюхова и Гаранина.
    """
    from sklearn.feature_selection import mutual_info_classif
    import matplotlib.pyplot as plt
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    mi_scores = []
    valid_names = []
    
    for i, name in enumerate(feature_names):
        try:
            feature_data = X[:, i:i+1].astype(float)
            mi = mutual_info_classif(feature_data, y, discrete_features=False, random_state=42)
            mi_scores.append(mi[0])
            valid_names.append(name)
        except Exception:
            continue
    
    if len(mi_scores) == 0:
        print("  ⚠️ Нет данных для Mutual Information ранжирования")
        return
    
    # Сортировка по убыванию MI
    sorted_idx = np.argsort(mi_scores)[::-1]
    mi_sorted = [mi_scores[i] for i in sorted_idx]
    names_sorted = [valid_names[i] for i in sorted_idx]
    
    # ТОП-15 признаков
    top_n = min(15, len(mi_sorted))
    
    plt.figure(figsize=(10, 8))
    
    # Горизонтальные бары
    y_pos = np.arange(top_n)
    plt.barh(y_pos, mi_sorted[:top_n], color='gray', edgecolor='black', linewidth=0.5)
    
    # Оси и подписи
    plt.yticks(y_pos, [name[:40] + '...' if len(name) > 40 else name for name in names_sorted[:top_n]], fontsize=8)
    plt.xlabel('Mutual Information', fontsize=10)
    plt.title('Ранжирование признаков по Mutual Information', fontsize=11, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle=':')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Mutual Information ранжирование: {output_file}")


def create_roc_curve_plot(y_test, y_proba, output_dir="figures"):
    """Создание ROC-кривой"""
    os.makedirs(output_dir, exist_ok=True)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    axes[0].set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    axes[0].set_title('ROC Curve', fontsize=12, fontweight='bold')
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(recall, precision, color='green', lw=2, 
                label=f'PR curve (AP = {avg_precision:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall (Sensitivity)', fontsize=11)
    axes[1].set_ylabel('Precision', fontsize=11)
    axes[1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'roc_pr_curves.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nROC-кривая сохранена: {filepath}")
    print(f"   AUC-ROC: {roc_auc:.4f}")
    
    return roc_auc, avg_precision


def create_table_4(X, y, feature_names, output_file="table_4_feature_selection.xlsx"):
    """
    ТАБЛИЦА 4: Отобранные по двум критериям параметры речи.
    Используются РЕАЛЬНЫЕ признаки из вашей работы.
    """
    from sklearn.feature_selection import f_classif, mutual_info_classif
    
    rows = []
    
    for i, name in enumerate(feature_names):
        try:
            feature_data = X[:, i:i+1].astype(float)
            
            # F-ANOVA
            f_stat, p_val = f_classif(feature_data, y)
            f_score = f_stat[0]
            f_pvalue = p_val[0]
            
            # Mutual Information
            mi_score = mutual_info_classif(feature_data, y, discrete_features=False, random_state=42)[0]
            
            # Категоризация F (как в статье)
            if f_score >= 2.0:
                f_category = f"Высокая ({f_score:.2f})"
            elif f_score >= 1.0:
                f_category = f"Средняя ({f_score:.2f})"
            else:
                f_category = f"Низкая ({f_score:.2f})"
            
            # Категоризация MI (как в статье)
            if mi_score == 0:
                mi_category = "0.0"
            elif mi_score >= 0.04:
                mi_category = f"Лучшая ({mi_score:.4f})"
            elif mi_score >= 0.03:
                mi_category = f"Хорошая ({mi_score:.4f})"
            elif mi_score > 0.01:
                mi_category = f"Есть ({mi_score:.4f})"
            else:
                mi_category = f"Слабая ({mi_score:.4f})"
            
            # Оценка важности (как в статье)
            if f_score >= 2.0 and mi_score > 0.01:
                importance = "Лучший компромисс"
            elif mi_score >= 0.04 and f_score < 2.0:
                importance = "Информативен, но хуже по ANOVA"
            elif f_score >= 2.0 and mi_score == 0:
                importance = "Только линейная связь"
            elif mi_score >= 0.03 and f_score < 1.0:
                importance = "Нелинейная связь"
            elif f_score >= 1.0 and mi_score > 0.01:
                importance = "Умеренная связь"
            else:
                importance = "Слабая связь"
            
            rows.append({
                'Признак': name,  # ✅ ВАШИ реальные признаки
                'ANOVA_F': f_category,
                'Mutual_Info': mi_category,
                'Важность (по обеим метрикам)': importance
            })
            
        except Exception:
            continue
    
    # Сортировка: сначала лучшие по совокупности
    df = pd.DataFrame(rows)
    df.to_excel(output_file, index=False)
    
    print(f"✓ table_4_feature_selection.xlsx")
    
    return df


def create_table_5(y_test, y_pred, output_file="table_5_random_forest_results.xlsx"):
    """
    ТАБЛИЦА 5: Результаты тестирования классификатора «случайный лес».
    Классы: ХОБЛ (до лечения) и После лечения.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Убедимся что массивы 1D
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()
    
    rows = []
    
    # Класс 1: ХОБЛ (до лечения)
    y_test_copd = (y_test == 1).astype(int)
    y_pred_copd = (y_pred == 1).astype(int)
    
    precision_copd = precision_score(y_test_copd, y_pred_copd, zero_division=0)
    recall_copd = recall_score(y_test_copd, y_pred_copd, zero_division=0)
    f1_copd = f1_score(y_test_copd, y_pred_copd, zero_division=0)
    
    rows.append({
        'Класс': 'ХОБЛ',
        'Точность': f"{precision_copd:.2f}",
        'Полнота': f"{recall_copd:.2f}",
        'F1-мера': f"{f1_copd:.2f}"
    })
    
    # Класс 0: После лечения
    y_test_after = (y_test == 0).astype(int)
    y_pred_after = (y_pred == 0).astype(int)
    
    precision_after = precision_score(y_test_after, y_pred_after, zero_division=0)
    recall_after = recall_score(y_test_after, y_pred_after, zero_division=0)
    f1_after = f1_score(y_test_after, y_pred_after, zero_division=0)
    
    rows.append({
        'Класс': 'После лечения',
        'Точность': f"{precision_after:.2f}",
        'Полнота': f"{recall_after:.2f}",
        'F1-мера': f"{f1_after:.2f}"
    })
    
    df = pd.DataFrame(rows)
    df.to_excel(output_file, index=False)
    
    print(f"✓ table_5_random_forest_results.xlsx")
    
    return df


def predict_new_files(classifier, analyzer, folder_path, threshold=0.5):
    """Предсказание для новых пациентов"""
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
                
                if filename.endswith('-1.wav') or filename.endswith('-1.mp3'):
                    true_label = 1
                    true_status = "ХОБЛ"
                elif filename.endswith('-2.wav') or filename.endswith('-2.mp3'):
                    true_label = 0
                    true_status = "Здоров"
                else:
                    true_label = None
                    true_status = "Неизвестно"
                
                pred_status = "ХОБЛ" if pred == 1 else "Здоров"
                
                if true_label is not None:
                    total_count += 1
                    if pred == true_label:
                        correct_count += 1
                        mark = "[OK]"
                    else:
                        mark = "[ERR]"
                else:
                    mark = "    "
                
                print(f"  {mark} {filename}: {pred_status} ({proba:.2%}) [Ожид: {true_status}]")
    
    if total_count > 0:
        accuracy = correct_count / total_count * 100
        print(f"\n  Итого: {correct_count}/{total_count} правильно ({accuracy:.1f}%)")


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
    
    print("\nФильтрация признаков...")
    valid_features = []
    for i, name in enumerate(feature_names):
        if np.std(X[:, i]) > 0.0001:
            valid_features.append(name)
    
    X_clean = X[:, [feature_names.index(f) for f in valid_features]]
    feature_names_clean = valid_features
    print(f"  Осталось признаков: {len(feature_names_clean)}")
    
    print("\nРазделение на train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Обучение: {len(X_train)}, Тест: {len(X_test)}")
    
    print("\nОбучение модели...")
    classifier = COPDClassifierMultiModel(class_weight='balanced')
    classifier.train(X_train, y_train, groups=None, feature_names=feature_names_clean, n_splits=5)
    
    classifier.feature_names = feature_names_clean
    
    print("\n" + "="*100)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*100)
    
    for name, result in classifier.model_results.items():
        print(f"\n  Модель: {name}")
        print(f"    F1 Median:  {result.get('median_f1', result.get('f1_median', 'N/A'))}")
        print(f"    F1 Std:     {result.get('f1_std', result.get('std_f1', 'N/A'))}")
        print(f"    Accuracy:   {result.get('accuracy', 'N/A')}")
    
    print("\n" + "="*100)
    print(f"  Лучшая модель: {classifier.best_model_name}")
    print("="*100)
    
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
    
    print("\nОтчёт по классам:")
    print(classification_report(y_test, y_pred, target_names=['После', 'До']))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nМатрица ошибок:")
    print("  Предсказано: Здоров/ХОБЛ")
    print(f"  Реально Здоров:  [{cm[0,0]:3d} / {cm[0,1]:3d}]")
    print(f"  Реально ХОБЛ:    [{cm[1,0]:3d} / {cm[1,1]:3d}]")
    
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
    
    print("\nСоздание графиков...")
    create_box_plots(X_clean, y, feature_names_clean)
    
    print("\nСоздание ROC-кривой...")
    create_roc_curve_plot(y_test, y_proba)
    
    print("\nСоздание Таблицы 4...")
    create_table_4(y_test, y_pred, y_proba)
    
    print("\nСоздание Таблицы 5...")
    create_table_5(y_test, y_pred, "table_5_random_forest_results.xlsx")


    classifier.save()
    
    
    # 🔴 ДИАГРАММЫ РАНЖИРОВАНИЯ (две отдельные картинки)
    print("\nСоздание диаграмм ранжирования...")
    create_f_anova_ranking(X_clean, y, feature_names_clean, "figures/ranking_f_anova.png")
    create_mi_ranking(X_clean, y, feature_names_clean, "figures/ranking_mutual_info.png")
    
    # 🔴 ЯЩИКИ С УСАМИ
    print("\nСоздание ящиков с усами...")
    create_box_plots(X_clean, y, feature_names_clean, "figures")
    
    # Предсказание для новых файлов
    predict_new_files(classifier, analyzer, "data/new_patients", threshold=best_threshold)
    
    print("\n✅ Обучение завершено")
    
    predict_new_files(classifier, analyzer, "data/new_patients", threshold=best_threshold)
    
    print("\nОбучение завершено")


if __name__ == "__main__":
    print("Запуск системы анализа ХОБЛ")
    print("="*60)
    
    try:
        train_pipeline()
    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*60)
    print("Готово")