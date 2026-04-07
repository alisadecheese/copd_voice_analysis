import sys, os, pandas as pd, numpy as np
from scipy import stats
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        ci0 = f"[{stats.t.interval(0.95, len(g0)-1, loc=med0, scale=stats.sem(g0))[0]:.2f}; {stats.t.interval(0.95, len(g0)-1, loc=med0, scale=stats.sem(g0))[1]:.2f}]" if len(g0) > 1 else "[N/A; N/A]"
        ci1 = f"[{stats.t.interval(0.95, len(g1)-1, loc=med1, scale=stats.sem(g1))[0]:.2f}; {stats.t.interval(0.95, len(g1)-1, loc=med1, scale=stats.sem(g1))[1]:.2f}]" if len(g1) > 1 else "[N/A; N/A]"

        p_str = "N/A"
        try:
            if len(g0) > 1 and len(g1) > 1:
                g0_clean = g0.replace([np.inf, -np.inf], np.nan).dropna()
                g1_clean = g1.replace([np.inf, -np.inf], np.nan).dropna()
                if len(g0_clean) > 1 and len(g1_clean) > 1 and g0_clean.std() > 0.0001 and g1_clean.std() > 0.0001:
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
    print(f"💾 Таблица сохранена в {output_file}")
    print("\n" + "="*100 + "\n📊 ТАБЛИЦА СТАТИСТИКИ\n" + "="*100)
    print(stats_df.to_string(index=False))
    print("="*100)
    return stats_df


def paired_analysis(X, y, groups, feature_names):
    """Парный анализ для пациентов с обеими записями"""
    print("\n🔍 ПАРНЫЙ АНАЛИЗ (пациенты с ДО и ПОСЛЕ)")
    print("="*80)
    
    patient_counts = Counter(groups)
    paired_patients = [pid for pid, cnt in patient_counts.items() if cnt == 2]
    print(f"Найдено парных пациентов: {len(paired_patients)}")
    
    if len(paired_patients) < 5:
        print("⚠️ Мало парных записей для надёжного анализа")
        return []
    
    print(f"ID парных пациентов: {paired_patients[:10]}")
    
    # Собираем данные для парного t-теста
    print("\n📊 РЕЗУЛЬТАТЫ ПАРНОГО T-ТЕСТА")
    print("-"*80)
    
    significant_paired = []
    
    for i, feat in enumerate(feature_names):
        before_vals, after_vals = [], []
        
        for pid in paired_patients:
            # Находим индексы этого пациента
            idx = [j for j, g in enumerate(groups) if g == pid]
            
            if len(idx) == 2:
                # Определяем какой "до" (y=1), какой "после" (y=0)
                if y[idx[0]] == 1 and y[idx[1]] == 0:
                    before_vals.append(X[idx[0], i])
                    after_vals.append(X[idx[1], i])
                elif y[idx[0]] == 0 and y[idx[1]] == 1:
                    before_vals.append(X[idx[1], i])
                    after_vals.append(X[idx[0], i])
                # else: метки не соответствуют ожидаемым (1 и 0)
        
        if len(before_vals) >= 5:
            try:
                stat, p_val = stats.ttest_rel(before_vals, after_vals)
                
                # Показываем все p < 0.1 для отладки
                if p_val < 0.1:
                    print(f"  {'✅' if p_val < 0.05 else '⚠️'} {feat}: p = {p_val:.4f} (парный, n={len(before_vals)})")
                
                if p_val < 0.05:
                    significant_paired.append((feat, p_val))
            except Exception as e:
                pass
    
    print(f"\n📈 ВСЕГО ЗНАЧИМЫХ В ПАРНОМ АНАЛИЗЕ (p < 0.05): {len(significant_paired)} из {len(feature_names)}")
    print("="*80)
    
    return significant_paired


def create_feature_plots(X, y, feature_names, stats_df, output_dir="figures"):
    """Создание графиков для значимых признаков"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Находим значимые признаки
    significant = []
    for _, row in stats_df.iterrows():
        p_val = row['p-value']
        if p_val != 'N/A' and float(p_val) < 0.05:
            significant.append(row['Показатель'])
    
    print(f"\n📈 СОЗДАНИЕ ГРАФИКОВ ДЛЯ {len(significant)} ЗНАЧИМЫХ ПРИЗНАКОВ")
    print("-"*80)
    
    for feat in significant[:5]:  # Топ-5 значимых
        if feat in feature_names:
            idx = feature_names.index(feat)
            feat_data = X[:, idx]
            
            plt.figure(figsize=(10, 6))
            df_plot = pd.DataFrame({
                'Значение': feat_data,
                'Группа': ['До лечения' if l == 1 else 'После' for l in y]
            })
            
            sns.boxplot(data=df_plot, x='Группа', y='Значение', palette='Set2', showfliers=False)
            sns.stripplot(data=df_plot, x='Группа', y='Значение', color='black', alpha=0.4, size=4, jitter=0.1)
            
            p_val = stats_df[stats_df['Показатель'] == feat]['p-value'].values[0]
            plt.title(f'{feat}\n(p = {p_val})', fontsize=12, fontweight='bold')
            plt.ylabel('Значение признака', fontsize=11)
            plt.xlabel('', fontsize=11)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Сохраняем с безопасным именем
            safe_name = "".join(c if c.isalnum() else '_' for c in feat[:50])
            filepath = os.path.join(output_dir, f'{safe_name}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ {filepath}")


def predict_new_files(classifier, analyzer, folder_path):
    """Предсказание для новых пациентов"""
    if not os.path.exists(folder_path):
        return
    print("\n🧪 ПРЕДСКАЗАНИЕ ДЛЯ НОВЫХ ПАЦИЕНТОВ")
    for filename in sorted([f for f in os.listdir(folder_path) if f.endswith('.wav')]):
        file_path = os.path.join(folder_path, filename)
        sound, _, _ = analyzer.load_audio(file_path)
        if sound:
            feats = analyzer.extract_features(sound, None, None, file_path)
            if feats and feats.get("valid"):
                for k, v in feats.items():
                    if isinstance(v, float) and np.isnan(v):
                        feats[k] = 0.0
                X_new = [[feats.get(col, 0.0) for col in classifier.feature_names]]
                pred = classifier.predict(X_new)[0]
                proba = classifier.predict_proba(X_new)[0][1]
                status = "🔴 ХОБЛ" if pred == 1 else "🟢 Здоров"
                print(f"  {filename}: {status} ({proba:.2%})")


if __name__ == "__main__":
    print("🚀 Запуск системы анализа ХОБЛ (eGeMAPS + LGDV)")
    analyzer = VoiceFeatureExtractor()
    X, y = analyzer.prepare_dataset("data")
    groups = analyzer.patient_groups

    if len(X) > 0:
        # ФИЛЬТРАЦИЯ
        non_zero = [i for i, col in enumerate(analyzer.feature_columns) 
                   if len(np.unique(X[:, i])) > 3 and np.var(X[:, i]) > 0.001]
        X_filtered = X[:, non_zero]
        feature_names = [analyzer.feature_columns[i] for i in non_zero]
        print(f"✅ Осталось признаков: {len(feature_names)}")

        # СТАТИСТИКА (на оригинальных данных)
        print("\n📈 ТАБЛИЦА СТАТИСТИКИ")
        stats_df = create_statistics_table(X_filtered, y, groups, feature_names)

        # ПАРНЫЙ АНАЛИЗ (для 7 пациентов)
        significant_paired = paired_analysis(X_filtered, y, groups, feature_names)

        # ГРАФИКИ
        create_feature_plots(X_filtered, y, feature_names, stats_df)

        # 🔴 РАЗДЕЛЯЕМ данные на обучение и тест (ЧЕСТНАЯ ОЦЕНКА!)
        from sklearn.model_selection import train_test_split
        
        print("\n🔧 РАЗДЕЛЕНИЕ НА ОБУЧЕНИЕ/ТЕСТ (80/20)")
        print("="*60)
        
        X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
            X_filtered, y, groups, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  Обучение: {len(X_train)} образцов")
        print(f"  Тест: {len(X_test)} образцов")
        
        # БАЛАНСИРОВКА (только для обучения!)
        print("\n🔧 БАЛАНСИРОВКА КЛАССОВ (только обучение)")
        if len(np.unique(y_train)) == 2:
            try:
                X_train_bal, y_train_bal = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_train, y_train)
                print(f"✅ После SMOTE: Класс 0={np.sum(y_train_bal==0)}, Класс 1={np.sum(y_train_bal==1)}")
            except Exception as e:
                print(f"⚠️ SMOTE ошибка: {e}")
                X_train_bal, y_train_bal = X_train, y_train
        else:
            X_train_bal, y_train_bal = X_train, y_train

        # ОБУЧЕНИЕ
        classifier = COPDClassifierMultiModel()
        classifier.train(X_train_bal, y_train_bal, groups=None, feature_names=feature_names, n_splits=5)

        # 🔴 ЧЕСТНАЯ ОЦЕНКА АНСАМБЛЯ (на тестовых данных!)
        print("\n🔬 УМНЫЙ АНСАМБЛЬ (исключаем плохие модели)")
        print("="*60)
        
        good_models = {}
        for name, result in classifier.model_results.items():
            if result['median_f1'] >= 0.3:
                good_models[name] = result
                print(f"  ✅ {name}: F1={result['median_f1']:.4f} (включена)")
            else:
                print(f"  ❌ {name}: F1={result['median_f1']:.4f} (исключена)")
        
        if len(good_models) >= 2 and len(X_test) > 0:
            print(f"\n📊 ОЦЕНКА АНСАМБЛЯ НА ТЕСТОВЫХ ДАННЫХ ({len(X_test)} образцов)")
            
            # Масштабируем тестовые данные
            X_test_scaled = classifier.scaler.transform(X_test)
            
            # Собираем предсказания от всех хороших моделей
            all_proba = []
            for name, result in good_models.items():
                proba = result['model'].predict_proba(X_test_scaled)[:, 1]
                all_proba.append(proba)
            
            # Усредняем
            ensemble_proba = np.mean(all_proba, axis=0)
            ensemble_pred = (ensemble_proba >= 0.5).astype(int)
            
            from sklearn.metrics import f1_score, accuracy_score, classification_report
            ensemble_f1 = f1_score(y_test, ensemble_pred)
            ensemble_acc = accuracy_score(y_test, ensemble_pred)
            
            print(f"\n  📈 F1 ансамбля (тест): {ensemble_f1:.4f}")
            print(f"  📈 Accuracy ансамбля (тест): {ensemble_acc:.4f}")
            print(f"\n📋 Отчёт по классам:")
            print(classification_report(y_test, ensemble_pred, target_names=['После', 'До']))
        else:
            print("⚠️ Мало данных для честной оценки ансамбля")

        # РЕЗУЛЬТАТЫ
        print("\n📊 СРАВНЕНИЕ МОДЕЛЕЙ (кросс-валидация на обучении)")
        print(classifier.get_model_comparison_table().to_string(index=False))
        
        print(f"\n🔬 ВАЖНОСТЬ ПРИЗНАКОВ ({classifier.best_model_name})")
        imp = classifier.get_feature_importance()
        if imp:
            for feat, val in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {feat}: {val:.4f}")

        classifier.save()
        predict_new_files(classifier, analyzer, "data/new_patients")
        print("\n✅ Готово!")
    else:
        print("❌ Нет данных")