# main.py
import os
import sys
import joblib
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_extractor import FeatureExtractor
from src.multicollinearity_check import run_analysis  # ← только этот импорт добавлен

DATA_DIR = "data"
FEATURES_RAW = "data/features_raw.csv"
FEATURES_CLEAN = "data/features_cleaned.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "copd_model.pkl")

def train_pipeline():
    print("🎤 1. Извлечение признаков...")
    extractor = FeatureExtractor()
    X, y = extractor.prepare_dataset(DATA_DIR)
    print(f"   ✅ Загружено: {X.shape[0]} записей, {X.shape[1]} признаков")
    
    # 🔽 БЛОК 1: ФИЛЬТРАЦИЯ МУЛЬТИКОЛЛИНЕАРНОСТИ (только это добавлено)
    print("🔍 2. Фильтрация мультиколлинеарности (VIF)...")
    df_raw = X.copy()
    df_raw["label"] = y
    df_raw.to_csv(FEATURES_RAW, index=False)
    
    X_clean, _, dropped = run_analysis(
        FEATURES_RAW,
        target_col="label",
        corr_threshold=0.85,
        vif_threshold=5.0
    )
    print(f"   🗑️ Удалено {len(dropped)} коррелирующих признаков")
    
    df_clean = X_clean.copy()
    df_clean["label"] = y
    df_clean.to_csv(FEATURES_CLEAN, index=False)
    # 🔼 КОНЕЦ БЛОКА 1
    
    print("📊 3. Разделение и масштабирование...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, stratify=y, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    print("🔧 4. Обучение моделей...")
    models = {
        "RF": RandomForestClassifier(n_estimators=300, max_depth=8, 
                                     class_weight='balanced',  # ← БЛОК 2: учёт имбаланса
                                     random_state=42, n_jobs=-1),
        "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
        "LR": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    }
    
    best_model, best_auc = None, -1
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        probs = model.predict_proba(X_test_sc)[:, 1]
        auc = roc_auc_score(y_test, probs)
        print(f"   {name}: AUC = {auc:.4f}")
        if auc > best_auc:
            best_auc, best_model = auc, model
    
    print("📈 5. Оценка лучшей модели...")
    y_pred = best_model.predict(X_test_sc)
    probs = best_model.predict_proba(X_test_sc)[:, 1]
    
    print(classification_report(y_test, y_pred, target_names=["После (0)", "До (1)"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    print("💾 6. Сохранение...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "feature_cols": X_clean.columns.tolist()
    }, MODEL_PATH)
    print(f"   ✅ Сохранено в {MODEL_PATH}")

def predict_files(file_paths):
    if not os.path.exists(MODEL_PATH):
        print("❌ Модель не найдена. Запустите обучение: python main.py")
        return
    cfg = joblib.load(MODEL_PATH)
    extractor = FeatureExtractor()
    print("\n🔎 Прогноз:")
    for fp in file_paths:
        feats = extractor.extract_single_file(fp)
        df = pd.DataFrame([feats]).reindex(columns=cfg["feature_cols"], fill_value=0)
        X_sc = cfg["scaler"].transform(df)
        prob = cfg["model"].predict_proba(X_sc)[0, 1]
        label = 1 if prob > 0.5 else 0
        status = "🔴 ХОБЛ" if label == 1 else "🟢 Здоров"
        print(f"  {os.path.basename(fp)}: {status} ({prob*100:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict_files(sys.argv[2:])
    else:
        train_pipeline()


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