import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.feature_extractor import VoiceFeatureExtractor
from ml_pipeline import COPDClassifierMultiModel

if __name__ == "__main__":
    print("Запуск системы анализа ХОБЛ")

    # 1. Извлечение признаков
    analyzer = VoiceFeatureExtractor()

#путь к папке с файлами
    DATA_DIR = "data"

    print("Подготовка датасета")
    X, y = analyzer.prepare_dataset(DATA_DIR)
    feature_names = analyzer.feature_columns

    if len(X) > 0:
        # 2. Обучение мульти-модели
        print("\n" + "=" * 50)
        classifier = COPDClassifierMultiModel()
        classifier.train(X, y, feature_names=feature_names, n_splits=2)

        # 3. Сравнение моделей
        print("\n" + "=" * 50)
        print("СРАВНЕНИЕ МОДЕЛЕЙ")
        print("=" * 50)
        comparison_df = classifier.get_model_comparison_table()
        print(comparison_df.to_string(index=False))

        # 4. Важность признаков (для лучшей модели)
        print("\n" + "=" * 50)
        print(f"ВАЖНОСТЬ ПРИЗНАКОВ ({classifier.best_model_name})")
        print("=" * 50)
        importance = classifier.get_feature_importance()
        if importance:
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for feat, imp in sorted_imp[:5]:  # Топ-5 признаков
                print(f"  {feat}: {imp:.4f}")

        # 5. Сохранение
        classifier.save()

        # 6. Тест предсказания
        print("\n" + "=" * 50)
        print("ТЕСТ ПРЕДСКАЗАНИЯ")
        print("=" * 50)
        if len(X) > 0:
            pred = classifier.predict(X[:1], use_best=True)
            proba = classifier.predict_proba(X[:1], use_best=True)
            print(f"Предсказание: {pred[0]}")
            print(f"Вероятность ХОБЛ: {proba[0][1]:.2%}")

        print("\nГотово!")
    else:
        print("Нет данных для обучения.")
