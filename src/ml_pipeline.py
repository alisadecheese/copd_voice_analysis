import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    make_scorer,
)
from scipy import stats
import joblib
import warnings

warnings.filterwarnings("ignore")


class COPDClassifierMultiModel:
    """
    Мульти-модельный классификатор для определения ХОБЛ.
    Включает: Random Forest, SVM, Gradient Boosting, Logistic Regression
    """

    def __init__(self):
        # Инициализация моделей с параметрами, подходящими для медицинских данных
        self.models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "SVM": SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                random_state=42,
                probability=True,  # Нужно для predict_proba
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            ),
            "LogisticRegression": LogisticRegression(
                C=1.0,
                class_weight="balanced",
                random_state=42,
                max_iter=1000,
                n_jobs=-1,
            ),
        }

        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.best_model_name = None
        self.best_model = None
        self.model_results = {}

    
    def train(self, X, y, groups=None, feature_names=None, n_splits=5):
        
        print("Масштабирование признаков...")
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = feature_names
        
        # ПРОВЕРКА НА ГРУППЫ
        if groups is None:
            print(" Предупреждение: groups не переданы! Используем обычную кросс-валидацию.")
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_groups = None
        else:
            print(f"👥 Используется GroupKFold (учет пациентов): {len(set(groups))} уникальных пациентов")
            skf = GroupKFold(n_splits=n_splits)
            cv_groups = groups
        
        # Адаптация n_splits под количество групп
        if groups is not None:
            unique_groups = len(set(groups))
            if n_splits > unique_groups:
                print(f"⚠️ n_splits={n_splits} больше числа пациентов ({unique_groups}). Уменьшено до {unique_groups}")
                n_splits = unique_groups
                skf = GroupKFold(n_splits=n_splits)
        
        scoring = make_scorer(f1_score)
        
        print(f"🧠 Обучение и оценка {len(self.models)} моделей (CV={n_splits})...\n")
        
        for name, model in self.models.items():
            print(f"  • Обучение модели: {name}")
            
            # 🔴 ПЕРЕДАЕМ groups В cross_val_score
            cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring=scoring, groups=cv_groups)
            
            median_score = np.median(cv_scores)
            n_scores = len(cv_scores)
            if n_scores >= 3 and np.std(cv_scores) > 0:
                ci = stats.t.interval(0.95, n_scores-1, loc=median_score, scale=stats.sem(cv_scores))
                ci_lower, ci_upper = ci[0], ci[1]
            else:
                ci_lower, ci_upper = float('nan'), float('nan')
            
            self.model_results[name] = {
                'cv_scores': cv_scores,
                'median_f1': median_score,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'mean_accuracy': np.mean(cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy', groups=cv_groups)),
                'model': model
            }
            
            print(f"    F1 (median): {median_score:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            model.fit(X_scaled, y)
        
        self.best_model_name = max(self.model_results.keys(), 
                                   key=lambda k: self.model_results[k]['median_f1'])
        self.best_model = self.model_results[self.best_model_name]['model']
        self.is_fitted = True
        
        print(f"\n✅ Лучшая модель: {self.best_model_name} (F1={self.model_results[self.best_model_name]['median_f1']:.4f})")
        return self

    def predict(self, X, use_best=True, model_name=None):
        """
        Предсказание класса.

        Parameters:
        -----------
        use_best : bool, default=True
            Использовать лучшую модель или конкретную
        model_name : str, optional
            Имя конкретной модели (если use_best=False)
        """
        if not self.is_fitted:
            raise Exception("Модель не обучена! Сначала вызовите train().")

        X_scaled = self.scaler.transform(X)

        if use_best:
            return self.best_model.predict(X_scaled)
        else:
            if model_name not in self.models:
                raise ValueError(f"Модель {model_name} не найдена")
            return self.models[model_name].predict(X_scaled)

    def predict_proba(self, X, use_best=True, model_name=None):
        """Вероятность принадлежности к классу"""
        if not self.is_fitted:
            raise Exception("Модель не обучена!")

        X_scaled = self.scaler.transform(X)

        if use_best:
            return self.best_model.predict_proba(X_scaled)
        else:
            if model_name not in self.models:
                raise ValueError(f"Модель {model_name} не найдена")
            return self.models[model_name].predict_proba(X_scaled)

    def get_model_comparison_table(self):
        """Возвращает таблицу сравнения всех моделей (DataFrame)"""
        if not self.model_results:
            raise Exception("Модели не обучены!")

        data = []
        for name, results in self.model_results.items():
            data.append(
                {
                    "Model": name,
                    "F1 Median": results["median_f1"],
                    "F1 CI Lower": results["ci_lower"],
                    "F1 CI Upper": results["ci_upper"],
                    "Accuracy Mean": results["mean_accuracy"],
                }
            )

        df = pd.DataFrame(data)
        return df.sort_values("F1 Median", ascending=False)

    def get_feature_importance(self, model_name=None):
        """
        Важность признаков (для Random Forest, Gradient Boosting, Logistic Regression).
        SVM не предоставляет importance напрямую.
        """
        if not self.is_fitted:
            return None

        if model_name is None:
            model_name = self.best_model_name

        model = self.model_results[model_name]["model"]

        if hasattr(model, "feature_importances_"):
            return dict(
                zip(
                    self.feature_names or range(len(model.feature_importances_)),
                    model.feature_importances_,
                )
            )
        elif hasattr(model, "coef_"):
            # Для Logistic Regression
            return dict(
                zip(
                    self.feature_names or range(model.coef_.shape[1]),
                    np.abs(model.coef_[0]),
                )
            )
        else:
            return None  # SVM не поддерживает

    def save(self, path="copd_multimodel.pkl"):
        """Сохранение всех моделей и скалера"""
        joblib.dump(
            {
                "models": self.models,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "best_model_name": self.best_model_name,
                "model_results": {
                    k: {kk: vv for kk, vv in v.items() if kk != "model"}
                    for k, v in self.model_results.items()
                },  # Исключаем объекты моделей для легковесности
                "is_fitted": self.is_fitted,
            },
            path,
        )
        print(f"Все модели сохранены в {path}")

    def load(self, path="copd_multimodel.pkl"):
        """Загрузка моделей"""
        data = joblib.load(path)
        self.models = data["models"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.best_model_name = data["best_model_name"]
        self.model_results = data["model_results"]
        # Восстанавливаем модели в results
        for name in self.model_results:
            self.model_results[name]["model"] = self.models[name]
        self.best_model = self.models[self.best_model_name]
        self.is_fitted = data["is_fitted"]
        print(f" Модели загружены из {path}")
        return self
