import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from scipy import stats
import pickle

class COPDClassifierMultiModel:
    """Мульти-модель для классификации ХОБЛ"""

    def __init__(self):
        self.scaler = RobustScaler()
        self.models = {
            'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', solver='lbfgs', random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=10, class_weight='balanced', random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
        }
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.is_fitted = False

    def train(self, X, y, groups=None, feature_names=None, n_splits=5):
        """Обучение моделей"""
        print("🔄 Масштабирование признаков...")
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = feature_names
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scoring = make_scorer(f1_score)

        print(f"🧠 Обучение {len(self.models)} моделей...\n")
        for name, model in self.models.items():
            print(f"  • {name}")
            try:
                cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring=scoring)
                median_f1 = np.median(cv_scores)
                std_f1 = np.std(cv_scores)
                n = len(cv_scores)
                ci = stats.t.interval(0.95, n-1, loc=median_f1, scale=stats.sem(cv_scores)) if n >= 3 and std_f1 > 0 else (np.nan, np.nan)
                
                self.model_results[name] = {
                    'median_f1': median_f1, 'std_f1': std_f1,
                    'ci_lower': ci[0], 'ci_upper': ci[1],
                    'accuracy': np.mean(cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')),
                    'model': model
                }
                print(f"    F1: {median_f1:.4f} ± {std_f1:.4f}, 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
                model.fit(X_scaled, y)
            except Exception as e:
                print(f"    ❌ Ошибка: {e}")

        if self.model_results:
            self.best_model_name = max(self.model_results, key=lambda k: self.model_results[k]['median_f1'])
            self.best_model = self.model_results[self.best_model_name]['model']
            self.is_fitted = True
            print(f"\n✅ Лучшая модель: {self.best_model_name}")
        return self

    def predict(self, X, use_best=True):
        if not self.is_fitted:
            raise ValueError("Модель не обучена!")
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled) if use_best else {n: m['model'].predict(X_scaled) for n, m in self.model_results.items()}

    def predict_proba(self, X, use_best=True):
        if not self.is_fitted:
            raise ValueError("Модель не обучена!")
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict_proba(X_scaled) if use_best else {n: m['model'].predict_proba(X_scaled) for n, m in self.model_results.items()}

    def get_model_comparison_table(self):
        return pd.DataFrame([{
            'Model': n, 'F1 Median': r['median_f1'], 'F1 Std': r['std_f1'],
            'CI Lower': r['ci_lower'], 'CI Upper': r['ci_upper'], 'Accuracy': r['accuracy']
        } for n, r in self.model_results.items()])

    def get_feature_importance(self):
        if not self.is_fitted:
            return None
        if hasattr(self.best_model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.best_model.feature_importances_))
        elif hasattr(self.best_model, 'coef_'):
            return dict(zip(self.feature_names, np.abs(self.best_model.coef_[0])))
        return None

    def save(self, path="copd_model.pkl"):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Модель сохранена в {path}")