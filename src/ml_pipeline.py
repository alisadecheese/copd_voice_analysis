import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer, accuracy_score, precision_score, recall_score
from scipy import stats
import pickle
import os


class COPDClassifierMultiModel:
    """Мульти-модель для классификации ХОБЛ (на основе обзора [10-13])"""

    def __init__(self):
        # 🔴 RobustScaler лучше для MFCC и акустических признаков
        self.scaler = RobustScaler()
        
        # Модели на основе обзора литературы
        self.models = {
            # SVM - лучшая по обзору (84% accuracy) [13]
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            ),
            
            # Logistic Regression - простая и эффективная [10]
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                C=0.1,
                class_weight='balanced',
                solver='lbfgs',
                random_state=42
            ),
            
            # Random Forest - хорош для малых выборок [10]
            'RandomForest': RandomForestClassifier(
                n_estimators=50,  # Уменьшено со 100
                max_depth=5,      # Уменьшено с 10
                min_samples_split=10,  # Увеличено (меньше переобучения)
                min_samples_leaf=5,    # Увеличено
                class_weight='balanced',
                random_state=42
            ),
            
            # KNN - хорош для регрессии [11]
            'KNN': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                metric='euclidean'
            ),
            
            # Neural Network - с регуляризацией
            'MLP': MLPClassifier(
                hidden_layer_sizes=(32, 16),  # Уменьшено
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.2,  # Увеличено
                alpha=0.01,  # L2 регуляризация
                random_state=42
            )
        }
        
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.is_fitted = False

    def train(self, X, y, groups=None, feature_names=None, n_splits=5):
        """Обучение моделей с кросс-валидацией"""
        print("🔄 Масштабирование признаков (RobustScaler)...")
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = feature_names

        # Stratified K-Fold для сохранения баланса классов
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scoring = make_scorer(f1_score)

        print(f"🧠 Обучение {len(self.models)} моделей (CV={n_splits})...\n")

        for name, model in self.models.items():
            print(f"  • Обучение модели: {name}")

            try:
                cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring=scoring)
                
                median_score = np.median(cv_scores)
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                
                n_scores = len(cv_scores)
                if n_scores >= 3 and std_score > 0:
                    ci = stats.t.interval(0.95, n_scores-1, loc=median_score, scale=stats.sem(cv_scores))
                    ci_lower, ci_upper = ci[0], ci[1]
                else:
                    ci_lower, ci_upper = float('nan'), float('nan')

                # Дополнительные метрики
                cv_accuracy = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
                cv_precision = cross_val_score(model, X_scaled, y, cv=skf, scoring='precision')
                cv_recall = cross_val_score(model, X_scaled, y, cv=skf, scoring='recall')

                self.model_results[name] = {
                    'cv_scores': cv_scores,
                    'median_f1': median_score,
                    'mean_f1': mean_score,
                    'std_f1': std_score,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'mean_accuracy': np.mean(cv_accuracy),
                    'mean_precision': np.mean(cv_precision),
                    'mean_recall': np.mean(cv_recall),
                    'model': model
                }

                print(f"    F1 (median): {median_score:.4f} ± {std_score:.4f}")
                print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                print(f"    Accuracy: {np.mean(cv_accuracy):.4f}")

                # Обучаем на всех данных
                model.fit(X_scaled, y)

            except Exception as e:
                print(f"    ❌ Ошибка: {e}")
                continue

        # Выбираем лучшую модель по F1
        if self.model_results:
            self.best_model_name = max(self.model_results.keys(),
                                       key=lambda k: self.model_results[k]['median_f1'])
            self.best_model = self.model_results[self.best_model_name]['model']
            self.is_fitted = True

            print(f"\n✅ Лучшая модель: {self.best_model_name}")
            print(f"   F1-Score: {self.model_results[self.best_model_name]['median_f1']:.4f}")
            print(f"   Accuracy: {self.model_results[self.best_model_name]['mean_accuracy']:.4f}")
        else:
            print("❌ Ни одна модель не обучилась!")

        return self

    def predict(self, X, use_best=True):
        """Предсказание"""
        if not self.is_fitted:
            raise ValueError("Модель не обучена!")

        X_scaled = self.scaler.transform(X)

        if use_best:
            return self.best_model.predict(X_scaled)
        else:
            return {name: result['model'].predict(X_scaled) for name, result in self.model_results.items()}

    def predict_proba(self, X, use_best=True):
        """Вероятности предсказания"""
        if not self.is_fitted:
            raise ValueError("Модель не обучена!")

        X_scaled = self.scaler.transform(X)

        if use_best:
            return self.best_model.predict_proba(X_scaled)
        else:
            return {name: result['model'].predict_proba(X_scaled) for name, result in self.model_results.items()}

    def get_model_comparison_table(self):
        """Таблица сравнения всех моделей"""
        data = []
        for name, result in self.model_results.items():
            data.append({
                'Model': name,
                'F1 Median': result['median_f1'],
                'F1 Mean': result['mean_f1'],
                'F1 Std': result['std_f1'],
                'F1 CI Lower': result['ci_lower'],
                'F1 CI Upper': result['ci_upper'],
                'Accuracy': result['mean_accuracy'],
                'Precision': result['mean_precision'],
                'Recall': result['mean_recall']
            })
        return pd.DataFrame(data)

    def get_feature_importance(self):
        """Важность признаков для лучшей модели"""
        if not self.is_fitted:
            return None

        if hasattr(self.best_model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.best_model.feature_importances_))
        elif hasattr(self.best_model, 'coef_'):
            return dict(zip(self.feature_names, np.abs(self.best_model.coef_[0])))
        else:
            return None

    def save(self, path="copd_multimodel.pkl"):
        """Сохранение модели"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Модели сохранены в {path}")

    @classmethod
    def load(cls, path="copd_multimodel.pkl"):
        """Загрузка модели"""
        with open(path, 'rb') as f:
            return pickle.load(f)