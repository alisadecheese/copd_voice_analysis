import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict


class StatisticalAnalyzer:
    """Статистический анализ с бутстреп доверительными интервалами"""

    def __init__(self, confidence_level: float = 0.95, n_bootstraps: int = 1000):
        self.confidence_level = confidence_level
        self.n_bootstraps = n_bootstraps

    def calculate_median_ci(
        self, data: np.ndarray
    ) -> Tuple[float, Tuple[float, float]]:
        """Медиана + доверительный интервал"""
        if len(data) == 0 or np.all(np.isnan(data)):
            return np.nan, (np.nan, np.nan)

        data = data[~np.isnan(data)]
        median = np.median(data)

        # Бутстреп ресемплинг
        bootstrapped_medians = []
        for _ in range(self.n_bootstraps):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrapped_medians.append(np.median(sample))

        # Перцентили [2.5; 97.5]
        lower_p = (1 - self.confidence_level) / 2 * 100
        upper_p = (1 + self.confidence_level) / 2 * 100
        ci_lower = np.percentile(bootstrapped_medians, lower_p)
        ci_upper = np.percentile(bootstrapped_medians, upper_p)

        return median, (ci_lower, ci_upper)

    def compare_groups(
        self,
        df: pd.DataFrame,
        feature: str,
        group1: str,
        group2: str,
        paired: bool = False,
    ) -> Dict:
        """Сравнение двух групп (Вилкоксон для парных, Манн-Уитни для независимых)"""
        data1 = df[df["label"] == group1][feature].dropna().values
        data2 = df[df["label"] == group2][feature].dropna().values

        if len(data1) == 0 or len(data2) == 0:
            return None

        # Проверка нормальности (Шапиро-Уилкс )
        _, p1 = stats.shapiro(data1[: min(50, len(data1))])  # Ограничение для Shapiro
        _, p2 = stats.shapiro(data2[: min(50, len(data2))])
        normal = p1 > 0.05 and p2 > 0.05

        # Выбор теста
        if paired and len(data1) == len(data2):
            stat, p_value = stats.wilcoxon(data1, data2)
            test_name = "Wilcoxon"
        else:
            stat, p_value = stats.mannwhitneyu(data1, data2, alternative="two-sided")
            test_name = "Mann-Whitney U"

        return {
            "median_1": np.median(data1),
            "median_2": np.median(data2),
            "p_value": p_value,
            "significant": p_value < 0.05,
            "test": test_name,
            "normal_distribution": normal,
        }

    def generate_report(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """Генерация отчета по всем признакам"""
        report = []
        groups = df["label"].unique()

        for feat in features:
            row = {"Feature": feat}
            for grp in groups:
                data = df[df["label"] == grp][feat].dropna().values
                median, ci = self.calculate_median_ci(data)
                row[f"{grp}_median"] = median
                row[f"{grp}_ci_low"] = ci[0]
                row[f"{grp}_ci_high"] = ci[1]
                row[f"{grp}_n"] = len(data)
            report.append(row)

        return pd.DataFrame(report)
