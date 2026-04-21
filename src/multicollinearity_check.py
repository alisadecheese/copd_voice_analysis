# src/multicollinearity_check.py
"""
Модуль для диагностики и фильтрации мультиколлинеарности признаков.
Используется в проекте диагностики ХОБЛ по голосу.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import warnings
warnings.filterwarnings('ignore')


def _clean_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """
    Предобработка числовых признаков:
    - оставляет только числовые столбцы
    - заменяет inf/-inf на NaN
    - заполняет пропуски медианой
    - удаляет константные признаки
    """
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    # Удаляем столбцы с нулевой или почти нулевой дисперсией
    return X.loc[:, X.std() > 1e-6].copy()


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Расчёт VIF (Variance Inflation Factor) для каждого признака.
    
    Parameters
    ----------
    X : pd.DataFrame
        Матрица признаков (только числовые)
    
    Returns
    -------
    pd.DataFrame с колонками ['feature', 'VIF'], отсортированная по убыванию VIF
    """
    X_clean = _clean_numeric(X)
    if X_clean.empty:
        return pd.DataFrame(columns=["feature", "VIF"])
    
    # Добавляем константу для корректного расчёта VIF
    X_const = add_constant(X_clean)
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_clean.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_const.values, i) 
        for i in range(1, X_const.shape[1])  # пропускаем константу (index 0)
    ]
    
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


def find_correlated_pairs(X: pd.DataFrame, threshold: float = 0.85) -> list:
    """
    Находит пары признаков с абсолютной корреляцией выше порога.
    
    Parameters
    ----------
    X : pd.DataFrame
        Матрица признаков
    threshold : float
        Порог корреляции (по умолчанию 0.85)
    
    Returns
    -------
    list of tuples: [(feat1, feat2, correlation), ...]
    """
    X_clean = _clean_numeric(X)
    if X_clean.empty:
        return []
    
    corr = X_clean.corr().abs()
    # Берём только верхний треугольник матрицы, чтобы не дублировать пары
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    
    pairs = []
    for col in upper.columns:
        for idx in upper.index:
            if upper.loc[idx, col] > threshold:
                pairs.append((idx, col, upper.loc[idx, col]))
    
    return sorted(pairs, key=lambda x: x[2], reverse=True)


def recursive_vif_elimination(X: pd.DataFrame, max_vif: float = 5.0) -> tuple:
    """
    Рекурсивно удаляет признаки с VIF > max_vif, начиная с самого проблемного.
    
    Parameters
    ----------
    X : pd.DataFrame
        Исходная матрица признаков
    max_vif : float
        Максимально допустимый VIF (по умолчанию 5.0)
    
    Returns
    -------
    tuple: (X_optimized, dropped_list)
        - X_optimized: pd.DataFrame с оставшимися признаками
        - dropped_list: list of tuples [(feature_name, vif_value), ...]
    """
    X_clean = X.copy()
    dropped = []
    
    iteration = 0
    while True:
        iteration += 1
        vif = compute_vif(X_clean)
        
        # Условие остановки: все VIF в норме или не осталось признаков
        if vif.empty or vif["VIF"].max() <= max_vif:
            break
        
        # Находим худший признак и удаляем его
        worst = vif.iloc[0]
        X_clean = X_clean.drop(columns=[worst["feature"]])
        dropped.append((worst["feature"], worst["VIF"]))
        
        # Защита от бесконечного цикла
        if iteration > 100:
            print(f"[WARN] Достигнут лимит итераций ({iteration}), остановка.")
            break
    
    return X_clean, dropped


def plot_diagnostics(X: pd.DataFrame, vif_df: pd.DataFrame, 
                     corr_threshold: float = 0.85, 
                     save_path: str = None):
    """
    Визуализация: бар-плот VIF и тепловая карта корреляций.
    
    Parameters
    ----------
    X : pd.DataFrame
        Матрица признаков для корреляционной матрицы
    vif_df : pd.DataFrame
        Результат compute_vif()
    corr_threshold : float
        Порог для подсветки сильных корреляций
    save_path : str, optional
        Если указан, сохраняет график в файл
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Bar plot: Top 20 признаков по VIF
    top_vif = vif_df.head(20)
    if not top_vif.empty:
        sns.barplot(data=top_vif, x="VIF", y="feature", ax=axes[0], palette="viridis")
        axes[0].set_title("Top 20 признаков по VIF")
        axes[0].axvline(x=5.0, color='r', linestyle='--', label='Порог VIF = 5')
        axes[0].axvline(x=10.0, color='orange', linestyle=':', label='Порог VIF = 10')
        axes[0].legend()
        axes[0].set_xlabel("VIF")
    
    # 2. Correlation heatmap
    X_clean = _clean_numeric(X)
    if not X_clean.empty and X_clean.shape[1] > 1:
        corr = X_clean.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, 
                    ax=axes[1], cbar_kws={'shrink': 0.8})
        axes[1].set_title(f"Корреляционная матрица\n(|r| > {corr_threshold} — сильные связи)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Графики сохранены в {save_path}")
    
    plt.show()


def run_analysis(data_path: str, target_col: str = "label", 
                 corr_threshold: float = 0.85, 
                 vif_threshold: float = 5.0,
                 save_plots: str = "vif_report.png") -> tuple:
    """
    Полный пайплайн анализа мультиколлинеарности.
    
    Parameters
    ----------
    data_path : str
        Путь к файлу с данными (.csv или .xlsx)
    target_col : str
        Имя столбца с целевой переменной
    corr_threshold : float
        Порог корреляции для поиска пар
    vif_threshold : float
        Максимальный допустимый VIF
    save_plots : str
        Путь для сохранения отчётных графиков
    
    Returns
    -------
    tuple: (X_optimized, y, dropped_features)
        - X_optimized: pd.DataFrame с отфильтрованными признаками
        - y: pd.Series с целевой переменной
        - dropped_features: list признаков, удалённых из-за мультиколлинеарности
    """
    # Чтение данных
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(data_path)
    else:
        raise ValueError("Поддерживаются только форматы .csv и .xlsx")
    
    if target_col not in df.columns:
        raise ValueError(f"Столбец '{target_col}' не найден в данных. Доступные: {list(df.columns)}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print("🔍 Запуск анализа мультиколлинеарности...")
    print(f"   Исходно признаков: {X.shape[1]}, наблюдений: {X.shape[0]}")
    
    # 1. Расчёт VIF
    vif_df = compute_vif(X)
    
    # 2. Поиск коррелирующих пар
    pairs = find_correlated_pairs(X, corr_threshold)
    
    # Вывод отчёта
    high_vif = vif_df[vif_df["VIF"] > vif_threshold]
    print(f"\n📊 Признаки с VIF > {vif_threshold} ({len(high_vif)}):")
    if not high_vif.empty:
        print(high_vif.to_string(index=False))
    else:
        print("   Нет признаков с высоким VIF ✅")
    
    if pairs:
        print(f"\n🔗 Сильно коррелирующие пары (|r| > {corr_threshold}):")
        for f1, f2, r in pairs[:15]:  # показываем топ-15
            print(f"   {f1} ↔ {f2}: r = {r:.3f}")
        if len(pairs) > 15:
            print(f"   ... и ещё {len(pairs) - 15} пар")
    
    # 3. Рекурсивная фильтрация
    print(f"\n🤖 Автоматическая оптимизация (рекурсивный VIF, порог={vif_threshold})...")
    X_optimized, dropped = recursive_vif_elimination(X, max_vif=vif_threshold)
    
    print(f"✅ Удалено {len(dropped)} признаков:")
    if dropped:
        for feat, v in dropped:
            print(f"   ❌ {feat} (VIF = {v:.2f})")
    else:
        print("   Все признаки уже в норме ✅")
    
    # 4. Визуализация
    print(f"\n📈 Построение диагностических графиков...")
    plot_diagnostics(X_optimized, compute_vif(X_optimized), corr_threshold, save_plots)
    
    # 5. Итоговая статистика
    print(f"\n📋 Итог:")
    print(f"   Было признаков: {X.shape[1]}")
    print(f"   Осталось признаков: {X_optimized.shape[1]}")
    print(f"   Удалено: {len(dropped)}")
    
    return X_optimized, y, dropped