import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def prepare_features(df: pd.DataFrame, target: str, is_prediction: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara las features para el modelo, incluyendo nuevas características de ingeniería avanzada.

    Args:
        df: DataFrame con los datos históricos
        target: Variable objetivo a predecir
        is_prediction: Si es True, prepara datos para predicción (sin target)

    Returns:
        Tuple con features (X) y target (y) si no es predicción
    """
    # Crear copia para no modificar el original
    df = df.copy()

    # 1. Procesamiento básico de fechas
    df = _process_dates(df)

    # 2. Features de tendencia y crecimiento
    df = _add_trend_features(df, target)

    # 3. Features de día de semana y mes
    df = _add_date_features(df)

    # 4. Features de ventanas móviles
    df = _add_window_features(df, target)

    # 5. Features cíclicas
    df = _add_cyclic_features(df)

    # 6. Features externas
    df = _add_external_features(df)

    # 7. Features de lag
    df = _add_lag_features(df, target)

    # 8. Normalización
    df = _normalize_features(df)

    # Preparar X e y
    features_to_drop = ['COD_SUC', 'FECHA', target] if not is_prediction else ['COD_SUC', 'FECHA']
    X = df.drop(columns=features_to_drop, errors='ignore')

    # Eliminar columnas con muchos nulos
    X = X.dropna(axis=1, thresh=0.7 * len(X))

    # Imputar valores faltantes
    for col in X.columns:
        if X[col].isna().any():
            if col.startswith('rolling'):
                X[col].fillna(method='ffill', inplace=True)
                X[col].fillna(method='bfill', inplace=True)
            else:
                X[col].fillna(X[col].median(), inplace=True)

    if is_prediction:
        return X, None
    else:
        y = df[target]
        return X, y


def _process_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Procesamiento básico de fechas"""
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df = df.sort_values('FECHA').reset_index(drop=True)
    df['dia_mes'] = df['FECHA'].dt.day
    df['dia_semana'] = df['FECHA'].dt.dayofweek
    df['mes'] = df['FECHA'].dt.month
    df['trimestre'] = df['FECHA'].dt.quarter
    df['es_fin_de_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
    return df


def _add_trend_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Añade features de tendencia y crecimiento"""
    # Tendencia respecto a promedio de 30 días
    df[f'{target}_rolling30'] = df[target].rolling(30, min_periods=1).mean()
    df[f'{target}_trend'] = df[target] - df[f'{target}_rolling30']

    # Cambio porcentual diario y semanal
    df[f'{target}_pct_change'] = df[target].pct_change()
    df[f'{target}_pct_change_7d'] = df[target].pct_change(7)

    # Días especiales del mes
    df['es_primer_dia_mes'] = (df['dia_mes'] == 1).astype(int)
    df['es_ultimo_dia_mes'] = (df['FECHA'].dt.is_month_end).astype(int)

    return df


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features basadas en día de semana y mes"""
    # Promedios históricos por día de semana
    for target in ['T_AO', 'T_AO_VENTA', 'T_VISITAS']:
        if target in df.columns:
            df[f'mean_{target}_weekday'] = df.groupby('dia_semana')[target].transform('mean')
            df[f'median_{target}_weekday'] = df.groupby('dia_semana')[target].transform('median')

    # Feriados chilenos
    cl_holidays = holidays.CountryHoliday('CL')
    df['es_feriado'] = df['FECHA'].apply(lambda x: x in cl_holidays).astype(int)

    return df


def _add_window_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Añade features de ventanas móviles"""
    windows = [3, 7, 14, 30]
    for window in windows:
        df[f'{target}_rolling{window}_mean'] = df[target].rolling(window, min_periods=1).mean()
        df[f'{target}_rolling{window}_std'] = df[target].rolling(window, min_periods=1).std()

    # Diferencia entre ventanas cortas y largas
    df[f'{target}_diff_rolling7_30'] = df[f'{target}_rolling7_mean'] - df[f'{target}_rolling30_mean']

    return df


def _add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features cíclicas para día de semana y mes"""
    # Codificación cíclica para día de semana
    df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)

    # Codificación cíclica para mes
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

    return df


def _add_external_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features externas (simuladas)"""
    # Días de pago (simulado - días 5 y 20 de cada mes)
    df['es_dia_pago'] = df['dia_mes'].isin([5, 20]).astype(int)

    # Eventos comerciales (simulado - últimos 5 días de cada mes)
    df['es_evento_comercial'] = (df['dia_mes'] >= 25).astype(int)

    return df


def _add_lag_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Añade features de lag"""
    lags = [1, 2, 3, 7, 14, 30]
    for lag in lags:
        df[f'{target}_lag{lag}'] = df[target].shift(lag)

    return df


def _normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalización de features numéricas"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_normalize = [col for col in numeric_cols if col not in ['COD_SUC', 'es_feriado', 'es_fin_de_semana',
                                                                    'es_primer_dia_mes', 'es_ultimo_dia_mes',
                                                                    'es_dia_pago', 'es_evento_comercial']]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in cols_to_normalize:
        if df[col].isnull().any():
            if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                df[col].fillna(df[col].median(), inplace=True)
            else: # para el caso de que la columna sea de otro tipo y tenga NaN
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)


    scaler = MinMaxScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

    return df