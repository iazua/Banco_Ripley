import pandas as pd
import numpy as np
import holidays


def prepare_features(df: pd.DataFrame, target: str, is_prediction: bool = False):
    """
    Prepara las features para entrenamiento de modelos predictivos o para predicción futura.
    Incluye variables temporales, rezagos y rolling averages.

    Args:
        df (pd.DataFrame): DataFrame de entrada con los datos.
        target (str): Nombre de la columna objetivo.
        is_prediction (bool): True si se está preparando datos para predicción futura,
                              False para entrenamiento.

    Returns:
        tuple: (X, y) donde X es el DataFrame de features e y es la Serie objetivo.
               Si is_prediction es True, y será una Serie de NaNs o un valor placeholder.
    """
    df = df.copy()

    # Asegurarse de que 'FECHA' sea datetime y ordenar
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df = df.sort_values("FECHA").reset_index(drop=True)

    # Variables temporales
    df["year"] = df["FECHA"].dt.year
    df["month"] = df["FECHA"].dt.month
    df["day"] = df["FECHA"].dt.day
    df["weekday"] = df["FECHA"].dt.weekday  # 0=Lunes, 6=Domingo
    df["dayofyear"] = df["FECHA"].dt.dayofyear
    df["weekofyear"] = df["FECHA"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["weekday"] >= 5

    # Añadir feriados chilenos (considerando un rango amplio de años)
    min_year = df['FECHA'].min().year - 1 if not df.empty else 2020
    max_year = df['FECHA'].max().year + 2 if not df.empty else 2025
    chile_holidays = holidays.Chile(years=range(min_year, max_year))
    df['is_holiday'] = df['FECHA'].apply(lambda x: x.date() in chile_holidays)

    # Columnas para rezagos y rolling averages
    # Incluimos explícitamente T_AO, T_AO_VENTA, T_VISITAS, DOTACION para los lags
    lag_cols = ["T_AO", "T_AO_VENTA", "T_VISITAS", "DOTACION"]

    for col in lag_cols:
        # Asegurarse de que las columnas existan, rellenando con 0 o NaN si no
        if col not in df.columns:
            df[col] = np.nan  # Inicializar si la columna no existe

        # Shift(1) para evitar data leakage, asegurando que solo usamos información pasada
        # Usamos .fillna(0) para las primeras filas después del shift si es numérico
        df[f"{col}_lag1"] = df[col].shift(1)
        # Considerar un min_periods de 1 para los rolling averages
        # Aseguramos que el rolling se calcule sobre los valores ya existentes o históricos
        df[f"{col}_rolling7"] = df[col].shift(1).rolling(window=7, min_periods=1).mean()
        df[f"{col}_rolling14"] = df[col].shift(1).rolling(window=14, min_periods=1).mean()

    # Definir las características que se usarán en el modelo
    features = [
        "year", "month", "day", "weekday", "dayofyear", "weekofyear",
        "is_weekend", "is_holiday"
    ]

    for col in lag_cols:
        features.extend([f"{col}_lag1", f"{col}_rolling7", f"{col}_rolling14"])

    # Asegurarse de que todas las columnas de características existan en el DataFrame antes de seleccionarlas
    for feat in features:
        if feat not in df.columns:
            df[feat] = np.nan  # Rellenar con NaN si no existen

    # Si estamos en modo de predicción, los valores 'y' no son conocidos.
    if is_prediction:
        X = df[features].copy()
        # Rellenar NaNs en las features. Para variables numéricas, podríamos usar 0 o una estrategia más avanzada.
        # Para booleanos, False es un buen valor por defecto.
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(0)  # Rellenar NaNs en features numéricas con 0
            elif pd.api.types.is_bool_dtype(X[col]):
                X[col] = X[col].fillna(False)
        y = pd.Series([np.nan] * len(df), index=df.index)  # El target es desconocido
    else:
        # Para entrenamiento, eliminamos las filas con NaNs en las características y el target
        # Esto asegura que el modelo se entrene solo con datos completos y válidos.
        # Es crucial que target exista en df para el entrenamiento
        if target not in df.columns:
            raise ValueError(f"La columna objetivo '{target}' no se encuentra en el DataFrame para el entrenamiento.")

        df_cleaned = df.dropna(subset=features + [target])
        X = df_cleaned[features]
        y = df_cleaned[target]

    return X, y
