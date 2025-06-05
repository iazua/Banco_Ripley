import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pyodbc
import holidays
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Configuración inicial
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Predicción de Visitas por Sucursal",
    layout="wide",
    page_icon="🔮"
)


# ----------- CLASES PERSONALIZADAS -----------
class NonNegativeXGB(XGBRegressor):
    def predict(self, X):
        preds = super().predict(X)
        return np.maximum(preds, 0).round().astype(int)

# ----------- FUNCIONES UTILITARIAS -----------
def es_feriado(fecha):
    return fecha in holidays.Chile(years=fecha.year)

def crear_features_temporales(df):
    df = df.copy()
    df['DIA_SEMANA'] = df['FECHA'].dt.dayofweek
    df['MES'] = df['FECHA'].dt.month
    df['DIA'] = df['FECHA'].dt.day
    df['AÑO'] = df['FECHA'].dt.year
    df['SEMANA_DEL_AÑO'] = df['FECHA'].dt.isocalendar().week
    df['TRIMESTRE'] = df['FECHA'].dt.quarter
    df['ES_FIN_DE_SEMANA'] = df['DIA_SEMANA'].isin([5, 6]).astype(int)
    df['ES_FERIADO'] = df['FECHA'].apply(es_feriado).astype(int)
    df['ES_HABIL'] = ((df['ES_FIN_DE_SEMANA'] == 0) & (df['ES_FERIADO'] == 0)).astype(int)
    df['HORA_SIN'] = np.sin(2 * np.pi * df['HORA'] / 24)
    df['HORA_COS'] = np.cos(2 * np.pi * df['HORA'] / 24)
    df['MES_SIN'] = np.sin(2 * np.pi * df['MES'] / 12)
    df['MES_COS'] = np.cos(2 * np.pi * df['MES'] / 12)
    df['HORA_FIN_SEMANA'] = df['HORA'] * df['ES_FIN_DE_SEMANA']
    df['HORA_FERIADO'] = df['HORA'] * df['ES_FERIADO']
    return df

def crear_features_series_temporales(df):
    df = df.copy().sort_values(['FECHA', 'HORA'])
    df['TIMESTAMP'] = df['FECHA'] + pd.to_timedelta(df['HORA'], unit='h')
    df = df.sort_values('TIMESTAMP')

    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f'LAG_{lag}_DIA'] = df['VISITAS'].shift(lag * 13)

    for window in [3, 7, 14, 21]:
        df[f'ROLLING_MEAN_{window}D'] = df['VISITAS'].rolling(window * 13, min_periods=1).mean()
        df[f'ROLLING_STD_{window}D'] = df['VISITAS'].rolling(window * 13, min_periods=1).std()
        df[f'ROLLING_MAX_{window}D'] = df['VISITAS'].rolling(window * 13, min_periods=1).max()

    df = df.fillna(method='bfill').fillna(0)
    return df.drop(columns=['TIMESTAMP'])

@st.cache_data(show_spinner=False)
def cargar_datos():
    conn_str = (
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=192.168.192.21;'
        'DATABASE=Auris_Personas;'
        'UID=iazuaz;'
        'PWD=192776775;'
    )
    query = "SELECT COD_SUC, FECHA, HORA, CANAL FROM TB_SALTA_OFERTA_SALON_DIARIO_MODELO_PY"
    conn = pyodbc.connect(conn_str)
    df = pd.read_sql(query, conn)
    conn.close()

    df.columns = df.columns.str.strip().str.upper()
    df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
    df['HORA'] = pd.to_numeric(df['HORA'], errors='coerce').astype('Int64')
    return df.dropna(subset=['COD_SUC', 'FECHA', 'HORA'])

def preparar_datos(df, sucursal):
    df = df[(df['COD_SUC'] == sucursal) & (df['HORA'].between(9, 21))]
    if df.empty:
        st.error("No hay datos disponibles para la sucursal seleccionada.")
        return None, None, None

    visitas = df.groupby(['FECHA', 'HORA']).size().reset_index(name='VISITAS')
    visitas = crear_features_temporales(visitas)
    visitas = crear_features_series_temporales(visitas)

    if len(visitas) < 100:
        st.warning("Datos insuficientes para modelar.")
        return None, None, None

    features = ['HORA', 'DIA_SEMANA', 'MES', 'DIA', 'AÑO', 'SEMANA_DEL_AÑO', 'TRIMESTRE',
                'ES_FIN_DE_SEMANA', 'ES_FERIADO', 'ES_HABIL', 'HORA_SIN', 'HORA_COS',
                'MES_SIN', 'MES_COS', 'HORA_FIN_SEMANA', 'HORA_FERIADO'] + \
                [f'LAG_{i}_DIA' for i in [1, 2, 3, 7, 14, 21, 28]] + \
                [f'ROLLING_MEAN_{i}D' for i in [3, 7, 14, 21]] + \
                [f'ROLLING_STD_{i}D' for i in [3, 7, 14, 21]] + \
                [f'ROLLING_MAX_{i}D' for i in [3, 7, 14, 21]]

    return visitas[features], visitas['VISITAS'], visitas

def evaluar_overfitting(modelo, X_train, y_train, X_test, y_test):
    st.subheader("🔍 Diagnóstico de Overfitting")
    train_score = modelo.score(X_train, y_train)
    test_score = modelo.score(X_test, y_test)
    col1, col2 = st.columns(2)
    col1.metric("R² Entrenamiento", f"{train_score:.3f}")
    col2.metric("R² Prueba", f"{test_score:.3f}")
    if train_score > test_score + 0.15:
        st.warning("Overfitting detectado: diferencia significativa entre entrenamiento y prueba")
    else:
        st.success("Modelo generaliza correctamente")

def plot_learning_curve(modelo, X, y):
    st.subheader("📉 Curva de Aprendizaje")
    train_sizes, train_scores, test_scores = learning_curve(
        modelo, X, y, cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_absolute_error')

    train_mean = -np.mean(train_scores, axis=1)
    test_mean = -np.mean(test_scores, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode='lines+markers', name='Entrenamiento'))
    fig.add_trace(go.Scatter(x=train_sizes, y=test_mean, mode='lines+markers', name='Validación'))
    fig.update_layout(title='Curva de Aprendizaje', xaxis_title='Cantidad de Muestras', yaxis_title='MAE')
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(modelo, X, y):  # <-- Añadir y como parámetro
    """Muestra importancia de características"""
    st.subheader("📌 Importancia de Características")

    # Importancia basada en ganancia
    if hasattr(modelo.named_steps['xgbregressor'], 'feature_importances_'):
        importances = modelo.named_steps['xgbregressor'].feature_importances_
        features = X.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        importance_df = importance_df.sort_values('Importance', ascending=False).head(20)

        fig = px.bar(importance_df, x='Importance', y='Feature',
                     title='Importancia de Características (Ganancia)',
                     labels={'Importance': 'Importancia', 'Feature': 'Característica'})
        st.plotly_chart(fig, use_container_width=True)

    # Importancia por permutación
    with st.spinner("Calculando importancia por permutación..."):
        result = permutation_importance(
            modelo, X.tail(500), y.tail(500),  # <-- Ahora y está definido
            n_repeats=10, random_state=42, n_jobs=-1)

        perm_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': result.importances_mean,
            'Std': result.importances_std
        }).sort_values('Importance', ascending=False).head(15)

        fig = px.bar(perm_importance_df, x='Importance', y='Feature',
                     error_x='Std',
                     title='Importancia de Características (Permutación)',
                     labels={'Importance': 'Importancia', 'Feature': 'Característica'})
        st.plotly_chart(fig, use_container_width=True)

def entrenar_modelo(X, y, sucursal):
    """Entrena el modelo XGBoost con validación cruzada temporal"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)


    tscv = TimeSeriesSplit(n_splits=11)
    params = {
        'xgbregressor__max_depth': [1, 2, 3, 4],
        'xgbregressor__learning_rate': [0.01, 0.05, 0.1],
        'xgbregressor__n_estimators': [10, 30, 50, 100, 200],
        'xgbregressor__subsample': [0.5, 0.8, 1.0],
        'xgbregressor__colsample_bytree': [0.5, 0.8, 1.0]
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgbregressor', NonNegativeXGB(objective='reg:squarederror', random_state=42))
    ])

    grid = GridSearchCV(pipeline, param_grid=params, scoring='neg_mean_absolute_error', cv=tscv, n_jobs=-1)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)

    # Métricas de evaluación
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Mostrar métricas
    st.subheader("📊 Métricas de Evaluación")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("R²", f"{r2:.2f}")

    # Gráfico de comparación real vs predicho
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test.values, mode='lines', name='Real', line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicho', line=dict(color='orange')))
    fig.update_layout(
        title="Comparación Real vs Predicción (Conjunto de Prueba)",
        xaxis_title="Índice",
        yaxis_title="Visitas",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True, key=f"comparacion_{sucursal}")

    # Evaluar overfitting y mostrar diagnósticos
    evaluar_overfitting(grid.best_estimator_, X_train, y_train, X_test, y_test)
    plot_learning_curve(grid.best_estimator_, X, y)
    plot_feature_importance(grid.best_estimator_, X, y)

    return grid.best_estimator_


def hacer_proyecciones(modelo, df_modelo, X, y, sucursal):
    """Genera y muestra proyecciones futuras"""
    st.subheader("📈 Proyecciones Futuras")

    # Configuración de parámetros
    dias_prediccion = st.slider("Selecciona días a predecir:", 7, 90, 35, key=f"slider_{sucursal}")
    horas_pred = sorted(df_modelo['HORA'].dropna().unique().tolist())
    ultima_fecha = pd.to_datetime(df_modelo['FECHA'].max())

    # Generar fechas futuras
    fechas_futuras = pd.date_range(
        start=ultima_fecha + pd.Timedelta(days=1),
        periods=dias_prediccion
    )

    # Crear DataFrame para predicciones
    df_futuro = pd.DataFrame(
        [(fecha, hora) for fecha in fechas_futuras for hora in horas_pred],
        columns=['FECHA', 'HORA']
    )

    # Crear características temporales
    df_futuro = crear_features_temporales(df_futuro)

    # Concatenar histórico y futuro para calcular lags y rolling
    historico = df_modelo.copy().sort_values(['FECHA', 'HORA'])
    historico = pd.concat([historico, df_futuro], ignore_index=True)
    historico = crear_features_series_temporales(historico)

    # Filtrar solo fechas futuras para predecir
    df_futuro = historico[historico['FECHA'].isin(fechas_futuras)].copy()
    X_futuro = df_futuro[X.columns]

    # Predecir visitas (asegurando no negativos y enteros)
    y_pred_futuro = modelo.predict(X_futuro)
    df_futuro['VISITAS_PREDICHAS'] = np.maximum(y_pred_futuro, 0).astype(int)

    # Preparar datos para mostrar
    df_tabla = df_futuro[['FECHA', 'HORA', 'VISITAS_PREDICHAS']].copy()
    df_tabla = df_tabla.sort_values(['FECHA', 'HORA'])
    df_tabla['FECHA'] = df_tabla['FECHA'].dt.strftime('%Y-%m-%d')

    # Mostrar tabla de predicciones
    st.markdown(f"### 📋 Predicciones por hora - Sucursal {sucursal}")
    st.dataframe(df_tabla, use_container_width=True, height=300)

    # Agrupar por día para gráfico
    df_plot_diario = df_futuro.groupby('FECHA', as_index=False)['VISITAS_PREDICHAS'].sum()

    # Gráfico combinado histórico + predicción
    st.markdown("### 📊 Histórico y Predicción de Visitas")

    # Preparar datos históricos
    df_historico = df_modelo.groupby('FECHA', as_index=False)['VISITAS'].sum()

    # SOLUCIÓN DEFINITIVA: Convertir todas las fechas a milisegundos desde epoch
    def date_to_ms(dt):
        return int(dt.timestamp() * 1000)

    # Convertir fechas
    fechas_historico = df_historico['FECHA'].apply(date_to_ms)
    fechas_prediccion = df_plot_diario['FECHA'].apply(date_to_ms)
    ultima_fecha_ms = date_to_ms(ultima_fecha)

    # Crear figura
    fig_combinado = go.Figure()

    # Añadir histórico
    fig_combinado.add_trace(go.Scatter(
        x=fechas_historico,
        y=df_historico['VISITAS'],
        name='Histórico',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Visitas: %{y}'
    ))

    # Añadir predicción
    fig_combinado.add_trace(go.Scatter(
        x=fechas_prediccion,
        y=df_plot_diario['VISITAS_PREDICHAS'],
        name='Predicción',
        line=dict(color='#ff7f0e', width=2, dash='dot'),
        hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Visitas predichas: %{y}'
    ))

    # Añadir línea vertical usando milisegundos
    fig_combinado.add_vline(
        x=ultima_fecha_ms,
        line_dash="dash",
        line_color="green",
        annotation_text="Inicio Predicción",
        annotation_position="top right",
        annotation_font_size=12,
        annotation_font_color="green"
    )

    # Configurar layout con formato de fechas
    fig_combinado.update_layout(
        title=f"Histórico y Predicción de Visitas Diarias - Sucursal {sucursal}",
        xaxis_title="Fecha",
        yaxis_title="Visitas",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(
            type='date',
            tickformat='%Y-%m-%d'
        )
    )

    st.plotly_chart(fig_combinado, use_container_width=True)

    # Gráfico de distribución horaria para el primer día proyectado
    primer_dia = fechas_futuras[0]
    df_horas = df_futuro[df_futuro['FECHA'] == primer_dia]

    fig_horas = px.bar(
        df_horas,
        x='HORA',
        y='VISITAS_PREDICHAS',
        title=f"🕒 Distribución Horaria Predicha - {primer_dia.strftime('%Y-%m-%d')}",
        labels={'HORA': 'Hora del día', 'VISITAS_PREDICHAS': 'Visitas predichas'},
        color='VISITAS_PREDICHAS',
        color_continuous_scale='Blues',
        text='VISITAS_PREDICHAS'
    )
    fig_horas.update_layout(
        template="plotly_white",
        xaxis=dict(tickmode='linear', dtick=1),
        yaxis=dict(title='Visitas predichas'),
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    fig_horas.update_traces(
        texttemplate='%{text}',
        textposition='outside'
    )
    st.plotly_chart(fig_horas, use_container_width=True)

# ------------------------------
# UI Principal
# ------------------------------
def main():
    st.title("🔮 Predicción de Visitas por Sucursal")

    # Cargar datos
    with st.spinner("Cargando datos..."):
        df = cargar_datos()

    if df is not None:
        # Selección de sucursal
        sucursales = sorted(df['COD_SUC'].unique())
        sucursal = st.selectbox("Selecciona una sucursal:", options=sucursales)

        if sucursal:
            with st.spinner("Preparando datos..."):
                X, y, df_modelo = preparar_datos(df, sucursal)

            if X is not None:
                with st.spinner("Entrenando modelo..."):
                    modelo_entrenado = entrenar_modelo(X, y, sucursal)

                hacer_proyecciones(modelo_entrenado, df_modelo, X, y, sucursal)


if __name__ == "__main__":
    main()