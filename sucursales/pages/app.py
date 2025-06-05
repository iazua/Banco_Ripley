import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from ngboost import NGBRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
import holidays
import pyodbc
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.ensemble import IsolationForest, VotingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import time
from utils.sidebar import set_sucursal_filter
import warnings

warnings.filterwarnings('ignore')

# -------------------- CONFIGURACIÓN DE PÁGINA --------------------
st.set_page_config(page_title="Predicción ACEPTA_OFERTA", layout="wide", page_icon="📊")

# -------------------- ESTILOS CSS PERSONALIZADOS --------------------
st.markdown("""
    <style>
    .metric-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f2f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-title {
        font-size: 14px;
        color: #555;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
    }
    .tab-content {
        padding: 20px 0;
    }
    .plot-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------- FUNCIONES DE UTILIDAD --------------------
def create_cyclical_features(df):
    """Crear características cíclicas para variables temporales."""
    df = df.copy()

    # Features cíclicos para hora
    df['HORA_SIN'] = np.sin(2 * np.pi * df['HORA'] / 24)
    df['HORA_COS'] = np.cos(2 * np.pi * df['HORA'] / 24)

    # Features cíclicos para día de la semana
    df['DIA_SEMANA_SIN'] = np.sin(2 * np.pi * df['DIA_SEMANA'] / 7)
    df['DIA_SEMANA_COS'] = np.cos(2 * np.pi * df['DIA_SEMANA'] / 7)

    # Features cíclicos para mes
    df['MES_SIN'] = np.sin(2 * np.pi * df['MES'] / 12)
    df['MES_COS'] = np.cos(2 * np.pi * df['MES'] / 12)

    return df


def create_lag_features(df, windows=[3, 7, 14, 30], lags=[1, 2, 3, 7]):
    """Crear características de rezago y ventana móvil."""
    df = df.copy()

    # Características de rezago
    for lag in lags:
        df[f'LAG_{lag}'] = df.groupby('COD_SUC')['ACEPTA_OFERTA'].shift(lag)

    # Características de ventana móvil
    for window in windows:
        df[f'ROLLING_MEAN_{window}'] = df.groupby('COD_SUC')['ACEPTA_OFERTA'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean())
        df[f'ROLLING_STD_{window}'] = df.groupby('COD_SUC')['ACEPTA_OFERTA'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std())
        df[f'ROLLING_MIN_{window}'] = df.groupby('COD_SUC')['ACEPTA_OFERTA'].transform(
            lambda x: x.rolling(window=window, min_periods=1).min())
        df[f'ROLLING_MAX_{window}'] = df.groupby('COD_SUC')['ACEPTA_OFERTA'].transform(
            lambda x: x.rolling(window=window, min_periods=1).max())

    return df


def create_interaction_features(df):
    """Crear características de interacción."""
    df = df.copy()

    # Interacciones hora-día
    df['HORA_DIA'] = df['HORA'] * df['DIA_SEMANA']
    df['HORA_MES'] = df['HORA'] * df['MES']

    # Interacciones con variables de negocio
    df['PICO_FIN_SEMANA'] = df['HORA_PICO'] * df['FIN_DE_SEMANA']
    df['TARDE_FIN_SEMANA'] = df['HORA_TARDE'] * df['FIN_DE_SEMANA']

    return df


# -------------------- CONEXIÓN SQL SERVER --------------------
@st.cache_data(ttl=3600)  # Cache por 1 hora
def cargar_datos():
    try:
        conn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=192.168.192.21;'
            'DATABASE=Auris_Personas;'
            'UID=iazuaz;'
            'PWD=192776775',
            timeout=30
        )

        query = """
                SELECT COD_SUC, \
                       FECHA, \
                       CANAL, \
                       HORA, \
                       ACEPTA_OFERTA
                FROM TB_SALTA_OFERTA_SALON_DIARIO_MODELO_PY
                    \
                """

        df = pd.read_sql(query, conn)
        df['FECHA'] = pd.to_datetime(df['FECHA'], format='%d-%m-%Y', errors='coerce')
        df['HORA'] = df['HORA'].astype(int)
        df.dropna(subset=['FECHA'], inplace=True)

        conn.close()
        return df

    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return None


# -------------------- CARGA INICIAL DE DATOS --------------------
df = cargar_datos()
if df is None:
    st.error("No se pudieron cargar los datos. Por favor, verifica la conexión.")
    st.stop()

# Sidebar con filtro transversal
set_sucursal_filter(df)

# Filtrar según selección
sucursal = st.session_state.get("COD_SUC")
df_filtrado = df[df["COD_SUC"] == sucursal]

# -------------------- INTERFAZ DE USUARIO --------------------
st.title("📊 Predicción Avanzada de ACEPTA_OFERTA por Tramo Horario")

# Configuración del modelo
col1, col2, col3 = st.columns(3)
sucursal_seleccionada = col1.selectbox("Selecciona una sucursal", sorted(df['COD_SUC'].unique()))
modelo_seleccionado = col2.selectbox("Selecciona el modelo", ["NGBoost", "TabNet", "Stacking", "Ensemble"])
dias_prediccion = col3.selectbox("Horizonte de predicción (días)", [7, 14, 21, 30])


# -------------------- PREPARACIÓN Y FILTRADO DE DATOS --------------------
@st.cache_data(ttl=3600)
def preparar_datos(df_suc, fecha_corte):
    df_suc = df_suc.copy()

    # Ordenar datos
    df_suc.sort_values(['FECHA', 'HORA'], inplace=True)

    # Filtrar por fecha
    df_suc = df_suc[df_suc['FECHA'] > fecha_corte].copy()

    # Crear características temporales básicas
    chile_holidays = holidays.Chile(years=[2023, 2024, 2025])
    df_suc['DIA_SEMANA'] = df_suc['FECHA'].dt.weekday
    df_suc['DIA_DEL_MES'] = df_suc['FECHA'].dt.day
    df_suc['MES'] = df_suc['FECHA'].dt.month
    df_suc['TRIMESTRE'] = df_suc['FECHA'].dt.quarter
    df_suc['SEMANA_DEL_ANO'] = df_suc['FECHA'].dt.isocalendar().week
    df_suc['DIA_DEL_ANO'] = df_suc['FECHA'].dt.dayofyear

    # Características de negocio mejoradas
    df_suc['DIA_LABORAL'] = df_suc['DIA_SEMANA'].apply(lambda x: 1 if x < 5 else 0)
    df_suc['FIN_DE_SEMANA'] = df_suc['DIA_SEMANA'].apply(lambda x: 1 if x >= 5 else 0)
    df_suc['ES_FERIADO'] = df_suc['FECHA'].dt.date.isin(chile_holidays).astype(int)
    df_suc['ES_VISPERA_FERIADO'] = (df_suc['FECHA'] + timedelta(days=1)).dt.date.isin(chile_holidays).astype(int)

    # Características de hora mejoradas
    df_suc['HORA_PICO'] = df_suc['HORA'].apply(lambda x: 1 if 12 <= x <= 14 else 0)
    df_suc['HORA_TARDE'] = df_suc['HORA'].apply(lambda x: 1 if 15 <= x <= 18 else 0)
    df_suc['HORA_MANANA'] = df_suc['HORA'].apply(lambda x: 1 if 9 <= x <= 11 else 0)

    # Aplicar transformaciones avanzadas
    df_suc = create_cyclical_features(df_suc)
    df_suc = create_lag_features(df_suc)
    df_suc = create_interaction_features(df_suc)

    # Características de tendencia mejoradas
    ventanas = [3, 7, 14, 30]
    for ventana in ventanas:
        df_suc[f'TENDENCIA_{ventana}'] = df_suc['ACEPTA_OFERTA'].rolling(window=ventana).mean() - \
                                         df_suc['ACEPTA_OFERTA'].rolling(window=ventana).mean().shift(1)

    # Eliminar valores nulos
    df_suc.dropna(inplace=True)

    return df_suc


# Preparar datos
df_suc = df[df['COD_SUC'] == sucursal_seleccionada].copy()
fecha_corte = df_suc['FECHA'].max() - timedelta(days=180)  # Aumentamos el período de entrenamiento
df_suc = preparar_datos(df_suc, fecha_corte)

# -------------------- FEATURE ENGINEERING AVANZADO --------------------
# Definir características para el modelo
features_base = [
    'HORA', 'DIA_LABORAL', 'FIN_DE_SEMANA', 'DIA_SEMANA', 'DIA_DEL_MES',
    'MES', 'TRIMESTRE', 'ES_FERIADO', 'ES_VISPERA_FERIADO', 'HORA_PICO',
    'HORA_TARDE', 'HORA_MANANA'
]

features_ciclicos = [
    'HORA_SIN', 'HORA_COS', 'DIA_SEMANA_SIN', 'DIA_SEMANA_COS',
    'MES_SIN', 'MES_COS'
]

features_lag = [f'LAG_{i}' for i in [1, 2, 3, 7]] + \
               [f'ROLLING_MEAN_{i}' for i in [3, 7, 14, 30]] + \
               [f'ROLLING_STD_{i}' for i in [3, 7, 14, 30]] + \
               [f'TENDENCIA_{i}' for i in [3, 7, 14, 30]]

features_interaccion = [
    'HORA_DIA', 'HORA_MES', 'PICO_FIN_SEMANA', 'TARDE_FIN_SEMANA'
]

# Combinar todas las características
features = features_base + features_ciclicos + features_lag + features_interaccion

# Preparar datos para el modelo
X = df_suc[features]
y = df_suc['ACEPTA_OFERTA']

# -------------------- VALIDACIÓN Y TRANSFORMACIONES --------------------
# Dividir datos temporalmente
train_size = int(len(df_suc) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Pipeline de preprocesamiento
preprocessor = Pipeline([
    ('scaler', RobustScaler()),
    ('power_transform', PowerTransformer(method='yeo-johnson')),
])

# Aplicar transformaciones
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# -------------------- CONFIGURACIÓN DE MODELOS --------------------
def configurar_modelos():
    # Modelo 1: NGBoost (Natural Gradient Boosting)
    ngboost_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'minibatch_frac': [0.5, 0.7, 1.0],
        'col_sample': [0.8, 0.9, 1.0]
    }

    # Modelo 2: TabNet (Deep Learning para datos tabulares)
    tabnet_params = {
        'n_d': [8, 16, 32],  # Dimensión de la capa de decisión
        'n_a': [8, 16, 32],  # Dimensión de la capa de atención
        'n_steps': [3, 5, 7],  # Número de pasos de decisión
        'gamma': [1.0, 1.3, 1.5],  # Factor de regularización
        'lambda_sparse': [0.001, 0.01, 0.1]  # Parámetro de esparsidad
    }

    # Modelo base 3 (para stacking): Random Forest avanzado
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'bootstrap': [True, False],
        'max_features': ['sqrt', 'log2', None]
    }

    # Modelo base 4 (para stacking): Gradient Boosting avanzado
    gb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }

    # Configurar los modelos base para el Stacking
    base_models = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('gb', GradientBoostingRegressor(random_state=42))
    ]

    # Modelo 3: StackingRegressor (modelo de ensamble avanzado)
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=GradientBoostingRegressor(random_state=42),
        cv=5
    )

    stacking_params = {
        'final_estimator__n_estimators': [100, 200],
        'final_estimator__learning_rate': [0.01, 0.05, 0.1]
    }

    return {
        'NGBoost': (NGBRegressor(random_state=42), ngboost_params),
        'TabNet': (TabNetRegressor(), tabnet_params),
        'Stacking': (stacking_model, stacking_params)
    }


# -------------------- ENTRENAMIENTO DEL MODELO --------------------
st.subheader("🔍 Entrenando modelo avanzado...")
progress_bar = st.progress(0)

# Configurar y entrenar modelo
modelos = configurar_modelos()

if modelo_seleccionado == "Ensemble":
    # Crear ensemble de modelos
    estimators = []
    for name, (model, _) in modelos.items():
        estimators.append((name, model))

    model = VotingRegressor(estimators=estimators)
    model.fit(X_train_processed, y_train)
    best_model = model
else:
    base_model, param_grid = modelos[modelo_seleccionado]

    # Búsqueda de hiperparámetros con validación cruzada temporal
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )

    # Simulación de progreso
    for i in range(1, 6):
        progress_bar.progress(i * 20)
        time.sleep(0.2)

    grid.fit(X_train_processed, y_train.values.reshape(-1, 1))
    best_model = grid.best_estimator_

progress_bar.progress(100)

# Predicciones y evaluación
y_pred = best_model.predict(X_test_processed)

# Métricas de evaluación
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Métricas específicas para picos
picos_mask = y_test > np.percentile(y_test, 90)
mae_picos = mean_absolute_error(y_test[picos_mask], y_pred[picos_mask])
r2_picos = r2_score(y_test[picos_mask], y_pred[picos_mask])

# -------------------- VISUALIZACIÓN DE RESULTADOS --------------------
st.success("✅ Entrenamiento completado")

# Crear pestañas
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Predicciones vs Real",
    "📊 Métricas y Rendimiento",
    "🔍 Análisis de Residuos",
    "📌 Importancia de Variables",
    "🔄 Comparación de Modelos"
])

with tab1:
    st.subheader("Comparación de Predicciones vs Valores Reales")

    results_df = pd.DataFrame({
        'Fecha': df_suc.iloc[train_size:]['FECHA'],
        'Hora': df_suc.iloc[train_size:]['HORA'],
        'Real': y_test,
        'Predicción': y_pred,
        'Residuo': y_test - y_pred
    })

    # Gráfico de líneas con bandas de confianza
    fig = go.Figure()

    # Valores reales
    fig.add_trace(go.Scatter(
        x=results_df['Fecha'],
        y=results_df['Real'],
        name='Real',
        line=dict(color='blue')
    ))

    # Predicciones
    fig.add_trace(go.Scatter(
        x=results_df['Fecha'],
        y=results_df['Predicción'],
        name='Predicción',
        line=dict(color='red')
    ))

    # Bandas de confianza
    std_pred = results_df['Predicción'].std()
    fig.add_trace(go.Scatter(
        x=results_df['Fecha'],
        y=results_df['Predicción'] + 1.96 * std_pred,
        fill=None,
        mode='lines',
        line_color='rgba(255,0,0,0.1)',
        name='Intervalo Superior'
    ))

    fig.add_trace(go.Scatter(
        x=results_df['Fecha'],
        y=results_df['Predicción'] - 1.96 * std_pred,
        fill='tonexty',
        mode='lines',
        line_color='rgba(255,0,0,0.1)',
        name='Intervalo Inferior'
    ))

    fig.update_layout(
        title='Predicciones vs Valores Reales con Intervalos de Confianza',
        xaxis_title='Fecha',
        yaxis_title='ACEPTA_OFERTA',
        height=600,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Métricas de Rendimiento del Modelo")

    col1, col2, col3, col4 = st.columns(4)

    # Métricas generales
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">MAE General</div>
                <div class="metric-value">{mae:.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">RMSE</div>
                <div class="metric-value">{rmse:.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">R² General</div>
                <div class="metric-value">{r2:.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">R² en Picos</div>
                <div class="metric-value">{r2_picos:.2f}</div>
            </div>
        """, unsafe_allow_html=True)

with tab3:
    st.subheader("Análisis de Residuos")

    # Gráfico de residuos
    fig = make_subplots(rows=2, cols=1)

    # Residuos vs Predicciones
    fig.add_trace(
        go.Scatter(
            x=results_df['Predicción'],
            y=results_df['Residuo'],
            mode='markers',
            name='Residuos'
        ),
        row=1, col=1
    )

    # Distribución de residuos
    fig.add_trace(
        go.Histogram(
            x=results_df['Residuo'],
            name='Distribución de Residuos'
        ),
        row=2, col=1
    )

    fig.update_layout(height=800, title_text="Análisis de Residuos")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Importancia de Variables")

    if hasattr(best_model, 'feature_importances_'):
        importances = pd.DataFrame({
            'Feature': features,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig = px.bar(importances, x='Importance', y='Feature',
                     orientation='h',
                     title='Importancia de Características')
        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Este modelo no proporciona importancia de características")

with tab5:
    st.subheader("Comparación entre Modelos")

    # Entrenar y evaluar todos los modelos
    model_metrics = []
    for name, (model, _) in modelos.items():
        model.fit(X_train_processed, y_train)
        y_pred_comp = model.predict(X_test_processed)

        model_metrics.append({
            'Modelo': name,
            'MAE': mean_absolute_error(y_test, y_pred_comp),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_comp)),
            'R²': r2_score(y_test, y_pred_comp)
        })

    metrics_df = pd.DataFrame(model_metrics)

    # Visualización comparativa
    fig = go.Figure()
    for metric in ['MAE', 'RMSE', 'R²']:
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df['Modelo'],
            y=metrics_df[metric]
        ))

    fig.update_layout(
        barmode='group',
        title='Comparación de Métricas entre Modelos',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(metrics_df)

# -------------------- PREDICCIÓN FUTURA --------------------
st.subheader("🔮 Predicción para los Próximos Días")

# Generar fechas futuras
ultima_fecha = df_suc['FECHA'].max()
fechas_futuras = pd.date_range(start=ultima_fecha + timedelta(days=1),
                               periods=dias_prediccion)

# Crear DataFrame para predicción
future_data = []
for fecha in fechas_futuras:
    for hora in range(9, 22):
        # Crear características para predicción futura
        data = {
            'FECHA': fecha,
            'HORA': hora,
            'DIA_SEMANA': fecha.weekday(),
            'DIA_DEL_MES': fecha.day,
            'MES': fecha.month,
            'TRIMESTRE': (fecha.month - 1) // 3 + 1,
            'ES_FERIADO': int(fecha.date() in chile_holidays),
            'ES_VISPERA_FERIADO': int((fecha + timedelta(days=1)).date() in chile_holidays),
            'HORA_PICO': 1 if 12 <= hora <= 14 else 0,
            'HORA_TARDE': 1 if 15 <= hora <= 18 else 0,
            'HORA_MANANA': 1 if 9 <= hora <= 11 else 0
        }

        # Agregar características cíclicas
        data.update({
            'HORA_SIN': np.sin(2 * np.pi * hora / 24),
            'HORA_COS': np.cos(2 * np.pi * hora / 24),
            'DIA_SEMANA_SIN': np.sin(2 * np.pi * data['DIA_SEMANA'] / 7),
            'DIA_SEMANA_COS': np.cos(2 * np.pi * data['DIA_SEMANA'] / 7),
            'MES_SIN': np.sin(2 * np.pi * data['MES'] / 12),
            'MES_COS': np.cos(2 * np.pi * data['MES'] / 12)
        })

        future_data.append(data)

future_df = pd.DataFrame(future_data)

# Preparar características para predicción
X_future = future_df[features]
X_future_processed = preprocessor.transform(X_future)
future_df['PREDICCION'] = best_model.predict(X_future_processed)

# Visualización de predicciones futuras
fig = px.line(future_df, x='FECHA', y='PREDICCION',
              color='HORA',
              title=f'Predicción de ACEPTA_OFERTA para los próximos {dias_prediccion} días')
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Mostrar tabla con predicciones
st.dataframe(future_df[['FECHA', 'HORA', 'PREDICCION']].sort_values(['FECHA', 'HORA']))

# -------------------- ANÁLISIS POR HORA --------------------
st.subheader("🕒 Análisis por Tramo Horario")

hourly_analysis = future_df.groupby('HORA').agg({
    'PREDICCION': ['mean', 'std', 'min', 'max']
}).reset_index()

hourly_analysis.columns = ['HORA', 'Media', 'Desv. Estándar', 'Mínimo', 'Máximo']

fig = go.Figure()
fig.add_trace(go.Bar(x=hourly_analysis['HORA'],
                     y=hourly_analysis['Media'],
                     name='Media',
                     error_y=dict(type='data',
                                  array=hourly_analysis['Desv. Estándar'])))

fig.update_layout(
    title='ACEPTA_OFERTA Promedio por Hora con Intervalos de Confianza',
    xaxis_title='Hora del Día',
    yaxis_title='ACEPTA_OFERTA Promedio',
    height=500
)

st.plotly_chart(fig, use_container_width=True)
st.dataframe(hourly_analysis)

# -------------------- EXPORTACIÓN DE RESULTADOS --------------------
st.subheader("📤 Exportar Resultados")

if st.button("Exportar Predicciones a CSV"):
    csv = future_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name=f"predicciones_{sucursal_seleccionada}_{modelo_seleccionado}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )

# -------------------- INFORMACIÓN ADICIONAL --------------------
with st.expander("ℹ️ Información del Modelo"):
    st.write("""
    ### Características del modelo:
    - **Período de entrenamiento**: Últimos 180 días
    - **Features utilizadas**: {} características
    - **Métricas en picos**: R² = {:.2f}, MAE = {:.2f}
    - **Mejor modelo**: {}
    """.format(len(features), r2_picos, mae_picos, modelo_seleccionado))

    if modelo_seleccionado != "Ensemble" and hasattr(best_model, 'get_params'):
        st.write("### Mejores hiperparámetros:")
        st.json(best_model.get_params())