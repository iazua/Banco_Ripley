import streamlit as st
import pandas as pd
import numpy as np
import holidays
from datetime import timedelta
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# --- CONFIGURACION ---
st.set_page_config(page_title="Predicción de Llamadas", layout="centered")

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    df = pd.read_pickle(r'C:\Users\iazuaz\PyCharmMiscProject\model_FRONT\data\data_limpia_llamadas.pkl')
    df = df.sort_values('Fecha').reset_index(drop=True)
    return df

df = load_data()

# --- FEATURES TEMPORALES ---
cl_holidays = holidays.Chile()

def crear_features(df):
    df['anio'] = df['Fecha'].dt.year
    df['mes'] = df['Fecha'].dt.month
    df['dia'] = df['Fecha'].dt.day
    df['dia_semana'] = df['Fecha'].dt.weekday
    df['fin_de_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
    df['es_feriado'] = df['Fecha'].isin(cl_holidays).astype(int)
    df['es_laboral'] = (~df['fin_de_semana'].astype(bool) & ~df['es_feriado'].astype(bool)).astype(int)
    return df

df = crear_features(df)

# --- ENTRENAMIENTO DEL MODELO ---
features = ['Es_Cyber', 'anio', 'mes', 'dia', 'dia_semana', 'fin_de_semana', 'es_feriado', 'es_laboral']
X = df[features]
y = df['Llamadas_Recibidas']

model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=7, random_state=42)
model.fit(X, y)

# --- PREDICCIÓN ---
st.title("📞 Predicción de Llamadas Recibidas")
st.markdown("Selecciona una fecha futura para proyectar los próximos **35 días** de llamadas recibidas.")

fecha_inicio = st.date_input("📅 Fecha de inicio de predicción", value=df['Fecha'].max() + timedelta(days=1), min_value=df['Fecha'].max() + timedelta(days=1))

if fecha_inicio:
    fechas_pred = pd.date_range(start=fecha_inicio, periods=35)
    df_futuro = pd.DataFrame({'Fecha': fechas_pred})
    df_futuro = crear_features(df_futuro)

    st.subheader("⚡ ¿Qué días serán Cyber?")
    fechas_str = df_futuro['Fecha'].dt.strftime('%Y-%m-%d').tolist()
    fechas_seleccionadas = st.multiselect("Selecciona las fechas Cyber", options=fechas_str, default=[])
    fechas_cyber = pd.to_datetime(fechas_seleccionadas)
    df_futuro['Es_Cyber'] = df_futuro['Fecha'].isin(fechas_cyber).astype(int)

    X_futuro = df_futuro[features]
    df_futuro['Llamadas_Predichas'] = model.predict(X_futuro)

    # Intervalos de predicción
    df_eval = df.copy()
    df_eval['Predicciones'] = model.predict(X)
    df_eval['Error'] = df_eval['Llamadas_Recibidas'] - df_eval['Predicciones']
    residual_std = df_eval['Error'][-35:].std()
    df_futuro['PI_Lower'] = df_futuro['Llamadas_Predichas'] - 1.96 * residual_std
    df_futuro['PI_Upper'] = df_futuro['Llamadas_Predichas'] + 1.96 * residual_std

    st.subheader("📊 Gráfico Interactivo con Histórico y Predicción")
    fig_interactivo = go.Figure()
    fig_interactivo.add_trace(go.Scatter(x=df['Fecha'], y=df['Llamadas_Recibidas'], mode='lines', name='Histórico', line=dict(color='blue')))
    fig_interactivo.add_trace(go.Scatter(x=df_futuro['Fecha'], y=df_futuro['Llamadas_Predichas'], mode='lines+markers', name='Predicción', line=dict(color='orange', dash='dash')))
    fig_interactivo.add_trace(go.Scatter(x=df_futuro['Fecha'], y=df_futuro['PI_Upper'], name='Límite Superior', line=dict(color='gray', dash='dot')))
    fig_interactivo.add_trace(go.Scatter(x=df_futuro['Fecha'], y=df_futuro['PI_Lower'], name='Límite Inferior', line=dict(color='gray', dash='dot'), fill='tonexty', fillcolor='rgba(200,200,200,0.2)'))
    fig_interactivo.update_layout(title="Predicción de Llamadas Recibidas (35 días)", xaxis_title="Fecha", yaxis_title="Llamadas", legend_title="Leyenda", template="plotly_white")
    st.plotly_chart(fig_interactivo)

    st.subheader("📋 Detalle de Predicciones")
    df_futuro['Llamadas_Predichas'] = df_futuro['Llamadas_Predichas'].round().astype(int)
    df_futuro['Fecha'] = df_futuro['Fecha'].dt.strftime('%d-%m-%Y')
    df_futuro['PI_Lower'] = df_futuro['PI_Lower'].round().astype(int)
    df_futuro['PI_Upper'] = df_futuro['PI_Upper'].round().astype(int)
    st.dataframe(df_futuro[['Fecha', 'Es_Cyber', 'Llamadas_Predichas', 'PI_Lower', 'PI_Upper']].set_index('Fecha'))




# --- EVALUACIÓN DEL MODELO ---
st.markdown("---")
st.header("📈 Evaluación d   el Modelo (Últimos 35 días)")
st.markdown("**Métricas de desempeño en los últimos 35 días:**")
st.write(f"R²: {r2_score(df_eval['Llamadas_Recibidas'][-35:], df_eval['Predicciones'][-35:]):.3f}")
st.write(f"MAE: {mean_absolute_error(df_eval['Llamadas_Recibidas'][-35:], df_eval['Predicciones'][-35:]):.2f}")
rmse = np.sqrt(mean_squared_error(df_eval['Llamadas_Recibidas'][-35:], df_eval['Predicciones'][-35:]))
st.write(f"RMSE: {rmse:.2f}")
# --- ANÁLISIS DE ERRORES ---
st.subheader("📊 Análisis de Errores")
anio_sel = st.selectbox("Selecciona el año", sorted(df_eval['anio'].unique(), reverse=True))
mes_sel = st.selectbox("Selecciona el mes", sorted(df_eval['mes'].unique()))

df_filtrado = df_eval[(df_eval['anio'] == anio_sel) & (df_eval['mes'] == mes_sel)]

fig1 = px.scatter(df_filtrado, x='Llamadas_Recibidas', y='Error', title='Error vs. Llamadas Reales')
st.plotly_chart(fig1)

fig2 = px.line(df_filtrado, x='Fecha', y='Error', title='Error a lo largo del tiempo')

st.plotly_chart(fig2)