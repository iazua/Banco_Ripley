import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import holidays
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Interacciones RRSS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar el modelo entrenado
@st.cache_resource
def load_model():
    model_path = r'C:\Users\iazuaz\PyCharmMiscProject\model_RRSS\model\consultas_model_v1.pkl'

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Cargar los datos limpios
@st.cache_data
def load_data():
    data_path = r'C:\Users\iazuaz\PyCharmMiscProject\model_RRSS\data\cleaned_data.pkl'
    with open(data_path, 'rb') as f:
        df = pickle.load(f)
    return df

# Ingeniería de variables de fecha
def create_date_features(df):
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['dia_semana'] = df['Fecha'].dt.dayofweek
    df['fin_de_semana'] = (df['dia_semana'] >= 5).astype(int)
    df['dia_mes'] = df['Fecha'].dt.day
    df['semana_mes'] = df['Fecha'].apply(lambda x: (x.day - 1) // 7 + 1)
    df['mes'] = df['Fecha'].dt.month
    df['trimestre'] = df['Fecha'].dt.quarter
    df['año'] = df['Fecha'].dt.year
    chile_holidays = holidays.CountryHoliday('CL')
    df['es_feriado'] = df['Fecha'].apply(lambda x: x in chile_holidays).astype(int)
    df['dia_laboral'] = ((df['fin_de_semana'] == 0) & (df['es_feriado'] == 0)).astype(int)
    df['dias_desde_inicio'] = (df['Fecha'] - df['Fecha'].min()).dt.days
    return df

# Predicción a futuro
def predict_future(model, last_date, days_to_predict, cyber_dates=None):
    future_dates = pd.date_range(start=last_date + timedelta(days=75), periods=days_to_predict)
    future_df = pd.DataFrame({'Fecha': future_dates})
    future_df = create_date_features(future_df)
    future_df['Es_Cyber'] = 0
    if cyber_dates is not None:
        cyber_dates = pd.to_datetime(cyber_dates)
        future_df['Es_Cyber'] = future_df['Fecha'].isin(cyber_dates).astype(int)
    expected_columns = [
        'Es_Cyber', 'dia_semana', 'fin_de_semana', 'dia_mes',
        'semana_mes', 'mes', 'trimestre', 'año',
        'es_feriado', 'dia_laboral', 'dias_desde_inicio'
    ]
    for col in expected_columns:
        if col not in future_df.columns:
            future_df[col] = 0
    X_future = future_df[expected_columns]
    future_df['Interacciones_Predichas'] = model.predict(X_future)
    return future_df

# Aplicación principal
def main():
    st.title("📊 Modelo de Predicción de Interacciones RRSS")
    model = load_model()
    df = load_data()
    df = create_date_features(df)

    # Controles de la barra lateral
    st.sidebar.header("Configuración de la Predicción")
    days_to_predict = st.sidebar.slider("Número de días a predecir", 1, 90, 30)
    st.sidebar.markdown("### Seleccionar Días Cyber")
    today = datetime.today().date()
    date_range = pd.date_range(today, today + timedelta(days=150)).to_list()
    cyber_days_selected = st.sidebar.multiselect(
        "Elige los días Cyber (solo afecta predicciones)",
        options=date_range,
        format_func=lambda x: x.strftime("%Y-%m-%d")
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Datos Históricos", "Desempeño del Modelo", "Predicciones Futuras", "Análisis de Variables"])

    with tab1:
        st.header("Resumen de Datos Históricos")
        col1, col2 = st.columns(2)
        min_date = df['Fecha'].min().date()
        max_date = df['Fecha'].max().date()
        with col1:
            start_date = st.date_input("Fecha de inicio", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("Fecha de término", max_date, min_value=min_date, max_value=max_date)
        filtered_df = df[(df['Fecha'].dt.date >= start_date) & (df['Fecha'].dt.date <= end_date)]
        st.subheader("Métricas Clave")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Interacciones", f"{filtered_df['Consultas_Recibidas'].sum():,.0f}")
        with col2:
            st.metric("Promedio Diario", f"{filtered_df['Consultas_Recibidas'].mean():,.0f}")
        with col3:
            st.metric("Máximo Diario", f"{filtered_df['Consultas_Recibidas'].max():,.0f}")
        st.subheader("Interacciones en el Tiempo")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(filtered_df['Fecha'], filtered_df['Consultas_Recibidas'], label='Reales')
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Interacciones Recibidas")
        ax.set_title("Histórico de Interacciones")
        ax.grid(True)
        st.pyplot(fig)
        if st.checkbox("Mostrar datos crudos"):
            st.dataframe(filtered_df)

    with tab2:
        st.header("Análisis del Desempeño del Modelo")
        X = df.drop(['Consultas_Recibidas', 'Fecha'], axis=1)
        y = df['Consultas_Recibidas']
        test_size = int(len(df) * 0.2)
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        st.subheader("Métricas del Modelo en Conjunto de Prueba")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"{mae:,.1f}")
        with col2:
            st.metric("MSE", f"{mse:,.1f}")
        with col3:
            st.metric("RMSE", f"{rmse:,.1f}")
        st.subheader("Valores Reales vs. Predichos")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax.set_xlabel('Reales')
        ax.set_ylabel('Predichos')
        ax.set_title('Valores Reales vs. Predichos')
        st.pyplot(fig)
        st.subheader("Comparación Temporal")
        test_dates = df.iloc[X_test.index]['Fecha']
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test_dates, y_test, label='Reales')
        ax.plot(test_dates, y_pred, label='Predichos', alpha=0.7)
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Interacciones Recibidas')
        ax.set_title('Reales vs. Predichos en el Tiempo')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        st.subheader("Distribución de Errores")
        errors = y_test - y_pred
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(errors, kde=True, ax=ax)
        ax.set_xlabel('Error de Predicción')
        ax.set_title('Distribución de Errores')
        st.pyplot(fig)

    with tab3:
        st.header("Predicciones Futuras")
        last_date = df['Fecha'].max()
        future_predictions = predict_future(model, last_date, days_to_predict, cyber_dates=cyber_days_selected)

        st.subheader("Interacciones Predichas para Fechas Futuras")

        # Round and convert before displaying
        display_predictions = future_predictions[['Fecha', 'Interacciones_Predichas']].copy()
        display_predictions['Interacciones_Predichas'] = display_predictions['Interacciones_Predichas'].round().astype(
            int)
        st.dataframe(display_predictions)

        # Use same rounded values for export
        predicciones_export = display_predictions.copy()
        csv = predicciones_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar Predicciones en CSV",
            data=csv,
            file_name=f"interacciones_predichas_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )

        # Plot original (or you could also plot rounded if preferred)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(future_predictions['Fecha'], future_predictions['Interacciones_Predichas'], label='Predichas')
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Interacciones Predichas")
        ax.set_title("Predicción de Interacciones Futuras")
        ax.grid(True)
        st.pyplot(fig)

    # El código anterior permanece igual hasta...

    with tab4:
        st.header("Análisis de Variables")
        st.subheader("Importancia de las Variables")
        try:
            importancias = model.named_steps['model'].feature_importances_
            columnas = X_train.columns
            importancia_df = pd.DataFrame({'Variable': columnas, 'Importancia': importancias})
            importancia_df = importancia_df.sort_values(by='Importancia', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importancia', y='Variable', data=importancia_df, ax=ax)
            ax.set_title("Importancia de las Variables en el Modelo")
            st.pyplot(fig)
        except Exception as e:
            st.warning("No se pudo calcular la importancia de las variables.")
            st.text(f"Detalle del error: {e}")

if __name__ == "__main__":
    main()
