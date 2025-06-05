import streamlit as st
from datetime import datetime
from utils.sidebar import set_sucursal_filter

st.set_page_config(page_title="Inicio", page_icon="📊", layout="centered")

# Título principal
st.title("📈 Sistema de Análisis y Predicción de Visitas")

# Subtítulo
st.subheader("Bienvenido/a a la aplicación")

st.markdown(
    """
    Esta aplicación permite:
    - 📊 Analizar el comportamiento histórico de visitas.
    - 🤖 Predecir visitas futuras utilizando modelos de machine learning como XGBoost.

    Usa la barra lateral para navegar entre las secciones.
    """
)

# Información adicional
st.info(f"🕒 Última actualización del sistema: {datetime.today().strftime('%d/%m/%Y')}")

# Imagen decorativa (opcional)
st.image("https://cdn-icons-png.flaticon.com/512/4149/4149656.png", width=200)

# Footer
st.markdown("---")
st.markdown(
    "Desarrollado por Ignacio Azua - Banco Ripley | [GitHub](https://github.com/iazua) | [LinkedIn](https://linkedin.com/in/iazua)")
