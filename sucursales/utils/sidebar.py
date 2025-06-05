import streamlit as st
import pandas as pd

def set_sucursal_filter(df: pd.DataFrame):
    # Obtener valores únicos del campo COD_SUC
    cod_sucursales = df["COD_SUC"].unique()
    cod_sucursales.sort()

    # Crear selectbox en la barra lateral
    seleccion = st.sidebar.selectbox(
        "🔎 Selecciona una sucursal",
        options=cod_sucursales,
        index=0,
        key="selected_sucursal"
    )

    # Guardar en session_state
    st.session_state["COD_SUC"] = seleccion
