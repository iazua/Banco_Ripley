import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pyodbc
from utils.sidebar import set_sucursal_filter
from plotly.subplots import make_subplots

# Configuración inicial
st.set_page_config(
    page_title="Análisis de Visitas a Sucursales",
    layout="wide",
    page_icon="🏪"
)

# Estilos CSS personalizados
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
    .tipo-tienda-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 20px;
    }
    .tienda-semana {
        background-color: #d4edda;
        color: #155724;
    }
    .tienda-fin-semana {
        background-color: #f8d7da;
        color: #721c24;
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


@st.cache_data(show_spinner=False)
def load_data():
    conn_str = (
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=192.168.192.21;'
        'DATABASE=Auris_Personas;'
        'UID=iazuaz;'
        'PWD=192776775;'
    )
    query = "SELECT * FROM TB_SALTA_OFERTA_SALON_DIARIO_MODELO_PY"
    conn = pyodbc.connect(conn_str)
    df = pd.read_sql(query, conn)
    conn.close()

    # Limpieza y preparación de datos
    df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True)
    df['DIA'] = df['FECHA'].dt.day_name()
    df['MES'] = df['FECHA'].dt.month_name()
    df['HORA'] = df['HORA'].astype(int)
    df['PERIODO'] = df['PERIODO'].astype(str)
    df['DIA_SEMANA_NUM'] = df['FECHA'].dt.dayofweek  # Lunes=0, Domingo=6
    df['ES_FIN_SEMANA'] = df['DIA_SEMANA_NUM'] >= 5  # 5 y 6 son sábado y domingo

    # Ordenar días de la semana y meses
    dias_semana = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    meses = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
             'August', 'September', 'October', 'November', 'December']

    df['DIA'] = pd.Categorical(df['DIA'], categories=dias_semana, ordered=True)
    df['MES'] = pd.Categorical(df['MES'], categories=meses, ordered=True)

    return df


data = load_data()

# Sidebar con filtro transversal
set_sucursal_filter(data)

# Filtrar según selección
sucursal = st.session_state.get("COD_SUC")
df_filtrado = data[data["COD_SUC"] == sucursal]


# Función para determinar tipo de tienda
def determinar_tipo_tienda(df_sucursal):
    visitas_por_dia = df_sucursal.groupby('ES_FIN_SEMANA').size()
    total_visitas = visitas_por_dia.sum()

    if total_visitas == 0:
        return "Sin datos suficientes", 0, 0

    porcentaje_fin_semana = (visitas_por_dia.get(True, 0) / total_visitas) * 100
    porcentaje_semana = (visitas_por_dia.get(False, 0) / total_visitas) * 100

    if porcentaje_fin_semana > 50:
        return "Tienda Fin de Semana", porcentaje_semana, porcentaje_fin_semana
    else:
        return "Tienda Semana", porcentaje_semana, porcentaje_fin_semana


# Título de la aplicación
st.title("🏪 Análisis de Visitas a Sucursales")

# Mostrar datos filtrados
with st.expander("🔍 Ver datos de la sucursal seleccionada", expanded=False):
    st.write(f"Mostrando datos para la sucursal: **{sucursal}**")
    st.dataframe(df_filtrado)

# Sidebar para filtros adicionales
st.sidebar.header("Filtros Adicionales")
sucursales = sorted(data['COD_SUC'].unique())
sucursal_seleccionada = st.sidebar.selectbox("Seleccione Sucursal", sucursales, index=sucursales.index(sucursal))

# Filtrar datos por sucursal seleccionada
data_filtrada = data[data['COD_SUC'] == sucursal_seleccionada]

# Determinar tipo de tienda
tipo_tienda, pct_semana, pct_fin_semana = determinar_tipo_tienda(data_filtrada)

# Mostrar métricas clave
st.header(f"📊 Métricas Clave - Sucursal {sucursal_seleccionada}")

# Tarjeta de tipo de tienda
st.markdown(f"""
    <div class="tipo-tienda-card {'tienda-fin-semana' if 'Fin de Semana' in tipo_tienda else 'tienda-semana'}">
        Tipo de Tienda: {tipo_tienda}<br>
        Semana: {pct_semana:.1f}% | Fin de Semana: {pct_fin_semana:.1f}%
    </div>
""", unsafe_allow_html=True)

# Columnas para métricas
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total Visitas</div>
            <div class="metric-value">{:,}</div>
        </div>
    """.format(len(data_filtrada)), unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Visitas con Venta</div>
            <div class="metric-value">{:,}</div>
        </div>
    """.format(data_filtrada['VENTA'].sum()), unsafe_allow_html=True)

with col3:
    tasa_conversion = (data_filtrada['VENTA'].sum() / len(data_filtrada)) * 100 if len(data_filtrada) > 0 else 0
    st.markdown("""
        <div class="metric-card">
            <div class="metric-title">% Aceptación/Visitas</div>
            <div class="metric-value">{:.1f}%</div>
        </div>
    """.format(tasa_conversion), unsafe_allow_html=True)

with col4:
    visitas_por_dia = len(data_filtrada) / data_filtrada['FECHA'].nunique() if data_filtrada[
                                                                                   'FECHA'].nunique() > 0 else 0
    st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Visitas Promedio por Día</div>
            <div class="metric-value">{:.1f}</div>
        </div>
    """.format(visitas_por_dia), unsafe_allow_html=True)

# Análisis temporal
st.header("📅 Análisis Temporal")

# Gráfico de visitas por día con tendencia
st.subheader("Evolución Diaria de Visitas")
visitas_por_dia = data_filtrada.groupby('FECHA').size().reset_index(name='Visitas')

fig = px.line(visitas_por_dia, x='FECHA', y='Visitas',
              title=f'Visitas Diarias - Sucursal {sucursal_seleccionada}',
              template='plotly_white')
fig.update_traces(line=dict(width=2.5, color='#1f77b4'))
fig.update_layout(
    hovermode='x unified',
    xaxis_title='Fecha',
    yaxis_title='Número de Visitas',
    height=400
)
st.plotly_chart(fig, use_container_width=True)

# Gráficos de distribución
st.subheader("Distribución de Visitas")
col1, col2 = st.columns(2)

with col1:
    # Visitas por día de la semana mejorado
    visitas_dia_semana = data_filtrada['DIA'].value_counts().sort_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=visitas_dia_semana.index,
        y=visitas_dia_semana.values,
        marker_color=['#4e79a7' if dia not in ['Saturday', 'Sunday'] else '#e15759' for dia in
                      visitas_dia_semana.index],
        name='Visitas'
    ))

    fig.update_layout(
        title=f'Visitas por Día de la Semana - Sucursal {sucursal_seleccionada}',
        xaxis_title='Día de la Semana',
        yaxis_title='Número de Visitas',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Visitas por mes mejorado
    visitas_mes = data_filtrada['MES'].value_counts().sort_index()

    fig = px.bar(visitas_mes, x=visitas_mes.index, y=visitas_mes.values,
                 labels={'x': 'Mes', 'y': 'Número de Visitas'},
                 title=f'Visitas por Mes - Sucursal {sucursal_seleccionada}',
                 color=visitas_mes.values,
                 color_continuous_scale='Blues')

    fig.update_layout(
        template='plotly_white',
        height=400,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

# Análisis por hora del día mejorado
st.subheader("Distribución Horaria")
col1, col2 = st.columns([2, 1])

with col1:
    visitas_hora = data_filtrada['HORA'].value_counts().sort_index()

    fig = px.area(visitas_hora, x=visitas_hora.index, y=visitas_hora.values,
                  labels={'x': 'Hora del Día', 'y': 'Número de Visitas'},
                  title=f'Distribución de Visitas por Hora - Sucursal {sucursal_seleccionada}')

    fig.update_layout(
        template='plotly_white',
        height=400,
        xaxis=dict(tickmode='linear', dtick=1)
    )
    fig.update_traces(fill='tozeroy', line=dict(width=1, color='#1f77b4'))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Boxplot de visitas por hora
    fig = px.box(data_filtrada, x='HORA', y='VENTA',
                 title='Distribución de Ventas por Hora',
                 labels={'HORA': 'Hora del Día', 'VENTA': 'Venta (0=No, 1=Sí)'})

    fig.update_layout(
        template='plotly_white',
        height=400,
        xaxis=dict(tickmode='linear', dtick=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# Análisis de canales mejorado
st.header("📶 Análisis por Canal")
canales = data_filtrada['CANAL'].value_counts()

col1, col2 = st.columns([1, 2])
with col1:
    fig = px.pie(canales, values=canales.values, names=canales.index,
                 title=f'Distribución de Visitas por Canal',
                 hole=0.4)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Tasa de conversión por canal
    conversion_canal = data_filtrada.groupby('CANAL')['VENTA'].agg(['sum', 'count']).reset_index()
    conversion_canal['Tasa Conversión'] = (conversion_canal['sum'] / conversion_canal['count']) * 100

    fig = px.bar(conversion_canal, x='CANAL', y='Tasa Conversión',
                 title='Tasa de Conversión por Canal',
                 labels={'CANAL': 'Canal', 'Tasa Conversión': 'Tasa de Conversión (%)'},
                 color='Tasa Conversión',
                 color_continuous_scale='Viridis')

    fig.update_layout(
        height=400,
        coloraxis_showscale=False,
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig, use_container_width=True)

# Comparación entre sucursales mejorada
st.header("🏬 Comparación entre Sucursales")

# Gráfico combinado de visitas y tasa de conversión
conversion_por_sucursal = data.groupby('COD_SUC')['VENTA'].agg(['sum', 'count']).reset_index()
conversion_por_sucursal['Tasa Conversión'] = (conversion_por_sucursal['sum'] / conversion_por_sucursal['count']) * 100
conversion_por_sucursal = conversion_por_sucursal.sort_values('Tasa Conversión', ascending=False)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Bar(
        x=conversion_por_sucursal['COD_SUC'],
        y=conversion_por_sucursal['count'],
        name='Total Visitas',
        marker_color='#3498db'
    ),
    secondary_y=False
)
fig.add_trace(
    go.Scatter(
        x=conversion_por_sucursal['COD_SUC'],
        y=conversion_por_sucursal['Tasa Conversión'],
        name='Tasa Conversión',
        mode='lines+markers',
        line=dict(color='#e74c3c', width=2)
    ),
    secondary_y=True
)

fig.update_layout(
    title_text='Comparación entre Sucursales: Visitas y Tasa de Conversión',
    height=500,
    template='plotly_white',
    xaxis_title='Sucursal',
    hovermode='x unified'
)
fig.update_yaxes(title_text="Total Visitas", secondary_y=False)
fig.update_yaxes(title_text="Tasa de Conversión (%)", secondary_y=True, range=[0, 100])

st.plotly_chart(fig, use_container_width=True)

# Análisis detallado por día y hora mejorado
st.header("🕒 Análisis Detallado: Día y Hora")

# Heatmap interactivo de visitas por día de semana y hora
visitas_dia_hora = data_filtrada.groupby(['DIA', 'HORA']).size().reset_index(name='Visitas')
visitas_dia_hora_pivot = visitas_dia_hora.pivot(index='DIA', columns='HORA', values='Visitas')

fig = px.imshow(
    visitas_dia_hora_pivot,
    labels=dict(x="Hora del Día", y="Día de la Semana", color="Visitas"),
    x=visitas_dia_hora_pivot.columns,
    y=visitas_dia_hora_pivot.index,
    color_continuous_scale='YlOrBr',
    aspect='auto'
)

fig.update_layout(
    title=f'Visitas por Día de Semana y Hora - Sucursal {sucursal_seleccionada}',
    height=500,
    xaxis=dict(tickmode='linear', dtick=1)
)

st.plotly_chart(fig, use_container_width=True)

# Análisis de tipo de tienda para todas las sucursales
st.header("📌 Clasificación de Tiendas por Patrón de Visitas")


# Calcular para todas las sucursales
def calcular_tipo_tienda_por_sucursal(df):
    resultados = []
    for suc in df['COD_SUC'].unique():
        df_suc = df[df['COD_SUC'] == suc]
        visitas_por_dia = df_suc.groupby('ES_FIN_SEMANA').size()
        total_visitas = visitas_por_dia.sum()

        if total_visitas == 0:
            resultados.append({'COD_SUC': suc, 'Tipo_Tienda': 'Sin datos', 'Porcentaje_Fin_Semana': 0})
            continue

        porcentaje_fin_semana = (visitas_por_dia.get(True, 0) / total_visitas) * 100

        if porcentaje_fin_semana > 50:
            tipo = "Fin de Semana"
        else:
            tipo = "Semana"

        resultados.append({'COD_SUC': suc, 'Tipo_Tienda': tipo, 'Porcentaje_Fin_Semana': porcentaje_fin_semana})

    return pd.DataFrame(resultados)


df_tipos_tienda = calcular_tipo_tienda_por_sucursal(data)

# Mostrar resultados
col1, col2 = st.columns(2)
with col1:
    st.subheader("Distribución de Tipos de Tienda")
    fig = px.pie(df_tipos_tienda, names='Tipo_Tienda', title='Proporción de Tiendas por Tipo')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Porcentaje de Visitas Fin de Semana por Sucursal")
    fig = px.bar(df_tipos_tienda.sort_values('Porcentaje_Fin_Semana'),
                 x='COD_SUC', y='Porcentaje_Fin_Semana',
                 color='Tipo_Tienda',
                 title='% Visitas Fin de Semana por Sucursal',
                 labels={'COD_SUC': 'Sucursal', 'Porcentaje_Fin_Semana': '% Visitas Fin de Semana'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Mostrar datos crudos si se desea
if st.checkbox('Mostrar datos detallados de la sucursal'):
    st.subheader('Datos Detallados')
    st.write(data_filtrada)

# Mostrar datos de clasificación de tiendas
with st.expander("Ver datos de clasificación de todas las sucursales"):
    st.dataframe(df_tipos_tienda.sort_values('Porcentaje_Fin_Semana', ascending=False))