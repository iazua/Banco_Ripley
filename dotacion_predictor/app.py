import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import timedelta
from preprocessing import prepare_features
from utils import calcular_efectividad, estimar_dotacion_optima, _modelo_efectividad, estimar_parametros_efectividad

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Predicción de Dotación Óptima", layout="wide")
st.image("https://www.fide.edu.pe/wp-content/uploads/2021/08/banco-ripley.jpg", width=200)


# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    df = pd.read_excel("data/DOTACION_EFECTIVIDAD.xlsx")
    df.columns = df.columns.str.strip().str.upper()
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    return df


df = load_data()

# --- INTERFAZ DE USUARIO ---
st.title("🔍 Predicción de Dotación y Efectividad por Sucursal")
sucursales = sorted(df["COD_SUC"].unique())
cod_suc = st.selectbox("Selecciona una sucursal", sucursales)

# --- PROCESAMIENTO DE DATOS ---
df_suc = df[df["COD_SUC"] == cod_suc].copy().sort_values("FECHA")

# Calcular P_EFECTIVIDAD histórica
if 'P_EFECTIVIDAD' not in df_suc.columns:
    if 'T_AO' in df_suc.columns and 'T_AO_VENTA' in df_suc.columns:
        df_suc['P_EFECTIVIDAD'] = calcular_efectividad(df_suc['T_AO'], df_suc['T_AO_VENTA'])
    else:
        df_suc['P_EFECTIVIDAD'] = np.nan

# Calcular promedio histórico de efectividad para DOTACION == 1
promedio_efectividad_dotacion_1 = np.nan
if 'DOTACION' in df_suc.columns and 'P_EFECTIVIDAD' in df_suc.columns:
    df_suc_dot1 = df_suc[df_suc["DOTACION"] == 1].copy()
    if not df_suc_dot1.empty and df_suc_dot1["P_EFECTIVIDAD"].notna().any():
        promedio_efectividad_dotacion_1 = df_suc_dot1["P_EFECTIVIDAD"].mean()

# --- CONFIGURACIÓN DE PARÁMETROS ---
n_dias_forecast = 35
efectividad_deseada = st.slider(
    "Selecciona la efectividad (P_EFECTIVIDAD) objetivo que deseas alcanzar:",
    min_value=0.0, max_value=1.0, value=0.8, step=0.01, format="%.2f"
)


# --- FUNCIONES AUXILIARES ---
def cargar_modelo(variable, cod_suc):
    path = f"models/predictor_{variable}_{cod_suc}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    st.error(f"No se encontró el modelo para {variable} en sucursal {cod_suc}")
    return None


def generar_predicciones(df_hist, model, target):
    df_pred = df_hist.copy()
    last_date = df_pred["FECHA"].max()
    predicciones = []
    fechas = []

    for _ in range(n_dias_forecast):
        next_date = last_date + timedelta(days=1)
        df_for_features = df_pred.tail(100).copy()
        temp_row = pd.DataFrame([df_for_features.iloc[-1]]).copy()
        temp_row['FECHA'] = next_date

        if target in temp_row.columns:
            temp_row[target] = np.nan

        for prev_target_col in ["T_AO", "T_AO_VENTA", "T_VISITAS", "DOTACION", "P_EFECTIVIDAD"]:
            predicted_col_name = f"{prev_target_col}_pred"
            if predicted_col_name in df_pred.columns and not df_pred[predicted_col_name].isnull().all():
                last_pred_val = df_pred[predicted_col_name].ffill().iloc[-1]
                temp_row[prev_target_col] = last_pred_val
            elif prev_target_col in df_pred.columns and not df_pred[prev_target_col].isnull().all():
                last_real_val = df_pred[prev_target_col].ffill().iloc[-1]
                temp_row[prev_target_col] = last_real_val
            elif prev_target_col in df_hist.columns and not df_hist[prev_target_col].isnull().all():
                temp_row[prev_target_col] = df_hist[prev_target_col].ffill().iloc[-1]

        df_for_features = pd.concat([df_for_features, temp_row], ignore_index=True).sort_values("FECHA").reset_index(
            drop=True)
        X_pred, _ = prepare_features(df_for_features, target, is_prediction=True)

        if X_pred is not None and len(X_pred) > 0:
            X_input = X_pred.iloc[[-1]]
            y_pred = model.predict(X_input)[0]
            predicciones.append(y_pred)
            fechas.append(next_date)

            new_pred_row_dict = {"FECHA": next_date, f"{target}_pred": y_pred}
            for col in df_hist.columns:
                if col not in new_pred_row_dict:
                    if col in temp_row.columns:
                        new_pred_row_dict[col] = temp_row[col].values[0]
                    elif col in df_pred.columns:
                        new_pred_row_dict[col] = df_pred[col].iloc[-1]

            df_pred = pd.concat([df_pred, pd.DataFrame([new_pred_row_dict])], ignore_index=True)
        else:
            st.warning(f"No se pudieron generar features para {target} en fecha {next_date}.")
            break
        last_date = next_date

    return pd.DataFrame({"FECHA": fechas, f"{target}_pred": predicciones})


# --- CARGA DE MODELOS Y GENERACIÓN DE PREDICCIONES ---
modelo_tao = cargar_modelo("T_AO", cod_suc)
modelo_venta = cargar_modelo("T_AO_VENTA", cod_suc)
modelo_visitas = cargar_modelo("T_VISITAS", cod_suc)

if modelo_tao and modelo_venta and modelo_visitas:
    pred_tao = generar_predicciones(df_suc, modelo_tao, "T_AO")
    pred_venta = generar_predicciones(df_suc, modelo_venta, "T_AO_VENTA")
    pred_visitas = generar_predicciones(df_suc, modelo_visitas, "T_VISITAS")

    # Combinar todas las predicciones
    df_pred_all = pred_tao.merge(pred_venta, on="FECHA", how="outer").merge(pred_visitas, on="FECHA", how="outer")

    # Calcular efectividad predicha por el modelo
    if "T_AO_pred" in df_pred_all.columns and "T_AO_VENTA_pred" in df_pred_all.columns:
        df_pred_all["P_EFECTIVIDAD_pred_modelo"] = calcular_efectividad(df_pred_all["T_AO_pred"],
                                                                        df_pred_all["T_AO_VENTA_pred"])
    else:
        st.warning("No se pudieron calcular todas las predicciones necesarias para la efectividad del modelo.")
        df_pred_all["P_EFECTIVIDAD_pred_modelo"] = np.nan

    # Ajustar según efectividad deseada
    if "T_AO_pred" in df_pred_all.columns and df_pred_all["T_AO_pred"].notna().any():
        df_pred_all["T_AO_VENTA_pred_ajustada"] = efectividad_deseada * df_pred_all["T_AO_pred"]
        df_pred_all["P_EFECTIVIDAD_pred_ajustada"] = calcular_efectividad(df_pred_all["T_AO_pred"],
                                                                          df_pred_all["T_AO_VENTA_pred_ajustada"])
    else:
        st.warning("No se pueden ajustar las ventas según la efectividad deseada.")
        df_pred_all["T_AO_VENTA_pred_ajustada"] = df_pred_all.get("T_AO_VENTA_pred", np.nan)
        df_pred_all["P_EFECTIVIDAD_pred_ajustada"] = df_pred_all.get("P_EFECTIVIDAD_pred_modelo", np.nan)

    # Estimación de parámetros de efectividad
    df_historico_para_parametros = df_suc[['DOTACION', 'T_AO', 'T_AO_VENTA']].copy().dropna()
    if not df_historico_para_parametros.empty and len(df_historico_para_parametros) > 1:
        params_efectividad_estimados = estimar_parametros_efectividad(df_historico_para_parametros)
    else:
        st.warning("No hay suficientes datos históricos para estimar parámetros de efectividad.")
        params_efectividad_estimados = {'L': 1.0, 'k': 0.5, 'x0_base': 5.0, 'x0_factor_t_ao_venta': 0.05}

    # Cálculo de dotación necesaria
    dotacion_necesaria_diaria = []
    efectividad_real_con_dotacion_optima = []
    col_t_ao_venta_para_dotacion = "T_AO_VENTA_pred_ajustada" if "T_AO_VENTA_pred_ajustada" in df_pred_all.columns else "T_AO_VENTA_pred"

    if "T_AO_pred" in df_pred_all.columns and col_t_ao_venta_para_dotacion in df_pred_all.columns:
        for idx, row in df_pred_all.iterrows():
            t_ao_p = row["T_AO_pred"]
            t_ao_venta_p = row[col_t_ao_venta_para_dotacion]

            if pd.notna(t_ao_p) and pd.notna(t_ao_venta_p) and t_ao_p > 0:
                dotacion_opt, efect_resultante_modelo = estimar_dotacion_optima(
                    np.array([t_ao_p]), np.array([t_ao_venta_p]),
                    efectividad_deseada, params_efectividad=params_efectividad_estimados
                )

                if dotacion_opt == 1 and pd.notna(promedio_efectividad_dotacion_1):
                    efect_resultante = promedio_efectividad_dotacion_1
                else:
                    efect_resultante = efect_resultante_modelo

                dotacion_necesaria_diaria.append(dotacion_opt)
                efectividad_real_con_dotacion_optima.append(efect_resultante)
            else:
                dotacion_necesaria_diaria.append(0)
                efectividad_real_con_dotacion_optima.append(0.0)
    else:
        st.warning("Faltan datos para calcular la dotación óptima.")
        dotacion_necesaria_diaria = [np.nan] * len(df_pred_all)
        efectividad_real_con_dotacion_optima = [np.nan] * len(df_pred_all)

    df_pred_all["DOTACION_NECESARIA"] = dotacion_necesaria_diaria
    df_pred_all["P_EFECTIVIDAD_OPT_DOT"] = efectividad_real_con_dotacion_optima

    # --- PRESENTACIÓN DE RESULTADOS ---
    dotacion_optima_promedio_general = np.nanmean(
        df_pred_all["DOTACION_NECESARIA"]) if "DOTACION_NECESARIA" in df_pred_all.columns and df_pred_all[
        "DOTACION_NECESARIA"].notna().any() else 0
    efectividad_promedio_con_dotacion_optima = np.nanmean(
        df_pred_all["P_EFECTIVIDAD_OPT_DOT"]) if "P_EFECTIVIDAD_OPT_DOT" in df_pred_all.columns and df_pred_all[
        "P_EFECTIVIDAD_OPT_DOT"].notna().any() else 0

    st.success(
        f"📈 Dotación necesaria para alcanzar {efectividad_deseada:.2%} de efectividad (promedio): **{dotacion_optima_promedio_general:.0f}** personas.")
    st.info(
        f"Con esta dotación, la efectividad promedio esperada es: **{efectividad_promedio_con_dotacion_optima:.2%}**")


    # --- PREPARACIÓN DE DATOS PARA VISUALIZACIÓN ---
    st.subheader("📊 Predicciones próximas 35 fechas")

    # Renombrar columnas según lo solicitado
    df_display = df_pred_all.rename(columns={
        "T_VISITAS_pred": "Dotación proyectada",
        "T_AO_pred": "Acepta Oferta esperada",
        "T_AO_VENTA_pred": "Ventas Concretadas esperada",
        "P_EFECTIVIDAD_pred_modelo": "% Efectividad esperada (Modelo)",
        "T_AO_VENTA_pred_ajustada": "Acepta Oferta requerida",
        "P_EFECTIVIDAD_pred_ajustada": "% Efectividad requerida",
        "DOTACION_NECESARIA": "Dotación requerida",
        "P_EFECTIVIDAD_OPT_DOT": "% Efectividad esperada (Óptima)"
    })

    # Agregar día de la semana
    if 'FECHA' in df_display.columns:
        df_display_copy = df_display.copy()
        df_display_copy['Día'] = pd.to_datetime(df_display_copy['FECHA']).dt.day_name().map({
            'Monday': 'Lunes',
            'Tuesday': 'Martes',
            'Wednesday': 'Miércoles',
            'Thursday': 'Jueves',
            'Friday': 'Viernes',
            'Saturday': 'Sábado',
            'Sunday': 'Domingo'
        })

        # Reordenar columnas
        cols = [col for col in df_display_copy.columns if col != 'Día']
        if 'FECHA' in cols:
            fecha_idx = cols.index('FECHA')
            new_order = cols[:fecha_idx + 1] + ['Día'] + cols[fecha_idx + 1:]
            df_display = df_display_copy[new_order]

    # Definir formato para las columnas
    format_dict = {
        'FECHA': '{:%d-%m-%Y}',
        'Acepta Oferta esperada': '{:.0f}',
        'Ventas Concretadas esperada': '{:.0f}',
        'Acepta Oferta requerida': '{:.0f}',
        'Dotación proyectada': '{:.0f}',
        '% Efectividad esperada (Modelo)': '{:.2%}',
        '% Efectividad requerida': '{:.2%}',
        'Dotación requerida': '{:.0f}',
        '% Efectividad esperada (Óptima)': '{:.2%}'
    }

    # Filtrar format_dict para columnas existentes
    existing_format_dict = {k: v for k, v in format_dict.items() if k in df_display.columns}

    # Mostrar tabla con los nuevos nombres
    if not df_display.empty:
        st.dataframe(df_display.style.format(existing_format_dict), use_container_width=True)
    else:
        st.write("No hay datos para mostrar.")
    # === NUEVA SECCIÓN: Análisis semanal histórico para la sucursal seleccionada ===
    st.header("📅 Análisis semanal histórico para la sucursal seleccionada")


    import pandas as pd
    import numpy as np
    import streamlit as st
    import plotly.express as px

    if not df_suc.empty:
        df_suc['DIA_SEMANA'] = df_suc['FECHA'].dt.dayofweek  # 0=Lunes, 6=Domingo
        dias_orden = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        df_suc['NOMBRE_DIA'] = pd.Categorical(
            df_suc['FECHA'].dt.day_name().map({
                'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
                'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
            }),
            categories=dias_orden,
            ordered=True

        )

        # Promedios por día para cada variable
        promedio_t_ao = df_suc.groupby('NOMBRE_DIA', observed=True)['T_AO'].mean().reset_index()
        promedio_t_ao_venta = df_suc.groupby('NOMBRE_DIA', observed=True)['T_AO_VENTA'].mean().reset_index()
        promedio_t_visitas = df_suc.groupby('NOMBRE_DIA', observed=True)['T_VISITAS'].mean().reset_index()

        # Mostrar tres gráficos horizontalmente
        st.subheader("📈 Promedio por día de la semana")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.caption("Promedio OFERTAS ACEPTADAS")
            st.bar_chart(promedio_t_ao.set_index('NOMBRE_DIA'))

        with col2:
            st.caption("Promedio VENTAS CONCRETADAS")
            st.bar_chart(promedio_t_ao_venta.set_index('NOMBRE_DIA'))

        with col3:
            st.caption("Promedio VISITAS")
            st.bar_chart(promedio_t_visitas.set_index('NOMBRE_DIA'))
        # Calcular porcentaje de registros Día de Semana vs. Fin de Semana
        df_suc['TIPO_DIA'] = np.where(df_suc['DIA_SEMANA'] < 5, 'Día de Semana', 'Fin de Semana')
        total_registros = len(df_suc)
        registros_dia_semana = len(df_suc[df_suc['TIPO_DIA'] == 'Día de Semana'])
        registros_fin_semana = total_registros - registros_dia_semana

        porcentaje_dia_semana = registros_dia_semana / total_registros * 100 if total_registros > 0 else 0
        porcentaje_fin_semana = registros_fin_semana / total_registros * 100 if total_registros > 0 else 0

        # Mostrar KPIs
        st.subheader("🗓️ Distribución de registros por tipo de día")
        col_kpi1, col_kpi2 = st.columns(2)
        col_kpi1.metric("📅 Día de Semana", f"{porcentaje_dia_semana:.2f}%")
        col_kpi2.metric("📆 Fin de Semana", f"{porcentaje_fin_semana:.2f}%")
    else:
        st.warning("No hay datos históricos para la sucursal seleccionada.")

    # === NUEVA FILA DE GRÁFICOS: DOTACION, P_EFECTIVIDAD, RELACIÓN DOTACION VS P_EFECTIVIDAD ===
    st.subheader("📊 Análisis adicional para la sucursal seleccionada")

    if not df_suc.empty:
        if 'NOMBRE_DIA' not in df_suc.columns:
            df_suc['DIA_SEMANA'] = df_suc['FECHA'].dt.dayofweek
            df_suc['NOMBRE_DIA'] = pd.Categorical(
                df_suc['FECHA'].dt.day_name().map({
                    'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
                    'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
                }),
                categories=dias_orden,
                ordered=True
            )

        promedio_dotacion = df_suc.groupby('NOMBRE_DIA', observed=True)['DOTACION'].mean().reset_index()
        promedio_p_efectividad = df_suc.groupby('NOMBRE_DIA', observed=True)['P_EFECTIVIDAD'].mean().reset_index()

        col4, col5, col6 = st.columns(3)

        with col4:
            st.caption("Promedio Dotación")
            st.bar_chart(promedio_dotacion.set_index('NOMBRE_DIA'))

        with col5:
            st.caption("Promedio Efectividad")
            st.bar_chart(promedio_p_efectividad.set_index('NOMBRE_DIA'))

        with col6:
            st.caption("Efectividad promedio por Dotación")
            df_scatter = df_suc[['DOTACION', 'P_EFECTIVIDAD']].dropna()
            if not df_scatter.empty:
                df_scatter_grouped = df_scatter.groupby('DOTACION', as_index=False)['P_EFECTIVIDAD'].mean()
                fig_scatter = px.scatter(
                    df_scatter_grouped, x='DOTACION', y='P_EFECTIVIDAD',
                    title='Promedio P_EFECTIVIDAD por DOTACION',
                    labels={'DOTACION': 'Dotación', 'P_EFECTIVIDAD': 'P. Efectividad'},
                    range_y=[0, 1]
                )
                fig_scatter.update_traces(marker=dict(size=10, color='green', opacity=0.8))
                fig_scatter.update_layout(width=350, height=350, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("No hay datos válidos para mostrar la relación DOTACION vs P_EFECTIVIDAD.")
    else:
        st.warning("No hay datos históricos para la sucursal seleccionada.")

    # --- GRÁFICOS ---
    st.subheader("📈 Histórico y Predicción de Acepta Oferta")
    if 'T_AO' in df_suc.columns and 'Acepta Oferta esperada' in df_display.columns:
        # Solución simplificada para el gráfico
        df_plot = pd.DataFrame({
            'Fecha': df_suc['FECHA'].tolist() + df_display['FECHA'].tolist(),
            'Valor': df_suc['T_AO'].tolist() + df_display['Acepta Oferta esperada'].tolist(),
            'Tipo': ['Histórico'] * len(df_suc) + ['Predicción'] * len(df_display)
        })
        st.line_chart(df_plot.pivot(index='Fecha', columns='Tipo', values='Valor'))
    else:
        st.info("No hay suficientes datos para generar el gráfico.")

    st.subheader("📈 Histórico y Predicción de Ventas Concretadas")
    # Solución simplificada para evitar el error
    plot_data = []

    # Datos históricos
    if 'T_AO_VENTA' in df_suc.columns:
        plot_data.append(
            df_suc[['FECHA', 'T_AO_VENTA']].rename(columns={'T_AO_VENTA': 'Ventas'}).assign(Tipo='Histórico'))

    # Datos del modelo
    if 'Ventas Concretadas esperada' in df_display.columns:
        plot_data.append(df_display[['FECHA', 'Ventas Concretadas esperada']].rename(
            columns={'Ventas Concretadas esperada': 'Ventas'}).assign(Tipo='Modelo'))

    # Datos objetivo
    if 'Acepta Oferta requerida' in df_display.columns:
        plot_data.append(df_display[['FECHA', 'Acepta Oferta requerida']].rename(
            columns={'Acepta Oferta requerida': 'Ventas'}).assign(Tipo='Objetivo'))

    if plot_data:
        df_plot_ventas = pd.concat(plot_data)
        st.line_chart(df_plot_ventas.pivot(index='FECHA', columns='Tipo', values='Ventas'))
    else:
        st.info("No hay suficientes datos para generar el gráfico.")

    st.subheader("Curva de Efectividad vs. Dotación")
    dotacion_rango = np.arange(0, 21)
    efectividad_esperada_curva = []

    avg_t_ao_venta_curva = np.nanmean(
        df_display['Acepta Oferta requerida']) if 'Acepta Oferta requerida' in df_display.columns else np.nanmean(
        df_display.get('Ventas Concretadas esperada', np.nan))

    if pd.notna(avg_t_ao_venta_curva) and avg_t_ao_venta_curva > 0:
        for dot_val in dotacion_rango:
            if dot_val == 1 and pd.notna(promedio_efectividad_dotacion_1):
                efectividad_esperada_curva.append(promedio_efectividad_dotacion_1)
            else:
                efectividad_esperada_curva.append(
                    _modelo_efectividad(dot_val, avg_t_ao_venta_curva, params_efectividad_estimados))

        df_curva = pd.DataFrame({
            "Dotación": dotacion_rango,
            "Efectividad Esperada": efectividad_esperada_curva
        })
        st.line_chart(df_curva.set_index("Dotación"))
        st.write(f"*(Curva basada en un promedio de Ventas Concretadas de {avg_t_ao_venta_curva:.0f})*")
    else:
        st.info("No hay datos válidos para generar la curva de efectividad.")

    # --- BOTÓN DE DESCARGA ---
    csv = df_display.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar resultados en CSV", csv, f"predicciones_{cod_suc}.csv", "text/csv")

else:
    st.warning("No se pudieron cargar todos los modelos necesarios para la sucursal seleccionada.")

