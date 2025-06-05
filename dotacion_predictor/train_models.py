import os
import pandas as pd
import joblib
import pyodbc
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from preprocessing import prepare_features
from sklearn.metrics import mean_absolute_error
import numpy as np
from utils import estimar_dotacion_optima, estimar_parametros_efectividad  # Importar las funciones necesarias

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data_from_excel(file_path):
    return pd.read_excel(file_path)



def train_model_per_branch(df, target):
    unique_branches = df["COD_SUC"].unique()
    for sucursal in unique_branches:
        suc_df = df[df["COD_SUC"] == sucursal].copy()
        X, y = prepare_features(suc_df, target)

        if len(X) < 10:
            print(f"⚠️ Insuficientes datos para {sucursal} - {target}.")
            continue

        tscv = TimeSeriesSplit(n_splits=5)
        model_cv = GridSearchCV(
            XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42),
            param_grid={"n_estimators": [50, 100], "max_depth": [3, 5]},
            cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        model_cv.fit(X, y)

        filename = f"{MODEL_DIR}/predictor_{target}_{sucursal}.pkl"
        joblib.dump(model_cv.best_estimator_, filename)
        print(f"✅ Modelo {target} para sucursal {sucursal} guardado en {filename}")


def predict_future_values(df, branch_code, future_dates, targets):
    predictions = {}
    suc_df = df[df["COD_SUC"] == branch_code].copy()

    # Asegurarse de que el DataFrame histórico tenga las columnas necesarias para los lags
    # y rolling means en prepare_features, incluso si son NaN para las fechas futuras.
    # Aquí vamos a crear un DataFrame combinado que incluya el histórico y las fechas futuras.

    # DataFrame para mantener el historial y las predicciones futuras
    df_combined_preds = suc_df.copy()

    for date_idx, current_date in enumerate(future_dates):
        # Crear una fila para la fecha actual en el futuro
        future_row = pd.DataFrame([{
            "FECHA": current_date,
            "COD_SUC": branch_code,
            # Inicializar las columnas target con NaN para que prepare_features las procese
            "T_AO": np.nan,
            "T_AO_VENTA": np.nan,
            "T_VISITAS": np.nan,
            "DOTACION": np.nan  # También la dotación si es una feature
        }])

        # Concatenar para que prepare_features pueda calcular lags y rolling averages
        # Utilizaremos el df_combined_preds que se va actualizando con las predicciones.
        df_temp = pd.concat([df_combined_preds.tail(30), future_row],
                            ignore_index=True)  # Solo las últimas 30 filas + la nueva para eficiencia
        df_temp = df_temp.sort_values("FECHA").reset_index(drop=True)

        for target in targets:
            model_filename = f"{MODEL_DIR}/predictor_{target}_{branch_code}.pkl"
            if not os.path.exists(model_filename):
                print(f"❌ Modelo para {target} y sucursal {branch_code} no encontrado.")
                # Si el modelo no existe, la predicción será NaN
                if target not in predictions:
                    predictions[target] = []
                predictions[target].append(np.nan)
                continue

            model = joblib.load(model_filename)

            # Preparar features para la fecha actual del futuro
            # Se pasa el df_temp para que los lags y rolling se calculen correctamente
            X_future, _ = prepare_features(df_temp, target, is_prediction=True)

            if X_future.empty:
                print(f"⚠️ No se pudieron preparar features para {current_date} - {target}.")
                if target not in predictions:
                    predictions[target] = []
                predictions[target].append(np.nan)
                continue

            # Tomar la última fila de X_future, que corresponde a current_date
            X_input = X_future.iloc[[-1]]

            pred = model.predict(X_input)[0]

            # Almacenar la predicción
            if target not in predictions:
                predictions[target] = []
            predictions[target].append(pred)

            # Actualizar el df_combined_preds con la predicción para el siguiente ciclo
            # Encuentra la fila correspondiente a current_date y actualiza el valor
            df_combined_preds.loc[df_combined_preds['FECHA'] == current_date, target] = pred

            # Si la fila de current_date no existe (es la primera vez que se predice para esta fecha),
            # la añadimos a df_combined_preds
            if (df_combined_preds['FECHA'] == current_date).sum() == 0:
                new_row = {"FECHA": current_date, "COD_SUC": branch_code}
                new_row[target] = pred
                # Rellenar otras columnas con valores por defecto o del histórico más reciente
                for col in suc_df.columns:
                    if col not in new_row and col != "FECHA" and col != "COD_SUC":
                        new_row[col] = suc_df[col].iloc[-1] if not suc_df.empty else np.nan
                df_combined_preds = pd.concat([df_combined_preds, pd.DataFrame([new_row])], ignore_index=True)
                df_combined_preds = df_combined_preds.sort_values("FECHA").reset_index(drop=True)

    return predictions


if __name__ == "__main__":
    print("🔄 Cargando datos desde Excel...")
    df = load_data_from_excel("data/DOTACION_EFECTIVIDAD.xlsx")
    df.columns = ["COD_SUC", "FECHA", "T_AO", "T_AO_VENTA", "DOTACION", "T_VISITAS", "P_EFECTIVIDAD"]  # Si tiene 7 columnas

    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df = df.dropna().reset_index(drop=True)
    print("✅ Datos cargados. Entrenando modelos T_AO, T_AO_VENTA, T_VISITAS y DOTACION...") # Modificado

    for target in ["T_AO", "T_AO_VENTA", "T_VISITAS", "DOTACION"]:  # Entrenar también para DOTACION # Modificado
        train_model_per_branch(df, target)
        print(f"🏁 Entrenamiento completado para {target}")

    print("\n🔮 Realizando predicciones futuras y calculando dotación necesaria...")
    unique_branches = df["COD_SUC"].unique()
    future_dates = pd.to_datetime(
        pd.date_range(start=df["FECHA"].max() + pd.Timedelta(days=1), periods=7, freq='D'))  # Próximos 7 días

    for branch_code in unique_branches:
        print(f"\n--- Predicciones para Sucursal: {branch_code} ---")
        # Modificado para incluir DOTACION en la predicción de prueba
        predicted_values = predict_future_values(df, branch_code, future_dates, ["T_AO", "T_AO_VENTA", "T_VISITAS", "DOTACION"])

        t_ao_pred_arr = np.array(predicted_values.get("T_AO", []))
        t_ao_venta_pred_arr = np.array(predicted_values.get("T_AO_VENTA", []))

        # Para estimar los parámetros de efectividad, usar un subconjunto del df histórico
        # que contenga los datos necesarios para la estimación del modelo sigmoide.
        df_historico_para_params = df[df["COD_SUC"] == branch_code][['DOTACION', 'T_AO', 'T_AO_VENTA']].dropna()
        if not df_historico_para_params.empty:
            params_efectividad_sucursal = estimar_parametros_efectividad(df_historico_para_params)
        else:
            params_efectividad_sucursal = None  # Esto hará que estimar_dotacion_optima use valores por defecto

        if np.any(~np.isnan(t_ao_pred_arr)) and np.any(~np.isnan(t_ao_venta_pred_arr)):
            # Pasar las predicciones de T_AO y T_AO_VENTA para el cálculo de la dotación óptima
            # Esto permitirá que estimar_dotacion_optima considere las particularidades de las predicciones.
            dotacion_optima_total, efectividad_promedio_total = estimar_dotacion_optima(
                t_ao_pred_arr,
                t_ao_venta_pred_arr,
                efectividad_deseada=0.8,  # Puedes ajustar la efectividad deseada aquí
                params_efectividad=params_efectividad_sucursal
            )
            print(f"  DOTACION Óptima Global (para el período futuro): {dotacion_optima_total}")
            print(f"  Efectividad Promedio Esperada (con dotación óptima): {efectividad_promedio_total:.2f}")
        else:
            print("  No hay suficientes predicciones válidas para calcular la dotación óptima global.")

        for i, date in enumerate(future_dates):
            t_ao_pred = predicted_values.get("T_AO", [np.nan])[i]
            t_ao_venta_pred = predicted_values.get("T_AO_VENTA", [np.nan])[i]
            t_visitas_pred = predicted_values.get("T_VISITAS", [np.nan])[i]
            t_dotacion_pred = predicted_values.get("DOTACION", [np.nan])[i] # Añadido para mostrar DOTACION predicha

            print(f"Fecha: {date.strftime('%Y-%m-%d')}")
            print(f"  T_AO Predicho: {t_ao_pred:.2f}")
            print(f"  T_AO_VENTA Predicho: {t_ao_venta_pred:.2f}")
            print(f"  T_VISITAS Predicho: {t_visitas_pred:.2f}")
            print(f"  DOTACION Predicha: {t_dotacion_pred:.2f}") # Añadido para mostrar DOTACION predicha


            # Calcular la dotación necesaria para cada día individualmente si es necesario
            if not np.isnan(t_ao_pred) and t_ao_pred > 0 and not np.isnan(t_ao_venta_pred):
                dotacion_necesaria_diaria, efectividad_diaria_resultante = estimar_dotacion_optima(
                    np.array([t_ao_pred]),
                    np.array([t_ao_venta_pred]),
                    efectividad_deseada=0.8,
                    params_efectividad=params_efectividad_sucursal
                )
                print(f"  DOTACION Necesaria Diaria: {dotacion_necesaria_diaria}")
                print(f"  Efectividad Diaria Esperada: {efectividad_diaria_resultante:.2f}")
            else:
                print("  No se puede calcular DOTACION necesaria (T_AO o T_AO_VENTA predicho es inválido o cero).")

    print("\n🏁 Proceso de entrenamiento, predicción y cálculo de dotación completado.")