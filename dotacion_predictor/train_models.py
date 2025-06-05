import os
import pandas as pd
import joblib
import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from preprocessing import prepare_features
from preprocessing import _add_lag_features
from utils import estimar_dotacion_optima, estimar_parametros_efectividad
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data_from_excel(file_path: str) -> pd.DataFrame:
    """Carga datos desde archivo Excel"""
    df = pd.read_excel(file_path)
    df.columns = ["COD_SUC", "FECHA", "T_AO", "T_AO_VENTA", "DOTACION", "T_VISITAS", "P_EFECTIVIDAD"]
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    return df.dropna().reset_index(drop=True)


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, tscv: TimeSeriesSplit) -> dict:
    """Evalúa el modelo usando validación cruzada temporal"""
    mae_scores = []
    r2_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    return {
        'mae': np.mean(mae_scores),
        'r2': np.mean(r2_scores),
        'mae_std': np.std(mae_scores),
        'r2_std': np.std(r2_scores)
    }


def objective(trial, X: pd.DataFrame, y: pd.Series, tscv: TimeSeriesSplit) -> float:
    """Función objetivo para optimización con Optuna"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
    }

    model = XGBRegressor(**params, objective='reg:squarederror', n_jobs=-1, random_state=42)

    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(mean_absolute_error(y_test, y_pred))

    return np.mean(scores)


def train_model_per_branch(df: pd.DataFrame, target: str) -> None:
    """Entrena modelos por sucursal usando ensamblado y optimización avanzada"""
    unique_branches = df["COD_SUC"].unique()

    for sucursal in unique_branches:
        suc_df = df[df["COD_SUC"] == sucursal].copy()
        X, y = prepare_features(suc_df, target)

        if len(X) < 30:  # Requerimos al menos 30 muestras
            print(f"⚠️ Insuficientes datos para {sucursal} - {target}.")
            continue

        # Validación cruzada temporal
        tscv = TimeSeriesSplit(n_splits=min(5, len(X) // 3))  # Asegurar al menos 3 muestras por fold

        # Optimización con Optuna solo para XGBoost
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X, y, tscv), n_trials=30, timeout=600)

        # Modelos base para stacking
        estimators = [
            ('xgb', XGBRegressor(**study.best_params, objective='reg:squarederror', n_jobs=-1, random_state=42)),
            ('lgbm', LGBMRegressor(random_state=42)),
            ('rf', RandomForestRegressor(random_state=42))
        ]

        # Modelo final con stacking
        final_model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(),
            n_jobs=-1
        )

        # Entrenamiento y evaluación
        final_model.fit(X, y)
        metrics = evaluate_model(final_model, X, y, tscv)

        print(f"✅ Modelo {target} para sucursal {sucursal} - "
              f"MAE: {metrics['mae']:.2f} ± {metrics['mae_std']:.2f}, "
              f"R²: {metrics['r2']:.2f} ± {metrics['r2_std']:.2f}")


        # Guardar modelo
        filename = f"{MODEL_DIR}/predictor_{target}_{sucursal}.pkl"
        joblib.dump(final_model, filename)


def predict_future_values(df: pd.DataFrame, branch_code: str, future_dates: list, targets: list) -> dict:
    """Predice valores futuros para una sucursal"""
    predictions = {}
    suc_df = df[df["COD_SUC"] == branch_code].copy()
    df_combined_preds = suc_df.copy()

    for current_date in future_dates:
        future_row = pd.DataFrame([{
            "FECHA": current_date,
            "COD_SUC": branch_code,
            "T_AO": np.nan,
            "T_AO_VENTA": np.nan,
            "T_VISITAS": np.nan,
            "DOTACION": np.nan
        }])

        df_temp = pd.concat([df_combined_preds.tail(30), future_row], ignore_index=True)
        df_temp = df_temp.sort_values("FECHA").reset_index(drop=True)

        for target in targets:
            model_filename = f"{MODEL_DIR}/predictor_{target}_{branch_code}.pkl"
            if not os.path.exists(model_filename):
                predictions.setdefault(target, []).append(np.nan)
                continue

            model = joblib.load(model_filename)
            X_future, _ = prepare_features(df_temp, target, is_prediction=True)

            if X_future.empty:
                predictions.setdefault(target, []).append(np.nan)
                continue

            pred = model.predict(X_future.iloc[[-1]])[0]
            predictions.setdefault(target, []).append(pred)

            # Actualizar df_combined_preds con la predicción
            mask = df_combined_preds['FECHA'] == current_date
            if mask.any():
                df_combined_preds.loc[mask, target] = pred
            else:
                new_row = {"FECHA": current_date, "COD_SUC": branch_code, target: pred}
                for col in suc_df.columns:
                    if col not in new_row and col != "FECHA" and col != "COD_SUC":
                        new_row[col] = suc_df[col].iloc[-1] if not suc_df.empty else np.nan
                df_combined_preds = pd.concat([df_combined_preds, pd.DataFrame([new_row])], ignore_index=True)
                df_combined_preds = df_combined_preds.sort_values("FECHA").reset_index(drop=True)

    return predictions


if __name__ == "__main__":
    print("🔄 Cargando datos desde Excel...")
    df = load_data_from_excel("data/DOTACION_EFECTIVIDAD.xlsx")
    print("✅ Datos cargados. Entrenando modelos...")

    for target in ["T_AO", "T_AO_VENTA", "T_VISITAS", "DOTACION"]:
        train_model_per_branch(df, target)
        print(f"🏁 Entrenamiento completado para {target}")

    print("\n🔮 Realizando predicciones futuras...")
    unique_branches = df["COD_SUC"].unique()
    future_dates = pd.date_range(
        start=df["FECHA"].max() + pd.Timedelta(days=1),
        periods=7
    ).tolist()

    for branch_code in unique_branches:
        print(f"\n--- Predicciones para Sucursal: {branch_code} ---")
        predicted_values = predict_future_values(
            df, branch_code, future_dates,
            ["T_AO", "T_AO_VENTA", "T_VISITAS", "DOTACION"]
        )

        # Estimación de parámetros de efectividad
        df_historico = df[df["COD_SUC"] == branch_code][['DOTACION', 'T_AO', 'T_AO_VENTA']].dropna()
        params_efectividad = estimar_parametros_efectividad(df_historico) if not df_historico.empty else None

        # Procesar predicciones
        t_ao_pred = np.array(predicted_values.get("T_AO", []))
        t_ao_venta_pred = np.array(predicted_values.get("T_AO_VENTA", []))

        if np.any(~np.isnan(t_ao_pred)) and np.any(~np.isnan(t_ao_venta_pred)):
            dotacion_optima, efectividad = estimar_dotacion_optima(
                t_ao_pred, t_ao_venta_pred,
                efectividad_deseada=0.8,
                params_efectividad=params_efectividad
            )
            print(f"  DOTACION Óptima Global: {dotacion_optima:.2f}")
            print(f"  Efectividad Esperada: {efectividad:.2f}")

        for i, date in enumerate(future_dates):
            print(f"\nFecha: {date.strftime('%Y-%m-%d')}")
            for target in ["T_AO", "T_AO_VENTA", "T_VISITAS", "DOTACION"]:
                val = predicted_values.get(target, [np.nan])[i]
                print(f"  {target} Predicho: {val:.2f}")

            # Cálculo diario de dotación óptima
            if (not np.isnan(t_ao_pred[i])) and (t_ao_pred[i] > 0) and (not np.isnan(t_ao_venta_pred[i])):
                dotacion_diaria, efectividad_diaria = estimar_dotacion_optima(
                    np.array([t_ao_pred[i]]),
                    np.array([t_ao_venta_pred[i]]),
                    efectividad_deseada=0.8,
                    params_efectividad=params_efectividad
                )
                print(f"  DOTACION Diaria Óptima: {dotacion_diaria:.2f}")
                print(f"  Efectividad Diaria: {efectividad_diaria:.2f}")

    print("\n🏁 Proceso completado.")