from fastapi import FastAPI, Query
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from typing import Literal, List
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns  # para el heatmap

from io import BytesIO
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras import layers, models



# Carga de datos

df = pd.read_csv("data/crypto_limpio.csv")
resumen = pd.read_csv("data/cluster_resumen_monedas.csv")
metricas_arima = pd.read_csv("data/metricas_arima.csv")
metricas_lstm = pd.read_csv("data/metricas_lstm.csv")


def _serie_close_por_moneda(coin: str) -> pd.Series:
    data = df[df["coin"] == coin].copy().sort_values("time")
    s = pd.Series(data["close"].astype(float).values, index=pd.to_datetime(data["time"]))
    s.name = "close"
    return s


def _serie_cierre(coin: str) -> pd.Series:
    s = (
        df[df["coin"] == coin]
        .copy()
        .sort_values("time")["close"]
        .astype(float)
        .reset_index(drop=True)
    )
    s.name = "close"
    return s


# Normalizar nombres de columna "coin" en métricas
if "coin" not in metricas_arima.columns:
    primera_col_arima = metricas_arima.columns[0]
    metricas_arima = metricas_arima.rename(columns={primera_col_arima: "coin"})

if "coin" not in metricas_lstm.columns:
    primera_col_lstm = metricas_lstm.columns[0]
    metricas_lstm = metricas_lstm.rename(columns={primera_col_lstm: "coin"})

tabla_coins = (
    resumen.merge(metricas_arima, on="coin", how="left", suffixes=("", "_arima"))
    .merge(metricas_lstm, on="coin", how="left", suffixes=("", "_lstm"))
).fillna(0.0)


# Modelos de entrada/salida

class PredictRequest(BaseModel):
    perfil_riesgo: Literal["bajo", "medio", "alto"]
    dias_inversion: int = 7


class RecommendationItem(BaseModel):
    coin: str
    cluster_kmeans: int
    ret_mean: float
    ret_std: float
    comentario: str


class RecommendationResponse(BaseModel):
    perfil_riesgo: str
    dias_inversion: int
    recomendaciones: List[RecommendationItem]


class HealthResponse(BaseModel):
    status: str
    n_registros: int
    n_monedas: int


history: List[RecommendationResponse] = []


# FastAPI

app = FastAPI(
    title="API Recomendador de Criptomonedas",
    description="API del proyecto de Inteligencia de Negocios.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Utilidades

def _png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")


def generar_recomendaciones(perfil_riesgo: str, dias_inversion: int) -> RecommendationResponse:
    tabla = tabla_coins.copy()

    if perfil_riesgo == "bajo":
        tabla = tabla.sort_values(["ret_std", "ret_mean"], ascending=[True, False])
        comentario_base = "Perfil conservador: se prioriza estabilidad."
    elif perfil_riesgo == "alto":
        tabla = tabla.sort_values("ret_mean", ascending=False)
        comentario_base = "Perfil agresivo: se prioriza retorno esperado."
    else:
        tabla["score"] = tabla["ret_mean"] / tabla["ret_std"].replace(0, 1)
        tabla = tabla.sort_values("score", ascending=False)
        comentario_base = "Perfil intermedio: equilibrio entre retorno y riesgo."

    top = tabla.head(2)
    recomendaciones: List[RecommendationItem] = []
    for _, row in top.iterrows():
        comentario = (
            f"{comentario_base} "
            f"Retorno medio diario ≈ {row['ret_mean']:.4f}, "
            f"desviación ≈ {row['ret_std']:.4f}."
        )
        recomendaciones.append(
            RecommendationItem(
                coin=str(row["coin"]),
                cluster_kmeans=int(row["cluster_kmeans"]),
                ret_mean=float(row["ret_mean"]),
                ret_std=float(row["ret_std"]),
                comentario=comentario,
            )
        )

    resp = RecommendationResponse(
        perfil_riesgo=perfil_riesgo,
        dias_inversion=dias_inversion,
        recomendaciones=recomendaciones,
    )
    history.append(resp)
    return resp



# Endpoints básicos

@app.get("/health", response_model=HealthResponse)
def health_check():
    n_registros = len(df)
    n_monedas = df["coin"].nunique()
    return HealthResponse(status="ok", n_registros=n_registros, n_monedas=n_monedas)


@app.post("/predict", response_model=RecommendationResponse)
def predict(req: PredictRequest):
    return generar_recomendaciones(req.perfil_riesgo, req.dias_inversion)


@app.get("/recommendations", response_model=RecommendationResponse)
def get_recommendations(
    perfil_riesgo: str = Query(...),
    dias_inversion: int = Query(7),
):
    return generar_recomendaciones(perfil_riesgo, dias_inversion)


@app.get("/history", response_model=List[RecommendationResponse])
def get_history():
    return history



# Endpoints auxiliares para el dashboard

@app.get("/coins")
def get_coins():
    """Devuelve la lista de monedas disponibles en el dataset."""
    coins = sorted(df["coin"].str.upper().unique().tolist())
    return {"coins": coins}


@app.get("/plot/heatmap")
def plot_heatmap():
    """Mapa de calor de correlaciones entre precios de cierre."""
    tabla = df.pivot_table(index="time", columns="coin", values="close")
    corr = tabla.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Mapa de calor de correlaciones entre criptomonedas")
    return _png(fig)


# Predicción ARIMA y RNN

def _pred_arima_serie(
    s: pd.Series,
    horizon: int = 7,
    hist_dias: int = 30,
    order=(1, 1, 1),
):
    s = s.astype(float).reset_index(drop=True)

    modelo = ARIMA(s, order=order)
    res = modelo.fit()

    start = max(0, len(s) - hist_dias)
    end = len(s) - 1

    pred_hist = res.predict(start=start, end=end)
    real_hist = s.iloc[start : end + 1]

    mae = float(mean_absolute_error(real_hist, pred_hist))
    rmse = float(mean_squared_error(real_hist, pred_hist) ** 0.5)

    # Pronóstico FUTUROdías hacia adelante
    forecast = res.forecast(steps=horizon)
    fut_vals = forecast.values.astype(float)
    y_hat = float(fut_vals[-1])

    idx_hist = list(range(start, end + 1))

    return idx_hist, real_hist.values, pred_hist.values, fut_vals, y_hat, mae, rmse


def _pred_rnn_serie(
    s: pd.Series,
    horizon: int = 7,
    ventana: int = 10,
    tipo: str = "lstm",
    epochs: int = 25,       
    batch_size: int = 16,
):
    s = s.astype(float).reset_index(drop=True)
    valores = s.values.reshape(-1, 1)

    escala = MinMaxScaler()
    val_norm = escala.fit_transform(valores)

    if len(val_norm) <= ventana + 5:
        raise ValueError("Serie demasiado corta para RNN")

    X = []
    y = []
    for i in range(len(val_norm) - ventana):
        X.append(val_norm[i : i + ventana])
        y.append(val_norm[i + ventana])

    X = np.array(X)
    y = np.array(y)

    corte = int(len(X) * 0.8)
    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]

    modelo = models.Sequential()
    modelo.add(layers.Input(shape=(ventana, 1)))
    if tipo == "lstm":
        modelo.add(layers.LSTM(32))
    else:
        modelo.add(layers.GRU(32))
    modelo.add(layers.Dense(1))
    modelo.compile(optimizer="adam", loss="mse")

    modelo.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    
    y_pred_norm = modelo.predict(X_test, verbose=0)
    y_test_real = escala.inverse_transform(y_test)
    y_pred_des = escala.inverse_transform(y_pred_norm)

    mae = float(mean_absolute_error(y_test_real, y_pred_des))
    rmse = float(mean_squared_error(y_test_real, y_pred_des) ** 0.5)

    start_idx = ventana + corte
    idx_hist = list(range(start_idx, start_idx + len(X_test)))

    
    ultimos = val_norm[-ventana:].copy()
    actual = ultimos.copy()
    fut_norm = []

    for _ in range(horizon):
        x_in = actual.reshape(1, ventana, 1)
        y_hat_step = float(modelo.predict(x_in, verbose=0)[0, 0])
        fut_norm.append(y_hat_step)
        actual = np.vstack([actual[1:], [[y_hat_step]]])

    fut_vals = escala.inverse_transform(np.array(fut_norm).reshape(-1, 1)).flatten()
    y_hat = float(fut_vals[-1])

    return idx_hist, y_test_real.flatten(), y_pred_des.flatten(), fut_vals, y_hat, mae, rmse


def _recomendacion(precio_actual: float, precio_futuro: float, umbral: float = 2.0):
    delta_pct = (precio_futuro - precio_actual) / precio_actual * 100
    if delta_pct > umbral:
        txt = "comprar"
    elif delta_pct < -umbral:
        txt = "no comprar"
    else:
        txt = "mantener / esperar"
    return float(delta_pct), txt


# Endpoint principal de predicción

@app.get("/series/predict")
def series_predict(
    coins: str = Query("btc,eth,ada"),
    model: str = Query("lstm", regex="^(arima|lstm|gru)$"),
    horizon: int = Query(7, ge=1, le=30),
):
    """
    Devuelve:
    - tramo histórico real (para métricas)
    - predicción sobre ese tramo
    - serie FUTURA de horizonte 'horizon'
    """
    coins_list = [c.strip().lower() for c in coins.split(",") if c.strip()]
    resultados = []

    for c in coins_list:
        s = _serie_cierre(c)
        if s.empty:
            continue

        fechas_full = _serie_close_por_moneda(c)

        if model == "arima":
            (
                idx_hist,
                y_real_hist,
                y_pred_hist,
                y_fut,
                y_hat,
                mae,
                rmse,
            ) = _pred_arima_serie(s, horizon=horizon)
        else:
            (
                idx_hist,
                y_real_hist,
                y_pred_hist,
                y_fut,
                y_hat,
                mae,
                rmse,
            ) = _pred_rnn_serie(s, horizon=horizon, tipo=model)

        fechas_hist = []
        for idx in idx_hist:
            if idx < len(fechas_full.index):
                fechas_hist.append(str(fechas_full.index[idx].date()))

        
        last_ts = fechas_full.index[-1]
        future_index = pd.date_range(start=last_ts + pd.Timedelta(days=1), periods=horizon, freq="D")
        fechas_fut = [str(d.date()) for d in future_index]

        precio_actual = float(s.iloc[-1])
        delta_pct, reco = _recomendacion(precio_actual, float(y_hat))

        resultados.append(
            {
                "coin": c.upper(),
                "fechas_hist": fechas_hist,
                "fechas_fut": fechas_fut,
                "y_real": [float(v) for v in y_real_hist],
                "y_pred": [float(v) for v in y_pred_hist],
                "y_pred_fut": [float(v) for v in y_fut],
                "precio_actual": precio_actual,
                "precio_pred_horizonte": float(y_hat),
                "variacion_pct": delta_pct,
                "recomendacion": reco,
                "mae": float(mae),
                "rmse": float(rmse),
            }
        )

    return JSONResponse({"model": model, "horizon": horizon, "resultados": resultados})
