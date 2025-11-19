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
from io import BytesIO
from statsmodels.tsa.arima.model import ARIMA

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

app = FastAPI(
    title="API Recomendador de Criptomonedas",
    description="API del proyecto de Inteligencia de Negocios (BTC, ETH, ADA).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/plot/price/all")
def plot_price_all(dias: int = 60):
    coins = ["btc", "eth", "ada"]
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 9), sharex=True)
    for ax, c in zip(axes, coins):
        s = _serie_cierre(c).tail(dias)
        ax.plot(s.values, label="Cierre")
        ax.set_title(f"Precio cierre – {c.upper()} (últimos {dias} días)")
        ax.set_ylabel("Precio")
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("Fecha (índice relativo)")
    fig.tight_layout()
    return _png(fig)

def _plot_arima_coin(coin: str, test_size: int, p: int, d: int, q: int):
    s = _serie_cierre(coin)
    n = len(s)
    corte = n - test_size if test_size > 0 else int(n * 0.2)
    train = s.iloc[:corte]
    test = s.iloc[corte:]
    modelo = ARIMA(train, order=(p, d, q))
    res = modelo.fit()
    pron = res.forecast(steps=len(test))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(train)), train.values, label="Train")
    ax.plot(range(len(train), len(train) + len(test)), test.values, label="Test")
    ax.plot(range(len(train), len(train) + len(test)), pron.values, label="Pronóstico")
    ax.set_title(f"ARIMA({p},{d},{q}) – {coin.upper()}")
    ax.set_ylabel("Precio")
    ax.legend(loc="upper left")
    return _png(fig)

@app.get("/plot/forecast/arima/btc")
def plot_arima_btc(test_size: int = 20, p: int = 1, d: int = 1, q: int = 1):
    return _plot_arima_coin("btc", test_size, p, d, q)

@app.get("/plot/forecast/arima/eth")
def plot_arima_eth(test_size: int = 20, p: int = 1, d: int = 1, q: int = 1):
    return _plot_arima_coin("eth", test_size, p, d, q)

@app.get("/plot/forecast/arima/ada")
def plot_arima_ada(test_size: int = 20, p: int = 1, d: int = 1, q: int = 1):
    return _plot_arima_coin("ada", test_size, p, d, q)

@app.get("/plot/forecast/arima/all")
def plot_arima_all(test_size: int = 20, p: int = 1, d: int = 1, q: int = 1):
    coins = ["btc", "eth", "ada"]
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=False)
    for ax, c in zip(axes, coins):
        s = _serie_cierre(c)
        n = len(s)
        corte = n - test_size if test_size > 0 else int(n * 0.2)
        train = s.iloc[:corte]
        test = s.iloc[corte:]
        modelo = ARIMA(train, order=(p, d, q))
        res = modelo.fit()
        pron = res.forecast(steps=len(test))
        ax.plot(range(len(train)), train.values, label="Train")
        ax.plot(range(len(train), len(train) + len(test)), test.values, label="Test")
        ax.plot(range(len(train), len(train) + len(test)), pron.values, label="Pronóstico")
        ax.set_title(f"ARIMA({p},{d},{q}) – {c.upper()}")
        ax.set_ylabel("Precio")
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("Fecha (índice relativo)")
    fig.tight_layout()
    return _png(fig)

@app.get("/series/price/all")
def series_price_all(dias: int = 60):
    coins = ["btc", "eth", "ada"]
    out = {}
    for c in coins:
        s = _serie_close_por_moneda(c).tail(dias)
        out[c] = {
            "x": [str(ts.date()) for ts in s.index],
            "y": [float(v) for v in s.values],
        }
    return JSONResponse(out)

@app.get("/series/forecast/arima/all")
def series_forecast_arima_all(test_size: int = 20, p: int = 1, d: int = 1, q: int = 1):
    coins = ["btc", "eth", "ada"]
    out = {}
    for c in coins:
        s = _serie_cierre(c)
        n = len(s)
        corte = n - test_size if test_size > 0 else int(n * 0.2)
        train = s.iloc[:corte].astype(float).tolist()
        test = s.iloc[corte:].astype(float).tolist()
        modelo = ARIMA(s.iloc[:corte], order=(p, d, q))
        res = modelo.fit()
        pron = res.forecast(steps=len(test)).astype(float).tolist()
        out[c.upper()] = {"train": train, "test": test, "forecast": pron}
    return JSONResponse(out)
