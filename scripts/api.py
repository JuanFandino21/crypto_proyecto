from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Literal, List
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi.responses import Response
from io import BytesIO
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_csv("data/crypto_limpio.csv")
resumen = pd.read_csv("data/cluster_resumen_monedas.csv")
metricas_arima = pd.read_csv("data/metricas_arima.csv")
metricas_lstm = pd.read_csv("data/metricas_lstm.csv")

# normalizo columnas coin en métricas por si vienen con otro nombre
if "coin" not in metricas_arima.columns:
    metricas_arima = metricas_arima.rename(columns={metricas_arima.columns[0]: "coin"})
if "coin" not in metricas_lstm.columns:
    metricas_lstm = metricas_lstm.rename(columns={metricas_lstm.columns[0]: "coin"})

# tabla combinada
tabla_coins = (
    resumen.merge(metricas_arima, on="coin", how="left", suffixes=("", "_arima"))
           .merge(metricas_lstm, on="coin", how="left", suffixes=("", "_lstm"))
).fillna(0.0)


def _pick_col(dframe: pd.DataFrame, opciones: list[str]) -> str:
    for c in opciones:
        if c in dframe.columns:
            return c
    raise KeyError(f"No se encontró ninguna de estas columnas: {opciones}")

def _serie_cierre(coin: str) -> pd.Series:
    fecha_col = _pick_col(df, ["time", "date", "Date", "fecha", "Fecha", "timestamp"])
    close_col = _pick_col(df, ["close", "Close", "precio_cierre", "Precio cierre", "price_close"])
    sub = df[df["coin"].str.lower() == coin.lower()].copy()
    if sub.empty:
        raise ValueError(f"Sin datos para '{coin}'")
    sub[fecha_col] = pd.to_datetime(sub[fecha_col], errors="coerce")
    sub = sub.dropna(subset=[fecha_col]).sort_values(fecha_col)
    s = sub.set_index(fecha_col)[close_col].astype(float)
    s.name = coin.upper()
    return s

# modelos de entrada/salida 
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

#app
app = FastAPI(
    title="API Recomendador de Criptomonedas",
    description="API del proyecto de Inteligencia de Negocios (BTC, ETH, ADA).",
    version="0.1.0",
)

# lógica simple de recomendación 
def generar_recomendaciones(perfil_riesgo: str, dias_inversion: int) -> RecommendationResponse:
    tabla = tabla_coins.copy()
    if perfil_riesgo == "bajo":
        tabla = tabla.sort_values(["ret_std", "ret_mean"], ascending=[True, False])
        comentario_base = "Perfil conservador: se priorizan monedas más estables."
    elif perfil_riesgo == "alto":
        tabla = tabla.sort_values("ret_mean", ascending=False)
        comentario_base = "Perfil agresivo: se priorizan monedas con más retorno esperado."
    else:
        tabla["score"] = tabla["ret_mean"] / tabla["ret_std"].replace(0, 1)
        tabla = tabla.sort_values("score", ascending=False)
        comentario_base = "Perfil intermedio: equilibrio entre retorno y riesgo."

    top = tabla.head(2)
    recomendaciones: List[RecommendationItem] = []
    for _, row in top.iterrows():
        comentario = (
            f"{comentario_base} Retorno medio diario ≈ {row['ret_mean']:.4f}, "
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

# endpoints 
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        n_registros=len(df),
        n_monedas=df["coin"].nunique(),
    )

@app.post("/predict", response_model=RecommendationResponse)
def predict(req: PredictRequest):
    return generar_recomendaciones(req.perfil_riesgo, req.dias_inversion)

@app.get("/recommendations", response_model=RecommendationResponse)
def get_recommendations(
    perfil_riesgo: str = Query(..., description="bajo | medio | alto"),
    dias_inversion: int = Query(7, description="días que pienso mantener la inversión"),
):
    return generar_recomendaciones(perfil_riesgo, dias_inversion)

@app.get("/history", response_model=List[RecommendationResponse])
def get_history():
    return history

# endpoints de gráficas
@app.get("/plot/price/all")
def plot_price_all(dias: int = 60):
    coins = sorted(df["coin"].str.lower().unique())
    n = len(coins)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 3*n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, c in zip(axes, coins):
        s = _serie_cierre(c).tail(dias)
        ax.plot(s.index, s.values, label=c.upper())
        ax.set_ylabel("Precio")
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("Fecha")
    fig.suptitle(f"Precio de cierre – todas (últimos {dias} días)")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")

@app.get("/plot/forecast/arima/all")
def plot_arima_all(test_size: int = 20, p: int = 1, d: int = 1, q: int = 1):
    coins = sorted(df["coin"].str.lower().unique())
    n = len(coins)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 3*n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, c in zip(axes, coins):
        s = _serie_cierre(c).dropna()
        npts = len(s)
        corte = max(1, min(npts - 1, int(npts * (1 - test_size/100))))  
        train = s.iloc[:corte]
        test  = s.iloc[corte:]

        modelo = ARIMA(train, order=(p, d, q)).fit()
        pron = modelo.forecast(steps=len(test))

        ax.plot(train.index, train.values, label="Train")
        ax.plot(test.index,  test.values,  label="Test")
        ax.plot(test.index,  pron.values,  label="Pronóstico")
        ax.set_title(f"ARIMA({p},{d},{q}) – {c.upper()}")
        ax.set_ylabel("Precio")
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("Fecha")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")
