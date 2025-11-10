from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Literal, List
import pandas as pd

df = pd.read_csv("data/crypto_limpio.csv")
resumen = pd.read_csv("data/cluster_resumen_monedas.csv")

metricas_arima = pd.read_csv("data/metricas_arima.csv")
metricas_lstm = pd.read_csv("data/metricas_lstm.csv")


if "coin" not in metricas_arima.columns:
    primera_col_arima = metricas_arima.columns[0]
    metricas_arima = metricas_arima.rename(columns={primera_col_arima: "coin"})

if "coin" not in metricas_lstm.columns:
    primera_col_lstm = metricas_lstm.columns[0]
    metricas_lstm = metricas_lstm.rename(columns={primera_col_lstm: "coin"})


tabla_coins = (
    resumen.merge(metricas_arima, on="coin", how="left", suffixes=("", "_arima"))
           .merge(metricas_lstm, on="coin", how="left", suffixes=("", "_lstm"))
)

tabla_coins = tabla_coins.fillna(0.0)

#modelos de entrada y de salida


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


#se crea la app


app = FastAPI(
    title="API Recomendador de Criptomonedas",
    description="API del proyecto de Inteligencia de Negocios (BTC, ETH, ADA).",
    version="0.1.0",
)

# Logica de negocio de recomendaciones


def generar_recomendaciones(perfil_riesgo: str, dias_inversion: int) -> RecommendationResponse:
    """
    Conecta el resumen del sprint 2 con una lógica sencilla de recomendación.
    Se usan los promedios y la volatilidad que ya se calculó.
    """

    tabla = tabla_coins.copy()

    if perfil_riesgo == "bajo":
        # Se ordena por volatilidad baja y dentro de eso por retorno medio alto
        tabla = tabla.sort_values(["ret_std", "ret_mean"], ascending=[True, False])
        comentario_base = "Perfil conservador: se priorizan monedas más estables."
    elif perfil_riesgo == "alto":
        # Se ordena por mayor retorno medio aceptando más volatilidad
        tabla = tabla.sort_values("ret_mean", ascending=False)
        comentario_base = "Perfil agresivo: se priorizan monedas con más retorno esperado."
    else:  # medio
        # Pequeño puntaje casero retorno / volatilidad
        tabla["score"] = tabla["ret_mean"] / tabla["ret_std"].replace(0, 1)
        tabla = tabla.sort_values("score", ascending=False)
        comentario_base = "Perfil intermedio: equilibrio entre retorno y riesgo."

    # se queda con las 2 mejores monedas
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
                coin=row["coin"],
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
    """
    Endpoint para saber si el servicio está vivo.
    """
    n_registros = len(df)
    n_monedas = df["coin"].nunique()
    return HealthResponse(
        status="ok",
        n_registros=n_registros,
        n_monedas=n_monedas,
    )


@app.post("/predict", response_model=RecommendationResponse)
def predict(req: PredictRequest):
    """
    Recibe el perfil de riesgo del usuario y devuelve recomendaciones.
    Por ahora se usa el resumen estadístico del sprint 2.
    """
    print(f"[PREDICT] perfil={req.perfil_riesgo}, dias_inversion={req.dias_inversion}")
    return generar_recomendaciones(req.perfil_riesgo, req.dias_inversion)


@app.get("/recommendations", response_model=RecommendationResponse)
def get_recommendations(
    perfil_riesgo: str = Query(..., description="Perfil de riesgo: bajo, medio o alto"),
    dias_inversion: int = Query(7, description="Cuántos días pienso mantener la inversión"),
):
    """
    Versión GET del recomendador, usando query params.
    """
    return generar_recomendaciones(perfil_riesgo, dias_inversion)


@app.get("/history", response_model=List[RecommendationResponse])
def get_history():
    """
    Devuelve todas las recomendaciones generadas desde que se levantó la API.
    Solo se guarda en memoria mientras el servidor está encendido.
    """
    return history
