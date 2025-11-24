**Proyecto: Análisis y recomendación de criptomonedas**

Este proyecto se centra en analizar varias criptomonedas reales y construir, paso a paso, un sistema que permita:

    Explorar el comportamiento histórico de precios.
    Probar modelos de predicción de series de tiempo.
    Generar señales sencillas de compra/mantenimiento/no compra.
    Visualizar todo en un dashboard web conectado a una API en FastAPI.

Se trabaja con un conjunto de 8 criptomonedas:

    ADA, BNB, BTC, DOGE, ETH, LTC, SOL y XRP

A continuación se describe lo que se hizo en cada sprint.

**Sprint 1 – Recolección y limpieza estricta de los datos**

En el primer sprint el objetivo fue conseguir datos reales de criptomonedas, limpiarlos muy bien y dejarlos listos en un solo archivo para los siguientes sprints.

**¿Qué se hizo?**

    Se escogió el proyecto de criptomonedas y predicción del comportamiento en el mercado.
    Se trabajó con un conjunto de monedas reales: ADA, BNB, BTC, DOGE, ETH, LTC, SOL y XRP.
    Se usaron dos APIs como fuentes de información: CryptoCompare y Binance:
        Una para obtener precios históricos (apertura, cierre, máximo, mínimo).
        Otra como fuente de respaldo para tener más flexibilidad.
    Se programó la descarga de datos por moneda y por día, guardando información como:
        Fecha
        Precio de apertura
        Precio de cierre
        Máximo
        Mínimo
        Volumen
        Nombre de la moneda

**Limpieza y estructura de la base de datos**

En este sprint se puso mucho cuidado en que el archivo final quedara totalmente limpio, porque de eso depende todo lo demás. Se hizo lo siguiente:

    Conversión de la columna de fechas a un formato de fecha correcto.
    Conversión de columnas numéricas para asegurar que realmente fueran números.
     Eliminación de:
        Filas duplicadas.
        Filas con valores vacíos o inconsistentes.
    Revisión de que las fechas estuvieran en orden y sin saltos extraños.
    Validaciones de consistencia, por ejemplo:
        El máximo ≥ el mínimo.
        El volumen no fuera negativo.

Al final se dejó un solo archivo principal:
    data/crypto_limpio.csv

con todas las monedas, ordenadas por moneda y por fecha.

**Mini análisis exploratorio**

Para verificar que todo estuviera bien, se hizo un primer vistazo:

    Se revisaron las primeras filas de la tabla.
    Se miraron estadísticas simples (promedios, mínimos, máximos).
    Se graficó el precio de cierre por moneda para ver que las curvas fueran sensatas.

**Resultado del Sprint 1**

    Dataset limpio, sin basura y sin huecos, en un solo CSV.
    Datos listos para análisis más avanzados (clústeres, modelos, API, dashboard).
    Estructura organizada dentro de la carpeta data/ para reutilizarla en los siguientes sprints.

**Sprint 2 – Análisis, clústeres y modelos de series de tiempo**

En el sprint 2 el objetivo fue empezar a sacarle jugo a los datos: agrupar las monedas según su comportamiento y probar modelos que intentan predecir el precio.

**Organización de los datos para este sprint**

    Se cargó el archivo limpio crypto_limpio.csv.
    Se reorganizaron los datos para trabajar por moneda y por fecha.
    Se construyeron tablas donde:
        Las filas son fechas.
        Las columnas son monedas.
        Los valores son precios de cierre o volúmenes.

**Resumen por moneda y clústeres**

Primero se creó un resumen por moneda, calculando:

    Retorno medio diario (qué tanto sube o baja en promedio).
    Desviación estándar del retorno (qué tan “nerviosa” es la moneda).
    Volumen medio negociado.
    Precio medio de cierre.

Con esos datos se armó una tabla resumen y se guardó en:

    data/cluster_resumen_monedas.csv

Sobre esa misma tabla se aplicaron técnicas de clusterización para agrupar las monedas según su comportamiento:

    K-Means → separa las monedas en grupos según sus estadísticas.
    DBSCAN → busca grupos por densidad.
    Jerárquico → agrupa monedas de forma progresiva según su parecido.

Cada moneda queda con etiquetas de grupo, lo que ayuda a responder:

    Qué monedas se parecen más entre sí.
    Cuáles son más estables y cuáles más volátiles.

**Modelos de series de tiempo (ARIMA)**

En este sprint también se usaron modelos de series de tiempo clásicos para tratar de predecir el precio de cierre.

Para cada moneda:

    Se tomó la serie de precios de cierre.
    Se dividió en:
        Una parte de entrenamiento (la mayor parte de la historia).
        Una parte final de prueba.
    Se ajustó un modelo ARIMA sencillo.
    Se generaron pronósticos sobre la parte de prueba.
    Se comparó lo pronosticado contra lo real usando:
        MAE (error medio absoluto).
        RMSE (error cuadrático medio).
        MAPE (error porcentual medio).

Las métricas se guardaron en:

    data/metricas_arima.csv

**Modelo LSTM (RNN) sencillo**

Además del modelo clásico, se probó una red neuronal recurrente LSTM:

    Se prepararon ventanas de varios días seguidos para que la red viera “tramos” de la serie.
    Se normalizaron los datos a un rango estable.
    Se dividió en entrenamiento y prueba.
    Se entrenó un modelo LSTM simple para predecir el siguiente valor de cierre.
    Se compararon los datos reales contra los predichos:
        Se graficó la curva real y la curva predicha.
        e calcularon métricas como RMSE, MAE y MAPE.

Los resultados de LSTM se guardaron en:

    data/metricas_lstm.csv

**Organización del proyecto con Poetry**

En este sprint también se organizó el proyecto para que todas las librerías se manejen con Poetry, de modo que:

    Las dependencias quedan definidas en pyproject.toml.
    Cualquiera que clone el proyecto puede instalar todo con un solo comando.

**Resultado del Sprint 2**

    Resumen estadístico por moneda para entender su comportamiento.
    Agrupación de monedas con varias técnicas de clusterización.
    Dos tipos de modelos para predicción de precios:
        Uno clásico (ARIMA).
        Uno basado en redes neuronales (LSTM).
    Métricas y resultados guardados en CSV para usarlos en la API y el dashboard.
    Proyecto estructurado con Poetry para manejo de dependencias.

**Sprint 3 – API de recomendaciones y modelos de predicción**

En el sprint 3 el objetivo fue sacar lo trabajado del cuaderno y llevarlo a una API, es decir, un servicio que otras aplicaciones puedan consumir.

**Estructura de la API**

Se creó el archivo:

    scripts/api.py

y se construyó una API usando FastAPI.
Al iniciar, la API carga:

    data/crypto_limpio.csv
    data/cluster_resumen_monedas.csv
    data/metricas_arima.csv
    data/metricas_lstm.csv

Con esos archivos se arma una tabla general tabla_coins que junta, por moneda:

    Estadísticas (retorno, desviación, volumen, etc.).
    Etiquetas de clúster.
    Métricas de modelos ARIMA y LSTM.

Endpoints principales

GET /health

    Verifica que el servicio esté vivo.
    Devuelve un JSON con:
        status → por ejemplo "ok".
        n_registros → cantidad de filas del dataset.
        n_monedas → cuántas monedas diferentes hay.

GET /coins

    Devuelve la lista de monedas disponibles en el dataset (ADA, BNB, BTC, DOGE, ETH, LTC, SOL, XRP).
    Se usa para llenar dinámicamente el selector del dashboard.

POST /predict

    Recibe un JSON con:
        perfil_riesgo: "bajo", "medio" o "alto".
        dias_inversion: número de días pensado para la inversión.

    Según ese perfil, la API consulta la tabla de monedas y arma una recomendación.
    Devuelve:

        El perfil recibido.
        Los días de inversión.
        Una lista de monedas recomendadas con:
            coin
            cluster_kmeans
            ret_mean
            ret_std
            comentario explicando la recomendación.

GET /recommendations

    Hace lo mismo que /predict, pero usando parámetros en la URL.

        Ejemplo:
        http://127.0.0.1:8000/recommendations?perfil_riesgo=medio&dias_inversion=7

GET /history

    Devuelve una lista con todas las recomendaciones generadas desde que se encendió la API.

GET /series/predict

    Endpoint dedicado al dashboard de predicciones.

    Parámetros:
        coins: lista separada por comas (por ejemplo btc,eth,ada).
        model: "arima", "lstm" o "gru".
        horizon: cuántos días hacia adelante se quiere predecir.
    Para cada moneda:
        Usa el modelo seleccionado (ARIMA, LSTM o GRU).
        Calcula un historial de valores reales y predichos.
        Obtiene un precio futuro al horizonte indicado.
        Calcula:
            Variación porcentual esperada.
            Señal sencilla: comprar, no comprar, mantener/esperar.
            Métricas de error (MAE y RMSE).

La respuesta es un JSON estructurado para que el frontend pueda dibujar las curvas y llenar la tabla de señales de compra.

**Resultado del Sprint 3**

    API funcional que se levanta en local y responde en formato JSON.
    Integración de datos limpios, estadísticas, clústeres y modelos de predicción.
    Endpoints claros:
        /health
        /coins
        /predict
        /recommendations
        /history
        /series/predict
    Documentación automática de FastAPI en http://127.0.0.1:8000/docs.

**Sprint 4 – Dashboard interactivo de predicciones y señales de compra**

En el sprint 4 el objetivo fue construir una interfaz visual simple pero clara, conectada directamente a la API, para que cualquier usuario pueda:

    Ver el estado general del servicio.
    Elegir un modelo de predicción y un horizonte de días.
    Seleccionar las monedas que quiere analizar.
    Ver las curvas de precios reales vs. predichos.
    Recibir señales de compra/mantenimiento/no compra.
    Explorar un mapa de calor de correlaciones entre las criptomonedas.

**Estructura general del dashboard**

El frontend está en la carpeta web/ y se compone principalmente de:

    web/index.html
    web/styles.css
    web/app.js

El tema visual es oscuro, con tarjetas, tipografía legible y un estilo alineado a un tablero de análisis financiero.

**Secciones del dashboard**

**Estado del servicio**

    Tarjeta superior que consume el endpoint /health y muestra:

        Número total de registros del dataset.
        Número de monedas analizadas.
        Estado del servicio (ok).

    También enciende un LED verde en la esquina superior derecha cuando el backend responde correctamente.

**Predicciones y señales de compra**

    Bloque principal del tablero. Incluye:

    Selector múltiple de monedas (<select multiple>), alimentado dinámicamente desde /coins.
    Selector de modelo:
        LSTM
        GRU
        ARIMA
    Campo Horizonte (días) para indicar cuántos días hacia adelante se quiere proyectar.
    Filtros opcionales de rango de fechas (Desde / Hasta) para acotar lo que se dibuja en el gráfico.
    Botón “Ver predicciones” que dispara la consulta a /series/predict.

    Una vez se recibe la respuesta, el dashboard muestra:

        Resumen arriba del gráfico
        modelo: LSTM • horizonte: 7 días
        Gráfico interactivo Plotly donde, para cada moneda seleccionada:
            La línea sólida corresponde a los valores reales.
            La línea punteada corresponde a los valores predichos por el modelo escogido.
            Se puede hacer zoom, mover el gráfico y descargar la figura.
        Tabla de señales de compra, con columnas:
            Moneda
            Precio actual
            Predicción (precio al horizonte)
            Variación %
            Recomendación (comprar / mantener / no comprar)
            MAE
            RMSE

    Esta sección es la que resume todo el trabajo de modelos del backend y lo lleva a algo entendible para un usuario: ver curvas y leer una señal simple.

Mapa de calor de correlaciones

    Sección inferior que muestra un heatmap generado en el backend con Matplotlib a partir del pivot de precios de cierre.

    Consume el endpoint /plot/heatmap.
    Cada eje muestra las monedas (ADA, BNB, BTC, DOGE, ETH, LTC, SOL, XRP).
    Cada celda indica qué tan correlacionadas están dos monedas:
        Valores cercanos a 1 → se mueven casi igual.
        Valores bajos → comportamientos más diferentes.
    Las celdas incluyen el valor numérico de la correlación y una barra de colores para orientar al usuario.

    Debajo se incluye un texto explicativo pensado para alguien que no es experto en trading ni en estadística.

**Ejecutar el proyecto**
**Backend API**

En la raíz del proyecto:

    poetry install
    poetry run uvicorn scripts.api:app --reload --host 127.0.0.1 --port 8000

La API quedará atendiendo peticiones en:

    http://127.0.0.1:8000

Frontend

    Opción rápida: abrir directamente el archivo web/index.html en el navegador.
    Opción recomendada: servirlo con un servidor HTTP sencillo.
    Opción 1 (Python 3)
    cd web
    python -m http.server 5500
    Luego navegar a:
    http://127.0.0.1:5500/index.html