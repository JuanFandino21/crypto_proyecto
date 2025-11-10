<<<<<<< HEAD
**Proyecto: Análisis y recomendación de criptomonedas**

Este proyecto se centra en analizar algunas criptomonedas (BTC, ETH y ADA) y construir, paso a paso, un sistema básico que permita recomendar monedas según el perfil de riesgo de una persona, apoyándose en datos históricos y en modelos que intentan predecir el comportamiento del precio.

A continuación se describe lo que se hizo en cada sprint:

**Sprint 1 – Recolección y limpieza estricta de los datos**

En el primer sprint el objetivo fue conseguir datos reales de criptomonedas, limpiarlos muy bien y dejarlos listos en un solo archivo para los siguientes sprints.
¿Qué se hizo?
    -Se escogió el proyecto de criptomonedas y predicción del comportamiento en el mercado.
    -Se trabajó con monedas como Bitcoin (BTC), Ethereum (ETH) y Cardano (ADA).
    -Se usaron dos APIs como fuentes de información:
        -Una para obtener precios históricos (apertura, cierre, máximo, mínimo).
        -Otra como fuente de respaldo para tener más flexibilidad.

    -Se programó la descarga de datos por moneda y por día, guardando información como:
        -Fecha
        -Precio de apertura
        -Precio de cierre
        -Máximo
        -Mínimo
        -Volumen    
        -Nombre de la moneda

Limpieza y estructura de la base de datos

En este sprint se puso mucho cuidado en que el archivo final quedara totalmente limpio, porque de eso depende todo lo demás. Para eso se hizo lo siguiente:

    -Se convirtió la columna de fechas a un formato de fecha correcto.
    -Se obligó a que las columnas numéricas realmente fueran números.
    -Se eliminaron:
        -Filas duplicadas.
        -Filas con valores vacíos o raros.
    -Se revisó que las fechas estuvieran en orden y que no hubiera saltos extraños.
    -Se validó que:
        -El máximo fuera mayor o igual que el mínimo.
        -El volumen no fuera negativo.
    -Al final se dejó un solo archivo principal:
    data/crypto_limpio.csv

con todas las monedas, ordenadas por moneda y por fecha.

Mini análisis exploratorio

Para verificar que todo estuviera bien, se hizo un primer vistazo:

Se revisaron las primeras filas de la tabla.
Se miraron estadísticas simples (promedios, mínimos, máximos).
Se graficó el precio de cierre por moneda para ver que las curvas fueran sensatas.

Resultado del Sprint 1

Al finalizar el sprint 1 se logró:

Tener un dataset limpio, sin basura y sin huecos, en un solo CSV.
Tener los datos listos para usarse en análisis más avanzados (clústeres, modelos, API, etc.).
Dejar todo organizado dentro de la carpeta data/ para reutilizarlo en los siguientes sprints.

**Sprint 2 – Análisis, clústeres y modelos de series de tiempo**

En el sprint 2 el objetivo fue empezar a sacarle jugo a los datos: agrupar las monedas según su comportamiento y probar modelos que intenten predecir el precio.

Organización de los datos para este sprint

Se volvió a cargar el archivo limpio crypto_limpio.csv.
Se reorganizaron los datos para poder trabajar por moneda y por fecha.
Se construyeron tablas donde:
    -Las filas son fechas.
    -Las columnas son monedas.
    -Los valores son precios de cierre o volúmenes.

Resumen por moneda y clústeres

Primero se creó un resumen por moneda, calculando cosas como:

    -Retorno medio diario (qué tanto sube o baja en promedio).
    -Desviación estándar del retorno (qué tan “nerviosa” es la moneda).
    -Volumen medio negociado.
    -Precio medio de cierre.

Con esos datos se armó una tabla resumen y se guardó en:
data/cluster_resumen_monedas.csv

Después, con esa misma tabla, se aplicaron varias técnicas de clusterización para agrupar las monedas según su comportamiento:

    -K-Means → se pidió que separara las monedas en varios grupos.
    -DBSCAN → intentó encontrar grupos por densidad.
    -Jerárquico → agrupó monedas de forma progresiva según su parecido.

El resultado fue que cada moneda quedó con varios labels o etiquetas de grupo, lo que ayuda a saber:

    -Qué monedas se parecen más entre sí.
    -Cuáles son más estables, cuáles más volátiles, etc.

Modelos de series de tiempo (ARIMA)

En este sprint también se usaron modelos de series de tiempo para tratar de predecir el precio de cierre.

Para cada moneda se hizo:

    -Se tomó la serie de precios de cierre.
    -Se dividió en:
        Una parte para “entrenar” el modelo (la mayor parte de la historia).
        Una parte final para “probarlo”.

    -Se ajustó un modelo ARIMA sencillo.
    -Se generaron pronósticos sobre la parte de prueba.
    -Se comparó lo pronosticado contra lo real usando:
        -MAE (error medio absoluto).
        -RMSE (error medio cuadrático).
        -MAPE (error porcentual medio).

Se generaron gráficas donde se ve:

    -En azul, los datos de entrenamiento.
    -En naranja, los datos de prueba reales.
    -En verde, lo que el modelo ARIMA cree que va a pasar.

Las métricas se guardaron en:
data/metricas_arima.csv

Modelo LSTM (RNN) sencillo

Además del modelo clásico, se quiso probar una red neuronal recurrente (LSTM) de forma sencilla.

Para cada moneda:

    -Se prepararon ventanas de varios días seguidos para que la red vea “tramos” de la serie.
    -Se normalizaron los datos para que trabajara en un rango estable.
    -Se dividió en entrenamiento y prueba.
    -Se entrenó un modelo LSTM simple para que intentara predecir el siguiente valor.
    -Se compararon de nuevo los datos reales contra los predichos:
    -Se graficó la curva real y la curva predicha.
    -Se calcularon las mismas métricas (RMSE, MAE, MAPE).
Los resultados de LSTM se guardaron en:
data/metricas_lstm.csv

Organización del proyecto con Poetry

En este sprint también se organizó el proyecto para que todas las librerías y dependencias se manejen con Poetry, de modo que:

    -Las librerías que se usan quedan escritas en pyproject.toml.
    -Cualquier persona que clone el proyecto puede instalar todo con un solo comando.

Resultado del Sprint 2

Al finalizar el sprint 2 se logró:

    -Tener un resumen estadístico por moneda que permite entender su comportamiento.
    -Agrupar las monedas con varias técnicas de clusterización.
    -Probar dos tipos de modelos para predicción de precios:
        -Uno clásico (ARIMA).
        -Uno basado en redes neuronales (LSTM).
    -Guardar las métricas y resultados en archivos CSV reutilizables para el sprint 3.
    -Dejar el proyecto mejor organizado usando Poetry.


**Sprint 3 – API de recomendaciones de criptomonedas**

En el sprint 3 el objetivo fue sacar lo que se había hecho del cuaderno y llevarlo a una API, es decir, a un servicio que otras aplicaciones puedan consumir para pedir recomendaciones de monedas.

Qué se buscaba

El sprint pedía:

    -Tener una API con rutas claras.
    -Que respondiera en formato JSON.
    -Que usara lo que ya se trabajó en los sprints anteriores (datos limpios, resumen, clústeres, modelos).
    -Probar los endpoints y documentarlos.

Estructura de la API

Se creó un archivo:
scripts/api.py

y se construyó una API usando FastAPI.
La API carga al inicio:

    crypto_limpio.csv
    cluster_resumen_monedas.csv
    metricas_arima.csv
    metricas_lstm.csv

Con esos archivos se arma una tabla general tabla_coins que junta, por moneda:

    -Sus estadísticas (retorno, desviación, etc.).
    -Sus grupos de clúster.

Sobre esa tabla se hace la lógica de recomendación.

Endpoints implementados

La API expone las siguientes rutas:

GET /health
    -Sirve para verificar que el servicio está vivo.
    -Devuelve un JSON sencillo con:
        -status → normalmente "ok".
        -n_registros → cantidad de filas del dataset.
        -n_monedas → cuántas monedas diferentes hay.

POST /predict

    -Recibe un JSON con los datos básicos del “usuario”:
        -perfil_riesgo: "bajo", "medio" o "alto".
        -dias_inversion: cuántos días piensa dejar la inversión.
    -Según ese perfil, la API consulta la tabla de monedas y arma una recomendación.
    -Devuelve un JSON que incluye:
        -El perfil que se envió.
        -Los días de inversión.
        -Una lista de monedas recomendadas con:
            coin
            cluster_kmeans
            ret_mean
            ret_std
            comentario explicando por qué se recomienda.

GET /recommendations

    -Hace lo mismo que /predict, pero usando parámetros en la URL.
    Ejemplo:
    http://127.0.0.1:8000/recommendations?perfil_riesgo=medio&dias_inversion=7

    -Es cómodo para probar desde el navegador o desde Postman.

GET /history

    -Devuelve una lista con todas las recomendaciones que se han generado desde que se encendió la API.
    -Sirve como historial sencillo de consultas.

Lógica de recomendación

La función central de este sprint toma:

    -El perfil de riesgo.
    -La información calculada en el sprint 2 (retornos, desviaciones, clústeres).

y hace una selección simple pero clara:

    -Para perfil bajo:
        -Se priorizan monedas con menos variación.
    -Para perfil medio:
        -Se busca un equilibrio entre retorno y riesgo.
    -Para perfil alto:
        -Se priorizan monedas con mayor retorno, aceptando más movimiento.
Al final se devuelven las mejores monedas para ese perfil, junto con un comentario en texto que explica la decisión.

Pruebas y documentación

    -La API se probó en local con:
        -Navegador (para /health y /recommendations).
        -Postman (para /predict y /history).

    -FastAPI genera automáticamente la documentación en:
    -http://127.0.0.1:8000/docs
    -donde se pueden ver:
        -Todos los endpoints.
        -Los modelos de entrada y salida.
        -Botones para probar las rutas y ver las respuestas JSON.
    -En consola se imprimen mensajes simples cuando se llama a /predict, lo que ayuda a ver que las peticiones sí llegan.

Resultado del Sprint 3

Al finalizar el sprint 3 se logró:

    -Tener una API funcional que se levanta en local y responde en formato JSON.
    -Integrar los resultados del sprint 1 y 2 en una lógica de recomendación sencilla.
    -Exponer endpoints claros:
        -/health
        -/predict
        -/recommendations
        -/history
    -Probar la API y dejarla documentada automáticamente con FastAPI.

Estado general del proyecto

Hasta este punto, el proyecto ya cuenta con:

    -Datos reales de criptomonedas limpios y organizados.
    -Un análisis previo que incluye:
        -Estadísticas por moneda.
        -Agrupación en clústeres.
        -Dos tipos de modelos de predicción de precios.
    -Una API que, a partir de un perfil de riesgo y unos días de inversión, devuelve recomendaciones de monedas en formato JSON, lista para ser consumida por una futura interfaz o dashboard.

La idea es que los siguientes sprints puedan seguir construyendo sobre esto, ya sea mejorando la lógica de recomendación, conectando directamente las predicciones de los modelos o creando una interfaz visual para el usuario final.

