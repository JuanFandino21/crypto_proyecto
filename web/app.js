const API = "http://127.0.0.1:8000";

const el = (q) => document.querySelector(q);

const statusDot = el("#statusDot");
const healthBox = el("#healthBox");

const frmPred = el("#frmPred");
const coinsPred = el("#coinsPred");
const modeloPred = el("#modeloPred");
const horizonPred = el("#horizonPred");
const desdePred = el("#desdePred");
const hastaPred = el("#hastaPred");
const predResumen = el("#predResumen");
const predPlot = el("#predPlot");
const predTablaBox = el("#predTablaBox");
const imgHeatmap = el("#imgHeatmap");

async function getJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} at ${url}`);
  return r.json();
}

function setDot(ok) {
  statusDot.classList.remove("dot-off", "dot-on");
  statusDot.classList.add(ok ? "dot-on" : "dot-off");
}

function num(n) {
  return typeof n === "number" ? n.toFixed(4) : n;
}

function pintaHealth(data) {
  const [reg, mon, st] = healthBox.querySelectorAll(".stat-k");
  reg.textContent = data.n_registros;
  mon.textContent = data.n_monedas;
  st.textContent = data.status;
  st.classList.toggle("ok", data.status === "ok");
}

// Utilidad para filtrar por rango de fechas

function filtraPorFecha(fechas, valores, desde, hasta) {
  const x = [];
  const y = [];
  for (let i = 0; i < fechas.length; i++) {
    const f = fechas[i];
    if (desde && f < desde) continue;
    if (hasta && f > hasta) continue;
    x.push(f);
    y.push(valores[i]);
  }
  return { x, y };
}

// Pintar predicciones histórico + futuro

function pintaPredicciones(data) {
  const { model, horizon, resultados } = data;
  const desde = desdePred.value || null;
  const hasta = hastaPred.value || null;
  const traces = [];
  const filas = [];

  resultados.forEach((r) => {
    // tramo histórico (para métricas)
    const realHist = filtraPorFecha(r.fechas_hist, r.y_real, desde, hasta);
    const predHist = filtraPorFecha(r.fechas_hist, r.y_pred, desde, hasta);

    // tramo FUTURO
    const predFut = filtraPorFecha(r.fechas_fut, r.y_pred_fut, desde, hasta);

    // real histórico
    traces.push({
      x: realHist.x,
      y: realHist.y,
      type: "scatter",
      mode: "lines",
      name: `${r.coin} real`,
    });

    // predicción sobre histórico
    traces.push({
      x: predHist.x,
      y: predHist.y,
      type: "scatter",
      mode: "lines",
      name: `${r.coin} pred ${model.toUpperCase()}`,
      line: { dash: "dot" },
    });

    // predicción FUTURA
    traces.push({
      x: predFut.x,
      y: predFut.y,
      type: "scatter",
      mode: "lines",
      name: `${r.coin} futuro ${model.toUpperCase()}`,
      line: { dash: "dashdot" },
    });

    filas.push(`
      <tr>
        <td>${r.coin}</td>
        <td>${num(r.precio_actual)}</td>
        <td>${num(r.precio_pred_horizonte)}</td>
        <td>${r.variacion_pct.toFixed(2)}%</td>
        <td>${r.recomendacion}</td>
        <td>${num(r.mae)}</td>
        <td>${num(r.rmse)}</td>
      </tr>
    `);
  });

  if (predPlot) {
    Plotly.newPlot(
      predPlot,
      traces,
      {
        title: `Predicciones – modelo ${model.toUpperCase()} (días a predecir: ${horizon})`,
        xaxis: { title: "Fecha" },
        yaxis: { title: "Precio de cierre" },
        margin: { t: 40, r: 20, b: 50, l: 60 },
      },
      { responsive: true, displaylogo: false }
    );
  }

  if (predTablaBox) {
    predTablaBox.innerHTML = `
      <table class="table">
        <thead>
          <tr>
            <th>Moneda</th>
            <th>Precio actual</th>
            <th>Predicción</th>
            <th>Variación %</th>
            <th>Recomendación</th>
            <th>MAE</th>
            <th>RMSE</th>
          </tr>
        </thead>
        <tbody>
          ${filas.join("")}
        </tbody>
      </table>
    `;
  }

  if (predResumen) {
    let resumen = `modelo: ${model.toUpperCase()} • días a predecir: ${horizon}`;
    if (desde || hasta) {
      resumen += ` • rango: ${desde || "inicio"} a ${hasta || "fin"}`;
    }
    predResumen.textContent = resumen;
  }
}


// Llamada a la API de predicciones

async function cargarPredicciones() {
  if (!frmPred) return;
  try {
    const seleccionadas = [...coinsPred.selectedOptions].map((o) => o.value);
    if (seleccionadas.length === 0) return;

    const coins = seleccionadas.join(",");
    const model = modeloPred.value;
    const horizon = parseInt(horizonPred.value || "7", 10);

    const url = `${API}/series/predict?coins=${encodeURIComponent(
      coins
    )}&model=${model}&horizon=${horizon}&_t=${Date.now()}`;

    const data = await getJson(url);
    pintaPredicciones(data);
  } catch (e) {
    console.error("Predicciones error:", e);
  }
}


// Heatmap de correlaciones

async function cargarHeatmap() {
  if (!imgHeatmap) return;
  try {
    const bust = `_t=${Date.now()}`;
    imgHeatmap.src = `${API}/plot/heatmap?${bust}`;
  } catch (e) {
    console.error("Error cargando heatmap:", e);
  }
}

// Cargar lista de monedas desde la API

async function cargarCoinsPred() {
  if (!coinsPred) return;
  try {
    const data = await getJson(`${API}/coins?_t=${Date.now()}`);
    const coins = data.coins || [];

    coinsPred.innerHTML = "";
    coins.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c.toLowerCase();
      opt.textContent = c.toUpperCase();
      opt.selected = true; // todas seleccionadas por defecto
      coinsPred.appendChild(opt);
    });
  } catch (e) {
    console.error("Error cargando monedas:", e);
  }
}


// Carga inicial (health + predicciones + heatmap)

async function cargar() {
  try {
    const h = await getJson(`${API}/health?_t=${Date.now()}`);
    setDot(true);
    pintaHealth(h);
  } catch {
    setDot(false);
  }

  await cargarPredicciones();
  await cargarHeatmap();
}


// Listeners

if (frmPred) {
  frmPred.addEventListener("submit", (e) => {
    e.preventDefault();
    cargarPredicciones();
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  await cargarCoinsPred();
  await cargar();
});
