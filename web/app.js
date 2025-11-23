const API = "http://127.0.0.1:8000";

const el = (q) => document.querySelector(q);
const tbody = el("#tbl tbody");
const statusDot = el("#statusDot");
const healthBox = el("#healthBox");
const resumenSel = el("#resumenSel");
const frm = el("#frm");
const perfil = el("#perfil");
const dias = el("#dias");
const imgPrecios = el("#imgPrecios");
const imgArimaBtc = el("#imgArimaBtc");
const imgArimaEth = el("#imgArimaEth");
const imgArimaAda = el("#imgArimaAda");
const pltLine = el("#pltLine");
const frmPred = el("#frmPred");
const coinsPred = el("#coinsPred");
const modeloPred = el("#modeloPred");
const horizonPred = el("#horizonPred");
const desdePred = el("#desdePred");
const hastaPred = el("#hastaPred");
const predResumen = el("#predResumen");
const predPlot = el("#predPlot");
const predTablaBox = el("#predTablaBox");

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

function pintaRecs(data) {
  resumenSel.textContent = `perfil: ${data.perfil_riesgo} • días: ${data.dias_inversion}`;
  tbody.innerHTML = "";
  data.recomendaciones.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.coin.toUpperCase()}</td>
      <td>${r.cluster_kmeans}</td>
      <td>${num(r.ret_mean)}</td>
      <td>${num(r.ret_std)}</td>
      <td>${r.comentario}</td>
    `;
    tbody.appendChild(tr);
  });
}

function pintaImgs() {
  const bust = `&_t=${Date.now()}`;
  const d = Number(dias.value) || 60;
  imgPrecios.src = `${API}/plot/price/all?dias=${d}${bust}`;
  imgArimaBtc.src = `${API}/plot/forecast/arima/btc?test_size=20&p=1&d=1&q=1${bust}`;
  imgArimaEth.src = `${API}/plot/forecast/arima/eth?test_size=20&p=1&d=1&q=1${bust}`;
  imgArimaAda.src = `${API}/plot/forecast/arima/ada?test_size=20&p=1&d=1&q=1${bust}`;
}

async function pintaPlotly() {
  const d = Number(dias.value) || 60;
  const url = `${API}/series/price/all?dias=${d}&_t=${Date.now()}`;
  const data = await getJson(url);

  const toTrace = (name, s) => {
    if (Array.isArray(s)) {
      const xs = s.map((p) => p[0]);
      const ys = s.map((p) => p[1]);
      return { x: xs, y: ys, type: "scatter", mode: "lines", name };
    }
    return { x: s.x, y: s.y, type: "scatter", mode: "lines", name };
  };

  const traces = [];
  if (data.btc) traces.push(toTrace("BTC", data.btc));
  if (data.eth) traces.push(toTrace("ETH", data.eth));
  if (data.ada) traces.push(toTrace("ADA", data.ada));

  Plotly.newPlot(
    pltLine,
    traces,
    { margin: { t: 20, r: 20, b: 50, l: 60 }, xaxis: { title: "Fecha" }, yaxis: { title: "Precio" } },
    { responsive: true, displaylogo: false }
  );

  setTimeout(() => Plotly.Plots.resize(pltLine), 50);
  window.addEventListener("resize", () => Plotly.Plots.resize(pltLine));
}

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

function pintaPredicciones(data) {
  const { model, horizon, resultados } = data;
  const desde = desdePred.value || null;
  const hasta = hastaPred.value || null;
  const traces = [];
  const filas = [];

  resultados.forEach((r) => {
    const realFiltrado = filtraPorFecha(r.fechas_hist, r.y_real, desde, hasta);
    const predFiltrado = filtraPorFecha(r.fechas_hist, r.y_pred, desde, hasta);

    traces.push({
      x: realFiltrado.x,
      y: realFiltrado.y,
      type: "scatter",
      mode: "lines",
      name: `${r.coin} real`,
    });

    traces.push({
      x: predFiltrado.x,
      y: predFiltrado.y,
      type: "scatter",
      mode: "lines",
      name: `${r.coin} pred ${model.toUpperCase()}`,
      line: { dash: "dot" },
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
        title: `Predicciones – modelo ${model.toUpperCase()} (horizonte ${horizon} días)`,
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
    predResumen.textContent = `modelo: ${model.toUpperCase()} • horizonte: ${horizon} días`;
  }
}

async function cargarPredicciones() {
  if (!frmPred) return;
  try {
    const coins = [...coinsPred.selectedOptions].map((o) => o.value).join(",");
    if (!coins) return;
    const model = modeloPred.value;
    const horizon = parseInt(horizonPred.value || "7", 10);
    const url = `${API}/series/predict?coins=${encodeURIComponent(coins)}&model=${model}&horizon=${horizon}&_t=${Date.now()}`;
    const data = await getJson(url);
    pintaPredicciones(data);
  } catch (e) {
    console.error("Predicciones error:", e);
  }
}

async function cargar() {
  try {
    const h = await getJson(`${API}/health`);
    setDot(true);
    pintaHealth(h);
  } catch {
    setDot(false);
  }

  try {
    const url = `${API}/recommendations?perfil_riesgo=${perfil.value}&dias_inversion=${dias.value}`;
    const recs = await getJson(url);
    pintaRecs(recs);
  } catch (e) {
    console.error("Recs error:", e);
  }

  pintaImgs();
  try {
    await pintaPlotly();
  } catch (e) {
    console.error("Plotly error:", e);
  }

  await cargarPredicciones();
}

frm.addEventListener("submit", (e) => {
  e.preventDefault();
  cargar();
});

el("#btnReset").addEventListener("click", () => {
  perfil.value = "medio";
  dias.value = 7;
  cargar();
});

if (frmPred) {
  frmPred.addEventListener("submit", (e) => {
    e.preventDefault();
    cargarPredicciones();
  });
}

document.addEventListener("DOMContentLoaded", cargar);
