const API = "http://127.0.0.1:8000";

const el = (q) => document.querySelector(q);
const tbody = el("#tbl tbody");
const statusDot = el("#statusDot");
const healthBox = el("#healthBox");
const resumenSel = el("#resumenSel");
const frm = el("#frm");
const perfil = el("#perfil");
const dias = el("#dias");

// imágenes
const imgPrecios = el("#imgPrecios");
const imgArimaBtc = el("#imgArimaBtc");
const imgArimaEth = el("#imgArimaEth");
const imgArimaAda = el("#imgArimaAda");

// plotly
const pltLine = el("#pltLine");

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
  console.log("Plotly fetch:", url);
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

  if (!traces.length) {
    console.warn("Plotly: no hay datos en /series/price/all");
  }

  Plotly.newPlot(
    pltLine,
    traces,
    { margin: { t: 20, r: 20, b: 50, l: 60 }, xaxis: { title: "Fecha" }, yaxis: { title: "Precio" } },
    { responsive: true, displaylogo: false }
  );

  // asegurar redimensionado
  setTimeout(() => Plotly.Plots.resize(pltLine), 50);
  window.addEventListener("resize", () => Plotly.Plots.resize(pltLine));
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

document.addEventListener("DOMContentLoaded", cargar);
