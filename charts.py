import json
import pandas as pd
from datetime import datetime

OVERLAY_INDICATORS = {"sma", "ema", "wma", "bbands", "bollinger"}
COLORS = ["#f0b429", "#a78bfa", "#34d399", "#fb7185", "#60a5fa", "#f97316"]


def _df_to_candle_json(df: pd.DataFrame) -> str:
    rows = []
    for ts, row in df.iterrows():
        if any(pd.isna([row['Open'], row['High'], row['Low'], row['Close']])):
            continue
        rows.append({
            "time": int(ts.timestamp()),
            "open": round(float(row['Open']), 2),
            "high": round(float(row['High']), 2),
            "low":  round(float(row['Low']),  2),
            "close":round(float(row['Close']),2),
        })
    return json.dumps(rows)


def _df_to_volume_json(df: pd.DataFrame) -> str:
    rows = []
    for ts, row in df.iterrows():
        if pd.isna(row['Volume']):
            continue
        color = "#26a69a" if float(row['Close']) >= float(row['Open']) else "#ef5350"
        rows.append({"time": int(ts.timestamp()), "value": round(float(row['Volume']), 0), "color": color})
    return json.dumps(rows)


def _indicator_series_data(indicator_df: pd.DataFrame) -> str:
    result = {}
    for col in indicator_df.columns:
        series = []
        for ts, val in indicator_df[col].items():
            if not pd.isna(val):
                series.append({"time": int(ts.timestamp()), "value": round(float(val), 4)})
        result[col] = series
    return json.dumps(result)


# ─────────────────────────────────────────────────────────
#  ОБЩИЙ CSS
# ─────────────────────────────────────────────────────────

COMMON_CSS = """
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:              #0d1117;
  --bg2:             #161b22;
  --border:          #30363d;
  --text:            #e6edf3;
  --text2:           #8b949e;
  --text3:           #484f58;
  --btn-bg:          #21262d;
  --btn-hover:       #30363d;
  --btn-active-bg:   #1f6feb;
  --btn-active-text: #ffffff;
  --up:   #26a69a;
  --down: #ef5350;
}
body.light {
  --bg:              #ffffff;
  --bg2:             #f6f8fa;
  --border:          #d0d7de;
  --text:            #1f2328;
  --text2:           #57606a;
  --text3:           #6e7781;
  --btn-bg:          #eaeef2;
  --btn-hover:       #d0d7de;
  --btn-active-bg:   #0969da;
  --btn-active-text: #ffffff;
  --up:   #1a7f64;
  --down: #cf222e;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
  transition: background .2s, color .2s;
}

.header {
  padding: 10px 16px;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
  flex-wrap: wrap;
  gap: 8px;
}
.header-left  { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.ticker       { font-size: 20px; font-weight: 700; color: var(--text); }
.badge        { background: #1f6feb; color: #fff; padding: 2px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.badge.alert  { background: #da3633; }
.price-block  { text-align: right; }
.price-main   { font-size: 18px; font-weight: 600; color: var(--text); }
.price-change { font-size: 12px; margin-top: 2px; }

.toolbar {
  padding: 6px 16px;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  flex-shrink: 0;
}
.toolbar-group { display: flex; gap: 3px; align-items: center; }
.toolbar-sep   { width: 1px; height: 20px; background: var(--border); margin: 0 4px; }
.toolbar-label { font-size: 11px; color: var(--text2); margin-right: 2px; white-space: nowrap; }

.btn {
  padding: 4px 10px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--btn-bg);
  color: var(--text2);
  font-size: 12px;
  cursor: pointer;
  transition: background .15s, color .15s;
  white-space: nowrap;
  user-select: none;
}
.btn:hover  { background: var(--btn-hover); color: var(--text); }
.btn.active { background: var(--btn-active-bg); color: var(--btn-active-text); border-color: transparent; }
.btn.icon   { padding: 4px 8px; font-size: 14px; }

#chart-price     { flex: 3; width: 100%; min-height: 0; }
#chart-volume    { flex: 1; width: 100%; border-top: 1px solid var(--border); min-height: 0; }
#chart-indicator { flex: 1.2; width: 100%; border-top: 1px solid var(--border); min-height: 0; }

#tooltip {
  position: fixed;
  top: 0; left: 0;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 12px;
  color: var(--text);
  pointer-events: none;
  opacity: 0;
  transition: opacity .1s;
  z-index: 100;
  min-width: 160px;
  line-height: 1.7;
}
#tooltip.visible { opacity: 1; }
#tooltip .tt-date  { color: var(--text2); font-size: 11px; margin-bottom: 4px; }
#tooltip .tt-row   { display: flex; justify-content: space-between; gap: 12px; }
#tooltip .tt-label { color: var(--text2); }
#tooltip .tt-up    { color: var(--up);   font-weight: 600; }
#tooltip .tt-down  { color: var(--down); font-weight: 600; }

.footer {
  padding: 5px 16px;
  background: var(--bg2);
  border-top: 1px solid var(--border);
  font-size: 11px;
  color: var(--text3);
  text-align: center;
  flex-shrink: 0;
}
</style>
"""

# ─────────────────────────────────────────────────────────
#  ОБЩИЙ JS
# ─────────────────────────────────────────────────────────
COMMON_JS = """
// ── Тема ──────────────────────────────────────────────
let isLight = false;

function themeLayout() {
  return isLight
    ? { background: { color: '#ffffff' }, textColor: '#1f2328' }
    : { background: { color: '#0d1117' }, textColor: '#8b949e' };
}
function themeGrid() {
  return isLight
    ? { vertLines: { color: '#eaeef2' }, horzLines: { color: '#eaeef2' } }
    : { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } };
}
function themeBorder() { return isLight ? '#d0d7de' : '#30363d'; }

function toggleTheme() {
  isLight = !isLight;
  document.body.classList.toggle('light', isLight);
  ALL_CHARTS.filter(Boolean).forEach(c => c.applyOptions({
    layout: themeLayout(),
    grid: themeGrid(),
    rightPriceScale: { borderColor: themeBorder() },
    timeScale: { borderColor: themeBorder() },
  }));
}

// ── Период ────────────────────────────────────────────
function setPeriod(days, btn) {
  document.querySelectorAll('.period-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  if (days === 0 || !ALL_CANDLES.length) {
    ALL_CHARTS.filter(Boolean).forEach(c => c.timeScale().fitContent());
    return;
  }
  const to   = ALL_CANDLES[ALL_CANDLES.length - 1].time;
  const from = to - days * 86400;
  ALL_CHARTS.filter(Boolean).forEach(c => c.timeScale().setVisibleRange({ from, to }));
}

// ── Tooltip ───────────────────────────────────────────
const tooltip = document.getElementById('tooltip');

function setupTooltip(chart, getSeries, containerEl) {
  chart.subscribeCrosshairMove(param => {
    if (!param.time || !param.point) { tooltip.classList.remove('visible'); return; }
    const c = param.seriesData.get(getSeries());
    if (!c) { tooltip.classList.remove('visible'); return; }

    const date = new Date(param.time * 1000)
      .toLocaleDateString('ru-RU', { day:'2-digit', month:'2-digit', year:'numeric' });

    const val  = c.close ?? c.value;
    const isUp = c.open !== undefined ? val >= c.open : true;
    const cls  = isUp ? 'tt-up' : 'tt-down';

    let html = `<div class="tt-date">${date}</div>`;
    if (c.open !== undefined) {
      html += `
        <div class="tt-row"><span class="tt-label">O</span><span class="${cls}">${c.open.toFixed(2)}</span></div>
        <div class="tt-row"><span class="tt-label">H</span><span class="${cls}">${c.high.toFixed(2)}</span></div>
        <div class="tt-row"><span class="tt-label">L</span><span class="${cls}">${c.low.toFixed(2)}</span></div>
        <div class="tt-row"><span class="tt-label">C</span><span class="${cls}">${c.close.toFixed(2)}</span></div>`;
    } else {
      html += `<div class="tt-row"><span class="tt-label">Цена</span><span class="${cls}">${val.toFixed(2)}</span></div>`;
    }

    tooltip.innerHTML = html;
    tooltip.classList.add('visible');

    const rect = containerEl.getBoundingClientRect();
    let x = rect.left + param.point.x + 16;
    let y = rect.top  + param.point.y - 10;
    if (x + 180 > window.innerWidth)  x = rect.left + param.point.x - 190;
    if (y + 130 > window.innerHeight) y = window.innerHeight - 140;
    tooltip.style.left = x + 'px';
    tooltip.style.top  = y + 'px';
  });
}

// ── Sync scroll ───────────────────────────────────────
function syncCharts(...charts) {
  const valid = charts.filter(Boolean);
  valid.forEach(src => {
    src.timeScale().subscribeVisibleLogicalRangeChange(r => {
      if (!r) return;
      valid.filter(c => c !== src).forEach(c => c.timeScale().setVisibleLogicalRange(r));
    });
  });
}

// ── Resize ────────────────────────────────────────────
function resizeAll() {
  ALL_CHARTS.filter(Boolean).forEach((chart, i) => {
    const el = CHART_ELS[i];
    if (el && el.clientHeight > 0) {
      chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
    }
  });
}
new ResizeObserver(resizeAll).observe(document.body);

// ── Рисование линий ───────────────────────────────────
let drawMode      = false;
let drawStart     = null;
let previewSeries = null;
const drawnLines  = [];

function toggleDrawMode(btn) {
  drawMode = btn.classList.toggle('active');
  priceEl.style.cursor = drawMode ? 'crosshair' : 'default';
  if (!drawMode) {
    drawStart = null;
    if (previewSeries) {
      priceChart.removeSeries(previewSeries);
      previewSeries = null;
    }
  }
}

function undoLine() {
  const last = drawnLines.pop();
  if (last) priceChart.removeSeries(last);
}

function clearLines() {
  drawnLines.forEach(s => priceChart.removeSeries(s));
  drawnLines.length = 0;
}

function getChartPoint(e) {
  const rect  = priceEl.getBoundingClientRect();
  const x     = e.clientX - rect.left;
  const y     = e.clientY - rect.top;

  const price = priceSeries.coordinateToPrice(y);

  let time = priceChart.timeScale().coordinateToTime(x);
  if (time === null) {
    const logical = priceChart.timeScale().coordinateToLogical(x);
    if (logical === null) return null;
    const idx = Math.max(0, Math.min(Math.round(logical), ALL_CANDLES.length - 1));
    time = ALL_CANDLES[idx]?.time ?? null;
  }

  if (time === null || price === null) return null;
  return { time, price };
}

priceEl.addEventListener('click', e => {
  if (!drawMode) return;
  const point = getChartPoint(e);
  if (!point) return;
  const { time, price } = point;

  if (!drawStart) {
    drawStart = { time, price };
  } else {
    if (previewSeries) {
      priceChart.removeSeries(previewSeries);
      previewSeries = null;
    }
    const p1 = drawStart.time <= time ? drawStart : { time, price };
    const p2 = drawStart.time <= time ? { time, price } : drawStart;

    const line = priceChart.addLineSeries({
      color: '#f0b429', lineWidth: 2, lineStyle: 0,
      lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false,
    });
    line.setData([
      { time: p1.time, value: p1.price },
      { time: p2.time, value: p2.price },
    ]);
    drawnLines.push(line);
    drawStart = null;
    drawMode  = false;
    priceEl.style.cursor = 'default';
    document.getElementById('btn-draw').classList.remove('active');
  }
});

priceEl.addEventListener('mousemove', e => {
  if (!drawMode || !drawStart) return;
  const point = getChartPoint(e);
  if (!point) return;
  const { time, price } = point;

  if (previewSeries) priceChart.removeSeries(previewSeries);

  const p1 = drawStart.time <= time ? drawStart : { time, price };
  const p2 = drawStart.time <= time ? { time, price } : drawStart;

  previewSeries = priceChart.addLineSeries({
    color: 'rgba(240,180,41,0.5)', lineWidth: 1, lineStyle: 1,
    lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false,
  });
  previewSeries.setData([
    { time: p1.time, value: p1.price },
    { time: p2.time, value: p2.price },
  ]);
});
"""



#  TOOLBAR — общие блоки
# ─────────────────────────────────────────────────────────

TOOLBAR_CHART_TYPE = """
  <div class="toolbar-group">
    <span class="toolbar-label">Тип:</span>
    <button class="btn type-btn active" onclick="setChartType('candlestick',this)">Свечи</button>
    <button class="btn type-btn"        onclick="setChartType('line',this)">Линия</button>
    <button class="btn type-btn"        onclick="setChartType('area',this)">Area</button>
    <button class="btn type-btn"        onclick="setChartType('bar',this)">Бары</button>
  </div>"""

TOOLBAR_PERIOD = """
  <div class="toolbar-group">
    <span class="toolbar-label">Период:</span>
    <button class="btn period-btn" onclick="setPeriod(30,this)">1М</button>
    <button class="btn period-btn" onclick="setPeriod(90,this)">3М</button>
    <button class="btn period-btn" onclick="setPeriod(180,this)">6М</button>
    <button class="btn period-btn" onclick="setPeriod(365,this)">1Г</button>
    <button class="btn period-btn active" onclick="setPeriod(0,this)">Всё</button>
  </div>"""

TOOLBAR_DRAWING = """
  <div class="toolbar-group">
    <button class="btn" id="btn-draw" onclick="toggleDrawMode(this)" title="Нарисовать линию">✏️ Линия</button>
    <button class="btn" onclick="undoLine()"  title="Удалить последнюю">↩️</button>
    <button class="btn" onclick="clearLines()" title="Удалить все линии">🗑️</button>
  </div>"""

TOOLBAR_THEME = """
  <div class="toolbar-group">
    <button class="btn icon" onclick="toggleTheme()" title="Тема">🌗</button>
  </div>"""

SEP = '\n  <div class="toolbar-sep"></div>'

# ─────────────────────────────────────────────────────────
#  ОБЩИЙ JS — создание серии цены и смена типа
# ─────────────────────────────────────────────────────────

PRICE_SERIES_JS = """
function createPriceSeries(type) {
  switch (type) {
    case 'line':
      return priceChart.addLineSeries({ color: '#60a5fa', lineWidth: 2 });
    case 'area':
      return priceChart.addAreaSeries({
        lineColor: '#60a5fa',
        topColor: 'rgba(96,165,250,0.3)',
        bottomColor: 'rgba(96,165,250,0)',
      });
    case 'bar':
      return priceChart.addBarSeries({ upColor: '#26a69a', downColor: '#ef5350' });
    default:
      return priceChart.addCandlestickSeries({
        upColor: '#26a69a', downColor: '#ef5350',
        borderUpColor: '#26a69a', borderDownColor: '#ef5350',
        wickUpColor:   '#26a69a', wickDownColor:   '#ef5350',
      });
  }
}

function setChartType(type, btn) {
  document.querySelectorAll('.type-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  priceChart.removeSeries(priceSeries);
  priceSeries = createPriceSeries(type);
  const flat = type === 'line' || type === 'area';
  priceSeries.setData(flat ? ALL_CANDLES.map(d => ({ time: d.time, value: d.close })) : ALL_CANDLES);
}

function toggleVolume(btn) {
  const show = volumeEl.style.display === 'none';
  volumeEl.style.display = show ? '' : 'none';
  btn.classList.toggle('active', show);
  resizeAll();
}

function makeChart(el) {
  return LightweightCharts.createChart(el, {
    layout: { background: { color: '#0d1117' }, textColor: '#8b949e' },
    grid:   { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#30363d' },
    timeScale: { borderColor: '#30363d', timeVisible: true, secondsVisible: false },
    handleScroll: true, handleScale: true,
  });
}
"""


# ─────────────────────────────────────────────────────────
#  build_price_html
# ─────────────────────────────────────────────────────────

def build_price_html(ticker: str, df: pd.DataFrame) -> str:
    candle_json  = _df_to_candle_json(df)
    volume_json  = _df_to_volume_json(df)
    period_start = df.index[0].strftime('%d.%m.%Y')
    period_end   = df.index[-1].strftime('%d.%m.%Y')
    last_close   = df['Close'].iloc[-1]
    change       = last_close - df['Close'].iloc[0]
    change_pct   = (last_close / df['Close'].iloc[0] - 1) * 100
    change_color = "#26a69a" if change >= 0 else "#ef5350"
    sign         = "+" if change >= 0 else ""

    toolbar = f"""<div class="toolbar">
{TOOLBAR_CHART_TYPE}{SEP}
{TOOLBAR_PERIOD}{SEP}
  <div class="toolbar-group">
    <button class="btn active" id="btn-vol" onclick="toggleVolume(this)">📊 Объём</button>
  </div>{SEP}
{TOOLBAR_DRAWING}{SEP}
{TOOLBAR_THEME}
</div>"""

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{ticker} — График цен</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
{COMMON_CSS}
</head>
<body>

<div class="header">
  <div class="header-left">
    <span class="ticker">📈 {ticker}</span>
    <span style="font-size:12px;color:var(--text2)">{period_start} — {period_end}</span>
  </div>
  <div class="price-block">
    <div class="price-main">{last_close:.2f} ₽</div>
    <div class="price-change" style="color:{change_color}">{sign}{change:.2f} ₽ &nbsp;({sign}{change_pct:.2f}%)</div>
  </div>
</div>

{toolbar}

<div id="chart-price"></div>
<div id="chart-volume"></div>
<div id="tooltip"></div>
<div class="footer">Данные Московской биржи (MOEX) · {period_start} — {period_end}</div>

<script>
const ALL_CANDLES = {candle_json};
const ALL_VOLUMES = {volume_json};

const priceEl  = document.getElementById('chart-price');
const volumeEl = document.getElementById('chart-volume');

{PRICE_SERIES_JS}

const priceChart  = makeChart(priceEl);
const volumeChart = makeChart(volumeEl);

const ALL_CHARTS = [priceChart, volumeChart];
const CHART_ELS  = [priceEl, volumeEl];

let priceSeries = createPriceSeries('candlestick');
priceSeries.setData(ALL_CANDLES);

const volSeries = volumeChart.addHistogramSeries({{ priceScaleId: '' }});
volSeries.priceScale().applyOptions({{ scaleMargins: {{ top: 0.1, bottom: 0 }} }});
volSeries.setData(ALL_VOLUMES);

{COMMON_JS}

setupTooltip(priceChart, () => priceSeries, priceEl);
syncCharts(priceChart, volumeChart);
resizeAll();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────
#  build_indicator_html
# ─────────────────────────────────────────────────────────

def build_indicator_html(ticker: str, price_df: pd.DataFrame, indicator_df: pd.DataFrame, indicator_name: str) -> str:
    candle_json   = _df_to_candle_json(price_df)
    volume_json   = _df_to_volume_json(price_df)
    ind_data_json = _indicator_series_data(indicator_df)
    is_overlay    = indicator_name.lower() in OVERLAY_INDICATORS
    period_start  = price_df.index[0].strftime('%d.%m.%Y')
    period_end    = price_df.index[-1].strftime('%d.%m.%Y')

    ind_buttons = "\n    ".join(
        f'<button class="btn ind-btn active" data-col="{col}" onclick="toggleSeries(\'{col}\',this)">{col}</button>'
        for col in indicator_df.columns
    )

    levels_js = ""
    ind_low = indicator_name.lower()
    if ind_low == "rsi":
        levels_js = """
  addLevel(indChart, seriesList[0], 70, '#ef5350', 'OB 70');
  addLevel(indChart, seriesList[0], 30, '#26a69a', 'OS 30');"""
    elif ind_low == "stoch":
        levels_js = """
  addLevel(indChart, seriesList[0], 80, '#ef5350', 'OB 80');
  addLevel(indChart, seriesList[0], 20, '#26a69a', 'OS 20');"""

    ind_panel_html = "" if is_overlay else '<div id="chart-indicator"></div>'
    ind_chart_js   = "" if is_overlay else """
  const indEl = document.getElementById('chart-indicator');
  indChart = makeChart(indEl);
  ALL_CHARTS.push(indChart);
  CHART_ELS.push(indEl);"""
    ind_target = "priceChart" if is_overlay else "indChart"

    toolbar = f"""<div class="toolbar">
{TOOLBAR_CHART_TYPE}{SEP}
{TOOLBAR_PERIOD}{SEP}
  <div class="toolbar-group">
    <span class="toolbar-label">Линии:</span>
    {ind_buttons}
  </div>{SEP}
  <div class="toolbar-group">
    <button class="btn active" id="btn-vol" onclick="toggleVolume(this)">📊 Объём</button>
  </div>{SEP}
{TOOLBAR_DRAWING}{SEP}
{TOOLBAR_THEME}
</div>"""

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{ticker} — {indicator_name.upper()}</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
{COMMON_CSS}
</head>
<body>

<div class="header">
  <div class="header-left">
    <span class="ticker">📊 {ticker}</span>
    <span class="badge">{indicator_name.upper()}</span>
    <span style="font-size:12px;color:var(--text2)">{period_start} — {period_end}</span>
  </div>
</div>

{toolbar}

<div id="chart-price"></div>
<div id="chart-volume"></div>
{ind_panel_html}
<div id="tooltip"></div>
<div class="footer">Данные Московской биржи (MOEX) · {period_start} — {period_end}</div>

<script>
const ALL_CANDLES  = {candle_json};
const ALL_VOLUMES  = {volume_json};
const IND_DATA     = {ind_data_json};
const COLORS       = {json.dumps(COLORS)};
const IS_OVERLAY   = {'true' if is_overlay else 'false'};

const priceEl  = document.getElementById('chart-price');
const volumeEl = document.getElementById('chart-volume');

{PRICE_SERIES_JS}

const priceChart  = makeChart(priceEl);
const volumeChart = makeChart(volumeEl);

const ALL_CHARTS = [priceChart, volumeChart];
const CHART_ELS  = [priceEl, volumeEl];

let indChart = null;
{ind_chart_js}

let priceSeries = createPriceSeries('candlestick');
priceSeries.setData(ALL_CANDLES);

const volSeries = volumeChart.addHistogramSeries({{ priceScaleId: '' }});
volSeries.priceScale().applyOptions({{ scaleMargins: {{ top: 0.1, bottom: 0 }} }});
volSeries.setData(ALL_VOLUMES);

const seriesList = [];
const seriesMap  = {{}};

function addLevel(chart, refSeries, value, color, title) {{
  const s = chart.addLineSeries({{
    color, lineWidth: 1, lineStyle: 2,
    title, lastValueVisible: false, priceLineVisible: false,
  }});
  s.setData(refSeries.data().map(d => ({{ time: d.time, value }})));
}}

(function buildIndSeries() {{
  const target = IS_OVERLAY ? priceChart : indChart;
  if (!target) return;
  Object.entries(IND_DATA).forEach(([col, data], i) => {{
    const s = target.addLineSeries({{
      color: COLORS[i % COLORS.length],
      lineWidth: 2, title: col,
      lastValueVisible: true, priceLineVisible: false,
    }});
    s.setData(data);
    seriesList.push(s);
    seriesMap[col] = s;
  }});
  {levels_js}
}})();

function toggleSeries(col, btn) {{
  const s = seriesMap[col];
  if (!s) return;
  const on = btn.classList.toggle('active');
  s.applyOptions({{ visible: on }});
}}

{COMMON_JS}

setupTooltip(priceChart, () => priceSeries, priceEl);
syncCharts(...ALL_CHARTS);
resizeAll();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────
#  build_alert_html
# ─────────────────────────────────────────────────────────

def build_alert_html(ticker: str, df: pd.DataFrame, current_price: float, target_value: float, condition: str) -> str:
    candle_json  = _df_to_candle_json(df)
    volume_json  = _df_to_volume_json(df)
    line_color   = "#26a69a" if condition == "выше" else "#ef5350"
    period_start = df.index[0].strftime('%d.%m.%Y')
    period_end   = df.index[-1].strftime('%d.%m.%Y')
    now_str      = datetime.now().strftime('%d.%m.%Y %H:%M:%S')

    toolbar = f"""<div class="toolbar">
{TOOLBAR_CHART_TYPE}{SEP}
{TOOLBAR_PERIOD}{SEP}
  <div class="toolbar-group">
    <button class="btn active" id="btn-trigger" onclick="toggleTrigger(this)">🎯 Триггер</button>
    <button class="btn active" id="btn-vol"     onclick="toggleVolume(this)">📊 Объём</button>
  </div>{SEP}
{TOOLBAR_DRAWING}{SEP}
{TOOLBAR_THEME}
</div>"""

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🔔 Алерт {ticker}</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
{COMMON_CSS}
</head>
<body>

<div class="header">
  <div class="header-left">
    <span class="ticker">🔔 {ticker}</span>
    <span class="badge alert">АЛЕРТ</span>
    <span style="font-size:12px;color:var(--text2)">{now_str}</span>
  </div>
  <div class="price-block">
    <div class="price-main">{current_price:.2f} ₽</div>
    <div class="price-change" style="color:{line_color}">{condition} {target_value:.2f} ₽</div>
  </div>
</div>

{toolbar}

<div id="chart-price"></div>
<div id="chart-volume"></div>
<div id="tooltip"></div>
<div class="footer">Данные Московской биржи (MOEX) · {period_start} — {period_end}</div>

<script>
const ALL_CANDLES  = {candle_json};
const ALL_VOLUMES  = {volume_json};
const TARGET_VALUE = {target_value};

const priceEl  = document.getElementById('chart-price');
const volumeEl = document.getElementById('chart-volume');

{PRICE_SERIES_JS}

const priceChart  = makeChart(priceEl);
const volumeChart = makeChart(volumeEl);

const ALL_CHARTS = [priceChart, volumeChart];
const CHART_ELS  = [priceEl, volumeEl];

let priceSeries = createPriceSeries('candlestick');
priceSeries.setData(ALL_CANDLES);

const triggerSeries = priceChart.addLineSeries({{
  color: '{line_color}', lineWidth: 2, lineStyle: 1,
  title: 'Триггер {target_value:.2f}',
  lastValueVisible: true, priceLineVisible: false,
}});
triggerSeries.setData(ALL_CANDLES.map(d => ({{ time: d.time, value: TARGET_VALUE }})));

const volSeries = volumeChart.addHistogramSeries({{ priceScaleId: '' }});
volSeries.priceScale().applyOptions({{ scaleMargins: {{ top: 0.1, bottom: 0 }} }});
volSeries.setData(ALL_VOLUMES);

function toggleTrigger(btn) {{
  const on = btn.classList.toggle('active');
  triggerSeries.applyOptions({{ visible: on }});
}}

{COMMON_JS}

setupTooltip(priceChart, () => priceSeries, priceEl);
syncCharts(priceChart, volumeChart);
resizeAll();
</script>
</body>
</html>"""