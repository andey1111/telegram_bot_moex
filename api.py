from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional
import pandas as pd
import pandas_ta as ta
from datetime import date, datetime, timedelta
import asyncio
import logging
import os

from cache import (
    init_cache_tables,
    get_candles_cached,
    get_indicator_cached,
    get_latest_indicator_value,
)

# Импортируем get_now_price из bot.py или дублируем здесь
import aiohttp
import time

logger = logging.getLogger(__name__)

app = FastAPI(title="MOEX Bot API", version="1.0.0")

# CORS — разрешаем запросы с любого origin (для демо)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статические файлы (HTML, CSS, JS)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Кэш цен
price_cache: dict = {}
PRICE_CACHE_TTL = 60


# ─────────────────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    await init_cache_tables()
    logger.info("✅ API запущен")


# ─────────────────────────────────────────────────────────
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────

async def get_now_price(ticker: str) -> Optional[float]:
    ticker = ticker.upper()
    current_time = time.time()
    if ticker in price_cache:
        cached_price, timestamp = price_cache[ticker]
        if current_time - timestamp < PRICE_CACHE_TTL:
            return cached_price

    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params={"iss.only": "marketdata"}, timeout=aiohttp.ClientTimeout(total=5), ssl=False) as response:
                if response.status != 200:
                    return None
                data = await response.json()
                marketdata = data.get("marketdata", {})
                if not marketdata.get("data"):
                    return None
                columns = marketdata["columns"]
                row = marketdata["data"][0]
                if "LAST" in columns:
                    price = row[columns.index("LAST")]
                    if price is not None:
                        price_float = float(price)
                        price_cache[ticker] = (price_float, current_time)
                        return price_float
    except Exception as e:
        logger.error(f"Ошибка цены {ticker}: {e}")
    return None


def df_to_candles(df: pd.DataFrame) -> list:
    """Конвертирует DataFrame в список словарей для lightweight-charts."""
    result = []
    for ts, row in df.iterrows():
        if any(pd.isna([row["Open"], row["High"], row["Low"], row["Close"]])):
            continue
        result.append({
            "time": int(ts.timestamp()),
            "open":  round(float(row["Open"]),  2),
            "high":  round(float(row["High"]),  2),
            "low":   round(float(row["Low"]),   2),
            "close": round(float(row["Close"]), 2),
        })
    return result


def df_to_volumes(df: pd.DataFrame) -> list:
    result = []
    for ts, row in df.iterrows():
        if pd.isna(row["Volume"]):
            continue
        color = "#26a69a" if float(row["Close"]) >= float(row["Open"]) else "#ef5350"
        result.append({
            "time":  int(ts.timestamp()),
            "value": round(float(row["Volume"]), 0),
            "color": color,
        })
    return result


def df_to_indicator(indicator_df: pd.DataFrame) -> dict:
    """Конвертирует DataFrame индикатора в словарь серий."""
    result = {}
    for col in indicator_df.columns:
        series = []
        for ts, val in indicator_df[col].items():
            if not pd.isna(val):
                series.append({
                    "time":  int(ts.timestamp()),
                    "value": round(float(val), 4),
                })
        result[col] = series
    return result


# ─────────────────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Главная страница."""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "MOEX API работает", "docs": "/docs"}


@app.get("/api/price/{ticker}")
async def get_price(ticker: str):
    """Текущая цена акции."""
    ticker = ticker.upper()
    price = await get_now_price(ticker)
    if price is None:
        raise HTTPException(status_code=404, detail=f"Цена для {ticker} не найдена")
    return {
        "ticker": ticker,
        "price":  price,
        "time":   datetime.now().isoformat(),
    }


@app.get("/api/candles/{ticker}")
async def get_candles(
    ticker: str,
    start: Optional[str] = Query(None, description="Дата начала DD.MM.YYYY"),
    end:   Optional[str] = Query(None, description="Дата окончания DD.MM.YYYY"),
):
    """
    Исторические свечи для графика.
    
    Пример: /api/candles/SBER?start=01.01.2024
    """
    ticker = ticker.upper()

    df = await get_candles_cached(ticker, start, end)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"Данные для {ticker} не найдены")

    current_price = await get_now_price(ticker)
    change = float(df["Close"].iloc[-1] - df["Close"].iloc[0])
    change_pct = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[0]) - 1) * 100

    return {
        "ticker":        ticker,
        "candles":       df_to_candles(df),
        "volumes":       df_to_volumes(df),
        "current_price": current_price,
        "change":        round(change, 2),
        "change_pct":    round(change_pct, 2),
        "period_start":  df.index[0].strftime("%d.%m.%Y"),
        "period_end":    df.index[-1].strftime("%d.%m.%Y"),
        "records":       len(df),
    }


@app.get("/api/indicator/{ticker}")
async def get_indicator(
    ticker:    str,
    metric:    str = Query(..., description="Индикатор с параметрами, например: rsi 14"),
    start:     Optional[str] = Query(None, description="Дата начала DD.MM.YYYY"),
    end:       Optional[str] = Query(None, description="Дата окончания DD.MM.YYYY"),
):
    """
    Расчёт индикатора.
    
    Примеры:
    - /api/indicator/SBER?metric=rsi 14
    - /api/indicator/SBER?metric=macd 12 26 9
    - /api/indicator/SBER?metric=sma 50
    """
    ticker = ticker.upper()
    ind_name = metric.strip().lower().split()[0]

    if not hasattr(ta, ind_name):
        raise HTTPException(status_code=400, detail=f"Индикатор '{ind_name}' не найден в pandas_ta")

    # Свечи
    price_df = await get_candles_cached(ticker, start, end)
    if price_df is None or price_df.empty:
        raise HTTPException(status_code=404, detail=f"Данные для {ticker} не найдены")

    # Индикатор
    indicator_df = await get_indicator_cached(ticker, metric, start, end)
    if indicator_df is None or indicator_df.empty:
        raise HTTPException(status_code=500, detail=f"Не удалось рассчитать {metric}")

    # Последние значения
    last_values = {}
    for col, val in indicator_df.iloc[-1].to_dict().items():
        if not pd.isna(val):
            last_values[col] = round(float(val), 4)

    return {
        "ticker":       ticker,
        "metric":       metric,
        "candles":      df_to_candles(price_df),
        "volumes":      df_to_volumes(price_df),
        "indicator":    df_to_indicator(indicator_df),
        "last_values":  last_values,
        "is_overlay":   ind_name in {"sma", "ema", "wma", "bbands", "bollinger"},
        "period_start": price_df.index[0].strftime("%d.%m.%Y"),
        "period_end":   price_df.index[-1].strftime("%d.%m.%Y"),
        "records":      len(price_df),
    }


@app.get("/api/indicators/list")
async def list_indicators():
    """Список всех доступных индикаторов из pandas_ta."""
    indicators = [name for name in dir(ta) if not name.startswith("_") and callable(getattr(ta, name))]
    return {
        "count":      len(indicators),
        "indicators": sorted(indicators),
    }


@app.get("/api/search/{query}")
async def search_ticker(query: str):
    """Поиск тикера на MOEX."""
    url = f"https://iss.moex.com/iss/securities.json"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params={"q": query, "limit": 10, "engine": "stock", "market": "shares"},
                timeout=aiohttp.ClientTimeout(total=5),
                ssl=False
            ) as response:
                data = await response.json()
                securities = data.get("securities", {})
                if not securities.get("data"):
                    return {"results": []}

                columns = securities["columns"]
                results = []
                for row in securities["data"][:10]:
                    item = dict(zip(columns, row))
                    results.append({
                        "ticker": item.get("secid", ""),
                        "name":   item.get("shortname", ""),
                        "type":   item.get("typename", ""),
                    })
                return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))