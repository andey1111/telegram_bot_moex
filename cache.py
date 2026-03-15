import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List
import pandas as pd
import pandas_ta as ta
import aiosqlite
import apimoex
import requests

logger = logging.getLogger(__name__)

DB_PATH = "alerts.db"

# После какого часа считаем что дневная свеча закрылась
# 19 — только основная сессия, 0 — с учётом вечерней
CANDLE_CLOSE_HOUR = 19


# ─────────────────────────────────────────────────────────
#  ИНИЦИАЛИЗАЦИЯ ТАБЛИЦ КЭША
# ─────────────────────────────────────────────────────────

async def init_cache_tables():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS candles_cache (
                ticker  TEXT NOT NULL,
                date    TEXT NOT NULL,
                open    REAL,
                high    REAL,
                low     REAL,
                close   REAL,
                volume  REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS indicator_cache (
                ticker     TEXT NOT NULL,
                date       TEXT NOT NULL,
                indicator  TEXT NOT NULL,
                value      REAL,
                PRIMARY KEY (ticker, date, indicator)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS cache_meta (
                ticker      TEXT PRIMARY KEY,
                last_update TEXT NOT NULL
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_candles_ticker ON candles_cache(ticker)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_indicator_ticker ON indicator_cache(ticker, indicator)")
        await db.commit()
        logger.info("✅ Таблицы кэша инициализированы")


# ─────────────────────────────────────────────────────────
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────

def _candle_is_closed_today() -> bool:
    """Закрылась ли уже сегодняшняя дневная свеча."""
    return datetime.now().hour >= CANDLE_CLOSE_HOUR


def _today() -> str:
    return date.today().isoformat()


def _yesterday() -> str:
    return (date.today() - timedelta(days=1)).isoformat()


async def _get_last_update(ticker: str) -> Optional[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT last_update FROM cache_meta WHERE ticker = ?", (ticker,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def _set_last_update(ticker: str, dt: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO cache_meta (ticker, last_update) VALUES (?, ?) "
            "ON CONFLICT(ticker) DO UPDATE SET last_update = ?",
            (ticker, dt, dt)
        )
        await db.commit()


async def _load_candles_from_db(ticker: str) -> Optional[pd.DataFrame]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT date, open, high, low, close, volume FROM candles_cache "
            "WHERE ticker = ? ORDER BY date ASC",
            (ticker,)
        ) as cursor:
            rows = await cursor.fetchall()

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["date", "Open", "High", "Low", "Close", "Volume"])
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


async def _save_candles_to_db(ticker: str, df: pd.DataFrame):
    async with aiosqlite.connect(DB_PATH) as db:
        for ts, row in df.iterrows():
            await db.execute(
                "INSERT INTO candles_cache (ticker, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(ticker, date) DO UPDATE SET "
                "open=excluded.open, high=excluded.high, low=excluded.low, "
                "close=excluded.close, volume=excluded.volume",
                (
                    ticker,
                    ts.strftime("%Y-%m-%d"),
                    float(row["Open"]) if pd.notna(row["Open"]) else None,
                    float(row["High"]) if pd.notna(row["High"]) else None,
                    float(row["Low"])  if pd.notna(row["Low"])  else None,
                    float(row["Close"])if pd.notna(row["Close"])else None,
                    float(row["Volume"])if pd.notna(row["Volume"])else None,
                )
            )
        await db.commit()


def _fetch_moex(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Синхронный запрос к MOEX ISS."""
    try:
        with requests.Session() as session:
            data = apimoex.get_board_history(
                session=session,
                security=ticker.upper(),
                board="TQBR",
                start=start,
                end=end,
                columns=("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "TRADEDATE")
            )
        if not data:
            return None
        df = pd.DataFrame(data)
        df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
        df.set_index("TRADEDATE", inplace=True)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        logger.error(f"Ошибка MOEX запроса {ticker}: {e}")
        return None


# ─────────────────────────────────────────────────────────
#  ОСНОВНАЯ ФУНКЦИЯ — ПОЛУЧИТЬ СВЕЧИ С КЭШОМ
# ─────────────────────────────────────────────────────────

async def get_candles_cached(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Получает дневные свечи с кэшированием.
    - Если кэш свежий (обновлён сегодня после CANDLE_CLOSE_HOUR) — возвращает из БД
    - Иначе — догружает только недостающие свечи от MOEX и обновляет кэш
    """
    ticker = ticker.upper()
    now_str = datetime.now().isoformat()
    today   = _today()

    last_update = await _get_last_update(ticker)
    cached_df   = await _load_candles_from_db(ticker)

    # Определяем нужно ли обновлять
    need_update = True
    if last_update and cached_df is not None:
        last_dt = datetime.fromisoformat(last_update)
        # Не обновляем если обновляли сегодня после закрытия свечи
        if (last_dt.date() == date.today()
                and last_dt.hour >= CANDLE_CLOSE_HOUR
                and _candle_is_closed_today()):
            need_update = False
        # Или обновляли в выходной/праздник и биржа не торговала
        elif last_dt.date() == date.today() and not _candle_is_closed_today():
            need_update = False

    if need_update:
        # Загружаем только недостающий период
        if cached_df is not None and len(cached_df) > 0:
            # Есть кэш — догружаем от последней даты
            fetch_start = (cached_df.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            # Нет кэша — грузим всё за последние 2 года
            fetch_start = (date.today() - timedelta(days=730)).isoformat()

        fetch_end = today
        logger.info(f"🔄 Обновляем кэш {ticker}: {fetch_start} → {fetch_end}")

        loop = asyncio.get_event_loop()
        new_df = await loop.run_in_executor(
            None, _fetch_moex, ticker, fetch_start, fetch_end
        )

        if new_df is not None and not new_df.empty:
            await _save_candles_to_db(ticker, new_df)
            await _set_last_update(ticker, now_str)
            # Инвалидируем кэш индикаторов для новых дат
            await _invalidate_indicator_cache(ticker, fetch_start)
            # Перечитываем полный кэш из БД
            cached_df = await _load_candles_from_db(ticker)
        elif cached_df is None:
            logger.warning(f"⚠️ Нет данных для {ticker}")
            return None
        else:
            # MOEX не вернул новых данных (выходной?) — используем кэш
            await _set_last_update(ticker, now_str)

    if cached_df is None or cached_df.empty:
        return None

    # Фильтруем по запрошенному периоду
    if start:
        cached_df = cached_df[cached_df.index >= pd.to_datetime(start)]
    if end:
        cached_df = cached_df[cached_df.index <= pd.to_datetime(end)]

    return cached_df if not cached_df.empty else None


# ─────────────────────────────────────────────────────────
#  КЭШИРОВАНИЕ ИНДИКАТОРОВ
# ─────────────────────────────────────────────────────────

def _indicator_key(metric: str) -> str:
    """Нормализует название индикатора для использования как ключ."""
    return metric.strip().lower().replace(" ", "_")


async def _invalidate_indicator_cache(ticker: str, from_date: str):
    """Удаляет устаревшие значения индикатора начиная с from_date."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM indicator_cache WHERE ticker = ? AND date >= ?",
            (ticker, from_date)
        )
        await db.commit()


async def _load_indicator_from_db(ticker: str, ind_key: str) -> Optional[pd.Series]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT date, value FROM indicator_cache "
            "WHERE ticker = ? AND indicator = ? ORDER BY date ASC",
            (ticker, ind_key)
        ) as cursor:
            rows = await cursor.fetchall()

    if not rows:
        return None

    index = pd.to_datetime([r[0] for r in rows])
    values = [r[1] for r in rows]
    return pd.Series(values, index=index, name=ind_key)


async def _save_indicator_to_db(ticker: str, ind_key: str, series: pd.Series):
    async with aiosqlite.connect(DB_PATH) as db:
        for ts, val in series.items():
            if pd.isna(val):
                continue
            await db.execute(
                "INSERT INTO indicator_cache (ticker, date, indicator, value) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(ticker, date, indicator) DO UPDATE SET value = excluded.value",
                (ticker, ts.strftime("%Y-%m-%d"), ind_key, float(val))
            )
        await db.commit()


# ─────────────────────────────────────────────────────────
#  ОСНОВНАЯ ФУНКЦИЯ — РАССЧИТАТЬ ИНДИКАТОР С КЭШОМ
# ─────────────────────────────────────────────────────────

async def get_indicator_cached(
    ticker: str,
    metric: str,
    start: Optional[str] = None,
    end: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Рассчитывает индикатор с кэшированием.
    Для многоколоночных индикаторов (MACD, BBands) каждая колонка кэшируется отдельно.
    """
    ticker  = ticker.upper()
    ind_key = _indicator_key(metric)

    # Получаем свежие свечи (с кэшем)
    df = await get_candles_cached(ticker)
    if df is None or df.empty:
        return None

    # Считаем индикатор для полного датасета
    loop = asyncio.get_event_loop()
    result_df = await loop.run_in_executor(None, _calc_indicator, df, metric)

    if result_df is None or result_df.empty:
        return None

    # Кэшируем каждую колонку отдельно
    for col in result_df.columns:
        col_key = f"{ind_key}__{col}"
        await _save_indicator_to_db(ticker, col_key, result_df[col].dropna())

    # Фильтруем по запрошенному периоду
    if start:
        result_df = result_df[result_df.index >= pd.to_datetime(start)]
    if end:
        result_df = result_df[result_df.index <= pd.to_datetime(end)]

    return result_df if not result_df.empty else None


def _calc_indicator(df: pd.DataFrame, metric: str) -> Optional[pd.DataFrame]:
    """Синхронный расчёт индикатора — вызывается через executor."""
    metric   = metric.strip().lower()
    parts    = metric.split()
    ind_name = parts[0]

    if not hasattr(ta, ind_name):
        return None

    try:
        func   = getattr(ta, ind_name)
        params = [int(p) for p in parts[1:] if p.isdigit()]

        if ind_name == "macd":
            fast   = params[0] if len(params) > 0 else 12
            slow   = params[1] if len(params) > 1 else 26
            signal = params[2] if len(params) > 2 else 9
            result = func(close=df["Close"], fast=fast, slow=slow, signal=signal)
            return pd.DataFrame(result)

        elif ind_name in ["bbands", "bollinger"]:
            length = params[0] if params else 20
            std    = params[1] if len(params) > 1 else 2
            result = func(close=df["Close"], length=length, std=std)
            return result if isinstance(result, pd.DataFrame) else pd.DataFrame({ind_name: result})

        elif ind_name == "stoch":
            k        = params[0] if len(params) > 0 else 14
            d        = params[1] if len(params) > 1 else 3
            smooth_k = params[2] if len(params) > 2 else 3
            result = func(high=df["High"], low=df["Low"], close=df["Close"],
                          k=k, d=d, smooth_k=smooth_k)
            return result if isinstance(result, pd.DataFrame) else pd.DataFrame({ind_name: result})

        elif ind_name in ["atr", "adx"]:
            length = params[0] if params else 14
            result = func(high=df["High"], low=df["Low"], close=df["Close"], length=length)
            return pd.DataFrame({ind_name: result}) if isinstance(result, pd.Series) else pd.DataFrame(result)

        elif ind_name in ["rsi", "cci", "willr"]:
            length = params[0] if params else 14
            result = func(close=df["Close"], length=length)
            return pd.DataFrame({ind_name: result}) if isinstance(result, pd.Series) else pd.DataFrame(result)

        elif ind_name in ["sma", "ema", "wma"]:
            length = params[0] if params else 20
            result = func(close=df["Close"], length=length)
            return pd.DataFrame({ind_name: result})

        else:
            kwargs = {
                "close":  df["Close"],
                "high":   df["High"],
                "low":    df["Low"],
                "volume": df["Volume"],
            }
            if params:
                kwargs["length"] = params[0]

            try:
                result = func(**kwargs)
            except TypeError:
                for drop in [("volume",), ("high", "low"), ("high", "low", "volume")]:
                    try:
                        kw = {k: v for k, v in kwargs.items() if k not in drop}
                        result = func(**kw)
                        break
                    except TypeError:
                        continue
                else:
                    kw = {"close": df["Close"]}
                    if params:
                        kw["length"] = params[0]
                    result = func(**kw)

            if isinstance(result, pd.Series):
                return pd.DataFrame({ind_name: result})
            elif isinstance(result, pd.DataFrame):
                return result
            else:
                return pd.DataFrame({ind_name: result})

    except Exception as e:
        logger.error(f"Ошибка расчёта {ind_name}: {e}")
        return None


# ─────────────────────────────────────────────────────────
#  УТИЛИТЫ
# ─────────────────────────────────────────────────────────

async def get_latest_indicator_value(ticker: str, metric: str) -> Optional[Dict]:
    result_df = await get_indicator_cached(ticker, metric)
    if result_df is None or result_df.empty:
        return None
    logger.info(f"Колонки индикатора {metric}: {list(result_df.columns)}")  # ← добавить
    last = result_df.iloc[-1].to_dict()
    return {k: v for k, v in last.items() if not pd.isna(v)}


async def clear_ticker_cache(ticker: str):
    """Полностью сбрасывает кэш для тикера."""
    ticker = ticker.upper()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM candles_cache WHERE ticker = ?", (ticker,))
        await db.execute("DELETE FROM indicator_cache WHERE ticker = ?", (ticker,))
        await db.execute("DELETE FROM cache_meta WHERE ticker = ?", (ticker,))
        await db.commit()
    logger.info(f"🗑 Кэш {ticker} очищен")


async def get_cache_stats() -> Dict:
    """Статистика кэша для отладки."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT COUNT(DISTINCT ticker) FROM candles_cache") as c:
            tickers = (await c.fetchone())[0]
        async with db.execute("SELECT COUNT(*) FROM candles_cache") as c:
            candles = (await c.fetchone())[0]
        async with db.execute("SELECT COUNT(*) FROM indicator_cache") as c:
            indicators = (await c.fetchone())[0]

    return {
        "tickers_cached": tickers,
        "candles_total":  candles,
        "indicators_total": indicators,
    }