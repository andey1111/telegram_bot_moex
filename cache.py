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

# если биржа закрылась — свеча сформировалась, можно не обновлять
# 19 часов это конец основной сессии MOEX
CANDLE_CLOSE_HOUR = 19


# =========================
# СОЗДАЁМ ТАБЛИЦЫ В БАЗЕ ДАННЫХ
# =========================

async def init_cache_tables():
    # создаю три таблицы если их ещё нет
    # candles_cache — храним свечи чтобы не дёргать MOEX каждый раз
    # indicator_cache — храним посчитанные индикаторы
    # cache_meta — запоминаем когда последний раз обновляли тикер
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
        # индексы чтобы поиск по тикеру был быстрее
        await db.execute("CREATE INDEX IF NOT EXISTS idx_candles_ticker ON candles_cache(ticker)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_indicator_ticker ON indicator_cache(ticker, indicator)")
        await db.commit()
        logger.info("✅ Таблицы кэша инициализированы")


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ШТУКИ
# =========================

def _candle_is_closed_today() -> bool:
    # просто проверяю — уже после 19 часов или нет
    return datetime.now().hour >= CANDLE_CLOSE_HOUR


def _today() -> str:
    # возвращаю сегодняшнюю дату в нужном формате
    return date.today().isoformat()


def _yesterday() -> str:
    # вчерашняя дата — иногда нужна
    return (date.today() - timedelta(days=1)).isoformat()


async def _get_last_update(ticker: str) -> Optional[str]:
    # смотрю в базу — когда последний раз обновляли этот тикер
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT last_update FROM cache_meta WHERE ticker = ?", (ticker,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def _set_last_update(ticker: str, dt: str):
    # записываю время обновления — если тикер уже есть то просто обновляю
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO cache_meta (ticker, last_update) VALUES (?, ?) "
            "ON CONFLICT(ticker) DO UPDATE SET last_update = ?",
            (ticker, dt, dt)
        )
        await db.commit()


async def _load_candles_from_db(ticker: str) -> Optional[pd.DataFrame]:
    # читаю свечи из базы и возвращаю как датафрейм
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
    # сохраняю свечи в базу — если свеча уже есть то обновляю её
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
    # запрашиваю данные у московской биржи через их апи
    # это синхронная функция — запускаю её через executor чтобы не блокировать бота
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


# =========================
# ГЛАВНАЯ ФУНКЦИЯ — ПОЛУЧИТЬ СВЕЧИ
# =========================

async def get_candles_cached(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None
) -> Optional[pd.DataFrame]:
    # главная идея — сначала смотрю в кэш, и только если надо — иду на биржу
    # это сильно ускоряет работу и не нагружает api мосбиржи лишними запросами
    ticker = ticker.upper()
    now_str = datetime.now().isoformat()
    today   = _today()

    last_update = await _get_last_update(ticker)
    cached_df   = await _load_candles_from_db(ticker)

    # проверяю нужно ли обновлять данные
    need_update = True
    if last_update and cached_df is not None:
        last_dt = datetime.fromisoformat(last_update)
        # если уже обновляли сегодня после закрытия биржи — не обновляем
        if (last_dt.date() == date.today()
                and last_dt.hour >= CANDLE_CLOSE_HOUR
                and _candle_is_closed_today()):
            need_update = False
        # если биржа ещё не закрылась сегодня — тоже не обновляем
        elif last_dt.date() == date.today() and not _candle_is_closed_today():
            need_update = False

    if need_update:
        # умная догрузка — беру только то чего не хватает а не всё заново
        if cached_df is not None and len(cached_df) > 0:
            # в кэше что-то есть — догружаю только новые свечи
            fetch_start = (cached_df.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            # кэша нет вообще — загружаю данные за 2 года
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
            # новые свечи появились — старые значения индикаторов уже неактуальны
            await _invalidate_indicator_cache(ticker, fetch_start)
            # перечитываю полный кэш из базы
            cached_df = await _load_candles_from_db(ticker)
        elif cached_df is None:
            # биржа ничего не вернула и кэша нет — не повезло
            logger.warning(f"⚠️ Нет данных для {ticker}")
            return None
        else:
            # биржа ничего не вернула но кэш есть — наверное выходной
            await _set_last_update(ticker, now_str)

    if cached_df is None or cached_df.empty:
        return None

    # обрезаю по нужному периоду если пользователь указал даты
    if start:
        cached_df = cached_df[cached_df.index >= pd.to_datetime(start)]
    if end:
        cached_df = cached_df[cached_df.index <= pd.to_datetime(end)]

    return cached_df if not cached_df.empty else None


# =========================
# КЭШИРОВАНИЕ ИНДИКАТОРОВ
# =========================

def _indicator_key(metric: str) -> str:
    # делаю из "RSI 14" нормальный ключ "rsi_14" для хранения в базе
    return metric.strip().lower().replace(" ", "_")


async def _invalidate_indicator_cache(ticker: str, from_date: str):
    # удаляю старые значения индикаторов начиная с даты когда пришли новые свечи
    # потому что индикаторы пересчитаются с учётом новых данных
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM indicator_cache WHERE ticker = ? AND date >= ?",
            (ticker, from_date)
        )
        await db.commit()


async def _load_indicator_from_db(ticker: str, ind_key: str) -> Optional[pd.Series]:
    # читаю значения индикатора из базы
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
    # сохраняю значения индикатора в базу — пропускаю NaN которые бывают в начале
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


# =========================
# ГЛАВНАЯ ФУНКЦИЯ — ПОСЧИТАТЬ ИНДИКАТОР
# =========================

async def get_indicator_cached(
    ticker,
    metric,
    start=None,
    end=None
):
    # считаю индикатор и кэширую результат
    # в следующий раз просто достану из базы без пересчёта
    ticker  = ticker.upper()
    ind_key = _indicator_key(metric)

    # сначала получаю свежие свечи (они тоже кэшируются)
    df = await get_candles_cached(ticker)
    if df is None or df.empty:
        return None

    # считаю индикатор в отдельном потоке чтобы не тормозить бота
    loop = asyncio.get_event_loop()
    result_df = await loop.run_in_executor(None, _calc_indicator, df, metric)

    if result_df is None or result_df.empty:
        return None

    # кэширую каждую колонку отдельно — у MACD например три колонки
    for col in result_df.columns:
        col_key = f"{ind_key}__{col}"
        await _save_indicator_to_db(ticker, col_key, result_df[col].dropna())

    # обрезаю по нужному периоду
    if start:
        result_df = result_df[result_df.index >= pd.to_datetime(start)]
    if end:
        result_df = result_df[result_df.index <= pd.to_datetime(end)]

    return result_df if not result_df.empty else None


def _calc_indicator(df: pd.DataFrame, metric: str) -> Optional[pd.DataFrame]:
    # здесь происходит сам расчёт индикатора через pandas-ta
    # для популярных индикаторов прописал параметры вручную
    # для всех остальных — универсальная ветка которая пробует разные комбинации
    metric   = metric.strip().lower()
    parts    = metric.split()
    ind_name = parts[0]

    if not hasattr(ta, ind_name):
        return None

    try:
        func   = getattr(ta, ind_name)
        params = [int(p) for p in parts[1:] if p.isdigit()]

        if ind_name == "macd":
            # macd принимает три параметра: быстрая, медленная, сигнальная
            fast   = params[0] if len(params) > 0 else 12
            slow   = params[1] if len(params) > 1 else 26
            signal = params[2] if len(params) > 2 else 9
            result = func(close=df["Close"], fast=fast, slow=slow, signal=signal)
            return pd.DataFrame(result)

        elif ind_name in ["bbands", "bollinger"]:
            # полосы боллинджера — длина и количество стандартных отклонений
            length = params[0] if params else 20
            std    = params[1] if len(params) > 1 else 2
            result = func(close=df["Close"], length=length, std=std)
            return result if isinstance(result, pd.DataFrame) else pd.DataFrame({ind_name: result})

        elif ind_name == "stoch":
            # стохастик — три параметра k, d и сглаживание
            k        = params[0] if len(params) > 0 else 14
            d        = params[1] if len(params) > 1 else 3
            smooth_k = params[2] if len(params) > 2 else 3
            result = func(high=df["High"], low=df["Low"], close=df["Close"],
                          k=k, d=d, smooth_k=smooth_k)
            return result if isinstance(result, pd.DataFrame) else pd.DataFrame({ind_name: result})

        elif ind_name in ["atr", "adx"]:
            # эти индикаторы требуют high и low помимо close
            length = params[0] if params else 14
            result = func(high=df["High"], low=df["Low"], close=df["Close"], length=length)
            return pd.DataFrame({ind_name: result}) if isinstance(result, pd.Series) else pd.DataFrame(result)

        elif ind_name in ["rsi", "cci", "willr"]:
            # осцилляторы — только close и период
            length = params[0] if params else 14
            result = func(close=df["Close"], length=length)
            return pd.DataFrame({ind_name: result}) if isinstance(result, pd.Series) else pd.DataFrame(result)

        elif ind_name in ["sma", "ema", "wma"]:
            # скользящие средние — самые простые
            length = params[0] if params else 20
            result = func(close=df["Close"], length=length)
            return pd.DataFrame({ind_name: result})

        else:
            # универсальная ветка для всех остальных индикаторов из pandas-ta
            # пробую передать все данные, если не работает — убираю лишнее
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
                # некоторые индикаторы не принимают все параметры — пробую варианты
                for drop in [("volume",), ("high", "low"), ("high", "low", "volume")]:
                    try:
                        kw = {k: v for k, v in kwargs.items() if k not in drop}
                        result = func(**kw)
                        break
                    except TypeError:
                        continue
                else:
                    # последний вариант — только close
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


# =========================
# УТИЛИТЫ
# =========================

async def get_latest_indicator_value(ticker: str, metric: str) -> Optional[Dict]:
    # беру последнее значение индикатора — нужно для проверки алертов
    result_df = await get_indicator_cached(ticker, metric, None, None)
    if result_df is None or result_df.empty:
        return None
    logger.info(f"Колонки индикатора {metric}: {list(result_df.columns)}")
    last = result_df.iloc[-1].to_dict()
    return {k: v for k, v in last.items() if not pd.isna(v)}


async def clear_ticker_cache(ticker: str):
    # полностью удаляю кэш тикера — полезно если данные слетели
    ticker = ticker.upper()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM candles_cache WHERE ticker = ?", (ticker,))
        await db.execute("DELETE FROM indicator_cache WHERE ticker = ?", (ticker,))
        await db.execute("DELETE FROM cache_meta WHERE ticker = ?", (ticker,))
        await db.commit()
    logger.info(f"🗑 Кэш {ticker} очищен")


async def get_cache_stats() -> Dict:
    # статистика кэша — сколько тикеров и записей храним
    # удобно для отладки чтобы понять не разрослась ли база
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