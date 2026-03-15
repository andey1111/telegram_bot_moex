import asyncio
import logging
import io
import time
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from charts import build_price_html, build_indicator_html, build_alert_html
from cache import init_cache_tables, get_candles_cached, get_indicator_cached, get_latest_indicator_value




import pandas as pd
import pandas_ta as ta
import apimoex
import requests
import aiohttp
import aiosqlite
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    Message, CallbackQuery, ReplyKeyboardMarkup, KeyboardButton,
    ReplyKeyboardRemove, InlineKeyboardMarkup, InlineKeyboardButton,
    BufferedInputFile
)
import os

import dotenv
dotenv.load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

# =========================
# КОНФИГУРАЦИЯ
# =========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = "alerts.db"
MAX_ALERTS_PER_USER = 50
PRICE_CACHE_TTL = 60
ALERT_CHECK_INTERVAL = 30

from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession

session = AiohttpSession(timeout=60)
router = Router()
user_locks = defaultdict(asyncio.Lock)
price_cache: Dict[str, Tuple[float, float]] = {}

# =========================
# БАЗА ДАННЫХ
# =========================

async def init_database():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                alerts_enabled INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                indicator TEXT NOT NULL,
                condition TEXT NOT NULL CHECK(condition IN ('выше', 'ниже')),
                value REAL NOT NULL,
                created_at TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                last_checked TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_active ON alerts(is_active)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts(user_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ticker ON alerts(ticker)")
        await db.commit()
        logger.info("✅ База данных инициализирована")

async def get_user_alerts_enabled(user_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT alerts_enabled FROM users WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return bool(row[0]) if row else True

async def set_user_alerts_enabled(user_id: int, enabled: bool):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO users (user_id, alerts_enabled) VALUES (?, ?)
               ON CONFLICT(user_id) DO UPDATE SET alerts_enabled = ?""",
            (user_id, int(enabled), int(enabled))
        )
        await db.commit()

async def count_user_alerts(user_id: int) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM alerts WHERE user_id = ? AND is_active = 1", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

async def add_alert(user_id: int, ticker: str, indicator: str, condition: str, value: float) -> bool:
    async with user_locks[user_id]:
        count = await count_user_alerts(user_id)
        if count >= MAX_ALERTS_PER_USER:
            return False
        async with aiosqlite.connect(DB_PATH) as db:
            created_at = datetime.now().isoformat()
            await db.execute(
                "INSERT INTO alerts (user_id, ticker, indicator, condition, value, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, ticker, indicator, condition, value, created_at)
            )
            await db.commit()
        return True

async def get_user_alerts(user_id: int) -> List[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT id, ticker, indicator, condition, value, created_at
               FROM alerts WHERE user_id = ? AND is_active = 1
               ORDER BY created_at DESC""",
            (user_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

async def delete_alert_by_id(alert_id: int, user_id: int) -> bool:
    async with user_locks[user_id]:
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute(
                "DELETE FROM alerts WHERE id = ? AND user_id = ?", (alert_id, user_id)
            )
            await db.commit()
            return cursor.rowcount > 0

async def get_all_active_alerts() -> List[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, user_id, ticker, indicator, condition, value FROM alerts WHERE is_active = 1"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

async def deactivate_alert(alert_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
        await db.commit()

# =========================
# MOEX
# =========================


def create_price_chart(ticker: str, df: pd.DataFrame) -> io.BytesIO:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(df.index, df['Close'], label='Цена закрытия', linewidth=2, color='blue', alpha=0.7)
    ax1.set_title(f'{ticker} - График цен', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Цена, руб.', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    colors = ['green' if close >= open_ else 'red'
              for close, open_ in zip(df['Close'], df['Open'])]
    ax2.bar(df.index, df['Volume'], color=colors, alpha=0.6)
    ax2.set_ylabel('Объем', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_indicator_chart(ticker: str, price_df: pd.DataFrame, indicator_df: pd.DataFrame, indicator_name: str) -> io.BytesIO:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Выравниваем индексы — берём только общие даты
    common_index = price_df.index.intersection(indicator_df.index)
    price_aligned = price_df.loc[common_index]
    indicator_aligned = indicator_df.loc[common_index]

    axes[0].plot(price_aligned.index, price_aligned['Close'], label='Цена закрытия', linewidth=2, color='blue', alpha=0.7)

    if indicator_name.lower() in ['sma', 'ema', 'wma']:
        if not indicator_aligned.empty:
            ind_col = indicator_aligned.columns[0]
            axes[0].plot(price_aligned.index, indicator_aligned[ind_col],
                        label=f'{indicator_name.upper()}', linewidth=2, color='orange', alpha=0.8)

    axes[0].set_title(f'{ticker} - {indicator_name.upper()}', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Цена, руб.', fontsize=12)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel('Значение индикатора', fontsize=12)
    if not indicator_aligned.empty:
        for col in indicator_aligned.columns:
            axes[1].plot(price_aligned.index, indicator_aligned[col], label=col, linewidth=2)

        if indicator_name.lower() == 'rsi':
            axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Перекупленность')
            axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Перепроданность')
            axes[1].set_ylim([0, 100])
        elif indicator_name.lower() == 'stoch':
            axes[1].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Перекупленность')
            axes[1].axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Перепроданность')
            axes[1].set_ylim([0, 100])

    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


def parse_date(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")
    except ValueError:
        return None

async def get_now_price(ticker: str) -> Optional[float]:
    ticker = ticker.upper().strip()
    current_time = time.time()
    if ticker in price_cache:
        cached_price, timestamp = price_cache[ticker]
        if current_time - timestamp < PRICE_CACHE_TTL:
            return cached_price

    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params={'iss.only': 'marketdata'}, timeout=5, ssl=False) as response:
                if response.status != 200:
                    return None
                data = await response.json()
                marketdata = data.get('marketdata', {})
                if not marketdata.get('data'):
                    return None
                columns = marketdata['columns']
                row = marketdata['data'][0]
                if 'LAST' in columns:
                    price = row[columns.index('LAST')]
                    if price is not None:
                        price_float = float(price)
                        price_cache[ticker] = (price_float, current_time)
                        return price_float
    except Exception as e:
        logger.error(f"Ошибка получения цены {ticker}: {e}")
    return None

async def get_historical_data(ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> Optional[pd.DataFrame]:
    try:
        today = date.today()
        start_date = parse_date(start) if start else (today - timedelta(days=350)).isoformat()
        end_date = parse_date(end) if end else today.isoformat()
        if not start_date or not end_date:
            return None

        loop = asyncio.get_event_loop()

        def fetch_data():
            with requests.Session() as session:
                data = apimoex.get_board_history(
                    session=session,
                    security=ticker.upper(),
                    board='TQBR',
                    start=start_date,
                    end=end_date,
                    columns=('OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'TRADEDATE')
                )
            return data

        data = await loop.run_in_executor(None, fetch_data)
        if not data:
            return None

        df = pd.DataFrame(data)
        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
        df.set_index('TRADEDATE', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        logger.error(f"Ошибка получения исторических данных для {ticker}: {e}")
        return None

async def calculate_indicator(ticker: str, metric: str, start: Optional[str] = None, end: Optional[str] = None):
    df = await get_candles_cached(ticker, start, end)
    if df is None:
        return None

    metric = metric.strip().lower()
    parts = metric.split()
    if not parts:
        return None

    ind_name = parts[0]

    try:
        if not hasattr(ta, ind_name):
            return None

        loop = asyncio.get_event_loop()

        def calculate():
            func = getattr(ta, ind_name)
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
                # Универсальная ветка — пробуем передать все доступные данные
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
                    # Убираем лишние аргументы по одному если функция не принимает
                    for drop in [("volume",), ("high", "low"), ("high", "low", "volume")]:
                        try:
                            kw = {k: v for k, v in kwargs.items() if k not in drop}
                            result = func(**kw)
                            break
                        except TypeError:
                            continue
                    else:
                        # Последний шанс — только close
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

        return await loop.run_in_executor(None, calculate)
    except Exception as e:
        logger.error(f"Ошибка расчета индикатора {ind_name}: {e}")
        return None

# =========================
# HTML-ГРАФИКИ (lightweight-charts)
# =========================

# Индикаторы, которые рисуются поверх свечей (overlay)
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
            "low": round(float(row['Low']), 2),
            "close": round(float(row['Close']), 2),
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


def _indicator_series_js(indicator_df: pd.DataFrame, target_chart: str, price_scale_id: str) -> str:
    """Генерирует JS-код для всех серий индикатора."""
    js = ""
    for i, col in enumerate(indicator_df.columns):
        series_data = []
        for ts, val in indicator_df[col].items():
            if pd.isna(val):
                continue
            series_data.append({"time": int(ts.timestamp()), "value": round(float(val), 4)})
        color = COLORS[i % len(COLORS)]
        js += f"""
const indSeries_{i} = {target_chart}.addLineSeries({{
    color: '{color}',
    lineWidth: 2,
    title: '{col}',
    priceScaleId: '{price_scale_id}',
    lastValueVisible: true,
    priceLineVisible: false,
}});
indSeries_{i}.setData({json.dumps(series_data)});
"""
    return js


def _overbought_oversold_js(indicator_name: str) -> str:
    """Горизонтальные уровни для RSI/Stoch."""
    ind = indicator_name.lower()
    if ind == "rsi":
        return """
const obLine = indicatorChart.addLineSeries({ color: '#ef5350', lineWidth: 1, lineStyle: 2, title: 'OB 70', lastValueVisible: false, priceLineVisible: false });
obLine.setData(indSeries_0.data().map(d => ({ time: d.time, value: 70 })));
const osLine = indicatorChart.addLineSeries({ color: '#26a69a', lineWidth: 1, lineStyle: 2, title: 'OS 30', lastValueVisible: false, priceLineVisible: false });
osLine.setData(indSeries_0.data().map(d => ({ time: d.time, value: 30 })));
"""
    elif ind == "stoch":
        return """
const obLine = indicatorChart.addLineSeries({ color: '#ef5350', lineWidth: 1, lineStyle: 2, title: 'OB 80', lastValueVisible: false, priceLineVisible: false });
obLine.setData(indSeries_0.data().map(d => ({ time: d.time, value: 80 })));
const osLine = indicatorChart.addLineSeries({ color: '#26a69a', lineWidth: 1, lineStyle: 2, title: 'OS 20', lastValueVisible: false, priceLineVisible: false });
osLine.setData(indSeries_0.data().map(d => ({ time: d.time, value: 20 })));
"""
    return ""



# =========================
# FSM
# =========================

class IndicatorForm(StatesGroup):
    ticker = State()
    metric = State()
    start_date = State()
    end_date = State()

class AlertForm(StatesGroup):
    ticker    = State()
    type      = State()     
    condition = State()
    value     = State()
    indicator = State()     
    ind_col   = State()      



# =========================
# КЛАВИАТУРЫ
# =========================


def get_alert_type_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="💰 На цену"), KeyboardButton(text="📊 На индикатор")],
            [KeyboardButton(text="Отмена")]
        ],
        resize_keyboard=True
    )

def get_main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📊 Рассчитать индикатор")],
            [KeyboardButton(text="🔔 Создать алерт"), KeyboardButton(text="📋 Мои алерты")],
            [KeyboardButton(text="❓ Справка")]
        ],
        resize_keyboard=True
    )

def get_single_menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text='Главное меню')]],
        resize_keyboard=True
    )

def get_skip_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Пропустить")]],
        resize_keyboard=True,
        one_time_keyboard=True
    )

def get_condition_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Выше"), KeyboardButton(text="Ниже")],
            [KeyboardButton(text="Отмена")]
        ],
        resize_keyboard=True
    )

def get_export_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🌐 HTML-графики", callback_data="export_html"),
                InlineKeyboardButton(text="📸 PNG-графики", callback_data="export_png"),
            ],
            [
                InlineKeyboardButton(text="📥 CSV", callback_data="export_csv"),
                InlineKeyboardButton(text="📊 Excel", callback_data="export_excel"),
            ],
            [
                InlineKeyboardButton(text="📈 Всё сразу (HTML + CSV)", callback_data="export_all"),
            ]
        ]
    )

def get_alert_management_keyboard(enabled: bool) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(
                text="🔕 Выключить" if enabled else "🔔 Включить",
                callback_data="toggle_alerts"
            )],
            [InlineKeyboardButton(text="🗑 Удалить алерт", callback_data="show_delete_alerts")]
        ]
    )

# =========================
# ХЭНДЛЕРЫ — ОСНОВНЫЕ КОМАНДЫ
# =========================

@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "👋 <b>Привет! Я бот для анализа акций Мосбиржи</b>\n\n"
        "📊 <b>Что я умею:</b>\n"
        "• Рассчитывать технические индикаторы\n"
        "• Отправлять интерактивные HTML-графики (lightweight-charts)\n"
        "• Отслеживать цены и отправлять уведомления\n"
        "• Экспортировать данные в CSV/Excel\n\n"
        "🔹 Выбери действие из меню ниже:",
        parse_mode="HTML",
        reply_markup=get_main_menu()
    )

@router.message(Command("help"))
@router.message(F.text == "❓ Справка")
async def cmd_help(message: Message):
    help_text = (
        "📚 <b>Доступные функции:</b>\n\n"
        "📊 <b>Расчёт индикаторов:</b>\n"
        "• SMA, EMA, WMA — скользящие средние\n"
        "• RSI, MACD, Bollinger Bands\n"
        "• ATR, ADX, Stochastic и другие\n\n"
        "🌐 <b>Графики:</b>\n"
        "• Интерактивные HTML-файлы на lightweight-charts\n"
        "• Открываются в любом браузере\n"
        "• Свечи, объём, линии индикаторов\n\n"
        "🔔 <b>Алерты:</b>\n"
        "• Уведомления при достижении цены\n"
        "• Проверка каждые 30 секунд\n"
        f"• Лимит: {MAX_ALERTS_PER_USER} алертов\n\n"
        "📋 <b>Примеры индикаторов:</b>\n"
        "• sma 50 · ema 20 · macd 12 26 9\n"
        "• rsi 14 · bbands 20 2 · atr 14\n\n"
        "⚙️ <b>Команды:</b>\n"
        "/start — главное меню\n"
        "/help — справка\n"
        "/cancel — отмена"
    )
    await message.answer(help_text, parse_mode="HTML", reply_markup=get_main_menu())

@router.message(Command("cancel"))
@router.message(F.text.casefold() == "отмена")
async def cmd_cancel(message: Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        await message.answer("❌ Нечего отменять", reply_markup=get_main_menu())
        return
    await state.clear()
    await message.answer("✅ Действие отменено", reply_markup=get_main_menu())

@router.message(F.text == 'Главное меню')
async def handle_main_menu_button(message: Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is not None:
        await state.clear()
    await message.answer("📋 <b>Главное меню</b>", parse_mode="HTML", reply_markup=get_main_menu())

# =========================
# ХЭНДЛЕРЫ — РАСЧЁТ ИНДИКАТОРОВ
# =========================



@router.message(Command("calc"))
@router.message(F.text == "📊 Рассчитать индикатор")
async def cmd_calc(message: Message, state: FSMContext):
    await state.set_state(IndicatorForm.ticker)
    await message.answer(
        "📊 <b>Введи тикер акции</b>\n\nПримеры: SBER, GAZP, YNDX, VTBR, ROSN",
        parse_mode="HTML",
        reply_markup=ReplyKeyboardRemove()
    )

@router.message(IndicatorForm.ticker)
async def process_ticker(message: Message, state: FSMContext):
    ticker = message.text.strip().upper()
    if len(ticker) < 2 or len(ticker) > 10:
        await message.answer("❌ Некорректный тикер. Попробуй еще раз:")
        return
    await state.update_data(ticker=ticker)
    await state.set_state(IndicatorForm.metric)
    await message.answer(
        f"✅ Тикер: <b>{ticker}</b>\n\n"
        "📈 <b>Введи индикатор с параметрами:</b>\n\n"
        "• sma 50 · ema 20 · macd 12 26 9\n"
        "• rsi 14 · bbands 20 2 · atr 14",
        parse_mode="HTML",
        reply_markup=get_single_menu_keyboard()
    )

@router.message(IndicatorForm.metric)
async def process_metric(message: Message, state: FSMContext):
    metric = message.text.strip()
    if not metric:
        await message.answer("❌ Индикатор не может быть пустым.")
        return
    await state.update_data(metric=metric)
    await state.set_state(IndicatorForm.start_date)
    await message.answer(
        f"✅ Индикатор: <b>{metric}</b>\n\n"
        "📅 <b>Введи дату начала</b> (ДД.ММ.ГГГГ)\n"
        "Или «Пропустить» для последних ~250 дней:",
        parse_mode="HTML",
        reply_markup=get_skip_keyboard()
    )

@router.message(IndicatorForm.start_date)
async def process_start_date(message: Message, state: FSMContext):
    start_date = None if message.text == "Пропустить" else message.text.strip()
    if start_date:
        try:
            datetime.strptime(start_date, "%d.%m.%Y")
        except ValueError:
            await message.answer("❌ Неверный формат! Используй ДД.ММ.ГГГГ")
            return
    await state.update_data(start_date=start_date)
    await state.set_state(IndicatorForm.end_date)
    await message.answer(
        f"✅ Дата начала: <b>{start_date or 'последние ~250 дней'}</b>\n\n"
        "📅 <b>Введи дату окончания</b> (ДД.ММ.ГГГГ)\n"
        "Или «Пропустить» для текущей даты:",
        parse_mode="HTML",
        reply_markup=get_skip_keyboard()
    )

@router.message(IndicatorForm.end_date)
async def process_end_date(message: Message, state: FSMContext):
    end_date = None if message.text == "Пропустить" else message.text.strip()
    if end_date:
        try:
            datetime.strptime(end_date, "%d.%m.%Y")
        except ValueError:
            await message.answer("❌ Неверный формат! Используй ДД.ММ.ГГГГ")
            return

    await state.update_data(end_date=end_date)
    data = await state.get_data()
    ticker = data['ticker']
    metric = data['metric']
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    processing_msg = await message.answer(
        "⏳ <b>Загружаю данные и рассчитываю индикатор...</b>",
        parse_mode="HTML"
    )

    try:
        price_df = await get_historical_data(ticker, start_date, end_date)
        if price_df is None or price_df.empty:
            await processing_msg.delete()
            await message.answer(
                f"❌ Не удалось получить данные для <b>{ticker}</b>\nПроверьте тикер и период.",
                parse_mode="HTML", reply_markup=get_main_menu()
            )
            await state.clear()
            return

        indicator_df = await calculate_indicator(ticker, metric, start_date, end_date)
        if indicator_df is None or indicator_df.empty:
            await processing_msg.delete()
            await message.answer(
                f"❌ Не удалось рассчитать индикатор <b>{metric}</b>\nПроверьте правильность написания.",
                parse_mode="HTML", reply_markup=get_main_menu()
            )
            await state.clear()
            return

        combined_df = price_df.copy()
        for col in indicator_df.columns:
            combined_df[col] = indicator_df[col]

        await state.update_data(
            price_df=price_df,
            indicator_df=indicator_df,
            combined_df=combined_df,
            indicator_name=metric.split()[0]
        )

        period_text = f"{start_date or 'последние ~250 дней'} — {end_date or 'сегодня'}"
        change_pct = (price_df['Close'].iloc[-1] / price_df['Close'].iloc[0] - 1) * 100

        result_text = (
            f"📊 <b>Результат для {ticker}</b>\n\n"
            f"📈 Индикатор: <b>{metric}</b>\n"
            f"📅 Период: {period_text}\n"
            f"📋 Записей: {len(price_df)}\n\n"
            f"💰 Текущая цена: <b>{price_df['Close'].iloc[-1]:.2f} ₽</b>\n"
            f"📉 Изменение: <b>{change_pct:+.2f}%</b>\n"
        )

        if not indicator_df.empty:
            last = indicator_df.iloc[-1].to_dict()
            result_text += "\n📈 <b>Последние значения:</b>\n"
            for key, value in last.items():
                if not pd.isna(value):
                    result_text += f"• {key}: {value:.3f}\n"

        result_text += "\n📤 <b>Выбери формат вывода:</b>"

        await processing_msg.delete()
        await message.answer(result_text, parse_mode="HTML", reply_markup=get_export_keyboard())

    except Exception as e:
        logger.error(f"Ошибка при расчете: {e}")
        await processing_msg.delete()
        await message.answer(
            f"❌ <b>Ошибка:</b>\n{str(e)[:200]}",
            parse_mode="HTML", reply_markup=get_main_menu()
        )
        await state.clear()

# =========================
# ХЭНДЛЕРЫ — ЭКСПОРТ
# =========================

@router.callback_query(F.data == "export_html")
async def export_html(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    ticker = data.get('ticker', 'data')
    price_df = data.get('price_df')
    indicator_df = data.get('indicator_df')
    indicator_name = data.get('indicator_name', 'indicator')

    if price_df is None:
        await callback.answer("❌ Данные не найдены", show_alert=True)
        return

    await callback.message.edit_text("⏳ Генерирую HTML-графики...")

    try:
        price_html = build_price_html(ticker, price_df)
        await callback.message.answer_document(
            document=BufferedInputFile(
                price_html.encode('utf-8'),
                filename=f"{ticker}_price_{datetime.now().strftime('%Y%m%d')}.html"
            ),
            caption=f"🌐 <b>График цен {ticker}</b> — откройте в браузере",
            parse_mode="HTML"
        )

        ind_html = build_indicator_html(ticker, price_df, indicator_df, indicator_name)
        await callback.message.answer_document(
            document=BufferedInputFile(
                ind_html.encode('utf-8'),
                filename=f"{ticker}_{indicator_name}_{datetime.now().strftime('%Y%m%d')}.html"
            ),
            caption=f"🌐 <b>{indicator_name.upper()} — {ticker}</b> — откройте в браузере",
            parse_mode="HTML"
        )

        await callback.answer("✅ HTML-файлы отправлены")
    except Exception as e:
        logger.error(f"Ошибка генерации HTML: {e}")
        await callback.answer("❌ Ошибка при генерации", show_alert=True)

    await callback.message.answer("✅ <b>Готово!</b>", parse_mode="HTML", reply_markup=get_single_menu_keyboard())
    await state.clear()


@router.callback_query(F.data == "export_png")
async def export_png(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    ticker         = data.get('ticker', 'data')
    price_df       = data.get('price_df')
    indicator_df   = data.get('indicator_df')
    indicator_name = data.get('indicator_name', 'indicator')

    if price_df is None:
        await callback.answer("❌ Данные не найдены", show_alert=True)
        return

    await callback.message.edit_text("⏳ Генерирую графики...")

    try:
        # График цен
        price_chart = create_price_chart(ticker, price_df)
        await callback.message.answer_photo(
            photo=BufferedInputFile(price_chart.read(), filename=f"{ticker}_price.png"),
            caption=f"📸 <b>График цен {ticker}</b>\n"
                    f"{price_df.index[0].strftime('%d.%m.%Y')} — "
                    f"{price_df.index[-1].strftime('%d.%m.%Y')}",
            parse_mode="HTML"
        )

        # График индикатора
        ind_chart = create_indicator_chart(ticker, price_df, indicator_df, indicator_name)
        await callback.message.answer_photo(
            photo=BufferedInputFile(ind_chart.read(), filename=f"{ticker}_{indicator_name}.png"),
            caption=f"📸 <b>{indicator_name.upper()} — {ticker}</b>",
            parse_mode="HTML"
        )

        await callback.answer("✅ Графики отправлены")

    except Exception as e:
        logger.error(f"Ошибка PNG: {e}")
        await callback.answer("❌ Ошибка при генерации", show_alert=True)

    await callback.message.answer("✅ <b>Готово!</b>", parse_mode="HTML", reply_markup=get_single_menu_keyboard())
    await state.clear()

@router.callback_query(F.data == "export_csv")
async def export_csv(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    combined_df = data.get('combined_df')
    ticker = data.get('ticker', 'data')
    indicator_name = data.get('indicator_name', 'indicator')

    if combined_df is None:
        await callback.answer("❌ Данные не найдены", show_alert=True)
        return

    await callback.message.edit_text("⏳ Создаю CSV файл...")

    try:
        csv_buffer = io.BytesIO()
        combined_df.to_csv(csv_buffer, encoding='utf-8-sig', index=True)
        csv_buffer.seek(0)
        await callback.message.answer_document(
            document=BufferedInputFile(
                csv_buffer.read(),
                filename=f"{ticker}_{indicator_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            ),
            caption=f"📥 Данные {ticker} — {indicator_name}"
        )
        await callback.answer("✅ CSV отправлен")
    except Exception as e:
        logger.error(f"Ошибка CSV: {e}")
        await callback.answer("❌ Ошибка при создании файла", show_alert=True)

    await callback.message.answer("✅ <b>Готово!</b>", parse_mode="HTML", reply_markup=get_single_menu_keyboard())
    await state.clear()


@router.callback_query(F.data == "export_excel")
async def export_excel(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    combined_df = data.get('combined_df')
    ticker = data.get('ticker', 'data')
    indicator_name = data.get('indicator_name', 'indicator')

    if combined_df is None:
        await callback.answer("❌ Данные не найдены", show_alert=True)
        return

    await callback.message.edit_text("⏳ Создаю Excel файл...")

    try:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            combined_df.to_excel(writer, sheet_name='Data', index=True)
            stats_df = pd.DataFrame({
                'Метрика': ['Тикер', 'Индикатор', 'Период', 'Записей', 'Начальная цена', 'Конечная цена', 'Изменение %'],
                'Значение': [
                    ticker, indicator_name,
                    f"{combined_df.index[0].strftime('%d.%m.%Y')} - {combined_df.index[-1].strftime('%d.%m.%Y')}",
                    len(combined_df),
                    f"{combined_df['Close'].iloc[0]:.2f}",
                    f"{combined_df['Close'].iloc[-1]:.2f}",
                    f"{((combined_df['Close'].iloc[-1] / combined_df['Close'].iloc[0] - 1) * 100):.2f}%"
                ]
            })
            stats_df.to_excel(writer, sheet_name='Статистика', index=False)
        excel_buffer.seek(0)
        await callback.message.answer_document(
            document=BufferedInputFile(
                excel_buffer.read(),
                filename=f"{ticker}_{indicator_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            ),
            caption=f"📊 Данные {ticker} — {indicator_name}"
        )
        await callback.answer("✅ Excel отправлен")
    except Exception as e:
        logger.error(f"Ошибка Excel: {e}")
        await callback.answer("❌ Ошибка при создании файла", show_alert=True)

    await callback.message.answer("✅ <b>Готово!</b>", parse_mode="HTML", reply_markup=get_single_menu_keyboard())
    await state.clear()


@router.callback_query(F.data == "export_all")
async def export_all(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    ticker = data.get('ticker', 'data')
    price_df = data.get('price_df')
    indicator_df = data.get('indicator_df')
    combined_df = data.get('combined_df')
    indicator_name = data.get('indicator_name', 'indicator')

    if price_df is None:
        await callback.answer("❌ Данные не найдены", show_alert=True)
        return

    await callback.message.edit_text("⏳ Генерирую всё...")

    try:
        # HTML цены
        price_html = build_price_html(ticker, price_df)
        await callback.message.answer_document(
            document=BufferedInputFile(
                price_html.encode('utf-8'),
                filename=f"{ticker}_price_{datetime.now().strftime('%Y%m%d')}.html"
            ),
            caption=f"🌐 <b>График цен {ticker}</b>",
            parse_mode="HTML"
        )

        # HTML индикатора
        ind_html = build_indicator_html(ticker, price_df, indicator_df, indicator_name)
        await callback.message.answer_document(
            document=BufferedInputFile(
                ind_html.encode('utf-8'),
                filename=f"{ticker}_{indicator_name}_{datetime.now().strftime('%Y%m%d')}.html"
            ),
            caption=f"🌐 <b>{indicator_name.upper()} — {ticker}</b>",
            parse_mode="HTML"
        )

        # CSV
        if combined_df is not None:
            csv_buffer = io.BytesIO()
            combined_df.to_csv(csv_buffer, encoding='utf-8-sig', index=True)
            csv_buffer.seek(0)
            await callback.message.answer_document(
                document=BufferedInputFile(
                    csv_buffer.read(),
                    filename=f"{ticker}_{indicator_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                ),
                caption=f"📥 CSV-данные {ticker}"
            )

        await callback.answer("✅ Всё отправлено")
    except Exception as e:
        logger.error(f"Ошибка export_all: {e}")
        await callback.answer("❌ Ошибка при генерации", show_alert=True)

    await callback.message.answer("✅ <b>Готово!</b>", parse_mode="HTML", reply_markup=get_single_menu_keyboard())
    await state.clear()

# =========================
# ХЭНДЛЕРЫ — АЛЕРТЫ
# =========================

@router.message(Command("alert"))
@router.message(F.text == "🔔 Создать алерт")
async def cmd_alert(message: Message, state: FSMContext):
    user_id = message.from_user.id
    count = await count_user_alerts(user_id)
    if count >= MAX_ALERTS_PER_USER:
        await message.answer(
            f"❌ <b>Достигнут лимит алертов ({MAX_ALERTS_PER_USER})!</b>\n\nУдалите ненужные через 📋 Мои алерты",
            parse_mode="HTML", reply_markup=get_main_menu()
        )
        return
    await state.set_state(AlertForm.ticker)
    await message.answer(
        f"🔔 <b>Создание алерта</b>\n\n"
        f"📊 Активных алертов: <b>{count}/{MAX_ALERTS_PER_USER}</b>\n\n"
        f"📈 Введи тикер для отслеживания:\nПример: SBER, GAZP, YNDX",
        parse_mode="HTML",
        reply_markup=ReplyKeyboardRemove()
    )

@router.message(AlertForm.ticker)
async def process_alert_ticker(message: Message, state: FSMContext):
    ticker = message.text.strip().upper()
    if len(ticker) < 2 or len(ticker) > 10:
        await message.answer("❌ Некорректный тикер. Попробуй еще раз:")
        return

    await state.update_data(ticker=ticker)
    await state.set_state(AlertForm.type)

    current_price = await get_now_price(ticker)
    price_info = f"\n💰 Текущая цена: <b>{current_price:.2f} ₽</b>" if current_price else ""

    await message.answer(
        f"✅ Тикер: <b>{ticker}</b>{price_info}\n\n"
        "🎯 <b>Выбери тип алерта:</b>",
        parse_mode="HTML",
        reply_markup=get_alert_type_keyboard()
    )


@router.message(AlertForm.type)
async def process_alert_type(message: Message, state: FSMContext):
    text = message.text.strip()

    if text == "Отмена":
        await state.clear()
        await message.answer("❌ Отменено", reply_markup=get_main_menu())
        return

    if text == "💰 На цену":
        await state.update_data(alert_type="price")
        await state.set_state(AlertForm.condition)
        await message.answer(
            "🎯 <b>Выбери условие:</b>",
            parse_mode="HTML",
            reply_markup=get_condition_keyboard()
        )

    elif text == "📊 На индикатор":
        await state.update_data(alert_type="indicator")
        await state.set_state(AlertForm.indicator)
        await message.answer(
            "📈 <b>Введи индикатор с параметрами:</b>\n\n"
            "• rsi 14\n"
            "• sma 50\n"
            "• macd 12 26 9\n"
            "• bbands 20 2",
            parse_mode="HTML",
            reply_markup=get_single_menu_keyboard()
        )
    else:
        await message.answer("❌ Выбери из клавиатуры", reply_markup=get_alert_type_keyboard())



@router.message(AlertForm.indicator)
async def process_alert_indicator(message: Message, state: FSMContext):
    logger.info(f"process_alert_indicator вызван: {message.text}")
    metric = message.text.strip().lower()
    ind_name = metric.split()[0]

    if not hasattr(ta, ind_name):
        await message.answer(
            f"❌ Индикатор <b>{ind_name}</b> не найден\nПопробуй ещё раз:",
            parse_mode="HTML"
        )
        return

    data   = await state.get_data()
    ticker = data['ticker']

    processing_msg = await message.answer("⏳ Считаю индикатор...")

    result_df = await get_indicator_cached(ticker, metric)
    if result_df is None or result_df.empty:
        await processing_msg.edit_text(
            f"❌ Не удалось рассчитать <b>{metric}</b>",
            parse_mode="HTML"
        )
        return

    await state.update_data(indicator=metric)
    cols = list(result_df.columns)
    logger.info(f"Колонки индикатора: {cols}")


    # Если одна колонка — сразу переходим к условию
    if len(cols) == 1:
        await state.update_data(ind_col=cols[0])
        await state.set_state(AlertForm.condition)

        last_val = result_df[cols[0]].iloc[-1]
        await processing_msg.edit_text(
            f"✅ <b>{metric.upper()}</b> · последнее значение: <b>{last_val:.3f}</b>\n\n"
            "🎯 <b>Выбери условие:</b>",
            parse_mode="HTML"
        )
        await message.answer("👇", reply_markup=get_condition_keyboard())

    else:
        # Несколько колонок (MACD, BBands) — спрашиваем какую отслеживать
        await state.update_data(ind_col=None, ind_cols=cols)
        await state.set_state(AlertForm.ind_col)

        last_vals = result_df.iloc[-1].to_dict()
        text = f"✅ <b>{metric.upper()}</b> имеет несколько линий:\n\n"
        for col, val in last_vals.items():
            if not pd.isna(val):
                text += f"• {col}: <b>{val:.3f}</b>\n"
        text += "\n<b>Какую линию отслеживать?</b>"

        keyboard = ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton(text=col)] for col in cols] + [[KeyboardButton(text="Отмена")]],
            resize_keyboard=True
        )
        await processing_msg.edit_text(text, parse_mode="HTML")
        await message.answer("👇", reply_markup=keyboard)


@router.message(AlertForm.ind_col)
async def process_alert_ind_col(message: Message, state: FSMContext):
    text = message.text.strip()

    if text == "Отмена":
        await state.clear()
        await message.answer("❌ Отменено", reply_markup=get_main_menu())
        return

    data = await state.get_data()
    cols = data.get('ind_cols', [])

    if text not in cols:
        await message.answer(f"❌ Выбери из списка: {', '.join(cols)}")
        return

    await state.update_data(ind_col=text)
    await state.set_state(AlertForm.condition)

    result_df = await get_indicator_cached(data['ticker'], data['indicator'])
    last_val  = result_df[text].iloc[-1] if result_df is not None else None
    val_str   = f" · последнее значение: <b>{last_val:.3f}</b>" if last_val else ""

    await message.answer(
        f"✅ Линия: <b>{text}</b>{val_str}\n\n"
        "🎯 <b>Выбери условие:</b>",
        parse_mode="HTML",
        reply_markup=get_condition_keyboard()
    )

@router.message(AlertForm.condition)
async def process_alert_condition(message: Message, state: FSMContext):
    condition_text = message.text.strip().lower()

    if condition_text == "отмена":
        await state.clear()
        await message.answer("❌ Создание алерта отменено", reply_markup=get_main_menu())
        return

    if condition_text not in ["выше", "ниже"]:
        await message.answer("❌ Выбери из клавиатуры: Выше или Ниже")
        return

    data       = await state.get_data()
    ticker     = data['ticker']
    alert_type = data.get('alert_type', 'price')

    await state.update_data(condition=condition_text)
    await state.set_state(AlertForm.value)

    if alert_type == "price":
        current_price = await get_now_price(ticker)
        price_info = f"\n💰 Текущая цена: <b>{current_price:.2f} ₽</b>" if current_price else ""
        await message.answer(
            f"✅ Тикер: <b>{ticker}</b>\n"
            f"✅ Условие: <b>{condition_text}</b>{price_info}\n\n"
            "💎 <b>Введи целевую цену:</b>\nПример: 250.5",
            parse_mode="HTML",
            reply_markup=ReplyKeyboardRemove()
        )
    else:
        indicator = data.get('indicator', '')
        ind_col   = data.get('ind_col', '')
        await message.answer(
            f"✅ Тикер: <b>{ticker}</b>\n"
            f"✅ Условие: <b>{condition_text}</b>\n"
            f"📊 {indicator.upper()} · {ind_col}\n\n"
            "💎 <b>Введи целевое значение индикатора:</b>\nПример: 70",
            parse_mode="HTML",
            reply_markup=ReplyKeyboardRemove()
        )

@router.message(AlertForm.value)
async def process_alert_value(message: Message, state: FSMContext):
    try:
        value = float(message.text.strip().replace(',', '.'))
    except ValueError:
        await message.answer("❌ Введи корректное число (например: 250.5)")
        return

    data = await state.get_data()
    if 'ticker' not in data or 'condition' not in data:
        await message.answer("❌ Ошибка. Начни заново", reply_markup=get_main_menu())
        await state.clear()
        return

    user_id    = message.from_user.id
    ticker     = data['ticker']
    condition  = data['condition']
    alert_type = data.get('alert_type', 'price')
    indicator  = data.get('indicator', 'price')
    ind_col    = data.get('ind_col', '')

    # indicator field в БД: "price" или "rsi_14__rsi_14" или "macd__MACD_12_26_9"
    ind_key = "price" if alert_type == "price" else f"{indicator}__{ind_col}"

    success = await add_alert(user_id, ticker, ind_key, condition, value)
    if not success:
        await message.answer(
            f"❌ <b>Достигнут лимит ({MAX_ALERTS_PER_USER})!</b>",
            parse_mode="HTML", reply_markup=get_main_menu()
        )
        await state.clear()
        return

    await state.clear()

    if alert_type == "price":
        current = await get_now_price(ticker)
        price_text = f"💰 Текущая цена: <b>{current:.2f} ₽</b>\n" if current else ""
        await message.answer(
            f"✅ <b>Алерт на цену создан!</b>\n\n"
            f"📈 Тикер: <b>{ticker}</b>\n"
            f"{price_text}"
            f"🎯 Условие: <b>{condition} {value:.2f} ₽</b>",
            parse_mode="HTML", reply_markup=get_main_menu()
        )
    else:
        await message.answer(
            f"✅ <b>Алерт на индикатор создан!</b>\n\n"
            f"📈 Тикер: <b>{ticker}</b>\n"
            f"📊 Индикатор: <b>{indicator.upper()}</b>\n"
            f"📌 Линия: <b>{ind_col}</b>\n"
            f"🎯 Условие: <b>{condition} {value:.3f}</b>\n\n"
            f"🔔 Проверка раз в день после 19:00",
            parse_mode="HTML", reply_markup=get_main_menu()
        )


@router.message(Command("myalerts"))
@router.message(F.text == "📋 Мои алерты")
async def cmd_my_alerts(message: Message):
    user_id = message.from_user.id
    user_alerts = await get_user_alerts(user_id)
    enabled = await get_user_alerts_enabled(user_id)

    if not user_alerts:
        await message.answer(
            "📋 <b>У тебя нет активных алертов</b>\n\nСоздай через 🔔 Создать алерт",
            parse_mode="HTML", reply_markup=get_main_menu()
        )
        return

    status = "🔔 <b>Включены</b>" if enabled else "🔕 <b>Выключены</b>"
    text = f"📋 <b>Активные алерты:</b>\n{status}\n\n"

    for i, alert in enumerate(user_alerts, 1):
        ticker = alert['ticker']
        condition = alert['condition']
        value = alert['value']
        current_price = await get_now_price(ticker)
        if current_price:
            diff = current_price - value
            trend = "📈" if diff > 0 else "📉"
            price_str = f"{current_price:.2f} ₽ ({trend} {abs(diff):.2f})"
        else:
            price_str = "н/д"
        text += (
            f"{i}. <b>{ticker}</b>\n"
            f"   💰 Текущая: {price_str}\n"
            f"   🎯 Условие: {condition} {value:.2f} ₽\n\n"
        )

    text += f"📊 Всего: <b>{len(user_alerts)}/{MAX_ALERTS_PER_USER}</b>"
    await message.answer(text, parse_mode="HTML", reply_markup=get_alert_management_keyboard(enabled))

@router.callback_query(F.data == "toggle_alerts")
async def toggle_alerts(callback: CallbackQuery):
    user_id = callback.from_user.id
    current_enabled = await get_user_alerts_enabled(user_id)
    new_enabled = not current_enabled
    await set_user_alerts_enabled(user_id, new_enabled)
    status = "включены" if new_enabled else "выключены"
    emoji = "🔔" if new_enabled else "🔕"
    await callback.answer(f"{emoji} Уведомления {status}", show_alert=True)
    await cmd_my_alerts(callback.message)

@router.callback_query(F.data == "show_delete_alerts")
async def show_delete_alerts(callback: CallbackQuery):
    user_id = callback.from_user.id
    user_alerts = await get_user_alerts(user_id)
    if not user_alerts:
        await callback.answer("У тебя нет алертов", show_alert=True)
        return
    keyboard = []
    for alert in user_alerts:
        keyboard.append([InlineKeyboardButton(
            text=f"🗑 {alert['ticker']} ({alert['condition']} {alert['value']:.2f} ₽)",
            callback_data=f"del_alert_{alert['id']}"
        )])
    keyboard.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_alerts")])
    await callback.message.edit_text(
        "🗑 <b>Выбери алерт для удаления:</b>",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard)
    )
    await callback.answer()

@router.callback_query(F.data.startswith("del_alert_"))
async def delete_alert(callback: CallbackQuery):
    user_id = callback.from_user.id
    alert_id = int(callback.data.split("_")[-1])
    success = await delete_alert_by_id(alert_id, user_id)
    if success:
        await callback.answer("✅ Алерт удалён", show_alert=True)
        remaining = await get_user_alerts(user_id)
        if remaining:
            await show_delete_alerts(callback)
        else:
            await callback.message.edit_text("✅ <b>Все алерты удалены</b>", parse_mode="HTML")
    else:
        await callback.answer("❌ Алерт не найден", show_alert=True)
    await callback.message.answer("Выберите действие:", reply_markup=get_single_menu_keyboard())
    await callback.answer()

@router.callback_query(F.data == "back_to_alerts")
async def back_to_alerts(callback: CallbackQuery):
    await cmd_my_alerts(callback.message)
    await callback.answer()

# =========================
# ФОНОВАЯ ЗАДАЧА ПРОВЕРКИ АЛЕРТОВ
# =========================


async def check_alerts_task():
    logger.info("🚀 Запуск фоновой задачи проверки алертов")
    await asyncio.sleep(10)

    while True:
        try:
            all_alerts = await get_all_active_alerts()
            if not all_alerts:
                await asyncio.sleep(ALERT_CHECK_INTERVAL)
                continue

            logger.info(f"🔍 Проверка {len(all_alerts)} алертов")

            alerts_by_ticker = defaultdict(list)
            for alert in all_alerts:
                alerts_by_ticker[alert['ticker']].append(alert)

            triggered_count = 0

            for ticker, ticker_alerts in alerts_by_ticker.items():
                for alert in ticker_alerts:
                    try:
                        alert_id  = alert['id']
                        user_id   = alert['user_id']
                        condition = alert['condition']
                        target    = alert['value']
                        indicator = alert['indicator']

                        if not await get_user_alerts_enabled(user_id):
                            continue

                        # ── Ценовой алерт — проверяем всегда ─────────
                        if indicator == "price":
                            current_val = await get_now_price(ticker)
                            if current_val is None:
                                continue

                            triggered = (
                                (condition == "выше" and current_val > target) or
                                (condition == "ниже" and current_val < target)
                            )

                            if not triggered:
                                continue

                            caption = (
                                f"🔔 <b>Алерт сработал!</b>\n\n"
                                f"📈 Тикер: <b>{ticker}</b>\n"
                                f"💰 Цена: <b>{current_val:.2f} ₽</b>\n"
                                f"🎯 Условие: {condition} <b>{target:.2f} ₽</b>\n"
                                f"⏰ {datetime.now().strftime('%H:%M:%S')}\n\n"
                                f"Откройте HTML-файл в браузере"
                            )

                        # ── Индикаторный алерт — только после 19:00 ──
                        else:
                            if datetime.now().hour < 19:
                                continue

                            if "__" not in indicator:
                                logger.error(f"Неверный формат индикатора: {indicator}")
                                continue

                            metric, ind_col = indicator.split("__", 1)

                            vals = await get_latest_indicator_value(ticker, metric)
                            logger.info(f"Индикатор {metric} для {ticker}: {vals}")

                            if vals is None:
                                logger.warning(f"Нет значений для {ticker} {metric}")
                                continue

                            # Ищем нужную колонку без учёта регистра
                            matched_key = None
                            for k in vals:
                                if k.lower() == ind_col.lower() or k == ind_col:
                                    matched_key = k
                                    break

                            if matched_key is None:
                                logger.warning(
                                    f"Колонка '{ind_col}' не найдена в {list(vals.keys())}"
                                )
                                continue

                            current_val = vals[matched_key]
                            triggered = (
                                (condition == "выше" and current_val > target) or
                                (condition == "ниже" and current_val < target)
                            )

                            if not triggered:
                                continue

                            caption = (
                                f"🔔 <b>Алерт сработал!</b>\n\n"
                                f"📈 Тикер: <b>{ticker}</b>\n"
                                f"📊 Индикатор: <b>{metric.upper()}</b>\n"
                                f"📌 Линия: <b>{matched_key}</b>\n"
                                f"📉 Значение: <b>{current_val:.3f}</b>\n"
                                f"🎯 Условие: {condition} <b>{target:.3f}</b>\n"
                                f"⏰ {datetime.now().strftime('%H:%M:%S')}\n\n"
                                f"Откройте HTML-файл в браузере"
                            )

                        # ── Отправка (общая для обоих типов) ─────────
                        price_df = await get_candles_cached(ticker)
                        if price_df is not None and len(price_df) > 10:
                            alert_html = build_alert_html(
                                ticker, price_df.tail(50),
                                current_val, target, condition
                            )
                            try:
                                await bot.send_document(
                                    chat_id=user_id,
                                    document=BufferedInputFile(
                                        alert_html.encode('utf-8'),
                                        filename=f"alert_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                                    ),
                                    caption=caption,
                                    parse_mode="HTML",
                                    reply_markup=get_main_menu()
                                )
                            except Exception as e:
                                logger.error(f"Ошибка отправки HTML: {e}")
                                await bot.send_message(
                                    user_id, caption,
                                    parse_mode="HTML",
                                    reply_markup=get_main_menu()
                                )
                        else:
                            await bot.send_message(
                                user_id, caption,
                                parse_mode="HTML",
                                reply_markup=get_main_menu()
                            )

                        await deactivate_alert(alert_id)
                        triggered_count += 1
                        logger.info(f"✅ Алерт {alert_id} сработал ({indicator})")

                    except Exception as e:
                        logger.error(f"❌ Ошибка алерта {alert.get('id')}: {e}")
                        continue

            logger.info(f"✅ Проверка завершена. Сработало: {triggered_count}/{len(all_alerts)}")
            await asyncio.sleep(ALERT_CHECK_INTERVAL)

        except Exception as e:
            logger.error(f"❌ Критическая ошибка в check_alerts_task: {e}")
            await asyncio.sleep(ALERT_CHECK_INTERVAL)


@router.message(F.text.regexp(r'(?i)^[A-Za-z]{2,10}\s+\w+'))
async def quick_calc(message: Message, state: FSMContext):
    """Быстрый расчёт одной строкой: SBER sma 50 [01.01.2024] [01.01.2025]"""
    text = message.text.strip()

    # Парсим: TICKER INDICATOR [PARAMS...] [DATE1] [DATE2]
    date_pattern = r'\d{2}\.\d{2}\.\d{4}'
    dates = re.findall(date_pattern, text)
    text_no_dates = re.sub(date_pattern, '', text).strip()

    parts = text_no_dates.split()
    if len(parts) < 2:
        return  # не наш формат — пропускаем

    ticker    = parts[0].upper()
    metric    = ' '.join(parts[1:]).strip()
    start_date = dates[0] if len(dates) > 0 else None
    end_date   = dates[1] if len(dates) > 1 else None

    # Проверяем что второе слово похоже на индикатор
    known = {'sma','ema','wma','rsi','macd','bbands','atr','adx','stoch','cci','willr','vwap','obv'}
    if parts[1].lower() not in known and not hasattr(ta, parts[1].lower()):
        return  # не наш формат

    processing_msg = await message.answer(
        f"⏳ <b>{ticker} · {metric}</b>\n"
        f"📅 {start_date or 'последние ~250 дней'} — {end_date or 'сегодня'}",
        parse_mode="HTML"
    )

    try:
        price_df = await get_historical_data(ticker, start_date, end_date)
        if price_df is None or price_df.empty:
            await processing_msg.edit_text(
                f"❌ Не удалось получить данные для <b>{ticker}</b>",
                parse_mode="HTML"
            )
            return

        indicator_df = await calculate_indicator(ticker, metric, start_date, end_date)
        if indicator_df is None or indicator_df.empty:
            await processing_msg.edit_text(
                f"❌ Не удалось рассчитать <b>{metric}</b>",
                parse_mode="HTML"
            )
            return

        combined_df = price_df.copy()
        for col in indicator_df.columns:
            combined_df[col] = indicator_df[col]

        change_pct = (price_df['Close'].iloc[-1] / price_df['Close'].iloc[0] - 1) * 100
        result_text = (
            f"📊 <b>{ticker} · {metric.upper()}</b>\n"
            f"📅 {start_date or 'последние ~250 дней'} — {end_date or 'сегодня'}\n"
            f"📋 Записей: {len(price_df)}\n\n"
            f"💰 Цена: <b>{price_df['Close'].iloc[-1]:.2f} ₽</b>  "
            f"<b>{change_pct:+.2f}%</b>\n"
        )

        last = indicator_df.iloc[-1].to_dict()
        result_text += "\n📈 <b>Последние значения:</b>\n"
        for key, val in last.items():
            if not pd.isna(val):
                result_text += f"• {key}: {val:.3f}\n"

        result_text += "\n📤 <b>Выбери формат:</b>"

        await state.update_data(
            ticker=ticker,
            metric=metric,
            price_df=price_df,
            indicator_df=indicator_df,
            combined_df=combined_df,
            indicator_name=metric.split()[0]
        )

        await processing_msg.edit_text(result_text, parse_mode="HTML", reply_markup=get_export_keyboard())

    except Exception as e:
        logger.error(f"Ошибка quick_calc: {e}")
        await processing_msg.edit_text(f"❌ Ошибка: {str(e)[:200]}", parse_mode="HTML")

# =========================
# ЗАПУСК
# =========================

async def on_startup():
    await init_database()
    await init_cache_tables()
    logger.info("✅ База данных инициализирована")
    logger.info("🚀 Бот запущен!")

async def on_shutdown():
    logger.info("🛑 Бот остановлен")


async def main():
    # Создавать здесь:
    global bot, dp
    
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    
    from aiogram.client.session.aiohttp import AiohttpSession
    session = AiohttpSession(timeout=60)
    bot = Bot(token=BOT_TOKEN, session=session)
    
    # Регистрируем роутер с хэндлерами
    dp.include_router(router)
    
    await on_startup()
    
    alert_task = asyncio.create_task(check_alerts_task())
    
    while True:
        try:
            logger.info("⏳ Запуск polling...")
            await dp.start_polling(
                bot,
                allowed_updates=dp.resolve_used_update_types(),
                polling_timeout=30
            )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Ошибка polling: {e}, перезапуск через 5 секунд...")
            await asyncio.sleep(5)
    
    alert_task.cancel()
    try:
        await alert_task
    except asyncio.CancelledError:
        pass
    
    await bot.session.close()
    logger.info("🛑 Бот остановлен")




if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Остановлено пользователем")


