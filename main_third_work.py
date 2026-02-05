import asyncio
import logging
import io
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
import concurrent.futures

import pandas as pd
import pandas_ta as ta
import apimoex 
import requests
import aiohttp
import aiosqlite
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    Message, CallbackQuery, ReplyKeyboardMarkup, KeyboardButton, 
    ReplyKeyboardRemove, InlineKeyboardMarkup, InlineKeyboardButton,
    BufferedInputFile
)
from dotenv import load_dotenv
import os

load_dotenv()  

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

# Настройки графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
user_locks = defaultdict(asyncio.Lock)
price_cache: Dict[str, Tuple[float, float]] = {}

# БАЗА ДАННЫХ

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
            "SELECT alerts_enabled FROM users WHERE user_id = ?",
            (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return bool(row[0]) if row else True

async def set_user_alerts_enabled(user_id: int, enabled: bool):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO users (user_id, alerts_enabled) 
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET alerts_enabled = ?
            """,
            (user_id, int(enabled), int(enabled))
        )
        await db.commit()

async def count_user_alerts(user_id: int) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM alerts WHERE user_id = ? AND is_active = 1",
            (user_id,)
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
                """
                INSERT INTO alerts (user_id, ticker, indicator, condition, value, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, ticker, indicator, condition, value, created_at)
            )
            await db.commit()
        return True

async def get_user_alerts(user_id: int) -> List[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT id, ticker, indicator, condition, value, created_at 
            FROM alerts 
            WHERE user_id = ? AND is_active = 1
            ORDER BY created_at DESC
            """,
            (user_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

async def delete_alert_by_id(alert_id: int, user_id: int) -> bool:
    async with user_locks[user_id]:
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute(
                "DELETE FROM alerts WHERE id = ? AND user_id = ?",
                (alert_id, user_id)
            )
            await db.commit()
            return cursor.rowcount > 0

async def get_all_active_alerts() -> List[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT id, user_id, ticker, indicator, condition, value 
            FROM alerts 
            WHERE is_active = 1
            """
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

async def deactivate_alert(alert_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
        await db.commit()

# ФУНКЦИИ ДЛЯ РАБОТЫ С MOEX

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
            async with session.get(
                url,
                params={'iss.only': 'marketdata'},
                timeout=5,
                ssl=False
            ) as response:
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
        
        if not start:
            start_date = (today - timedelta(days=350)).isoformat()
        else:
            start_date = parse_date(start)
            if not start_date:
                return None
        
        if not end:
            end_date = today.isoformat()
        else:
            end_date = parse_date(end)
            if not end_date:
                return None
        
        # Используем ThreadPoolExecutor для запуска синхронного кода в отдельном потоке
        loop = asyncio.get_event_loop()
        
        # Функция для запуска в отдельном потоке
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
        
        # Запускаем в отдельном потоке
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
    # Получаем данные асинхронно
    df = await get_historical_data(ticker, start, end)
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
        
        # Запускаем расчет индикатора в отдельном потоке
        loop = asyncio.get_event_loop()
        
        def calculate():
            func = getattr(ta, ind_name)
            params = [int(p) for p in parts[1:] if p.isdigit()]
            
            if ind_name == "macd":
                fast = params[0] if len(params) > 0 else 12
                slow = params[1] if len(params) > 1 else 26
                signal = params[2] if len(params) > 2 else 9
                result = func(close=df["Close"], fast=fast, slow=slow, signal=signal)
                return pd.DataFrame(result)
            
            elif ind_name in ["bbands", "bollinger"]:
                length = params[0] if params else 20
                std = params[1] if len(params) > 1 else 2
                result = func(close=df["Close"], length=length, std=std)
                if isinstance(result, pd.DataFrame):
                    return result
                else:
                    return pd.DataFrame({ind_name: result})
            
            elif ind_name == "stoch":
                k = params[0] if len(params) > 0 else 14
                d = params[1] if len(params) > 1 else 3
                smooth_k = params[2] if len(params) > 2 else 3
                result = func(high=df["High"], low=df["Low"], close=df["Close"], 
                             k=k, d=d, smooth_k=smooth_k)
                if isinstance(result, pd.DataFrame):
                    return result
                else:
                    return pd.DataFrame({ind_name: result})
            
            elif ind_name in ["atr", "adx", "rsi", "cci", "willr"]:
                length = params[0] if params else 14
                if ind_name in ["atr", "adx"]:
                    result = func(high=df["High"], low=df["Low"], close=df["Close"], length=length)
                else:
                    result = func(close=df["Close"], length=length)
                
                if isinstance(result, pd.Series):
                    return pd.DataFrame({ind_name: result})
                else:
                    return pd.DataFrame(result)
            
            elif ind_name in ["sma", "ema", "wma"]:
                length = params[0] if params else 20
                result = func(close=df["Close"], length=length)
                return pd.DataFrame({ind_name: result})
            
            else:
                if params:
                    result = func(close=df["Close"], length=params[0])
                else:
                    result = func(close=df["Close"])
                
                if isinstance(result, pd.Series):
                    return pd.DataFrame({ind_name: result})
                else:
                    return pd.DataFrame(result)
        
        # Запускаем расчет в отдельном потоке
        result = await loop.run_in_executor(None, calculate)
        return result
        
    except Exception as e:
        logger.error(f"Ошибка расчета индикатора {ind_name}: {e}")
        return None

# ФУНКЦИИ ДЛЯ СОЗДАНИЯ ГРАФИКОВ


def create_price_chart(ticker: str, df: pd.DataFrame) -> io.BytesIO:
    """Создать график цен"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # График цен
    ax1.plot(df.index, df['Close'], label='Цена закрытия', linewidth=2, color='blue', alpha=0.7)
    ax1.set_title(f'{ticker} - График цен', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Цена, руб.', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Форматирование дат
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # График объемов
    colors = ['green' if close >= open_ else 'red' 
              for close, open_ in zip(df['Close'], df['Open'])]
    
    ax2.bar(df.index, df['Volume'], color=colors, alpha=0.6)
    ax2.set_ylabel('Объем', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Форматирование дат
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
    """Создать график индикатора"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # График цен с индикатором
    axes[0].plot(price_df.index, price_df['Close'], label='Цена закрытия', linewidth=2, color='blue', alpha=0.7)
    
    if indicator_name.lower() in ['sma', 'ema', 'wma']:
        if not indicator_df.empty:
            ind_col = indicator_df.columns[0]
            axes[0].plot(price_df.index, indicator_df[ind_col], 
                        label=f'{indicator_name.upper()}', linewidth=2, color='orange', alpha=0.8)
    
    axes[0].set_title(f'{ticker} - {indicator_name.upper()}', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Цена, руб.', fontsize=12)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # График индикатора
    axes[1].set_ylabel('Значение индикатора', fontsize=12)
    
    if not indicator_df.empty:
        for col in indicator_df.columns:
            axes[1].plot(price_df.index, indicator_df[col], label=col, linewidth=2)
        
        # Особые случаи
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
    
    # Форматирование дат
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


# FSM 


class IndicatorForm(StatesGroup):
    ticker = State()
    metric = State()
    start_date = State()
    end_date = State()

class AlertForm(StatesGroup):
    ticker = State()
    condition = State()
    value = State()

# КЛАВИАТУРЫ

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
        keyboard=[
            [KeyboardButton(text='Главное меню')]
        ],
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
                InlineKeyboardButton(text="📥 CSV", callback_data="export_csv"),
                InlineKeyboardButton(text="📊 Excel", callback_data="export_excel")
            ],
            [
                InlineKeyboardButton(text="🖼️ Графики", callback_data="export_charts"),
                InlineKeyboardButton(text="📈 Все данные", callback_data="export_all")
            ]
        ]
    )

def get_alert_management_keyboard(enabled: bool) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔕 Выключить" if enabled else "🔔 Включить",
                    callback_data="toggle_alerts"
                )
            ],
            [InlineKeyboardButton(text="🗑 Удалить алерт", callback_data="show_delete_alerts")]
        ]
    )

# HANDLER ОСНОВНЫЕ КОМАНДЫ


@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "👋 <b>Привет! Я бот для анализа акций Мосбиржи</b>\n\n"
        "📊 <b>Что я умею:</b>\n"
        "• Рассчитывать технические индикаторы\n"
        "• Создавать графики цен и индикаторов\n"
        "• Отслеживать цены и отправлять уведомления\n"
        "• Экспортировать данные в CSV/Excel\n\n"
        "🔹 Для начала выбери действие из меню ниже:",
        parse_mode="HTML",
        reply_markup=get_main_menu()
    )

@dp.message(Command("help"))
@dp.message(F.text == "❓ Справка")
async def cmd_help(message: Message):
    help_text = (
        "📚 <b>Доступные функции:</b>\n\n"
        
        "📊 <b>Расчёт индикаторов:</b>\n"
        "• SMA, EMA, WMA - скользящие средние\n"
        "• RSI, MACD, Bollinger Bands\n"
        "• ATR, ADX, Stochastic и другие\n\n"
        
        "🖼️ <b>Графики:</b>\n"
        "• Автоматическое создание графиков\n"
        "• Графики цен с объемами\n"
        "• Графики индикаторов\n\n"
        
        "🔔 <b>Алерты:</b>\n"
        "• Создавай уведомления на достижение цены\n"
        "• Бот проверяет условия каждые 30 секунд\n"
        "• При срабатывании отправляется график\n"
        f"• Лимит: {MAX_ALERTS_PER_USER} алертов\n\n"
        
        "📋 <b>Примеры команд:</b>\n"
        "• sma 50 - простая скользящая средняя\n"
        "• macd 12 26 9\n"
        "• rsi 14\n"
        "• bbands 20 2\n\n"
        
        "⚙️ <b>Управление:</b>\n"
        "/start - главное меню\n"
        "/help - эта справка\n"
        "/cancel - отмена действия"
    )
    await message.answer(help_text, parse_mode="HTML", reply_markup=get_main_menu())

@dp.message(Command("cancel"))
@dp.message(F.text.casefold() == "отмена")
async def cmd_cancel(message: Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        await message.answer("❌ Нечего отменять", reply_markup=get_main_menu())
        return
    
    await state.clear()
    await message.answer("✅ Действие отменено", reply_markup=get_main_menu())

@dp.message(F.text == 'Главное меню')
async def handle_main_menu_button(message: Message, state: FSMContext):
    current_state = await state.get_state()
    
    if current_state is not None:
        await state.clear()
        await message.answer("✅ Возвращаюсь в главное меню", reply_markup=get_main_menu())
    else:
        await message.answer("📋 <b>Главное меню</b>\n\nВыберите действие:", 
                           parse_mode="HTML", reply_markup=get_main_menu())

# HANDLER РАСЧЁТ ИНДИКАТОРОВ

@dp.message(Command("calc"))
@dp.message(F.text == "📊 Рассчитать индикатор")
async def cmd_calc(message: Message, state: FSMContext):
    """Начало расчёта индикатора"""
    await state.set_state(IndicatorForm.ticker)
    await message.answer(
        "📊 <b>Введи тикер акции</b>\n\n"
        "Примеры: SBER, GAZP, YNDX, VTBR, ROSN\n"
        "Тикер должен быть в верхнем регистре",
        parse_mode="HTML",
        reply_markup=ReplyKeyboardRemove()
    )

@dp.message(IndicatorForm.ticker)
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
        "📋 <b>Примеры:</b>\n"
        "• sma 50 - скользящая средняя\n"
        "• ema 20 - экспоненциальная средняя\n"
        "• macd 12 26 9\n"
        "• rsi 14\n"
        "• bbands 20 2 - полосы Боллинджера\n"
        "• atr 14 - Average True Range",
        parse_mode="HTML",
        reply_markup=get_single_menu_keyboard()
    )

@dp.message(IndicatorForm.metric)
async def process_metric(message: Message, state: FSMContext):
    metric = message.text.strip()
    
    if not metric:
        await message.answer("❌ Индикатор не может быть пустым. Введи индикатор:")
        return
    
    await state.update_data(metric=metric)
    await state.set_state(IndicatorForm.start_date)
    
    await message.answer(
        f"✅ Индикатор: <b>{metric}</b>\n\n"
        "📅 <b>Введи дату начала</b> (формат: ДД.ММ.ГГГГ)\n"
        "Или нажми «Пропустить» для загрузки последних ~250 дней:",
        parse_mode="HTML",
        reply_markup=get_skip_keyboard()
    )

@dp.message(IndicatorForm.start_date)
async def process_start_date(message: Message, state: FSMContext):
    start_date = None if message.text == "Пропустить" else message.text.strip()
    
    if start_date:
        try:
            datetime.strptime(start_date, "%d.%m.%Y")
        except ValueError:
            await message.answer("❌ Неверный формат даты! Используй ДД.ММ.ГГГГ (например: 01.01.2024)")
            return
    
    await state.update_data(start_date=start_date)
    await state.set_state(IndicatorForm.end_date)
    
    await message.answer(
        f"✅ Дата начала: <b>{start_date or 'последние ~250 дней'}</b>\n\n"
        "📅 <b>Введи дату окончания</b> (формат: ДД.ММ.ГГГГ)\n"
        "Или нажми «Пропустить» для текущей даты:",
        parse_mode="HTML",
        reply_markup=get_skip_keyboard()
    )

@dp.message(IndicatorForm.end_date)
async def process_end_date(message: Message, state: FSMContext):
    """Обработка даты окончания и расчёт"""
    end_date = None if message.text == "Пропустить" else message.text.strip()
    
    if end_date:
        try:
            datetime.strptime(end_date, "%d.%m.%Y")
        except ValueError:
            await message.answer("❌ Неверный формат даты! Используй ДД.ММ.ГГГГ (например: 31.12.2024)")
            return
    
    await state.update_data(end_date=end_date)
    
    data = await state.get_data()
    ticker = data['ticker']
    metric = data['metric']
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    processing_msg = await message.answer("⏳ <b>Рассчитываю индикатор и создаю графики...</b>\n"
                                        "Это может занять несколько секунд", 
                                        parse_mode="HTML")
    
    try:
        # Получаем исторические данные АСИНХРОННО
        price_df = await get_historical_data(ticker, start_date, end_date)
        if price_df is None or price_df.empty:
            await processing_msg.delete()
            await message.answer(
                f"❌ Не удалось получить данные для {ticker}\n"
                f"Проверьте тикер и период",
                reply_markup=get_main_menu()
            )
            await state.clear()
            return
        
        # Рассчитываем индикатор АСИНХРОННО
        indicator_df = await calculate_indicator(ticker, metric, start_date, end_date)
        if indicator_df is None or indicator_df.empty:
            await processing_msg.delete()
            await message.answer(
                f"❌ Не удалось рассчитать индикатор {metric}\n"
                f"Проверьте правильность написания",
                reply_markup=get_main_menu()
            )
            await state.clear()
            return
        
        # Объединяем данные
        combined_df = price_df.copy()
        for col in indicator_df.columns:
            combined_df[col] = indicator_df[col]
        
        # Сохраняем данные для экспорта
        await state.update_data(
            price_df=price_df,
            indicator_df=indicator_df,
            combined_df=combined_df,
            indicator_name=metric.split()[0]
        )
        
        # Формируем информацию о запросе
        period_text = f"{start_date or 'последние ~250 дней'} — {end_date or 'сегодня'}"
        rows_count = len(price_df)
        
        result_text = (
            f"📊 <b>Результат для {ticker}</b>\n\n"
            f"📈 Индикатор: {metric}\n"
            f"📅 Период: {period_text}\n"
            f"📋 Записей: {rows_count}\n\n"
        )
        
        # Добавляем статистику
        result_text += f"💰 <b>Статистика цен:</b>\n"
        result_text += f"• Текущая цена: {price_df['Close'].iloc[-1]:.2f} ₽\n"
        result_text += f"• Изменение за период: {(price_df['Close'].iloc[-1] - price_df['Close'].iloc[0]):.2f} ₽ "
        change_pct = ((price_df['Close'].iloc[-1] / price_df['Close'].iloc[0] - 1) * 100)
        result_text += f"({change_pct:+.2f}%)\n"
        
        if not indicator_df.empty:
            last_indicator = indicator_df.iloc[-1].to_dict()
            result_text += f"\n📈 <b>Последние значения индикатора:</b>\n"
            for key, value in last_indicator.items():
                if not pd.isna(value):
                    result_text += f"• {key}: {value:.4f}\n"
        
        result_text += "\n📥 <b>Выбери формат экспорта:</b>"
        
        await processing_msg.delete()
        await message.answer(result_text, parse_mode="HTML", reply_markup=get_export_keyboard())
        
    except Exception as e:
        logger.error(f"Ошибка при расчете индикатора: {e}")
        await processing_msg.delete()
        await message.answer(
            f"❌ <b>Произошла ошибка:</b>\n{str(e)[:200]}",
            parse_mode="HTML",
            reply_markup=get_main_menu()
        )
        await state.clear()


# HANDLER ЭКСПОРТ ДАННЫХ

@dp.callback_query(F.data == "export_csv")
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
        
        file = BufferedInputFile(
            file=csv_buffer.read(),
            filename=f"{ticker}_{indicator_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        await callback.message.answer_document(
            document=file,
            caption=f"📥 Данные {ticker} с индикатором {indicator_name}"
        )
        
        await callback.answer("✅ CSV файл отправлен")
        
    except Exception as e:
        logger.error(f"Ошибка при создании CSV: {e}")
        await callback.answer("❌ Ошибка при создании файла", show_alert=True)
    
    await callback.message.answer("✅ <b>CSV-файл отправлен!</b>", parse_mode="HTML", reply_markup=get_single_menu_keyboard())
    await state.clear()

@dp.callback_query(F.data == "export_excel")
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
                    ticker,
                    indicator_name,
                    f"{combined_df.index[0].strftime('%d.%m.%Y')} - {combined_df.index[-1].strftime('%d.%m.%Y')}",
                    len(combined_df),
                    f"{combined_df['Close'].iloc[0]:.2f}",
                    f"{combined_df['Close'].iloc[-1]:.2f}",
                    f"{((combined_df['Close'].iloc[-1] / combined_df['Close'].iloc[0] - 1) * 100):.2f}%"
                ]
            })
            stats_df.to_excel(writer, sheet_name='Статистика', index=False)
        
        excel_buffer.seek(0)
        
        file = BufferedInputFile(
            file=excel_buffer.read(),
            filename=f"{ticker}_{indicator_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        
        await callback.message.answer_document(
            document=file,
            caption=f"📊 Данные {ticker} с индикатором {indicator_name}"
        )
        
        await callback.answer("✅ Excel файл отправлен")
        
    except Exception as e:
        logger.error(f"Ошибка при создании Excel: {e}")
        await callback.answer("❌ Ошибка при создании файла", show_alert=True)
    
    await callback.message.answer("✅ <b>Excel-файл отправлен!</b>", parse_mode="HTML", reply_markup=get_single_menu_keyboard())
    await state.clear()

@dp.callback_query(F.data == "export_charts")
async def export_charts(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    
    ticker = data.get('ticker', 'data')
    price_df = data.get('price_df')
    indicator_df = data.get('indicator_df')
    indicator_name = data.get('indicator_name', 'indicator')
    
    if price_df is None:
        await callback.answer("❌ Данные не найдены", show_alert=True)
        return
    
    await callback.message.edit_text("⏳ Создаю графики...")
    
    try:
        # Создаем график цен
        price_chart = create_price_chart(ticker, price_df)
        price_chart.seek(0)
        
        await callback.message.answer_photo(
            photo=BufferedInputFile(
                price_chart.read(),
                filename=f"{ticker}_price_chart.png"
            ),
            caption=f"📊 <b>График цен {ticker}</b>\n"
                   f"Период: {price_df.index[0].strftime('%d.%m.%Y')} - "
                   f"{price_df.index[-1].strftime('%d.%m.%Y')}",
            parse_mode="HTML"
        )
        
        # Создаем график индикатора
        indicator_chart = create_indicator_chart(ticker, price_df, indicator_df, indicator_name)
        indicator_chart.seek(0)
        
        await callback.message.answer_photo(
            photo=BufferedInputFile(
                indicator_chart.read(),
                filename=f"{ticker}_{indicator_name}_chart.png"
            ),
            caption=f"📈 <b>График индикатора {indicator_name.upper()}</b>\n"
                   f"Тикер: {ticker}",
            parse_mode="HTML"
        )
        
        await callback.answer("✅ Графики отправлены")
        
    except Exception as e:
        logger.error(f"Ошибка при создании графиков: {e}")
        await callback.answer("❌ Ошибка при создании графиков", show_alert=True)
    
    await callback.message.answer("✅ <b>Графики отправлены!</b>", parse_mode="HTML", reply_markup=get_single_menu_keyboard())
    await state.clear()

@dp.callback_query(F.data == "export_all")
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
    
    await callback.message.edit_text("⏳ Создаю файлы и графики...")
    
    try:
        # CSV файл
        csv_buffer = io.BytesIO()
        if combined_df is not None:
            combined_df.to_csv(csv_buffer, encoding='utf-8-sig', index=True)
            csv_buffer.seek(0)
            
            file = BufferedInputFile(
                file=csv_buffer.read(),
                filename=f"{ticker}_{indicator_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            await callback.message.answer_document(
                document=file,
                caption=f"📥 Данные {ticker} с индикатором {indicator_name}"
            )
        
        # График цен
        price_chart = create_price_chart(ticker, price_df)
        price_chart.seek(0)
        
        await callback.message.answer_photo(
            photo=BufferedInputFile(
                price_chart.read(),
                filename=f"{ticker}_price_chart.png"
            ),
            caption=f"📊 <b>График цен {ticker}</b>",
            parse_mode="HTML"
        )
        
        # График индикатора
        indicator_chart = create_indicator_chart(ticker, price_df, indicator_df, indicator_name)
        indicator_chart.seek(0)
        
        await callback.message.answer_photo(
            photo=BufferedInputFile(
                indicator_chart.read(),
                filename=f"{ticker}_{indicator_name}_chart.png"
            ),
            caption=f"📈 <b>График индикатора {indicator_name.upper()}</b>",
            parse_mode="HTML"
        )
        
        await callback.answer("✅ Все данные отправлены")
        
    except Exception as e:
        logger.error(f"Ошибка при экспорте всех данных: {e}")
        await callback.answer("❌ Ошибка при создании файлов", show_alert=True)
    
    await callback.message.answer("✅ <b>Все данные отправлены!</b>", parse_mode="HTML", reply_markup=get_single_menu_keyboard())
    await state.clear()

# HANDLERS - АЛЕРТЫ

@dp.message(Command("alert"))
@dp.message(F.text == "🔔 Создать алерт")
async def cmd_alert(message: Message, state: FSMContext):
    user_id = message.from_user.id
    count = await count_user_alerts(user_id)
    
    if count >= MAX_ALERTS_PER_USER:
        await message.answer(
            f"❌ <b>Достигнут лимит алертов ({MAX_ALERTS_PER_USER})!</b>\n\n"
            f"Удалите ненужные алерты через 📋 Мои алерты",
            parse_mode="HTML",
            reply_markup=get_main_menu()
        )
        return
    
    await state.set_state(AlertForm.ticker)
    await message.answer(
        f"🔔 <b>Создание алерта</b>\n\n"
        f"📊 Активных алертов: <b>{count}/{MAX_ALERTS_PER_USER}</b>\n\n"
        f"📈 <b>Введи тикер</b> для отслеживания:\n"
        f"Пример: SBER, GAZP, YNDX, VTBR",
        parse_mode="HTML",
        reply_markup=ReplyKeyboardRemove()
    )

@dp.message(AlertForm.ticker)
async def process_alert_ticker(message: Message, state: FSMContext):
    ticker = message.text.strip().upper()
    
    if len(ticker) < 2 or len(ticker) > 10:
        await message.answer("❌ Некорректный тикер. Попробуй еще раз:")
        return
    
    await state.update_data(ticker=ticker)
    await state.set_state(AlertForm.condition)
    
    await message.answer(
        f"✅ Тикер: <b>{ticker}</b>\n\n"
        "🎯 <b>Выбери условие:</b>\n"
        "• Выше - когда цена поднимется выше указанного значения\n"
        "• Ниже - когда цена опустится ниже указанного значения",
        parse_mode="HTML",
        reply_markup=get_condition_keyboard()
    )

@dp.message(AlertForm.condition)
async def process_alert_condition(message: Message, state: FSMContext):
    condition_text = message.text.strip().lower()
    
    if condition_text == "отмена":
        await state.clear()
        await message.answer("❌ Создание алерта отменено", reply_markup=get_main_menu())
        return
    
    if condition_text not in ["выше", "ниже"]:
        await message.answer("❌ Выбери условие из клавиатуры: Выше или Ниже")
        return
    
    condition = condition_text
    data = await state.get_data()
    ticker = data['ticker']
    
    current_price = await get_now_price(ticker)
    price_info = ""
    if current_price:
        price_info = f"\n💰 Текущая цена: <b>{current_price:.2f} ₽</b>"
    
    await state.update_data(condition=condition)
    await state.set_state(AlertForm.value)
    
    await message.answer(
        f"✅ Тикер: <b>{ticker}</b>\n"
        f"✅ Условие: <b>{condition}</b>{price_info}\n\n"
        "💎 <b>Введи целевое значение цены:</b>\n"
        "Пример: 250.5 или 250,5",
        parse_mode="HTML",
        reply_markup=ReplyKeyboardRemove()
    )

@dp.message(AlertForm.value)
async def process_alert_value(message: Message, state: FSMContext):
    try:
        value = float(message.text.strip().replace(',', '.'))
    except ValueError:
        await message.answer("❌ Введи корректное число (например: 250.5 или 250,5)")
        return

    data = await state.get_data()
    
    if 'ticker' not in data or 'condition' not in data:
        await message.answer(
            "❌ Произошла ошибка при создании алерта\n"
            "Попробуй начать заново через 🔔 Создать алерт",
            reply_markup=get_main_menu()
        )
        await state.clear()
        return

    user_id = message.from_user.id
    ticker = data['ticker']
    condition = data['condition']
    
    success = await add_alert(user_id, ticker, "price", condition, value)
    
    if not success:
        await message.answer(
            f"❌ <b>Достигнут лимит алертов ({MAX_ALERTS_PER_USER})!</b>\n\n"
            "Удалите ненужные через 📋 Мои алерты",
            parse_mode="HTML",
            reply_markup=get_main_menu()
        )
        await state.clear()
        return

    await state.clear()
    
    current_price = await get_now_price(ticker)
    price_text = (
        f"💰 Текущая цена: <b>{current_price:.2f} ₽</b>\n"
        if current_price is not None
        else ""
    )
    
    await message.answer(
        f"✅ <b>Алерт успешно создан!</b>\n\n"
        f"📈 Тикер: <b>{ticker}</b>\n"
        f"{price_text}"
        f"🎯 Условие: <b>{condition.capitalize()} {value:.2f} ₽</b>\n\n"
        f"🔔 Бот проверяет условия каждые <b>30 секунд</b>\n"
        f"Вы получите уведомление сразу при срабатывании!",
        parse_mode="HTML",
        reply_markup=get_main_menu()
    )

@dp.message(Command("myalerts"))
@dp.message(F.text == "📋 Мои алерты")
async def cmd_my_alerts(message: Message):
    user_id = message.from_user.id
    user_alerts = await get_user_alerts(user_id)
    enabled = await get_user_alerts_enabled(user_id)
    
    if not user_alerts:
        await message.answer(
            "📋 <b>У тебя нет активных алертов</b>\n\n"
            "Создай новый через 🔔 Создать алерт",
            parse_mode="HTML",
            reply_markup=get_main_menu()
        )
        return
    
    status = "🔔 <b>Включены</b>" if enabled else "🔕 <b>Выключены</b>"
    text = f"📋 <b>Твои активные алерты:</b>\n{status}\n\n"
    
    for i, alert in enumerate(user_alerts, 1):
        ticker = alert['ticker']
        condition = alert['condition']
        value = alert['value']
        
        current_price = await get_now_price(ticker)
        if current_price:
            difference = current_price - value
            trend = "📈" if difference > 0 else "📉"
            price_str = f"{current_price:.2f} ₽ ({trend} {abs(difference):.2f})"
        else:
            price_str = "н/д"
        
        text += (
            f"{i}. <b>{ticker}</b>\n"
            f"   💰 Текущая: {price_str}\n"
            f"   🎯 Условие: {condition} {value:.2f} ₽\n\n"
        )
    
    text += f"📊 Всего: <b>{len(user_alerts)}/{MAX_ALERTS_PER_USER}</b> алертов"
    
    await message.answer(
        text,
        parse_mode="HTML",
        reply_markup=get_alert_management_keyboard(enabled)
    )

@dp.callback_query(F.data == "toggle_alerts")
async def toggle_alerts(callback: CallbackQuery):
    user_id = callback.from_user.id
    current_enabled = await get_user_alerts_enabled(user_id)
    new_enabled = not current_enabled
    
    await set_user_alerts_enabled(user_id, new_enabled)
    
    status = "включены" if new_enabled else "выключены"
    emoji = "🔔" if new_enabled else "🔕"
    
    await callback.answer(f"{emoji} Уведомления {status}", show_alert=True)
    await cmd_my_alerts(callback.message)

@dp.callback_query(F.data == "show_delete_alerts")
async def show_delete_alerts(callback: CallbackQuery):
    user_id = callback.from_user.id
    user_alerts = await get_user_alerts(user_id)
    
    if not user_alerts:
        await callback.answer("У тебя нет алертов", show_alert=True)
        return
    
    keyboard = []
    for alert in user_alerts:
        alert_id = alert['id']
        ticker = alert['ticker']
        condition = alert['condition']
        value = alert['value']
        
        keyboard.append([InlineKeyboardButton(
            text=f"🗑 {ticker} ({condition} {value:.2f} ₽)",
            callback_data=f"del_alert_{alert_id}"
        )])
    
    keyboard.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_alerts")])
    
    await callback.message.edit_text(
        "🗑 <b>Выбери алерт для удаления:</b>",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard)
    )
    await callback.answer()

@dp.callback_query(F.data.startswith("del_alert_"))
async def delete_alert(callback: CallbackQuery):
    user_id = callback.from_user.id
    alert_id = int(callback.data.split("_")[-1])
    
    success = await delete_alert_by_id(alert_id, user_id)
    
    if success:
        await callback.answer("✅ Алерт удалён", show_alert=True)
        remaining_alerts = await get_user_alerts(user_id)
        
        if remaining_alerts:
            await show_delete_alerts(callback)
        else:
            await callback.message.edit_text(
                "✅ <b>Все алерты удалены</b>",
                parse_mode="HTML"
            )
    else:
        await callback.answer("❌ Алерт не найден", show_alert=True)
    
    await callback.message.answer("Выберите действие:", reply_markup=get_single_menu_keyboard())
    await callback.answer()

@dp.callback_query(F.data == "back_to_alerts")
async def back_to_alerts(callback: CallbackQuery):
    await cmd_my_alerts(callback.message)
    await callback.answer()

# ФОНТОВАЯ ЗАДАЧА ПРОВЕРКИ АЛЕРТОВ

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
                ticker = alert['ticker']
                alerts_by_ticker[ticker].append(alert)
            
            triggered_count = 0
            for ticker, ticker_alerts in alerts_by_ticker.items():
                current_price = await get_now_price(ticker)
                if current_price is None:
                    continue
                
                for alert in ticker_alerts:
                    try:
                        alert_id = alert['id']
                        user_id = alert['user_id']
                        condition = alert['condition']
                        target_value = alert['value']
                        
                        enabled = await get_user_alerts_enabled(user_id)
                        if not enabled:
                            continue
                        
                        triggered = False
                        if condition == "выше" and current_price > target_value:
                            triggered = True
                        elif condition == "ниже" and current_price < target_value:
                            triggered = True
                        
                        if triggered:
                            price_df = await get_historical_data(ticker, None, None)
                            if price_df is not None and len(price_df) > 10:
                                try:
                                    chart_buf = create_price_chart(ticker, price_df.tail(50))
                                    chart_buf.seek(0)
                                    
                                    await bot.send_photo(
                                        chat_id=user_id,
                                        photo=BufferedInputFile(
                                            chart_buf.read(),
                                            filename=f"alert_{ticker}.png"
                                        ),
                                        caption=(
                                            f"🔔 <b>!Алерт сработал!</b>\n\n"
                                            f"📈 Тикер: <b>{ticker}</b>\n"
                                            f"💰 Цена: <b>{current_price:.2f} ₽</b>\n"
                                            f"🎯 Условие: {condition} <b>{target_value:.2f} ₽</b>\n"
                                            f"⏰ Время: {datetime.now().strftime('%H:%M:%S')}\n\n"
                                            f"✅ Условие выполнено!"
                                        ),
                                        parse_mode="HTML"
                                    )
                                except Exception as e:
                                    logger.error(f"Ошибка при создании графика алерта: {e}")
                                    await bot.send_message(
                                        user_id,
                                        f"🔔 <b>!Алерт сработал!</b>\n\n"
                                        f"📈 Тикер: <b>{ticker}</b>\n"
                                        f"💰 Цена: <b>{current_price:.2f} ₽</b>\n"
                                        f"🎯 Условие: {condition} <b>{target_value:.2f} ₽</b>\n"
                                        f"⏰ Время: {datetime.now().strftime('%H:%M:%S')}",
                                        parse_mode="HTML"
                                    )

                                    await bot.send_message(
                                                user_id,
                                                "📋 <b>Вернуться в меню:</b>",
                                                parse_mode="HTML",
                                                reply_markup=get_main_menu()
                                            )
                            else:
                                await bot.send_message(
                                    user_id,
                                    f"🔔 <b>!Алерт сработал!</b>\n\n"
                                    f"📈 Тикер: <b>{ticker}</b>\n"
                                    f"💰 Цена: <b>{current_price:.2f} ₽</b>\n"
                                    f"🎯 Условие: {condition} <b>{target_value:.2f} ₽</b>\n"
                                    f"⏰ Время: {datetime.now().strftime('%H:%M:%S')}",
                                    parse_mode="HTML"
                                )
                            
                            await deactivate_alert(alert_id)
                            triggered_count += 1
                            logger.info(f"✅ Алерт {alert_id} сработал")
                    except Exception as e:
                        logger.error(f"❌ Ошибка обработки алерта: {e}")
                        continue
            
            logger.info(f"✅ Проверка завершена. Сработало: {triggered_count}/{len(all_alerts)}")
            await asyncio.sleep(ALERT_CHECK_INTERVAL)
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка в check_alerts_task: {e}")
            await asyncio.sleep(ALERT_CHECK_INTERVAL)


# ЗАПУСК БОТА


async def on_startup():
    await init_database()
    logger.info("✅ База данных инициализирована")
    logger.info("🚀 Бот запущен!")

async def on_shutdown():
    logger.info("🛑 Бот остановлен")

async def main():
    await on_startup()
    
    try:
        # Запускаем фоновую задачу
        alert_task = asyncio.create_task(check_alerts_task())
        
        logger.info("⏳ Запуск polling...")
        await dp.start_polling(bot)
        
    except KeyboardInterrupt:
        logger.info("Получен сигнал KeyboardInterrupt")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        # Отменяем фоновые задачи
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        await on_shutdown()

if __name__ == "__main__":
    asyncio.run(main())