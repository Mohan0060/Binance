# breakout_scannerv4.py
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import os
from dateutil.relativedelta import relativedelta
import pytz
from zoneinfo import ZoneInfo
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import numpy as np
import itertools

# ==============================================
# PROXY SUPPORT
# ==============================================
class ProxyManager:
    """Manages proxy rotation with round-robin support for multiple proxies."""
    def __init__(self):
        self.proxies = []
        self.mode = "none"  # "none", "single", "multiple"
        self._cycle = None
        self._current_index = 0

    def set_proxies(self, mode, proxy_list):
        """Configure proxies. mode: 'none', 'single', 'multiple'."""
        self.mode = mode
        self.proxies = [p.strip() for p in proxy_list if p.strip()] if proxy_list else []
        if self.mode == "multiple" and self.proxies:
            self._cycle = itertools.cycle(self.proxies)
            self._current_index = 0
        else:
            self._cycle = None
            self._current_index = 0

    def get_proxy(self):
        """Return proxy dict for requests, or None if no proxy configured."""
        if self.mode == "none" or not self.proxies:
            return None
        if self.mode == "single":
            return {"http": self.proxies[0], "https": self.proxies[0]}
        if self.mode == "multiple" and self._cycle:
            proxy = next(self._cycle)
            self._current_index = (self._current_index + 1) % len(self.proxies)
            return {"http": proxy, "https": proxy}
        return None

    def get_proxy_display(self):
        """Human-readable proxy status."""
        if self.mode == "none":
            return "No proxy"
        if self.mode == "single":
            return f"Single proxy: {self.proxies[0] if self.proxies else 'none'}"
        if self.mode == "multiple":
            return f"Multiple proxies ({len(self.proxies)}): round-robin"
        return "No proxy"


# Global proxy manager instance
_proxy_manager = ProxyManager()


def api_get(url, params=None, headers=None, timeout=10, _retries=4, **kwargs):
    """
    Wrapper around requests.get with proxy support, optional Binance API key
    (env BINANCE_API_KEY), and retry with exponential backoff on 429/418/5xx.
    Klines endpoints transparently fall back to data.binance.vision static
    files when Binance rate-limits/bans this IP, avoiding hard failures.
    """
    import random as _random
    proxies = _proxy_manager.get_proxy()
    headers = dict(headers or {})
    api_key = os.environ.get("BINANCE_API_KEY", "")
    if api_key and "binance.com" in url and "X-MBX-APIKEY" not in headers:
        headers["X-MBX-APIKEY"] = api_key
    last_exc = None
    response = None
    for attempt in range(_retries):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
                proxies=proxies,
                **kwargs,
            )
            if response.status_code in (429, 418) or response.status_code >= 500:
                ra = response.headers.get("Retry-After")
                try:
                    wait = float(ra) if ra else float(2 ** attempt)
                except (TypeError, ValueError):
                    wait = float(2 ** attempt)
                time.sleep(wait + _random.random() * 0.5)
                if _proxy_manager.mode == "multiple" and _proxy_manager.proxies:
                    proxies = _proxy_manager.get_proxy()
                continue
            rows = _maybe_vision_klines(url, params, response)
            if rows:
                return _StaticKlinesResponse(rows)
            return response
        except requests.RequestException as e:
            last_exc = e
            if _proxy_manager.mode == "multiple" and _proxy_manager.proxies:
                proxies = _proxy_manager.get_proxy()
            time.sleep((2 ** attempt) * 0.5 + _random.random() * 0.3)
    rows = _maybe_vision_klines(url, params, response)
    if rows:
        return _StaticKlinesResponse(rows)
    if last_exc is not None:
        raise last_exc
    return response


class _StaticKlinesResponse:
    """Minimal response-like wrapper for kline rows from data.binance.vision."""

    def __init__(self, rows):
        self._rows = rows
        self.status_code = 200
        self.headers = {}
        self.content = b""
        self.text = ""

    def json(self):
        return self._rows

    def raise_for_status(self):
        return None


def _maybe_vision_klines(url, params, response):
    """If this is a Binance klines call that hit an IP restriction
    (429/418, or 200/400/403 with an error body), fetch the same klines from
    data.binance.vision static ZIP files instead (no rate limits)."""
    try:
        if "binance.com" not in url or "/klines" not in url:
            return None
        if response is None:
            return None
        if response.status_code in (418, 429):
            pass  # rate limited / banned -> use static fallback
        elif response.status_code == 200:
            body = response.json()
            if not (isinstance(body, dict) and "code" in body):
                return None
        elif response.status_code in (400, 403):
            body = response.json()
            if not (isinstance(body, dict) and body.get("code") in (-1003, -1021)):
                return None
        else:
            return None
        p = params or {}
        rows = _vision_klines_fallback(
            p.get("symbol"), p.get("interval"), p.get("startTime"), p.get("endTime"), p.get("limit", 500)
        )
        return rows or None
    except Exception:
        return None


# Set page configuration
st.set_page_config(
    page_title="Binance Scanner Pro - Breakout + Volume Analysis",
    page_icon="🚀",
    layout="wide"
)

# Binance API configuration
BINANCE_API_URL = "https://fapi.binance.com"
BINANCE_SPOT_API_URL = "https://api.binance.com"
CMC_API_URL = "https://pro-api.coinmarketcap.com"

# Timezone configuration
IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

# Timeframe mapping to Binance intervals
TIMEFRAME_MAP = {
    "1H": "1h",
    "4H": "4h",
    "Daily": "1d",
    "Weekly": "1w",
    "Monthly": "1M"
}

# ==============================================
# CACHED API FUNCTIONS
# ==============================================

@st.cache_data(ttl=3600)
def get_perpetual_symbols():
    """Fetch all USDT perpetual trading pairs from Binance"""
    try:
        response = api_get(f"{BINANCE_API_URL}/fapi/v1/exchangeInfo", timeout=10)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or 'symbols' not in data:
            err_msg = data.get('msg', data) if isinstance(data, dict) else str(data)
            st.error(f"Binance exchangeInfo returned no symbol list: {err_msg}")
            return []
        perpetual_symbols = []
        for symbol_info in data['symbols']:
            if (symbol_info.get('contractType') == 'PERPETUAL' and
                symbol_info.get('quoteAsset') == 'USDT' and
                symbol_info.get('status') == 'TRADING'):
                perpetual_symbols.append(symbol_info['symbol'])
        return sorted(perpetual_symbols)
    except requests.RequestException as e:
        st.error(f"Error fetching symbols (network): {e}")
        return []
    except Exception as e:
        st.error(f"Error fetching symbols: {e}")
        return []

@st.cache_data(ttl=3600)
def get_spot_symbols():
    """Fetch all USDT spot trading pairs from Binance"""
    try:
        response = api_get(f"{BINANCE_SPOT_API_URL}/api/v3/exchangeInfo", timeout=10)
        data = response.json()

        spot_symbols = []
        for symbol_info in data['symbols']:
            if (symbol_info['quoteAsset'] == 'USDT' and
                symbol_info['status'] == 'TRADING' and
                symbol_info['isSpotTradingAllowed']):
                spot_symbols.append(symbol_info['symbol'])

        return set(spot_symbols)
    except Exception as e:
        return set()

@st.cache_data(ttl=300)
def get_top_volume_symbols(limit=500):
    """Get symbols sorted by 24h USDT volume"""
    try:
        response = api_get(f"{BINANCE_API_URL}/fapi/v1/ticker/24hr", timeout=10)
        data = response.json()

        volume_data = []
        for ticker in data:
            symbol = ticker['symbol']
            if symbol.endswith('USDT'):
                quote_volume = float(ticker.get('quoteVolume', 0))
                volume_data.append({
                    'symbol': symbol,
                    'volume': quote_volume,
                    'priceChangePercent': float(ticker.get('priceChangePercent', 0))
                })

        volume_data.sort(key=lambda x: x['volume'], reverse=True)
        return volume_data
    except Exception as e:
        return []

@st.cache_data(ttl=300)
def get_top_volume_symbols_list(limit=500):
    """Get just the symbols sorted by volume"""
    volume_data = get_top_volume_symbols(limit)
    return [item['symbol'] for item in volume_data[:limit]]

@st.cache_data(ttl=60)
def get_open_interest(symbol):
    """Fetch current open interest for a symbol"""
    try:
        params = {'symbol': symbol}
        response = api_get(f"{BINANCE_API_URL}/fapi/v1/openInterest", params=params, timeout=5)
        data = response.json()
        return float(data.get('openInterest', 0))
    except Exception as e:
        return None

# ==============================================
# TIMEFRAME-BASED OPEN INTEREST CALCULATION
# ==============================================

@st.cache_data(ttl=60)
def get_historical_open_interest(symbol, timeframe="Daily", limit=2):
    """
    Fetch historical open interest for comparison BASED ON SELECTED TIMEFRAME
    timeframe: Monthly, Weekly, Daily, 4H, 1H
    """
    try:
        # Map your timeframe to Binance OI period
        period_map = {
            "Monthly": "1M",   # Compare this month vs last month
            "Weekly": "1w",    # Compare this week vs last week
            "Daily": "1d",     # Compare today vs yesterday
            "4H": "4h",        # Compare this 4h vs previous 4h
            "1H": "1h"         # Compare this hour vs previous hour
        }

        # Get the appropriate Binance period based on selected timeframe
        binance_period = period_map.get(timeframe, "1d")

        params = {
            'symbol': symbol,
            'period': binance_period,
            'limit': limit
        }

        response = api_get(f"{BINANCE_API_URL}/futures/data/openInterestHist", params=params, timeout=5)
        data = response.json()

        if data and len(data) >= 2:
            # Get OI values for current and previous period
            current_oi = float(data[-1]['sumOpenInterest'])   # Current period
            previous_oi = float(data[-2]['sumOpenInterest'])  # Previous period

            if previous_oi > 0:
                oi_change = ((current_oi - previous_oi) / previous_oi) * 100

                # Return both change and period info
                return {
                    'change_percent': round(oi_change, 2),
                    'current_oi': current_oi,
                    'previous_oi': previous_oi,
                    'period': binance_period,
                    'timeframe': timeframe,
                    'comparison_text': f"{timeframe} ({binance_period})"
                }
        return {
            'change_percent': 0,
            'current_oi': 0,
            'previous_oi': 0,
            'period': binance_period,
            'timeframe': timeframe,
            'comparison_text': f"{timeframe} ({binance_period})"
        }
    except Exception as e:
        return {
            'change_percent': 0,
            'current_oi': 0,
            'previous_oi': 0,
            'period': 'N/A',
            'timeframe': timeframe,
            'comparison_text': f"{timeframe} (Error)"
        }

# Binance /futures/data/openInterestHist only keeps ~30 days and timestamps are period END.
BINANCE_OI_MAX_LOOKBACK_DAYS = 29
OI_PERIOD_SECONDS = {"1h": 3600, "4h": 14400, "1d": 86400}


def _oi_period_from_interval(interval):
    if interval in ("1h", "1H"):
        return "1h"
    if interval in ("4h", "4H"):
        return "4h"
    return "1d"


def _oi_usd_value(item):
    if not isinstance(item, dict):
        return 0.0
    raw = item.get("sumOpenInterestValue")
    if raw is None:
        raw = item.get("sumOpenInterest")
    try:
        return float(raw or 0)
    except (TypeError, ValueError):
        return 0.0


def _oi_period_start_sec(item, oi_period):
    """Convert Binance OI timestamp (period end, ms) to period start unix seconds."""
    end_ms = int(item.get("timestamp") or 0)
    if end_ms <= 0:
        return 0
    return int(end_ms / 1000) - OI_PERIOD_SECONDS.get(oi_period, 86400)


def index_oi_by_period_start(oi_data, interval):
    """Map period-start unix seconds -> OI USDT value."""
    oi_period = _oi_period_from_interval(interval)
    indexed = {}
    for item in oi_data or []:
        if not isinstance(item, dict) or "timestamp" not in item:
            continue
        start_sec = _oi_period_start_sec(item, oi_period)
        if start_sec <= 0:
            continue
        indexed[start_sec] = _oi_usd_value(item)
    return indexed


def lookup_open_interest(oi_by_ts, target_ts_sec, timeframe="Daily"):
    """Match a candle/period open time to OI, with a half-period tolerance."""
    if not oi_by_ts or not target_ts_sec:
        return 0.0
    target_ts_sec = int(target_ts_sec)
    if target_ts_sec in oi_by_ts:
        return oi_by_ts[target_ts_sec]
    max_diff = {"Daily": 43200, "1H": 1800, "4H": 7200}.get(timeframe, 3600)
    nearest_ts = min(oi_by_ts.keys(), key=lambda t: abs(t - target_ts_sec))
    if abs(nearest_ts - target_ts_sec) <= max_diff:
        return oi_by_ts[nearest_ts]
    return 0.0


def _fetch_oi_hist_chunk(symbol, oi_period, start_ts, end_ts, retries=4):
    """One openInterestHist request with retries. Returns a list or []."""
    params = {
        "symbol": symbol,
        "period": oi_period,
        "startTime": int(start_ts * 1000),
        "endTime": int(end_ts * 1000),
        "limit": 500,
    }
    url = f"{BINANCE_API_URL}/futures/data/openInterestHist"
    for attempt in range(retries):
        try:
            response = api_get(url, params=params, timeout=15)
            if response.status_code in (418, 429):
                time.sleep(1.2 * (attempt + 1))
                continue
            data = response.json()
            if isinstance(data, list):
                return data
            # HTTP 400 "startTime is invalid" when the window is too old / too wide
            if response.status_code >= 400 and attempt == 0:
                # Retry without startTime: latest available history
                fallback = api_get(
                    url,
                    params={"symbol": symbol, "period": oi_period, "limit": 500},
                    timeout=15,
                )
                fb = fallback.json()
                return fb if isinstance(fb, list) else []
            if attempt < retries - 1:
                time.sleep(0.4 * (attempt + 1))
                continue
            return []
        except (requests.RequestException, ValueError, TypeError):
            time.sleep(0.5 * (attempt + 1))
    return []


@st.cache_data(ttl=120)
def get_binance_historical_oi(symbol, interval, start_ts, end_ts):
    """Fetch historical open interest from Binance Futures (last ~30 days only)."""
    symbol = symbol.upper().replace(".P", "")
    oi_period = _oi_period_from_interval(interval)
    period_sec = OI_PERIOD_SECONDS[oi_period]

    now_ts = time.time()
    min_start = now_ts - BINANCE_OI_MAX_LOOKBACK_DAYS * 86400
    fetch_start = max(int(start_ts), int(min_start))
    fetch_end = max(int(end_ts), fetch_start + period_sec)
    fetch_end = min(fetch_end, int(now_ts) + period_sec)
    if fetch_end <= fetch_start:
        return []

    # Stay under Binance's ~1 month window and 500-row cap per call
    max_span_sec = min(BINANCE_OI_MAX_LOOKBACK_DAYS * 86400, 500 * period_sec)
    all_rows = []
    cursor = fetch_start
    while cursor < fetch_end:
        chunk_end = min(cursor + max_span_sec, fetch_end)
        chunk = _fetch_oi_hist_chunk(symbol, oi_period, cursor, chunk_end)
        if chunk:
            all_rows.extend(chunk)
            last_end_sec = int(chunk[-1].get("timestamp") or 0) / 1000
            next_cursor = int(last_end_sec) + 1
            cursor = next_cursor if next_cursor > cursor else chunk_end
        else:
            cursor = chunk_end
        if cursor < fetch_end:
            time.sleep(0.12)

    by_ts = {}
    for item in all_rows:
        if isinstance(item, dict) and item.get("timestamp") is not None:
            by_ts[item["timestamp"]] = item
    return list(by_ts.values())

# ==============================================
# HELPER FUNCTIONS
# ==============================================

def format_large_number(num):
    """Format large numbers with K, M, B suffixes"""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return f"{num:.2f}"

def format_volume_ratio(ratio):
    """Format volume ratio as 1:X format"""
    if ratio == float('inf'):
        return "∞:1"
    elif ratio >= 1:
        return f"1:{ratio:.1f}"
    else:
        return f"{1/ratio:.1f}:1"

def calculate_volume_percentile(current_volume, historical_volumes):
    """Calculate where current volume stands compared to history"""
    if not historical_volumes:
        return 0
    sorted_volumes = sorted(historical_volumes)
    count_less = sum(1 for v in sorted_volumes if v < current_volume)
    percentile = (count_less / len(sorted_volumes)) * 100
    return percentile

def get_candle_volumes(kline_data):
    """Extract both trade volume and USDT volume from kline"""
    if not kline_data or len(kline_data) < 8:
        return 0, 0
    trade_volume = float(kline_data[5])
    usdt_volume = float(kline_data[7])
    return trade_volume, usdt_volume

def check_spot_match_fast(symbol):
    """Fast spot market check with caching"""
    try:
        spot_symbols = get_spot_symbols()
        return symbol in spot_symbols
    except:
        return False

def get_historical_klines(symbol, interval, start_time=None, end_time=None, limit=500):
    """Fetch historical kline data for a specific time range"""
    try:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        if start_time:
            if isinstance(start_time, datetime):
                if start_time.tzinfo:
                    start_time_utc = start_time.astimezone(UTC)
                else:
                    start_time_utc = start_time.replace(tzinfo=UTC)
                params['startTime'] = int(start_time_utc.timestamp() * 1000)
            else:
                params['startTime'] = start_time

        if end_time:
            if isinstance(end_time, datetime):
                if end_time.tzinfo:
                    end_time_utc = end_time.astimezone(UTC)
                else:
                    end_time_utc = end_time.replace(tzinfo=UTC)
                params['endTime'] = int(end_time_utc.timestamp() * 1000)
            else:
                params['endTime'] = end_time

        response = api_get(f"{BINANCE_API_URL}/fapi/v1/klines", params=params, timeout=5)
        data = response.json()

        if isinstance(data, dict) and 'code' in data:
            return None

        return data
    except Exception as e:
        return None

def get_btc_daily_volume(date):
    """Get BTC daily volume for comparison"""
    try:
        btc_symbol = "BTCUSDT"
        start_date = datetime(date.year, date.month, date.day, 0, 0, 0, tzinfo=IST)
        end_date = datetime(date.year, date.month, date.day, 23, 59, 59, tzinfo=IST)

        klines = get_historical_klines(btc_symbol, "1d", start_date, end_date, limit=1)

        if klines and len(klines) > 0:
            _, usdt_volume = get_candle_volumes(klines[0])
            return usdt_volume
        return 0
    except:
        return 0

def get_btc_historical_volumes(days=30):
    """Get BTC historical volumes for the last N days"""
    try:
        end_date = datetime.now(IST)
        start_date = end_date - timedelta(days=days)

        klines = get_historical_klines("BTCUSDT", "1d", start_date, end_date, limit=days + 5)

        btc_volumes = {}
        if klines:
            for kline in klines:
                try:
                    timestamp = kline[0]
                    candle_date = datetime.fromtimestamp(timestamp / 1000, tz=UTC)
                    candle_date_ist = candle_date.astimezone(IST)
                    date_str = candle_date_ist.strftime('%Y-%m-%d')
                    _, usdt_volume = get_candle_volumes(kline)
                    btc_volumes[date_str] = usdt_volume
                except:
                    continue
        return btc_volumes
    except:
        return {}

@st.cache_data(ttl=86400)
def get_coingecko_id(symbol):
    """
    Search CoinGecko for a symbol and return the best matching coin ID based on market cap rank.
    """
    # Normalize symbol
    base_symbol = symbol.upper()
    if base_symbol.endswith(".P"):
        base_symbol = base_symbol[:-2]
    if base_symbol.endswith("USDT"):
        base_symbol = base_symbol[:-4]

    # Custom mapping dictionary for cases where symbol or ID doesn't match standard search
    COINGECKO_ID_MAP = {
        "AVAAI": "ava-ai",
        "LUNC": "terra-luna-2",
        "1000LUNC": "terra-luna-2",
        "1000PEPE": "pepe",
        "1000FLOKI": "floki",
        "1000XEC": "ecash",
        "1000RATS": "rats",
        "1000CAT": "simon-s-cat",
        "1000CHEEMS": "cheems",
        "10000LADYS": "milady-meme-coin",
        "1000000PEIPEI": "peipei",
        "1000000MOG": "mog-coin",
    }

    if base_symbol in COINGECKO_ID_MAP:
        return COINGECKO_ID_MAP[base_symbol]

    import re
    stripped_symbol = base_symbol
    match = re.match(r"^(\d+)(.+)$", base_symbol)
    if match:
        stripped_symbol = match.group(2)

    try:
        url = f"https://api.coingecko.com/api/v3/search?query={base_symbol}"
        r = api_get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        coins = data.get("coins", [])
        # Filter for exact symbol match or stripped symbol match
        matches = [c for c in coins if c.get("symbol", "").upper() in (base_symbol, stripped_symbol)]

        if not matches and stripped_symbol != base_symbol:
            # Try searching with the stripped symbol
            url = f"https://api.coingecko.com/api/v3/search?query={stripped_symbol}"
            r = api_get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            coins = data.get("coins", [])
            matches = [c for c in coins if c.get("symbol", "").upper() in (base_symbol, stripped_symbol)]

        if matches:
            # Sort by market_cap_rank, treating None as a very large number
            matches.sort(key=lambda x: x.get("market_cap_rank") if x.get("market_cap_rank") is not None else 999999)
            return matches[0]["id"]
    except Exception:
        pass

    # Fallback search in coins list
    try:
        url_list = "https://api.coingecko.com/api/v3/coins/list"
        r_list = api_get(url_list, timeout=10)
        r_list.raise_for_status()
        coins_list = r_list.json()

        matches = [c for c in coins_list if c.get("symbol", "").upper() in (base_symbol, stripped_symbol)]
        if matches:
            matches.sort(key=lambda x: len(x.get("id", "")))
            return matches[0]["id"]
    except Exception:
        pass

    return base_symbol.lower()

@st.cache_data(ttl=86400)
def get_coingecko_coin_details(coin_id):
    """Fetch current static supply details for a coin from CoinGecko"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "false"
        }
        r = api_get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        md = data.get("market_data", {})
        return {
            "total_supply": md.get("total_supply"),
            "max_supply": md.get("max_supply"),
            "current_fdv": md.get("fully_diluted_valuation", {}).get("usd")
        }
    except Exception:
        return {}

def _parse_coingecko_market_chart(data):
    """Turn CoinGecko market_chart JSON into one UTC daily row per date."""
    price_dict = {item[0]: item[1] for item in data.get("prices", [])}
    mc_dict = {item[0]: item[1] for item in data.get("market_caps", [])}
    vol_dict = {item[0]: item[1] for item in data.get("total_volumes", [])}

    by_date = {}
    for ts in sorted(mc_dict.keys()):
        dt = datetime.fromtimestamp(ts / 1000, tz=UTC)
        date_key = dt.strftime("%Y-%m-%d")
        price = price_dict.get(ts, 0)
        mc = mc_dict[ts]
        vol = vol_dict.get(ts, 0)
        v_mc = (vol / mc * 100) if mc > 0 else 0
        circ_supply = (mc / price) if price > 0 else 0
        by_date[date_key] = {
            "Date": dt.strftime("%d-%b-%y 00:00:00 UTC"),
            "timestamp": ts,
            "Price": price,
            "Market Cap": mc,
            "Volume": vol,
            "V/MC": v_mc,
            "Circulating Supply": circ_supply,
        }
    return [by_date[k] for k in sorted(by_date.keys())]


def _parse_coingecko_market_chart_points(data):
    """All CoinGecko market_chart points (not collapsed to daily)."""
    price_dict = {item[0]: item[1] for item in data.get("prices", [])}
    mc_dict = {item[0]: item[1] for item in data.get("market_caps", [])}
    vol_dict = {item[0]: item[1] for item in data.get("total_volumes", [])}
    all_ts = sorted(set(price_dict) | set(mc_dict) | set(vol_dict))
    points = []
    for ts in all_ts:
        price = price_dict.get(ts, 0)
        mc = mc_dict.get(ts, 0)
        vol = vol_dict.get(ts, 0)
        points.append({
            "timestamp": ts,
            "Price": price,
            "Market Cap": mc,
            "Volume": vol,
            "Circulating Supply": (mc / price) if price > 0 else 0,
        })
    return points


def nearest_coingecko_point(cg_points, target_ts_ms):
    if not cg_points:
        return None
    return min(cg_points, key=lambda p: abs(p["timestamp"] - target_ts_ms))


@st.cache_data(ttl=3600)
def get_coingecko_historical_points(coin_id, start_ts=None, end_ts=None, days=None):
    """Fetch raw CoinGecko market cap / price points for intraday alignment."""
    try:
        if start_ts is not None and end_ts is not None:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
            params = {"vs_currency": "usd", "from": int(start_ts), "to": int(end_ts)}
        else:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {"vs_currency": "usd", "days": days if days is not None else 10}
        r = api_get(url, params=params, timeout=20)
        r.raise_for_status()
        return _parse_coingecko_market_chart_points(r.json())
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_coingecko_historical_data(coin_id, days=None, start_ts=None, end_ts=None):
    """Fetch historical daily price, market cap, and volume from CoinGecko.

    Use start_ts/end_ts (unix seconds) for a specific UTC date range
    (CoinMarketCap-style history). Otherwise fetch the last `days` days.
    """
    try:
        if start_ts is not None and end_ts is not None:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
            params = {
                "vs_currency": "usd",
                "from": int(start_ts),
                "to": int(end_ts),
            }
        else:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": days if days is not None else 10,
            }
        r = api_get(url, params=params, timeout=20)
        r.raise_for_status()
        return _parse_coingecko_market_chart(r.json())
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_binance_daily_klines(symbol, start_ts, end_ts):
    """Daily Binance USDT-M perpetual klines: close price and USDT quote volume."""
    symbol = symbol.upper().replace(".P", "")
    start_dt = datetime.fromtimestamp(int(start_ts), tz=UTC)
    end_dt = datetime.fromtimestamp(int(end_ts), tz=UTC)
    all_klines = []
    cursor = start_dt
    while cursor <= end_dt:
        batch = get_historical_klines(symbol, "1d", cursor, end_dt, limit=1500)
        if not batch:
            break
        all_klines.extend(batch)
        last_open_ms = int(batch[-1][0])
        next_dt = datetime.fromtimestamp(last_open_ms / 1000, tz=UTC) + timedelta(days=1)
        if next_dt <= cursor or len(batch) < 1500:
            break
        cursor = next_dt

    by_date = {}
    for kline in all_klines:
        dt = datetime.fromtimestamp(int(kline[0]) / 1000, tz=UTC)
        date_key = dt.strftime("%Y-%m-%d")
        coin_volume, usdt_volume = get_candle_volumes(kline)
        by_date[date_key] = {
            "Date": dt.strftime("%d-%b-%y 00:00:00 UTC"),
            "timestamp": int(kline[0]),
            "Price": float(kline[4]),
            "Coin Volume": usdt_volume,
            "Base Volume": coin_volume,
        }
    return by_date


def rvol_signal(rvol):
    """Human-readable RVOL bucket."""
    if rvol < 0.8:
        return "Low"
    if rvol < 1.2:
        return "Normal"
    if rvol < 2.0:
        return "Elevated"
    if rvol < 3.0:
        return "High"
    return "Very High"


@st.cache_data(ttl=300)
def get_daily_usdt_volumes(symbol, start_ts, end_ts):
    """UTC date (YYYY-MM-DD) -> USDT quote volume from Binance daily klines."""
    by_date = get_binance_daily_klines(symbol, start_ts, end_ts)
    volumes = {k: v["Coin Volume"] for k, v in by_date.items()}
    return volumes, by_date


def compute_rvol(symbol, target_date, avg_days):
    """
    Relative Volume = volume on target_date / average of prior `avg_days` daily volumes.

    target_date: calendar date matching the UTC daily candle (same as V/MC daily table).
    avg_days: number of completed days BEFORE target_date used for the average.
    """
    symbol = symbol.upper().replace(".P", "")
    target_key = target_date.strftime("%Y-%m-%d")
    start_dt = datetime.combine(
        target_date - timedelta(days=avg_days + 15),
        datetime.min.time(),
        tzinfo=UTC,
    )
    end_dt = datetime.combine(target_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    volumes, details = get_daily_usdt_volumes(
        symbol, int(start_dt.timestamp()), int(end_dt.timestamp())
    )
    if target_key not in volumes:
        return None

    today_vol = float(volumes[target_key])
    prior_dates = sorted(d for d in volumes if d < target_key)
    if not prior_dates:
        return None
    use_dates = prior_dates[-avg_days:] if len(prior_dates) >= avg_days else prior_dates
    avg_vol = sum(float(volumes[d]) for d in use_dates) / len(use_dates)
    rvol = (today_vol / avg_vol) if avg_vol > 0 else 0.0
    bar = details.get(target_key, {})
    return {
        "Date": bar.get("Date", target_key),
        "Price": bar.get("Price"),
        "Target Volume": today_vol,
        "Avg Volume": avg_vol,
        "Avg Days Used": len(use_dates),
        "Avg Days Requested": int(avg_days),
        "RVOL": round(rvol, 2),
        "Signal": rvol_signal(rvol),
    }


def fetch_rvol_worker(symbol, target_date, avg_days):
    try:
        row = compute_rvol(symbol, target_date, avg_days)
        if not row:
            return None
        row["Coin"] = f"{symbol}.P" if not str(symbol).endswith(".P") else symbol
        return row
    except Exception:
        return None


def scan_rvol_parallel(symbols, target_date, avg_days, max_workers=30):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(symbols)
    if total == 0:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_rvol_worker, symbol, target_date, avg_days): symbol
            for symbol in symbols
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 20 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(f"RVOL scan {completed}/{total}...")
            try:
                row = future.result(timeout=10)
                if row:
                    results.append(row)
            except Exception:
                pass
    progress_bar.empty()
    status_text.empty()
    return results


def build_binance_vmc_rows(symbol, cg_rows, details, start_ts, end_ts, timezone_name="IST"):
    """
    Momentum uses Binance volume (matches Binance charts) and market cap.

    Market Cap = Binance close × circulating supply (from CoinGecko).
    Momentum   = (Binance USDT volume / Market Cap) × 100
    """
    binance_by_date = get_binance_daily_klines(symbol, start_ts, end_ts)
    cg_by_date = {}
    for row in cg_rows or []:
        dt = datetime.fromtimestamp(row["timestamp"] / 1000, tz=UTC)
        cg_by_date[dt.strftime("%Y-%m-%d")] = row

    current_circ = None
    if details:
        total = details.get("total_supply")
        # Prefer implied latest circulating supply from CoinGecko rows
        if cg_rows:
            current_circ = cg_rows[-1].get("Circulating Supply")
        if not current_circ:
            current_circ = total

    # Fetch historical Open Interest (daily). Binance timestamps are period END.
    oi_by_ts = index_oi_by_period_start(
        get_binance_historical_oi(symbol, "Daily", start_ts, end_ts),
        "Daily",
    )
    oi_by_date = {}
    for start_sec, oi_val in oi_by_ts.items():
        dt = datetime.fromtimestamp(start_sec, tz=UTC)
        oi_by_date[dt.strftime("%Y-%m-%d")] = oi_val

    rows = []
    for date_key in sorted(binance_by_date.keys()):
        b = binance_by_date[date_key]
        cg = cg_by_date.get(date_key, {})
        circ = cg.get("Circulating Supply") or current_circ or 0
        price = b["Price"]
        volume = b["Coin Volume"]
        market_cap = cg.get("Market Cap") or (price * circ if circ else 0)
        momentum = (volume / market_cap * 100) if market_cap else 0

        oi = oi_by_date.get(date_key, 0.0)
        if not oi:
            oi = lookup_open_interest(oi_by_ts, int(b["timestamp"]) / 1000, "Daily")
        moi = (oi / market_cap * 100) if market_cap and oi else 0.0
        voi = (volume / oi * 100) if oi and volume else 0.0

        rows.append({
            "Date": format_vmc_daily_label(b["timestamp"], timezone_name),
            "timestamp": b["timestamp"],
            "Price": price,
            "Volume": volume,
            "Market Cap": market_cap,
            "V/MC": momentum,
            "Circulating Supply": circ,
            "Open Interest": oi,
            "MOI": moi,
            "VOI": voi,
        })
    return rows


@st.cache_data(ttl=300)
def get_coingecko_markets_by_symbol(pages=4):
    """Map base symbol -> circulating supply / market cap (highest MC wins duplicates)."""
    by_symbol = {}
    try:
        for page in range(1, pages + 1):
            r = api_get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 250,
                    "page": page,
                    "sparkline": "false",
                },
                timeout=20,
            )
            r.raise_for_status()
            coins = r.json()
            if not isinstance(coins, list):
                break
            for coin in coins:
                symbol = str(coin.get("symbol") or "").upper()
                if not symbol or symbol in by_symbol:
                    continue
                by_symbol[symbol] = {
                    "id": coin.get("id"),
                    "circ": coin.get("circulating_supply") or 0,
                    "total": coin.get("total_supply") or 0,
                    "max": coin.get("max_supply") or 0,
                    "gecko_mc": coin.get("market_cap") or 0,
                }
            if len(coins) < 250:
                break
    except Exception:
        pass
    return by_symbol


def activity_score(vmc, vol_chg, mc_chg, price_chg):
    """ChatGPT overlapping thresholds: 0–3 Weak, 4–6 Interesting, 7–9 Strong, 10+ Extreme."""
    score = 0
    if vmc > 100:
        score += 1
    if vmc > 200:
        score += 2
    if vmc > 300:
        score += 3
    if vol_chg > 50:
        score += 1
    if vol_chg > 100:
        score += 2
    if mc_chg > 5:
        score += 1
    if mc_chg > 10:
        score += 2
    if price_chg > 5:
        score += 1
    if price_chg > 10:
        score += 2
    return score


def activity_band(score):
    if score >= 10:
        return "Extreme"
    if score >= 7:
        return "Strong"
    if score >= 4:
        return "Interesting"
    return "Weak"


def classify_activity_regime(vmc, vol_chg, price_chg, mc_chg):
    expansion = vmc > 100 and vol_chg > 100 and mc_chg > 10 and price_chg > 0
    if expansion and price_chg >= 50:
        return "Climax — do not chase"
    if expansion:
        return "Activity expansion"
    if vmc >= 100 and price_chg < 0:
        return "Dump / shakeout"
    if vmc >= 100 and price_chg > 0 and vol_chg < 0:
        return "Distribution (dying volume)"
    if vmc >= 100 and price_chg > 0:
        return "Hot — needs confirmation"
    if vmc < 25:
        return "Quiet"
    return "Watch"


def newton_force_score(force, p50=0.0005, p75=0.002, p90=0.01):
    """Map |F| to -3…+3. Thresholds default to small-cap-safe floors; pass series percentiles when ranking a history."""
    if force is None or (isinstance(force, float) and (math.isnan(force) or math.isinf(force))):
        return None
    mag = abs(force)
    sign = 1 if force > 0 else -1 if force < 0 else 0
    if mag < p50 or sign == 0:
        return 0
    if mag < p75:
        return sign * 1
    if mag < p90:
        return sign * 2
    return sign * 3


def newton_score_label(score):
    labels = {
        3: "+3 very strong bullish force",
        2: "+2 bullish force",
        1: "+1 weak bullish force",
        0: "0 neutral",
        -1: "-1 weak bearish force",
        -2: "-2 bearish force",
        -3: "-3 very strong bearish force",
    }
    return labels.get(score, "—")


def newton_candle_regime(velocity, accel, vmc, prev_vmc):
    """Price / V/MC / acceleration sign table from the Newton-candle model."""
    if None in (velocity, accel, vmc, prev_vmc):
        return None
    price_up = velocity > 0
    vmc_up = vmc > prev_vmc
    accel_up = accel > 0
    if price_up and vmc_up and accel_up:
        return "Strong bullish force"
    if price_up and vmc_up and not accel_up:
        return "Bullish but losing force"
    if price_up and not vmc_up and accel_up:
        return "Potentially weak rally"
    if (not price_up) and vmc_up and (not accel_up):
        return "Strong bearish force"
    if (not price_up) and vmc_up and accel_up:
        return "Selling pressure weakening"
    if (not price_up) and (not vmc_up) and (not accel_up):
        return "Weak bearish move"
    return "Mixed"


def compute_newton_fields(price, prev_price, prev2_price, volume, market_cap, prev_volume, prev_market_cap):
    """
    V_t = (C_t - C_{t-1}) / C_{t-1}
    A_t = V_t - V_{t-1}
    F_t = A_t × (Volume_t / MarketCap_t)
    """
    if not prev_price or not prev2_price or prev_price <= 0 or prev2_price <= 0:
        return None
    velocity = (price - prev_price) / prev_price
    prev_velocity = (prev_price - prev2_price) / prev2_price
    accel = velocity - prev_velocity
    turnover = (volume / market_cap) if market_cap else 0.0
    force = accel * turnover
    vmc = (volume / market_cap * 100) if market_cap else 0.0
    prev_vmc = (prev_volume / prev_market_cap * 100) if prev_market_cap else None
    score = newton_force_score(force)
    return {
        "Velocity": velocity * 100,
        "Acceleration": accel * 100,
        "Force": force,
        "Newton Score": score,
        "Newton Regime": newton_candle_regime(velocity, accel, vmc, prev_vmc),
    }


def attach_newton_to_df(df):
    """Add Newton velocity / acceleration / force columns to a V/MC history frame."""
    if df is None or df.empty or "Price" not in df.columns or "V/MC" not in df.columns:
        return df
    out = df.copy()
    if "timestamp" in out.columns:
        out = out.sort_values("timestamp").reset_index(drop=True)
    prices = pd.to_numeric(out["Price"], errors="coerce")
    vmc = pd.to_numeric(out["V/MC"], errors="coerce")
    velocity = prices.pct_change()
    accel = velocity.diff()
    force = accel * (vmc / 100.0)
    prev_vmc = vmc.shift(1)
    abs_f = force.abs().dropna()
    if len(abs_f) >= 10:
        p50, p75, p90 = float(abs_f.quantile(0.50)), float(abs_f.quantile(0.75)), float(abs_f.quantile(0.90))
    else:
        p50, p75, p90 = 0.0005, 0.002, 0.01
    scores = []
    regimes = []
    for i in range(len(out)):
        f = force.iloc[i]
        scores.append(
            newton_force_score(None if pd.isna(f) else float(f), p50, p75, p90)
        )
        v = None if pd.isna(velocity.iloc[i]) else float(velocity.iloc[i])
        a = None if pd.isna(accel.iloc[i]) else float(accel.iloc[i])
        m = None if pd.isna(vmc.iloc[i]) else float(vmc.iloc[i])
        pm = None if pd.isna(prev_vmc.iloc[i]) else float(prev_vmc.iloc[i])
        regimes.append(newton_candle_regime(v, a, m, pm))
    out["Velocity"] = velocity * 100
    out["Acceleration"] = accel * 100
    out["Force"] = force
    out["Newton Score"] = scores
    out["Newton Regime"] = regimes
    return out


VMC_TIMEFRAME_MAP = {
    "Daily": "1d",
    "1H": "1h",
    "4H": "4h",
}

# Crypto day: UTC midnight == 05:30 AM IST (CoinGecko daily timestamps are UTC).
CRYPTO_SESSION_HOUR = 5
CRYPTO_SESSION_MINUTE = 30
CRYPTO_4H_START_HOURS = {1, 5, 9, 13, 17, 21}

VMC_TZ_CONFIG = {
    "UTC": {
        "zone": UTC,
        "label": "UTC",
        "session_hour": 0,
        "session_minute": 0,
        "four_h_hours": {0, 4, 8, 12, 16, 20},
    },
    "IST": {
        "zone": IST,
        "label": "IST",
        "session_hour": CRYPTO_SESSION_HOUR,
        "session_minute": CRYPTO_SESSION_MINUTE,
        "four_h_hours": CRYPTO_4H_START_HOURS,
    },
}


def vmc_tz_config(timezone_name="IST"):
    return VMC_TZ_CONFIG.get(timezone_name, VMC_TZ_CONFIG["IST"])


def get_crypto_day_start(dt=None, timezone_name="IST"):
    """Return crypto-day open (00:00 UTC or 05:30 IST) containing dt."""
    cfg = vmc_tz_config(timezone_name)
    zone = cfg["zone"]
    if dt is None:
        dt = datetime.now(zone)
    else:
        dt = dt.astimezone(zone)
    day_start = dt.replace(
        hour=cfg["session_hour"],
        minute=cfg["session_minute"],
        second=0,
        microsecond=0,
    )
    if dt < day_start:
        day_start -= timedelta(days=1)
    return day_start


def get_crypto_range_bounds(range_mode, start_date=None, end_date=None, no_of_days=1, timezone_name="IST"):
    """Resolve query window using crypto-day boundaries in UTC or IST."""
    cfg = vmc_tz_config(timezone_name)
    zone = cfg["zone"]
    now = datetime.now(zone)
    if range_mode == "Today":
        return get_crypto_day_start(now, timezone_name), now
    if range_mode == "Last N days":
        crypto_start = get_crypto_day_start(now, timezone_name)
        days = max(int(no_of_days), 1)
        range_start = crypto_start - timedelta(days=days - 1)
        return range_start, now
    range_start = datetime(
        start_date.year, start_date.month, start_date.day,
        cfg["session_hour"], cfg["session_minute"], tzinfo=zone,
    )
    end_day_close = datetime(
        end_date.year, end_date.month, end_date.day,
        cfg["session_hour"], cfg["session_minute"], tzinfo=zone,
    ) + timedelta(days=1)
    return range_start, min(now, end_day_close)


def kline_open_tz(kline, timezone_name="IST"):
    cfg = vmc_tz_config(timezone_name)
    return datetime.fromtimestamp(int(kline[0]) / 1000, tz=UTC).astimezone(cfg["zone"])


def kline_open_ist(kline):
    return kline_open_tz(kline, "IST")


def is_crypto_period_start(dt_local, timeframe, timezone_name="IST"):
    """True when dt_local is a valid period open in the selected timezone."""
    cfg = vmc_tz_config(timezone_name)
    if dt_local.minute != cfg["session_minute"]:
        return False
    if timeframe == "1H":
        return True
    if timeframe == "4H":
        return dt_local.hour in cfg["four_h_hours"]
    return False


def crypto_period_starts_in_range(range_start, range_end, timeframe, timezone_name="IST"):
    """Period-open datetimes in [range_start, range_end] for UTC or IST."""
    starts = []
    day_cursor = get_crypto_day_start(range_start, timezone_name)
    while day_cursor <= range_end:
        if timeframe == "Daily":
            if range_start <= day_cursor <= range_end:
                starts.append(day_cursor)
        elif timeframe == "1H":
            ps = day_cursor
            day_end = day_cursor + timedelta(days=1)
            while ps < day_end:
                if range_start <= ps <= range_end:
                    starts.append(ps)
                ps += timedelta(hours=1)
        else:
            ps = day_cursor
            for _ in range(6):
                if range_start <= ps <= range_end:
                    starts.append(ps)
                ps += timedelta(hours=4)
        day_cursor += timedelta(days=1)
    return starts


def format_crypto_stars_period(start_dt, timeframe, timezone_name="IST"):
    """Format a period label in UTC or IST (same instants; CoinGecko is UTC)."""
    cfg = vmc_tz_config(timezone_name)
    start_dt = start_dt.astimezone(cfg["zone"])
    tz_label = cfg["label"]
    if timeframe == "Daily":
        if timezone_name == "UTC":
            return start_dt.strftime("%d-%b-%y 00:00 UTC")
        return start_dt.strftime("%d-%b-%y 05:30 AM IST")
    hours = 1 if timeframe == "1H" else 4
    end_dt = start_dt + timedelta(hours=hours)
    if timezone_name == "UTC":
        return (
            f"{start_dt.strftime('%d-%b-%y %H:%M')} - "
            f"{end_dt.strftime('%H:%M')} UTC"
        )
    return (
        f"{start_dt.strftime('%d-%b-%y %I:%M %p')} - "
        f"{end_dt.strftime('%I:%M %p')} {tz_label}"
    )


def format_vmc_daily_label(ts_ms, timezone_name="IST"):
    """Label a CoinGecko/Binance daily timestamp (UTC midnight == 05:30 IST)."""
    ts = ts_ms / 1000 if ts_ms > 1e12 else ts_ms
    cfg = vmc_tz_config(timezone_name)
    dt = datetime.fromtimestamp(ts, tz=UTC).astimezone(cfg["zone"])
    if timezone_name == "UTC":
        return dt.strftime("%d-%b-%y 00:00:00 UTC")
    return dt.strftime("%d-%b-%y 05:30:00 IST")


def format_vmc_period_ist(kline, timeframe):
    """Label a period using Crypto Stars 05:30 AM IST session boundaries."""
    start_dt = kline_open_ist(kline)
    if timeframe == "Daily":
        return start_dt.strftime("%d-%b-%y 05:30 AM IST")
    return format_crypto_stars_period(start_dt, timeframe, "IST")


@st.cache_data(ttl=300)
def get_binance_interval_klines(symbol, interval, start_ts, end_ts):
    """Binance USDT-M perpetual klines for 1h/4h (or any interval) over a UTC range."""
    symbol = symbol.upper().replace(".P", "")
    start_dt = datetime.fromtimestamp(int(start_ts), tz=UTC)
    end_dt = datetime.fromtimestamp(int(end_ts), tz=UTC)
    all_klines = []
    cursor = start_dt
    while cursor <= end_dt:
        batch = get_historical_klines(symbol, interval, cursor, end_dt, limit=1500)
        if not batch:
            break
        all_klines.extend(batch)
        last_open_ms = int(batch[-1][0])
        step = timedelta(hours=1) if interval == "1h" else timedelta(hours=4) if interval == "4h" else timedelta(days=1)
        next_dt = datetime.fromtimestamp(last_open_ms / 1000, tz=UTC) + step
        if next_dt <= cursor or len(batch) < 1500:
            break
        cursor = next_dt

    rows = []
    seen = set()
    for kline in all_klines:
        ts = int(kline[0])
        if ts in seen:
            continue
        seen.add(ts)
        coin_volume, usdt_volume = get_candle_volumes(kline)
        rows.append({
            "timestamp": ts,
            "Price": float(kline[4]),
            "Coin Volume": usdt_volume,
            "Base Volume": coin_volume,
            "kline": kline,
        })
    rows.sort(key=lambda x: x["timestamp"])
    return rows


def build_binance_vmc_rows_interval(symbol, cg_points, details, start_ts, end_ts, timeframe, timezone_name="IST"):
    """
    V/MC rows per 1H/4H period in UTC or IST (same candle instants).
    Volume + Price from Binance futures; Market Cap from nearest CoinGecko snapshot
    (Binance API does not provide market cap).
    """
    interval = VMC_TIMEFRAME_MAP.get(timeframe, "1h")
    fetch_start_ts = start_ts - 4 * 3600
    kline_rows = get_binance_interval_klines(symbol, interval, fetch_start_ts, end_ts)
    if not kline_rows:
        return []

    zone = vmc_tz_config(timezone_name)["zone"]
    range_start = datetime.fromtimestamp(int(start_ts), tz=UTC).astimezone(zone)
    range_end = datetime.fromtimestamp(int(end_ts), tz=UTC).astimezone(zone)

    fallback_circ = None
    if details:
        fallback_circ = details.get("total_supply")

    # Fetch historical Open Interest (align on period START, not Binance end timestamp)
    oi_by_ts = index_oi_by_period_start(
        get_binance_historical_oi(symbol, timeframe, start_ts, end_ts),
        timeframe,
    )

    rows = []
    for item in kline_rows:
        kline = item["kline"]
        period_start = kline_open_tz(kline, timezone_name)
        if not is_crypto_period_start(period_start, timeframe, timezone_name):
            continue
        if period_start < range_start or period_start > range_end:
            continue
        price = item["Price"]
        volume = item["Coin Volume"]
        cg_point = nearest_coingecko_point(cg_points, item["timestamp"])
        if cg_point and cg_point.get("Market Cap"):
            market_cap = cg_point["Market Cap"]
            circ = cg_point.get("Circulating Supply") or fallback_circ or 0
        else:
            circ = fallback_circ or 0
            market_cap = price * circ if circ else 0
        momentum = (volume / market_cap * 100) if market_cap else 0

        oi = lookup_open_interest(oi_by_ts, int(item["timestamp"]) / 1000, timeframe)
        moi = (oi / market_cap * 100) if market_cap and oi else 0.0
        voi = (volume / oi * 100) if oi and volume else 0.0

        rows.append({
            "Date": format_crypto_stars_period(period_start, timeframe, timezone_name),
            "timestamp": item["timestamp"],
            "Price": price,
            "Volume": volume,
            "Market Cap": market_cap,
            "V/MC": momentum,
            "Circulating Supply": circ,
            "Open Interest": oi,
            "MOI": moi,
            "VOI": voi,
        })
    return rows


def build_coingecko_vmc_rows(symbol, coin_id, details, start_ts, end_ts, timeframe="Daily", timezone_name="IST"):
    """
    CoinGecko mode: Volume + Market Cap from CoinGecko (API timestamps are UTC).
    Momentum = (CoinGecko volume / CoinGecko market cap) × 100
    """
    if timeframe != "Daily":
        return build_coingecko_vmc_rows_interval(
            symbol, coin_id, details, start_ts, end_ts, timeframe, timezone_name
        )

    cg_rows = get_coingecko_historical_data(coin_id, start_ts=start_ts, end_ts=end_ts)
    if not cg_rows:
        return []

    oi_by_ts = index_oi_by_period_start(
        get_binance_historical_oi(symbol, "Daily", start_ts, end_ts),
        "Daily",
    )
    oi_by_date = {}
    for start_sec, oi_val in oi_by_ts.items():
        dt = datetime.fromtimestamp(start_sec, tz=UTC)
        oi_by_date[dt.strftime("%Y-%m-%d")] = oi_val

    rows = []
    for row in cg_rows:
        ts_sec = int(row["timestamp"] / 1000)
        date_key = datetime.fromtimestamp(ts_sec, tz=UTC).strftime("%Y-%m-%d")
        volume = row["Volume"]
        market_cap = row["Market Cap"]
        momentum = (volume / market_cap * 100) if market_cap else 0.0
        oi = oi_by_date.get(date_key, 0.0)
        if not oi:
            oi = lookup_open_interest(oi_by_ts, ts_sec, "Daily")
        moi = (oi / market_cap * 100) if market_cap and oi else 0.0
        voi = (volume / oi * 100) if oi and volume else 0.0
        rows.append({
            "Date": format_vmc_daily_label(row["timestamp"], timezone_name),
            "timestamp": row["timestamp"],
            "Price": row["Price"],
            "Volume": volume,
            "Market Cap": market_cap,
            "V/MC": momentum,
            "Circulating Supply": row.get("Circulating Supply", 0),
            "Open Interest": oi,
            "MOI": moi,
            "VOI": voi,
        })
    return rows


def build_coingecko_vmc_rows_interval(symbol, coin_id, details, start_ts, end_ts, timeframe, timezone_name="IST"):
    """CoinGecko 1H/4H rows aligned to UTC or IST period opens (same instants)."""
    zone = vmc_tz_config(timezone_name)["zone"]
    range_start = datetime.fromtimestamp(int(start_ts), tz=UTC).astimezone(zone)
    range_end = datetime.fromtimestamp(int(end_ts), tz=UTC).astimezone(zone)
    fetch_start_ts = start_ts - 4 * 3600
    cg_points = get_coingecko_historical_points(
        coin_id, start_ts=fetch_start_ts, end_ts=end_ts
    )
    if not cg_points:
        return []

    oi_by_ts = index_oi_by_period_start(
        get_binance_historical_oi(symbol, timeframe, start_ts, end_ts),
        timeframe,
    )
    fallback_circ = details.get("total_supply") if details else None
    period_starts = crypto_period_starts_in_range(range_start, range_end, timeframe, timezone_name)

    rows = []
    for period_start in period_starts:
        target_ms = int(period_start.astimezone(UTC).timestamp() * 1000)
        pt = nearest_coingecko_point(cg_points, target_ms)
        if not pt or not pt.get("Market Cap"):
            continue
        price = pt["Price"]
        volume = pt["Volume"]
        market_cap = pt["Market Cap"]
        circ = pt.get("Circulating Supply") or fallback_circ or 0
        momentum = (volume / market_cap * 100) if market_cap else 0.0
        oi = lookup_open_interest(oi_by_ts, target_ms / 1000, timeframe)
        moi = (oi / market_cap * 100) if market_cap and oi else 0.0
        voi = (volume / oi * 100) if oi and volume else 0.0
        rows.append({
            "Date": format_crypto_stars_period(period_start, timeframe, timezone_name),
            "timestamp": target_ms,
            "Price": price,
            "Volume": volume,
            "Market Cap": market_cap,
            "V/MC": momentum,
            "Circulating Supply": circ,
            "Open Interest": oi,
            "MOI": moi,
            "VOI": voi,
        })
    return rows


# ==============================================
# COINMARKETCAP API (Single-coin history — CMC mode)
# ==============================================

def get_cmc_api_key():
    """Resolve CMC API key from session, env, or Streamlit secrets."""
    session_key = st.session_state.get("cmc_api_key")
    if session_key:
        return session_key.strip()
    env_key = os.environ.get("CMC_API_KEY")
    if env_key:
        return env_key.strip()
    try:
        if hasattr(st, "secrets"):
            if st.secrets.get("CMC_API_KEY"):
                return str(st.secrets["CMC_API_KEY"]).strip()
            cmc_secrets = st.secrets.get("coinmarketcap")
            if cmc_secrets and cmc_secrets.get("api_key"):
                return str(cmc_secrets["api_key"]).strip()
    except Exception:
        pass
    return None


def symbol_to_cmc_base(symbol):
    base = symbol.upper().replace(".P", "")
    if base.endswith("USDT"):
        base = base[:-4]
    return base


def parse_cmc_timestamp(ts_value):
    if isinstance(ts_value, (int, float)):
        return datetime.fromtimestamp(ts_value / 1000 if ts_value > 1e12 else ts_value, tz=UTC)
    ts_str = str(ts_value)
    if ts_str.endswith("Z"):
        ts_str = ts_str.replace("Z", "+00:00")
    return datetime.fromisoformat(ts_str).astimezone(UTC)


def cmc_api_get(path, params, api_key):
    headers = {
        "Accept": "application/json",
        "X-CMC_PRO_API_KEY": api_key,
    }
    response = api_get(f"{CMC_API_URL}{path}", headers=headers, params=params, timeout=30)
    payload = response.json()
    status = payload.get("status", {})
    if status.get("error_code", 0) != 0:
        raise RuntimeError(status.get("error_message") or f"CMC API error ({status.get('error_code')})")
    if response.status_code >= 400:
        raise RuntimeError(f"CMC HTTP {response.status_code}")
    return payload


def extract_cmc_quotes(payload):
    raw = payload.get("data")
    if isinstance(raw, list):
        if not raw:
            return []
        return raw[0].get("quotes", []) or []
    if isinstance(raw, dict):
        for entry in raw.values():
            if isinstance(entry, dict):
                return entry.get("quotes", []) or []
    return []


@st.cache_data(ttl=3600)
def get_cmc_coin_details(symbol, api_key):
    """Fetch supply metadata from CoinMarketCap for FDV / supply columns."""
    try:
        payload = cmc_api_get(
            "/v2/cryptocurrency/info",
            {"symbol": symbol, "aux": "urls,logo,description,tags,platform,date_added,notice"},
            api_key,
        )
        raw = payload.get("data", {})
        info = None
        if isinstance(raw, dict):
            entry = raw.get(symbol.upper())
            if isinstance(entry, list) and entry:
                info = entry[0]
            elif isinstance(entry, dict):
                info = entry
            elif raw:
                first_val = next(iter(raw.values()))
                if isinstance(first_val, list) and first_val:
                    info = first_val[0]
                elif isinstance(first_val, dict):
                    info = first_val
        if not info:
            return {}
        return {
            "total_supply": info.get("total_supply"),
            "max_supply": info.get("max_supply"),
            "circulating_supply": info.get("circulating_supply"),
        }
    except Exception:
        return {}


@st.cache_data(ttl=86400)
def get_cmc_id_for_symbol(symbol, api_key):
    """Resolve CMC numeric ID for a symbol (handles duplicate tickers like ACE)."""
    try:
        payload = cmc_api_get(
            "/v1/cryptocurrency/map",
            {"symbol": symbol.upper(), "listing_status": "active"},
            api_key,
        )
        entries = payload.get("data") or []
        if not entries:
            return None
        symbol_upper = symbol.upper()
        exact = [e for e in entries if str(e.get("symbol", "")).upper() == symbol_upper]
        pool = exact or entries
        pool.sort(key=lambda e: e.get("rank") or 999999)
        return pool[0].get("id")
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_cmc_quotes_historical(symbol, time_start_iso, time_end_iso, interval, api_key, cmc_id=None):
    params = {
        "time_start": time_start_iso,
        "time_end": time_end_iso,
        "interval": interval,
        "convert": "USD",
    }
    if cmc_id:
        params["id"] = str(cmc_id)
    else:
        params["symbol"] = symbol
    payload = cmc_api_get("/v3/cryptocurrency/quotes/historical", params, api_key)
    return extract_cmc_quotes(payload)


def cmc_quote_interval(timeframe):
    if timeframe == "Daily":
        return "1d"
    if timeframe == "1H":
        return "1h"
    return "4h"


def fetch_cmc_quotes_for_timeframe(symbol, time_start_iso, time_end_iso, timeframe, api_key):
    """Fetch CMC historical quotes; works on Basic+ plans (no OHLCV endpoint needed)."""
    interval = cmc_quote_interval(timeframe)
    cmc_id = get_cmc_id_for_symbol(symbol, api_key)
    try:
        quotes = fetch_cmc_quotes_historical(
            symbol, time_start_iso, time_end_iso, interval, api_key, cmc_id=cmc_id
        )
        if not quotes and cmc_id:
            quotes = fetch_cmc_quotes_historical(
                symbol, time_start_iso, time_end_iso, interval, api_key, cmc_id=None
            )
        return quotes, interval
    except RuntimeError as exc:
        message = str(exc).lower()
        if timeframe != "Daily" and ("subscription" in message or "plan" in message or "endpoint" in message):
            raise RuntimeError(
                f"Your CoinMarketCap plan does not support {timeframe} historical quotes. "
                "Try **Daily** timeframe, or upgrade your CMC API plan for 1H/4H data."
            ) from exc
        raise


def build_cmc_vmc_rows(symbol, start_ts, end_ts, timeframe, api_key, timezone_name="IST"):
    """
    CMC mode using /v3/cryptocurrency/quotes/historical (Basic-plan compatible).
    Volume = CMC volume_24h at each snapshot, Market Cap = CMC market_cap.
    Momentum = Volume / Market Cap × 100
    """
    cmc_symbol = symbol_to_cmc_base(symbol)
    zone = vmc_tz_config(timezone_name)["zone"]
    range_start = datetime.fromtimestamp(int(start_ts), tz=UTC).astimezone(zone)
    range_end = datetime.fromtimestamp(int(end_ts), tz=UTC).astimezone(zone)
    quote_interval = cmc_quote_interval(timeframe)

    time_start_iso = range_start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    time_end_iso = range_end.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    quote_snaps, quote_interval = fetch_cmc_quotes_for_timeframe(
        cmc_symbol, time_start_iso, time_end_iso, timeframe, api_key
    )

    # Fetch historical Open Interest from Binance Futures
    oi_by_ts = index_oi_by_period_start(
        get_binance_historical_oi(symbol, timeframe, start_ts, end_ts),
        timeframe,
    )

    rows = []
    seen = set()
    for snap in quote_snaps:
        period_start = parse_cmc_timestamp(snap.get("timestamp")).astimezone(zone)
        period_start = period_start.replace(second=0, microsecond=0)
        if timeframe != "Daily" and not is_crypto_period_start(period_start, timeframe, timezone_name):
            continue
        if period_start < range_start or period_start > range_end:
            continue
        key = period_start.isoformat()
        if key in seen:
            continue
        seen.add(key)

        usd = snap.get("quote", {}).get("USD", {})
        market_cap = float(usd.get("market_cap") or 0)
        price = float(usd.get("price") or 0)
        volume = float(usd.get("volume_24h") or usd.get("volume_24h_reported") or 0)
        momentum = (volume / market_cap * 100) if market_cap else 0
        circ_supply = (market_cap / price) if price else 0

        oi = lookup_open_interest(oi_by_ts, int(period_start.timestamp()), timeframe)
        moi = (oi / market_cap * 100) if market_cap and oi else 0.0
        voi = (volume / oi * 100) if oi and volume else 0.0

        period_label = format_crypto_stars_period(period_start, timeframe, timezone_name)

        rows.append({
            "Date": period_label,
            "timestamp": int(period_start.timestamp() * 1000),
            "Price": price,
            "Volume": volume,
            "Market Cap": market_cap,
            "V/MC": momentum,
            "Circulating Supply": circ_supply,
            "CMC Interval": quote_interval,
            "Open Interest": oi,
            "MOI": moi,
            "VOI": voi,
        })

    rows.sort(key=lambda x: x["timestamp"])
    return rows


def vmc_activity_worker(symbol, circ_supply, use_live_candle, timeframe="Daily"):
    """Compare current vs prior Binance candle; MC = close × circulating supply."""
    try:
        interval = VMC_TIMEFRAME_MAP.get(timeframe, "1d")
        klines = get_historical_klines(symbol, interval, limit=5)
        if not klines or len(klines) < 4:
            return None
        if use_live_candle:
            cur, prev, prev2 = klines[-1], klines[-2], klines[-3]
        else:
            cur, prev, prev2 = klines[-2], klines[-3], klines[-4]
        price = float(cur[4])
        prev_price = float(prev[4])
        prev2_price = float(prev2[4])
        prev_high = float(prev[2])
        _, volume = get_candle_volumes(cur)
        _, prev_volume = get_candle_volumes(prev)
        if circ_supply <= 0 or prev_price <= 0:
            return None
        market_cap = price * circ_supply
        prev_mc = prev_price * circ_supply
        vmc = (volume / market_cap * 100) if market_cap else 0
        vol_chg = ((volume - prev_volume) / prev_volume * 100) if prev_volume else 0
        price_chg = (price / prev_price - 1) * 100
        mc_chg = (market_cap / prev_mc - 1) * 100 if prev_mc else 0
        score = activity_score(vmc, vol_chg, mc_chg, price_chg)
        expansion = vmc > 100 and vol_chg > 100 and mc_chg > 10 and price_chg > 0
        newton = compute_newton_fields(
            price, prev_price, prev2_price, volume, market_cap, prev_volume, prev_mc
        ) or {}
        return {
            "Coin": f"{symbol}.P",
            "Period (IST)": format_vmc_period_ist(cur, timeframe),
            "Timeframe": timeframe,
            "Price": price,
            "Price %": round(price_chg, 2),
            "Coin Volume": volume,
            "Vol %": round(vol_chg, 1),
            "Market Cap": market_cap,
            "MC %": round(mc_chg, 2),
            "Vol/MCap %": round(vmc, 2),
            "Score": score,
            "Band": activity_band(score),
            "Regime": classify_activity_regime(vmc, vol_chg, price_chg, mc_chg),
            "Expansion": "Yes" if expansion else "No",
            "Close > Prior High": "Yes" if price > prev_high else "No",
            "Circ Supply": circ_supply,
            "Velocity %": round(newton["Velocity"], 2) if newton else None,
            "Accel pp": round(newton["Acceleration"], 2) if newton else None,
            "Force": newton.get("Force"),
            "Newton Score": newton.get("Newton Score"),
            "Newton Regime": newton.get("Newton Regime") or "—",
        }
    except Exception:
        return None


def scan_vmc_activity_parallel(symbol_circ_pairs, use_live_candle, timeframe="Daily", max_workers=40):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(symbol_circ_pairs)
    if total == 0:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(vmc_activity_worker, symbol, circ, use_live_candle, timeframe): symbol
            for symbol, circ in symbol_circ_pairs
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 15 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(f"Scanning {completed}/{total} coins... hits: {len(results)}")
            try:
                row = future.result(timeout=8)
                if row:
                    results.append(row)
            except Exception:
                pass
    progress_bar.empty()
    status_text.empty()
    return results

# ==============================================
# BREAKOUT DETECTION FUNCTIONS (Original)
# ==============================================

def fetch_single_breakout_worker(symbol, timeframe):
    """Worker function for parallel breakout scanning with Open Interest"""
    try:
        # Determine number of candles needed based on timeframe
        if timeframe == "Monthly":
            limit = 5
        elif timeframe == "Weekly":
            limit = 10
        elif timeframe == "Daily":
            limit = 30
        else:  # 4H, 1H
            limit = 50

        params = {
            'symbol': symbol,
            'interval': TIMEFRAME_MAP[timeframe],
            'limit': limit
        }

        response = api_get(f"{BINANCE_API_URL}/fapi/v1/klines", params=params, timeout=5)
        data = response.json()

        if not data or len(data) < 3:
            return None

        # Get last two COMPLETED candles (not current)
        last_candle = data[-2]
        prev_candle = data[-3]

        # Extract OHLCV
        last_open = float(last_candle[1])
        last_high = float(last_candle[2])
        last_low = float(last_candle[3])
        last_close = float(last_candle[4])
        last_volume = float(last_candle[5])
        last_usdt_volume = float(last_candle[7])

        prev_high = float(prev_candle[2])
        prev_low = float(prev_candle[3])
        prev_volume = float(prev_candle[5])
        prev_usdt_volume = float(prev_candle[7])

        # Check for bullish breakout
        bullish_breakout = (last_close > prev_high and last_high > prev_high)

        # Check for bearish breakout
        bearish_breakout = (last_close < prev_low and last_low < prev_low)

        if not bullish_breakout and not bearish_breakout:
            return None

        # Calculate volume ratio
        if prev_volume > 0:
            vol_ratio = last_volume / prev_volume
            if vol_ratio < 1:
                return None
            vol_ratio_str = f"1:{int(vol_ratio)}"
            vol_ratio_num = int(vol_ratio)
        else:
            return None

        # Get Open Interest change - NOW TIMEFRAME AWARE!
        oi_data = get_historical_open_interest(symbol, timeframe)
        oi_change = oi_data['change_percent']

        result = {
            'Coin Name': symbol,
            'USDT Volume Ratio': f"1:{int(last_usdt_volume / prev_usdt_volume)}" if prev_usdt_volume > 0 else "1:1",
            'Coin volume Ratio': vol_ratio_str,
            'Volume_Ratio_Num': vol_ratio_num,
            'Open_Interest_change': oi_change,
            'OI_Period': oi_data['comparison_text']
        }

        if bullish_breakout:
            tested_low = last_low <= prev_low
            result['Breakout Type'] = 'Trap' if tested_low else 'Normal'
            result['Direction'] = 'Bullish'
            result['Open to High Change %'] = round(((last_high - last_open) / last_open) * 100, 2)
            result['Open to close Change %'] = round(((last_close - last_open) / last_open) * 100, 2)
            result['Open to Low Change %'] = '-'
        else:
            tested_high = last_high >= prev_high
            result['Breakout Type'] = 'Trap' if tested_high else 'Normal'
            result['Direction'] = 'Bearish'
            result['Open to Low Change %'] = round(((last_low - last_open) / last_open) * 100, 2)
            result['Open to close Change %'] = round(((last_close - last_open) / last_open) * 100, 2)
            result['Open to High Change %'] = '-'

        result['Spot Match'] = 'Yes' if check_spot_match_fast(symbol) else 'No'
        candle_time = datetime.fromtimestamp(last_candle[0] / 1000, tz=UTC)
        result['Candle_Date'] = candle_time.astimezone(IST).strftime('%Y-%m-%d %H:%M')

        return result

    except Exception as e:
        return None

def scan_breakouts_parallel(symbols, timeframe, max_workers=25):
    """Parallel breakout scanning"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(fetch_single_breakout_worker, symbol, timeframe): symbol
            for symbol in symbols
        }

        completed = 0
        total = len(symbols)

        for future in as_completed(future_to_symbol):
            completed += 1
            if completed % 10 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(f"Scanning {completed}/{total} pairs... Found: {len(results)} breakouts")

            try:
                result = future.result(timeout=5)
                if result:
                    results.append(result)
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()
    return results

# ==============================================
# CANDLE PATTERN: BULLISH LOW SWEEP
# ==============================================

CANDLE_PATTERN_TIMEFRAMES = ["Monthly", "Weekly", "Daily", "4H", "1H"]

# Binance 4H/1H candles are UTC-aligned → IST opens at :30 past the hour.
# Maps slot label → (close_day_offset, close_hour, close_minute) on the backtest date.
CANDLE_PATTERN_4H_SLOTS_IST = {
    "01:30 AM - 05:30 AM": (0, 5, 30),
    "05:30 AM - 09:30 AM": (0, 9, 30),
    "09:30 AM - 01:30 PM": (0, 13, 30),
    "01:30 PM - 05:30 PM": (0, 17, 30),
    "05:30 PM - 09:30 PM": (0, 21, 30),
    "09:30 PM - 01:30 AM": (1, 1, 30),
}


def candle_pattern_1h_slots_ist():
    """Build 24 x 1H IST slot labels keyed by close (day_offset, hour, minute)."""
    slots = {}
    for h in range(24):
        open_total = (CRYPTO_SESSION_HOUR * 60 + CRYPTO_SESSION_MINUTE) + h * 60
        close_total = open_total + 60
        open_day = open_total // (24 * 60)
        close_day = close_total // (24 * 60)
        open_mins = open_total % (24 * 60)
        close_mins = close_total % (24 * 60)
        open_dt = datetime(2000, 1, 1, open_mins // 60, open_mins % 60)
        close_dt = datetime(2000, 1, 1, close_mins // 60, close_mins % 60)
        label = (
            f"{open_dt.strftime('%I:%M %p')} - {close_dt.strftime('%I:%M %p')}"
        )
        slots[label] = (close_day, close_mins // 60, close_mins % 60)
    return slots


CANDLE_PATTERN_1H_SLOTS_IST = candle_pattern_1h_slots_ist()


def candle_pattern_reference_time_ist(backtest_date, timeframe, slot_label):
    """IST close time of the selected candle on backtest_date."""
    if timeframe == "4H":
        slot_map = CANDLE_PATTERN_4H_SLOTS_IST
    elif timeframe == "1H":
        slot_map = CANDLE_PATTERN_1H_SLOTS_IST
    else:
        return datetime.combine(backtest_date, datetime.min.time(), tzinfo=IST) + timedelta(
            hours=23, minutes=59, seconds=59
        )
    close_day_offset, close_hour, close_minute = slot_map[slot_label]
    slot_close_date = backtest_date + timedelta(days=close_day_offset)
    return datetime.combine(slot_close_date, datetime.min.time(), tzinfo=IST) + timedelta(
        hours=close_hour, minutes=close_minute
    )


def candle_pattern_period_hours(timeframe):
    return {"1H": 1, "4H": 4, "Daily": 24, "Weekly": 168, "Monthly": 720}.get(timeframe, 1)


def get_candle_slot_label_ist(kline_or_ms, timeframe):
    """Return the IST slot label (e.g., '05:30 AM - 09:30 AM') for a candle or candle open timestamp."""
    if isinstance(kline_or_ms, (list, tuple)):
        ms = int(kline_or_ms[0])
    else:
        ms = int(kline_or_ms)
    ist_open = datetime.fromtimestamp(ms / 1000, tz=UTC).astimezone(IST)
    hours = candle_pattern_period_hours(timeframe)
    ist_close = ist_open + timedelta(hours=hours)
    return f"{ist_open.strftime('%I:%M %p')} - {ist_close.strftime('%I:%M %p')}"


def find_kline_index_at_reference(klines, reference_time_ist, timeframe):
    """Index of the candle that closes at reference_time_ist."""
    ref = reference_time_ist.astimezone(IST)
    target_open = ref - timedelta(hours=candle_pattern_period_hours(timeframe))
    target_ms = int(target_open.timestamp() * 1000)
    for i, kline in enumerate(klines):
        if int(kline[0]) == target_ms:
            return i
    for i, kline in enumerate(klines):
        if abs(int(kline[0]) - target_ms) < 60_000:
            return i
    return None


def detect_bullish_low_sweep(prev_kline, cur_kline, require_second_bullish=False):
    """
    Bullish low-sweep reclaim (2-candle pattern):
    1) Previous candle is bullish (close > open)
    2) Current low sweeps below previous low
    3) Current close is above previous close
  """
    prev_open = float(prev_kline[1])
    prev_low = float(prev_kline[3])
    prev_close = float(prev_kline[4])
    cur_open = float(cur_kline[1])
    cur_low = float(cur_kline[3])
    cur_close = float(cur_kline[4])

    if prev_close <= prev_open:
        return False
    if cur_low >= prev_low:
        return False
    if cur_close <= prev_close:
        return False
    if require_second_bullish and cur_close <= cur_open:
        return False
    return True


def detect_bearish_high_sweep(prev_kline, cur_kline, require_second_bearish=False):
    """
    Bearish high-sweep reclaim (2-candle pattern) for SELL entries:
    1) Previous candle is bearish (close < open)
    2) Current high sweeps above previous high
    3) Current close is below previous close
    """
    prev_open = float(prev_kline[1])
    prev_high = float(prev_kline[2])
    prev_close = float(prev_kline[4])
    cur_open = float(cur_kline[1])
    cur_high = float(cur_kline[2])
    cur_close = float(cur_kline[4])

    if prev_close >= prev_open:
        return False
    if cur_high <= prev_high:
        return False
    if cur_close >= prev_close:
        return False
    if require_second_bearish and cur_close >= cur_open:
        return False
    return True


def _candle_pattern_kline_limit(timeframe, historical=False):
    if historical:
        return {"Monthly": 60, "Weekly": 120, "Daily": 500, "4H": 500, "1H": 500}.get(timeframe, 500)
    return {"Monthly": 8, "Weekly": 12, "Daily": 30, "4H": 50, "1H": 50}.get(timeframe, 50)


def _candle_pattern_lookback_days(timeframe):
    return {"1H": 14, "4H": 45, "Daily": 120, "Weekly": 365, "Monthly": 900}.get(timeframe, 60)


def _forward_close_return_pct(klines, idx, periods):
    if idx + periods >= len(klines):
        return None
    entry = float(klines[idx][4])
    if entry <= 0:
        return None
    exit_close = float(klines[idx + periods][4])
    return round((exit_close / entry - 1) * 100, 2)


def build_bullish_low_sweep_row(
    symbol,
    prev_kline,
    cur_kline,
    timeframe,
    scan_mode="Live",
    klines=None,
    cur_index=None,
):
    prev_open = float(prev_kline[1])
    prev_high = float(prev_kline[2])
    prev_low = float(prev_kline[3])
    prev_close = float(prev_kline[4])
    cur_open = float(cur_kline[1])
    cur_high = float(cur_kline[2])
    cur_low = float(cur_kline[3])
    cur_close = float(cur_kline[4])
    _, prev_usdt_vol = get_candle_volumes(prev_kline)
    _, cur_usdt_vol = get_candle_volumes(cur_kline)

    sweep_pct = ((prev_low - cur_low) / prev_low * 100) if prev_low > 0 else 0.0
    close_ext_pct = ((cur_close - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
    cur_body_pct = ((cur_close - cur_open) / cur_open * 100) if cur_open > 0 else 0.0
    vol_ratio = (cur_usdt_vol / prev_usdt_vol) if prev_usdt_vol > 0 else None

    candle_time = datetime.fromtimestamp(int(cur_kline[0]) / 1000, tz=UTC).astimezone(IST)
    row = {
        "Coin": f"{symbol}.P",
        "Timeframe": timeframe,
        "Pattern Date": candle_time.strftime("%Y-%m-%d %H:%M"),
        "Prev Open": round(prev_open, 6),
        "Prev Close": round(prev_close, 6),
        "Prev Low": round(prev_low, 6),
        "Cur Open": round(cur_open, 6),
        "Cur Close": round(cur_close, 6),
        "Cur Low": round(cur_low, 6),
        "Sweep Below Low %": round(sweep_pct, 2),
        "Close Above Prev %": round(close_ext_pct, 2),
        "Candle 2 Body %": round(cur_body_pct, 2),
        "Candle 2 Type": "Bullish" if cur_close > cur_open else "Bearish",
        "USDT Volume": cur_usdt_vol,
        "USDT Vol Display": format_large_number(cur_usdt_vol),
        "Vol Ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
        "Close > Prev High": "Yes" if cur_close > prev_high else "No",
        "Spot Match": "Yes" if check_spot_match_fast(symbol) else "No",
        "Scan Mode": scan_mode,
    }
    if timeframe in ("4H", "1H"):
        row["IST Slot"] = get_candle_slot_label_ist(cur_kline[0], timeframe)

    if klines is not None and cur_index is not None:
        row["Fwd +1 %"] = _forward_close_return_pct(klines, cur_index, 1)
        row["Fwd +3 %"] = _forward_close_return_pct(klines, cur_index, 3)
        row["Fwd +5 %"] = _forward_close_return_pct(klines, cur_index, 5)

    return row


def build_bearish_high_sweep_row(
    symbol,
    prev_kline,
    cur_kline,
    timeframe,
    scan_mode="Live",
    klines=None,
    cur_index=None,
):
    prev_open = float(prev_kline[1])
    prev_high = float(prev_kline[2])
    prev_low = float(prev_kline[3])
    prev_close = float(prev_kline[4])
    cur_open = float(cur_kline[1])
    cur_high = float(cur_kline[2])
    cur_low = float(cur_kline[3])
    cur_close = float(cur_kline[4])
    _, prev_usdt_vol = get_candle_volumes(prev_kline)
    _, cur_usdt_vol = get_candle_volumes(cur_kline)

    sweep_pct = ((cur_high - prev_high) / prev_high * 100) if prev_high > 0 else 0.0
    close_ext_pct = ((prev_close - cur_close) / prev_close * 100) if prev_close > 0 else 0.0
    cur_body_pct = ((cur_open - cur_close) / cur_open * 100) if cur_open > 0 else 0.0
    vol_ratio = (cur_usdt_vol / prev_usdt_vol) if prev_usdt_vol > 0 else None

    candle_time = datetime.fromtimestamp(int(cur_kline[0]) / 1000, tz=UTC).astimezone(IST)
    row = {
        "Coin": f"{symbol}.P",
        "Timeframe": timeframe,
        "Pattern Date": candle_time.strftime("%Y-%m-%d %H:%M"),
        "Prev Open": round(prev_open, 6),
        "Prev Close": round(prev_close, 6),
        "Prev High": round(prev_high, 6),
        "Cur Open": round(cur_open, 6),
        "Cur Close": round(cur_close, 6),
        "Cur High": round(cur_high, 6),
        "Sweep Above High %": round(sweep_pct, 2),
        "Close Below Prev %": round(close_ext_pct, 2),
        "Candle 2 Body %": round(cur_body_pct, 2),
        "Candle 2 Type": "Bearish" if cur_close < cur_open else "Bullish",
        "USDT Volume": cur_usdt_vol,
        "USDT Vol Display": format_large_number(cur_usdt_vol),
        "Vol Ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
        "Close < Prev Low": "Yes" if cur_close < prev_low else "No",
        "Spot Match": "Yes" if check_spot_match_fast(symbol) else "No",
        "Scan Mode": scan_mode,
    }
    if timeframe in ("4H", "1H"):
        row["IST Slot"] = get_candle_slot_label_ist(cur_kline[0], timeframe)

    if klines is not None and cur_index is not None:
        row["Fwd +1 %"] = _forward_close_return_pct(klines, cur_index, 1)
        row["Fwd +3 %"] = _forward_close_return_pct(klines, cur_index, 3)
        row["Fwd +5 %"] = _forward_close_return_pct(klines, cur_index, 5)

    return row


def fetch_bullish_low_sweep_live_worker(
    symbol,
    timeframe,
    require_second_bullish=False,
    backtest_date=None,
    reference_time=None,
    min_usdt_volume=0,
):
    """Scan the latest completed 2-candle window for the bullish low-sweep pattern."""
    try:
        interval = TIMEFRAME_MAP[timeframe]
        if reference_time is not None:
            end_dt = reference_time.astimezone(IST)
            start_dt = end_dt - timedelta(days=_candle_pattern_lookback_days(timeframe))
            klines = get_historical_klines(
                symbol, interval, start_dt, end_dt, limit=_candle_pattern_kline_limit(timeframe, True)
            )
            scan_mode = "Backtest"
        elif backtest_date:
            end_dt = datetime.combine(backtest_date, datetime.max.time(), tzinfo=IST)
            start_dt = end_dt - timedelta(days=_candle_pattern_lookback_days(timeframe))
            klines = get_historical_klines(
                symbol, interval, start_dt, end_dt, limit=_candle_pattern_kline_limit(timeframe, True)
            )
            scan_mode = "Backtest"
        else:
            klines = get_historical_klines(
                symbol, interval, limit=_candle_pattern_kline_limit(timeframe, False)
            )
            scan_mode = "Live"

        if not klines or len(klines) < 4:
            return None

        if reference_time is not None:
            cur_idx = find_kline_index_at_reference(klines, reference_time, timeframe)
            if cur_idx is None or cur_idx < 1:
                return None
            cur_kline = klines[cur_idx]
            prev_kline = klines[cur_idx - 1]
        else:
            cur_kline = klines[-2]
            prev_kline = klines[-3]

        if not detect_bullish_low_sweep(prev_kline, cur_kline, require_second_bullish):
            return None

        _, cur_usdt_vol = get_candle_volumes(cur_kline)
        if cur_usdt_vol < min_usdt_volume:
            return None

        row = build_bullish_low_sweep_row(
            symbol, prev_kline, cur_kline, timeframe, scan_mode=scan_mode
        )
        if reference_time is not None:
            row["IST Slot"] = reference_time.astimezone(IST).strftime("%Y-%m-%d %I:%M %p IST")
        return row
    except Exception:
        return None


def scan_bullish_low_sweep_history(
    symbol,
    timeframe,
    start_date,
    end_date,
    require_second_bullish=False,
    min_usdt_volume=0,
    selected_slot=None,
):
    """Find every bullish low-sweep occurrence for one symbol inside a date range."""
    try:
        interval = TIMEFRAME_MAP[timeframe]
        start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=IST)
        end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=IST)
        klines = get_historical_klines(
            symbol, interval, start_dt, end_dt, limit=_candle_pattern_kline_limit(timeframe, True)
        )
        if not klines or len(klines) < 3:
            return []

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        hits = []
        for i in range(2, len(klines)):
            prev_kline = klines[i - 1]
            cur_kline = klines[i]
            cur_open_ms = int(cur_kline[0])
            if cur_open_ms < start_ms or cur_open_ms > end_ms:
                continue
            if selected_slot and selected_slot != "All slots":
                if get_candle_slot_label_ist(cur_open_ms, timeframe) != selected_slot:
                    continue
            if not detect_bullish_low_sweep(prev_kline, cur_kline, require_second_bullish):
                continue
            _, cur_usdt_vol = get_candle_volumes(cur_kline)
            if cur_usdt_vol < min_usdt_volume:
                continue
            hits.append(
                build_bullish_low_sweep_row(
                    symbol,
                    prev_kline,
                    cur_kline,
                    timeframe,
                    scan_mode="Backtest",
                    klines=klines,
                    cur_index=i,
                )
            )
        return hits
    except Exception:
        return []


def scan_bullish_low_sweep_parallel(
    symbols,
    timeframe,
    require_second_bullish=False,
    backtest_date=None,
    reference_time=None,
    min_usdt_volume=0,
    max_workers=25,
):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(symbols)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                fetch_bullish_low_sweep_live_worker,
                symbol,
                timeframe,
                require_second_bullish,
                backtest_date,
                reference_time,
                min_usdt_volume,
            ): symbol
            for symbol in symbols
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 10 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(
                    f"Pattern scan {completed}/{total}... matches: {len(results)}"
                )
            try:
                row = future.result(timeout=8)
                if row:
                    results.append(row)
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()
    return results


def scan_bullish_low_sweep_history_parallel(
    symbols,
    timeframe,
    start_date,
    end_date,
    require_second_bullish=False,
    min_usdt_volume=0,
    max_workers=25,
    selected_slot=None,
):
    all_rows = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(symbols)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                scan_bullish_low_sweep_history,
                symbol,
                timeframe,
                start_date,
                end_date,
                require_second_bullish,
                min_usdt_volume,
                selected_slot,
            ): symbol
            for symbol in symbols
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 10 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(
                    f"Backtest scan {completed}/{total}... hits: {len(all_rows)}"
                )
            try:
                rows = future.result(timeout=12) or []
                all_rows.extend(rows)
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()
    return all_rows


# --- 5m entry strategy backtest (HTF setup + LTF entry) ---

LTF_ENTRY_INTERVAL = "5m"


def _format_trade_duration(total_minutes):
    """Format duration in minutes into a human-readable string like '5 min', '1 hr', '1 hr 15 min', '1d 2 hr'."""
    if total_minutes is None:
        return None
    try:
        if isinstance(total_minutes, (float, np.floating)) and (math.isnan(total_minutes) or np.isnan(total_minutes)):
            return None
        total_minutes = int(round(float(total_minutes)))
    except (ValueError, TypeError):
        return None
    if total_minutes <= 0:
        return "0 min"

    days = total_minutes // 1440
    rem = total_minutes % 1440
    hours = rem // 60
    minutes = rem % 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours} hr")
    if minutes > 0 or (days == 0 and hours == 0):
        parts.append(f"{minutes} min")

    return " ".join(parts)


def _htf_window_bounds(cur_kline, timeframe):
    """IST open/close of the HTF candle 2 window."""
    open_dt = datetime.fromtimestamp(int(cur_kline[0]) / 1000, tz=UTC)
    close_dt = open_dt + timedelta(hours=candle_pattern_period_hours(timeframe))
    return open_dt, close_dt


def _fetch_5m_klines(symbol, start_dt, end_dt, limit=1500):
    return get_historical_klines(symbol, LTF_ENTRY_INTERVAL, start_dt, end_dt, limit=limit) or []


def _simulate_5m_entry_and_trade(
    prev_htf,
    cur_htf,
    m5_window,
    m5_after,
    rr_ratio,
    market_near_pct,
):
    """
    5m logic inside HTF candle 2:
    - Sweep prev HTF low must occur on 5m before entry
    - Entry when 5m close > prev HTF close OR > cur HTF open
    - SL = cur HTF low | TP = entry + RR * (entry - SL)
  """
    prev_close = float(prev_htf[4])
    prev_low = float(prev_htf[3])
    cur_open = float(cur_htf[1])
    cur_low = float(cur_htf[3])

    if not m5_window:
        return None

    swept = False
    signal_idx = None
    for i, k in enumerate(m5_window):
        low = float(k[3])
        close = float(k[4])
        if low < prev_low:
            swept = True
        if not swept:
            continue
        if close > prev_close or close > cur_open:
            signal_idx = i
            break

    if signal_idx is None:
        return {"status": "No 5m entry", "reason": "No reclaim close above C1 close / C2 open after sweep"}

    sig = m5_window[signal_idx]
    sig_close = float(sig[4])
    dist_pct = ((sig_close - prev_close) / prev_close * 100) if prev_close > 0 else 0
    order_type = "Market" if dist_pct <= market_near_pct else "Limit"
    limit_price = prev_close

    entry_price = None
    entry_idx = None
    entry_source = None

    if order_type == "Market":
        entry_price = sig_close
        entry_idx = signal_idx
        entry_source = "5m close"
    else:
        for j in range(signal_idx, len(m5_window)):
            if float(m5_window[j][3]) <= limit_price:
                entry_price = limit_price
                entry_idx = j
                entry_source = "Limit fill"
                break
        if entry_price is None and m5_after:
            for j, k in enumerate(m5_after):
                if float(k[3]) <= limit_price:
                    entry_price = limit_price
                    entry_idx = len(m5_window) + j
                    entry_source = "Limit fill (after C2)"
                    break

    if entry_price is None:
        return {
            "status": "No fill",
            "order_type": order_type,
            "reason": "Limit not touched",
            "signal_5m_time": datetime.fromtimestamp(int(sig[0]) / 1000, tz=UTC).astimezone(IST).strftime(
                "%Y-%m-%d %H:%M"
            ),
        }

    risk = entry_price - cur_low
    if risk <= 0:
        return {"status": "Invalid", "reason": "Entry below/equal SL"}

    tp = entry_price + rr_ratio * risk
    sl = cur_low

    all_m5 = m5_window + m5_after
    trail = all_m5[entry_idx + 1 :]

    outcome = "Open"
    exit_price = None
    bars_held = 0
    exit_trail_idx = None
    for ti, k in enumerate(trail):
        bars_held += 1
        low = float(k[3])
        high = float(k[2])
        if low <= sl:
            outcome = "SL"
            exit_price = sl
            exit_trail_idx = ti
            break
        if high >= tp:
            outcome = "TP"
            exit_price = tp
            exit_trail_idx = ti
            break

    pnl_pct = ((exit_price - entry_price) / entry_price * 100) if exit_price else None
    r_mult = ((exit_price - entry_price) / risk) if exit_price and risk else None

    post_tp_extra_r = None
    post_tp_total_r = None
    post_tp_bars_to_sl = None
    if outcome == "TP" and exit_trail_idx is not None:
        after_tp = trail[exit_trail_idx + 1 :]
        peak_high = tp
        bars_to_sl = 0
        sl_reached = False
        for k in after_tp:
            bars_to_sl += 1
            high = float(k[2])
            low = float(k[3])
            if high > peak_high:
                peak_high = high
            if low <= sl:
                sl_reached = True
                break
        post_tp_extra_r = round((peak_high - tp) / risk, 2) if peak_high > tp else 0.0
        post_tp_total_r = round((peak_high - entry_price) / risk, 2)
        post_tp_bars_to_sl = bars_to_sl if after_tp else 0
        if not sl_reached and after_tp:
            post_tp_bars_to_sl = None  # SL not hit within forward window

    sig_time = datetime.fromtimestamp(int(sig[0]) / 1000, tz=UTC).astimezone(IST)
    entry_time = datetime.fromtimestamp(int(all_m5[entry_idx][0]) / 1000, tz=UTC).astimezone(IST)

    close_time = None
    duration_str = None
    duration_minutes = None
    if exit_trail_idx is not None and exit_trail_idx < len(trail):
        exit_kline = trail[exit_trail_idx]
        exit_dt = datetime.fromtimestamp(int(exit_kline[0]) / 1000, tz=UTC).astimezone(IST)
        close_time = exit_dt.strftime("%Y-%m-%d %H:%M")
        diff_mins = int(round((exit_dt - entry_time).total_seconds() / 60))
        duration_minutes = diff_mins if diff_mins > 0 else (bars_held * 5)
        duration_str = _format_trade_duration(duration_minutes)

    return {
        "status": "Trade",
        "outcome": outcome,
        "order_type": order_type,
        "entry_source": entry_source,
        "entry_price": round(entry_price, 6),
        "sl": round(sl, 6),
        "tp": round(tp, 6),
        "risk_pct": round(risk / entry_price * 100, 3),
        "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
        "r_multiple": round(r_mult, 2) if r_mult is not None else None,
        "bars_held_5m": bars_held,
        "post_tp_extra_r": post_tp_extra_r,
        "post_tp_total_r": post_tp_total_r,
        "post_tp_bars_to_sl": post_tp_bars_to_sl,
        "signal_5m_time": sig_time.strftime("%Y-%m-%d %H:%M"),
        "entry_time": entry_time.strftime("%Y-%m-%d %H:%M"),
        "close_time": close_time,
        "duration": duration_str,
        "duration_minutes": duration_minutes,
        "reclaim_dist_pct": round(dist_pct, 3),
    }


def _simulate_5m_entry_and_trade_short(
    prev_htf,
    cur_htf,
    m5_window,
    m5_after,
    rr_ratio,
    market_near_pct,
):
    """
    5m logic inside HTF candle 2 for SELL entries:
    - Sweep prev HTF high must occur on 5m before entry
    - Entry when 5m close < prev HTF close OR < cur HTF open
    - SL = cur HTF high | TP = entry - RR * (SL - entry)
    """
    prev_close = float(prev_htf[4])
    prev_high = float(prev_htf[2])
    cur_open = float(cur_htf[1])
    cur_high = float(cur_htf[2])

    if not m5_window:
        return None

    swept = False
    signal_idx = None
    for i, k in enumerate(m5_window):
        high = float(k[2])
        close = float(k[4])
        if high > prev_high:
            swept = True
        if not swept:
            continue
        if close < prev_close or close < cur_open:
            signal_idx = i
            break

    if signal_idx is None:
        return {"status": "No 5m entry", "reason": "No reclaim close below C1 close / C2 open after sweep"}

    sig = m5_window[signal_idx]
    sig_close = float(sig[4])
    dist_pct = ((prev_close - sig_close) / prev_close * 100) if prev_close > 0 else 0
    order_type = "Market" if dist_pct <= market_near_pct else "Limit"
    limit_price = prev_close

    entry_price = None
    entry_idx = None
    entry_source = None

    if order_type == "Market":
        entry_price = sig_close
        entry_idx = signal_idx
        entry_source = "5m close"
    else:
        for j in range(signal_idx, len(m5_window)):
            if float(m5_window[j][2]) >= limit_price:
                entry_price = limit_price
                entry_idx = j
                entry_source = "Limit fill"
                break
        if entry_price is None and m5_after:
            for j, k in enumerate(m5_after):
                if float(k[2]) >= limit_price:
                    entry_price = limit_price
                    entry_idx = len(m5_window) + j
                    entry_source = "Limit fill (after C2)"
                    break

    if entry_price is None:
        return {
            "status": "No fill",
            "order_type": order_type,
            "reason": "Limit not touched",
            "signal_5m_time": datetime.fromtimestamp(int(sig[0]) / 1000, tz=UTC).astimezone(IST).strftime(
                "%Y-%m-%d %H:%M"
            ),
        }

    risk = cur_high - entry_price
    if risk <= 0:
        return {"status": "Invalid", "reason": "Entry above/equal SL"}

    tp = entry_price - rr_ratio * risk
    sl = cur_high

    all_m5 = m5_window + m5_after
    trail = all_m5[entry_idx + 1 :]

    outcome = "Open"
    exit_price = None
    bars_held = 0
    exit_trail_idx = None
    for ti, k in enumerate(trail):
        bars_held += 1
        low = float(k[3])
        high = float(k[2])
        if high >= sl:
            outcome = "SL"
            exit_price = sl
            exit_trail_idx = ti
            break
        if low <= tp:
            outcome = "TP"
            exit_price = tp
            exit_trail_idx = ti
            break

    pnl_pct = ((entry_price - exit_price) / entry_price * 100) if exit_price else None
    r_mult = ((entry_price - exit_price) / risk) if exit_price and risk else None

    post_tp_extra_r = None
    post_tp_total_r = None
    post_tp_bars_to_sl = None
    if outcome == "TP" and exit_trail_idx is not None:
        after_tp = trail[exit_trail_idx + 1 :]
        peak_low = tp
        bars_to_sl = 0
        sl_reached = False
        for k in after_tp:
            bars_to_sl += 1
            high = float(k[2])
            low = float(k[3])
            if low < peak_low:
                peak_low = low
            if high >= sl:
                sl_reached = True
                break
        post_tp_extra_r = round((tp - peak_low) / risk, 2) if peak_low < tp else 0.0
        post_tp_total_r = round((entry_price - peak_low) / risk, 2)
        post_tp_bars_to_sl = bars_to_sl if after_tp else 0
        if not sl_reached and after_tp:
            post_tp_bars_to_sl = None

    sig_time = datetime.fromtimestamp(int(sig[0]) / 1000, tz=UTC).astimezone(IST)
    entry_time = datetime.fromtimestamp(int(all_m5[entry_idx][0]) / 1000, tz=UTC).astimezone(IST)

    close_time = None
    duration_str = None
    duration_minutes = None
    if exit_trail_idx is not None and exit_trail_idx < len(trail):
        exit_kline = trail[exit_trail_idx]
        exit_dt = datetime.fromtimestamp(int(exit_kline[0]) / 1000, tz=UTC).astimezone(IST)
        close_time = exit_dt.strftime("%Y-%m-%d %H:%M")
        diff_mins = int(round((exit_dt - entry_time).total_seconds() / 60))
        duration_minutes = diff_mins if diff_mins > 0 else (bars_held * 5)
        duration_str = _format_trade_duration(duration_minutes)

    return {
        "status": "Trade",
        "outcome": outcome,
        "order_type": order_type,
        "entry_source": entry_source,
        "entry_price": round(entry_price, 6),
        "sl": round(sl, 6),
        "tp": round(tp, 6),
        "risk_pct": round(risk / entry_price * 100, 3),
        "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
        "r_multiple": round(r_mult, 2) if r_mult is not None else None,
        "bars_held_5m": bars_held,
        "post_tp_extra_r": post_tp_extra_r,
        "post_tp_total_r": post_tp_total_r,
        "post_tp_bars_to_sl": post_tp_bars_to_sl,
        "signal_5m_time": sig_time.strftime("%Y-%m-%d %H:%M"),
        "entry_time": entry_time.strftime("%Y-%m-%d %H:%M"),
        "close_time": close_time,
        "duration": duration_str,
        "duration_minutes": duration_minutes,
        "reclaim_dist_pct": round(dist_pct, 3),
    }


def backtest_sweep_strategy_setup(
    symbol,
    prev_htf,
    cur_htf,
    timeframe,
    rr_ratio=2.0,
    market_near_pct=0.5,
    require_second_bullish=False,
    entry_type="Buy",
):
    """Run full HTF + 5m strategy backtest for one setup."""
    if entry_type == "Buy":
        if not detect_bullish_low_sweep(prev_htf, cur_htf, require_second_bullish):
            return None
        open_dt, close_dt = _htf_window_bounds(cur_htf, timeframe)
        m5_window = _fetch_5m_klines(symbol, open_dt, close_dt)
        forward_end = close_dt + timedelta(hours=72)
        m5_after = _fetch_5m_klines(symbol, close_dt, forward_end)
        sim = _simulate_5m_entry_and_trade(
            prev_htf, cur_htf, m5_window, m5_after, rr_ratio, market_near_pct
        )
        if not sim:
            return None
        base = build_bullish_low_sweep_row(symbol, prev_htf, cur_htf, timeframe, scan_mode="Strategy")
        base["Entry Type"] = "Buy"
    elif entry_type == "Sell":
        if not detect_bearish_high_sweep(prev_htf, cur_htf, require_second_bullish):
            return None
        open_dt, close_dt = _htf_window_bounds(cur_htf, timeframe)
        m5_window = _fetch_5m_klines(symbol, open_dt, close_dt)
        forward_end = close_dt + timedelta(hours=72)
        m5_after = _fetch_5m_klines(symbol, close_dt, forward_end)
        sim = _simulate_5m_entry_and_trade_short(
            prev_htf, cur_htf, m5_window, m5_after, rr_ratio, market_near_pct
        )
        if not sim:
            return None
        base = build_bearish_high_sweep_row(symbol, prev_htf, cur_htf, timeframe, scan_mode="Strategy")
        base["Entry Type"] = "Sell"
    else:
        return None

    base.update(sim)
    return base


def scan_strategy_backtest_history(
    symbol,
    timeframe,
    start_date,
    end_date,
    rr_ratio=2.0,
    market_near_pct=0.5,
    require_second_bullish=False,
    min_usdt_volume=0,
    selected_slot=None,
    entry_type="Buy",
):
    interval = TIMEFRAME_MAP[timeframe]
    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=IST)
    end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=IST)
    klines = get_historical_klines(
        symbol, interval, start_dt, end_dt, limit=_candle_pattern_kline_limit(timeframe, True)
    )
    if not klines or len(klines) < 3:
        return []

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    rows = []
    for i in range(2, len(klines)):
        prev_k = klines[i - 1]
        cur_k = klines[i]
        if int(cur_k[0]) < start_ms or int(cur_k[0]) > end_ms:
            continue
        if selected_slot and selected_slot != "All slots":
            if get_candle_slot_label_ist(cur_k[0], timeframe) != selected_slot:
                continue
        _, vol = get_candle_volumes(cur_k)
        if vol < min_usdt_volume:
            continue
        types_to_check = ["Buy", "Sell"] if entry_type == "Both" else [entry_type]
        for et in types_to_check:
            row = backtest_sweep_strategy_setup(
                symbol, prev_k, cur_k, timeframe, rr_ratio, market_near_pct, require_second_bullish, entry_type=et
            )
            if row:
                rows.append(row)
    return rows


def fetch_strategy_backtest_date_worker(
    symbol, timeframe, backtest_date, rr_ratio, market_near_pct,
    require_second_bullish, min_usdt_volume, entry_type="Buy",
):
    try:
        interval = TIMEFRAME_MAP[timeframe]
        end_dt = datetime.combine(backtest_date, datetime.max.time(), tzinfo=IST)
        start_dt = end_dt - timedelta(days=_candle_pattern_lookback_days(timeframe))
        klines = get_historical_klines(
            symbol, interval, start_dt, end_dt, limit=_candle_pattern_kline_limit(timeframe, True)
        )
        if not klines or len(klines) < 4:
            return None
        cur_k = klines[-2]
        prev_k = klines[-3]
        _, vol = get_candle_volumes(cur_k)
        if vol < min_usdt_volume:
            return None
        types_to_check = ["Buy", "Sell"] if entry_type == "Both" else [entry_type]
        for et in types_to_check:
            row = backtest_sweep_strategy_setup(
                symbol, prev_k, cur_k, timeframe, rr_ratio, market_near_pct, require_second_bullish, entry_type=et
            )
            if row:
                return row
        return None
    except Exception:
        return None


def fetch_strategy_backtest_reference_worker(
    symbol,
    timeframe,
    reference_time,
    rr_ratio,
    market_near_pct,
    require_second_bullish,
    min_usdt_volume,
    entry_type="Buy",
):
    try:
        interval = TIMEFRAME_MAP[timeframe]
        end_dt = reference_time.astimezone(IST)
        start_dt = end_dt - timedelta(days=_candle_pattern_lookback_days(timeframe))
        klines = get_historical_klines(
            symbol, interval, start_dt, end_dt, limit=_candle_pattern_kline_limit(timeframe, True)
        )
        if not klines:
            return None
        cur_idx = find_kline_index_at_reference(klines, reference_time, timeframe)
        if cur_idx is None or cur_idx < 1:
            return None
        cur_k = klines[cur_idx]
        prev_k = klines[cur_idx - 1]
        _, vol = get_candle_volumes(cur_k)
        if vol < min_usdt_volume:
            return None
        types_to_check = ["Buy", "Sell"] if entry_type == "Both" else [entry_type]
        for et in types_to_check:
            row = backtest_sweep_strategy_setup(
                symbol, prev_k, cur_k, timeframe, rr_ratio, market_near_pct, require_second_bullish, entry_type=et
            )
            if row:
                row["IST Slot"] = reference_time.astimezone(IST).strftime("%Y-%m-%d %I:%M %p IST")
                return row
        return None
    except Exception:
        return None


def scan_strategy_backtest_parallel(
    symbols,
    timeframe,
    start_date=None,
    end_date=None,
    backtest_date=None,
    reference_time=None,
    rr_ratio=2.0,
    market_near_pct=0.5,
    require_second_bullish=False,
    min_usdt_volume=0,
    max_workers=20,
    selected_slot=None,
    entry_type="Buy",
):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(symbols)

    def _worker(sym):
        if reference_time is not None:
            return fetch_strategy_backtest_reference_worker(
                sym, timeframe, reference_time, rr_ratio, market_near_pct,
                require_second_bullish, min_usdt_volume, entry_type=entry_type,
            )
        if backtest_date is not None:
            return fetch_strategy_backtest_date_worker(
                sym, timeframe, backtest_date, rr_ratio, market_near_pct,
                require_second_bullish, min_usdt_volume, entry_type=entry_type,
            )
        rows = scan_strategy_backtest_history(
            sym, timeframe, start_date, end_date, rr_ratio, market_near_pct,
            require_second_bullish, min_usdt_volume,
            selected_slot=selected_slot, entry_type=entry_type,
        )
        return rows if rows else None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, s): s for s in symbols}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 5 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(f"Strategy backtest {completed}/{total}... trades: {len(results)}")
            try:
                row = future.result(timeout=20)
                if row:
                    if isinstance(row, list):
                        results.extend(row)
                    else:
                        results.append(row)
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()
    return results


def _strategy_trades_chronological(closed_trades):
    """Sort closed trades by entry time for equity-curve stats."""
    df = closed_trades.copy()
    if "entry_time" in df.columns:
        df["_sort_ts"] = pd.to_datetime(df["entry_time"], format="%Y-%m-%d %H:%M", errors="coerce")
    elif "Pattern Date" in df.columns:
        df["_sort_ts"] = pd.to_datetime(df["Pattern Date"], errors="coerce")
    else:
        df["_sort_ts"] = pd.RangeIndex(len(df))
    return df.sort_values("_sort_ts").reset_index(drop=True)


def _compute_strategy_summary_stats(closed_trades):
    """Cumulative R equity curve, max drawdown, expectancy, profit factor."""
    if closed_trades.empty or "r_multiple" not in closed_trades.columns:
        return None

    df = _strategy_trades_chronological(closed_trades)
    r = pd.to_numeric(df["r_multiple"], errors="coerce").fillna(0.0)
    equity = r.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    wins = r[r > 0]
    losses = r[r < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = None

    return {
        "sorted_df": df,
        "equity_r": equity,
        "drawdown_r": drawdown,
        "max_dd_r": round(max_dd, 2),
        "expectancy_r": round(float(r.mean()), 3),
        "profit_factor": round(profit_factor, 2) if profit_factor not in (None, float("inf")) else profit_factor,
        "gross_profit_r": round(gross_profit, 2),
        "gross_loss_r": round(gross_loss, 2),
    }


def _render_strategy_equity_curve(stats):
    df = stats["sorted_df"]
    x_vals = df["entry_time"].tolist() if "entry_time" in df.columns else list(range(1, len(df) + 1))
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Cumulative R (1R risk per trade)", "Drawdown (R)"),
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=stats["equity_r"],
            mode="lines+markers",
            name="Equity (R)",
            line=dict(color="#22c55e", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=stats["drawdown_r"],
            mode="lines",
            name="Drawdown (R)",
            fill="tozeroy",
            line=dict(color="#ef4444", width=1),
        ),
        row=2,
        col=1,
    )
    fig.update_layout(height=480, showlegend=False, margin=dict(l=40, r=20, t=40, b=40))
    fig.update_yaxes(title_text="R", row=1, col=1)
    fig.update_yaxes(title_text="R", row=2, col=1)
    fig.update_xaxes(title_text="Entry time", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


def _render_strategy_backtest_results(df):
    trades = df[df["status"] == "Trade"] if "status" in df.columns else df
    others = df[df["status"] != "Trade"] if "status" in df.columns else pd.DataFrame()
    st.success(
        f"**{len(df)}** HTF setup(s) — **{len(trades)}** with 5m entry, "
        f"**{len(others)}** no entry / no fill."
    )

    if not trades.empty:
        closed = trades[trades["outcome"].isin(["TP", "SL"])]
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Closed trades", len(closed))
        with m2:
            if len(closed):
                wr = (closed["outcome"] == "TP").mean() * 100
                st.metric("Win rate", f"{wr:.1f}%")
            else:
                st.metric("Win rate", "—")
        with m3:
            if closed["r_multiple"].notna().any():
                st.metric("Avg R", f"{closed['r_multiple'].mean():.2f}")
            else:
                st.metric("Avg R", "—")
        with m4:
            mkt = (trades["order_type"] == "Market").sum()
            st.metric("Market / Limit", f"{mkt} / {len(trades) - mkt}")
        with m5:
            if "duration_minutes" in closed.columns and closed["duration_minutes"].notna().any():
                avg_dur = closed["duration_minutes"].dropna().mean()
                st.metric("Avg duration", _format_trade_duration(avg_dur))
            else:
                st.metric("Avg duration", "—")

        stats = _compute_strategy_summary_stats(closed)
        if stats and len(closed) >= 2:
            st.subheader("Summary stats")
            s1, s2, s3, s4, s5 = st.columns(5)
            with s1:
                pf = stats["profit_factor"]
                st.metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}" if pf is not None else "—")
            with s2:
                st.metric("Expectancy (R)", f"{stats['expectancy_r']:.2f}")
            with s3:
                st.metric("Max drawdown (R)", f"{stats['max_dd_r']:.2f}")
            with s4:
                st.metric("Gross profit (R)", f"{stats['gross_profit_r']:.2f}")
            with s5:
                st.metric("Gross loss (R)", f"{stats['gross_loss_r']:.2f}")
            _render_strategy_equity_curve(stats)

        tp_trades = closed[closed["outcome"] == "TP"]
        if not tp_trades.empty and "post_tp_extra_r" in tp_trades.columns:
            extra = pd.to_numeric(tp_trades["post_tp_extra_r"], errors="coerce").dropna()
            if not extra.empty:
                st.caption(
                    f"**Post TP extra R** (avg on winners): **{extra.mean():.2f}R** — "
                    "extra upside from TP until original SL is touched."
                )

        show_cols = [
            "Coin", "Timeframe", "Pattern Date", "IST Slot", "Entry Type", "outcome", "order_type",
            "entry_price", "sl", "tp", "r_multiple", "pnl_pct",
            "post_tp_extra_r", "post_tp_total_r", "post_tp_bars_to_sl",
            "signal_5m_time", "entry_time", "close_time", "duration",
            "reclaim_dist_pct", "Sweep Below Low %", "Sweep Above High %", "USDT Vol Display",
        ]
        show_cols = [c for c in show_cols if c in trades.columns]
        st.subheader("Trades")
        st.dataframe(
            trades[show_cols].rename(
                columns={
                    "post_tp_extra_r": "Post TP Extra R",
                    "post_tp_total_r": "Post TP Total R",
                    "post_tp_bars_to_sl": "Post TP Bars to SL",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        if len(closed) >= 3:
            st.subheader("Outcome breakdown")
            agg_dict = {
                "count": ("outcome", "count"),
                "avg_r": ("r_multiple", "mean"),
                "avg_pnl": ("pnl_pct", "mean"),
            }
            if "duration_minutes" in closed.columns and closed["duration_minutes"].notna().any():
                agg_dict["avg_duration_min"] = ("duration_minutes", "mean")
            summary = closed.groupby("outcome").agg(**agg_dict).reset_index()
            if "avg_duration_min" in summary.columns:
                summary["avg_duration"] = summary["avg_duration_min"].apply(
                    lambda m: _format_trade_duration(m) if pd.notna(m) else "—"
                )
            st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.warning("HTF patterns found but no 5m entries triggered.")

    if not others.empty:
        st.subheader("Setups without trade")
        fail_cols = [c for c in ["Coin", "Pattern Date", "status", "reason", "signal_5m_time", "order_type"] if c in others.columns]
        st.dataframe(others[fail_cols], use_container_width=True, hide_index=True)


def render_candle_pattern_scanner():
    st.header("🕯️ Candle Pattern — Bullish Low Sweep")
    st.markdown(
        """
        **HTF setup (2 candles):** C1 bullish → C2 sweeps C1 low → C2 close above C1 close (BUY).
        C1 bearish → C2 sweeps C1 high → C2 close below C1 close (SELL).

        **Strategy backtest (5m):** Inside C2 window, wait for 5m sweep → enter on reclaim → SL at C2 extreme → TP at selected R:R (1:2 to 1:10).
        Market if reclaim is tight; limit if price already extended.
        Entry type: Buy (long), Sell (short), or Both.
        """
    )

    mode = st.radio(
        "Scanner mode",
        [
            "Live multi-coin scan",
            "Backtest on date",
            "Backtest date range",
            "Strategy backtest (5m entry)",
        ],
        horizontal=True,
        key="candle_pattern_mode",
    )

    strategy_mode = mode == "Strategy backtest (5m entry)"
    rr_ratio = 2.0
    market_near_pct = 0.5
    entry_type = "Buy"
    if strategy_mode:
        s1, s2, s3 = st.columns(3)
        with s1:
            rr_choice = st.selectbox(
                "Take profit (R:R)",
                ["1:2", "1:3", "1:4", "1:5", "1:6", "1:7", "1:8", "1:9", "1:10"],
                index=0,
                key="strategy_rr",
            )
            rr_ratio = float(rr_choice.split(":")[1])
        with s2:
            market_near_pct = st.number_input(
                "Market if within % of C1 close",
                min_value=0.05,
                max_value=5.0,
                value=0.5,
                step=0.05,
                key="strategy_market_near",
                help="5m close within this % of C1 close → market. Above → limit at C1 close.",
            )
        with s3:
            entry_type = st.radio(
                "Entry type",
                ["Buy", "Sell", "Both"],
                index=0,
                key="strategy_entry_type",
                horizontal=True,
                help="Buy = long (low sweep), Sell = short (high sweep), Both = check both.",
            )
            st.caption("LTF entry uses **5m** candles inside HTF candle 2 window.")

    c1, c2 = st.columns(2)
    with c1:
        timeframe = st.selectbox(
            "Timeframe",
            CANDLE_PATTERN_TIMEFRAMES,
            index=2,
            key="candle_pattern_tf",
        )
    with c2:
        require_second_bullish = st.checkbox(
            "Require candle 2 same direction",
            value=True,
            key="candle_pattern_c2_bull",
            help="When checked, candle 2 must close in the same direction as candle 1 (bullish for Buy, bearish for Sell).",
        )

    all_symbols = get_perpetual_symbols()
    if not all_symbols:
        st.error("Failed to load symbols from Binance.")
        return

    st.caption(f"**{len(all_symbols)}** USDT perpetual pairs available on Binance.")

    f1, f2 = st.columns(2)
    with f1:
        use_min_vol = st.checkbox(
            "Apply min USDT volume filter (candle 2)",
            value=False,
            key="candle_pattern_use_min_vol",
        )
        min_usdt_volume = 0
        if use_min_vol:
            min_usdt_volume = st.number_input(
                "Min USDT volume (candle 2)",
                min_value=0,
                max_value=50_000_000,
                value=100_000,
                step=10_000,
                key="candle_pattern_min_vol",
            )
    with f2:
        limit_universe = st.checkbox(
            "Limit universe size (top by 24h volume)",
            value=False,
            key="candle_pattern_limit_universe",
            disabled=False,
        )
        scan_n = len(all_symbols)
        if limit_universe:
            scan_n = st.selectbox(
                "Universe size",
                [50, 100, 200, 300, 500],
                index=2,
                key="candle_pattern_scan_n",
            )

    parallel_workers = st.slider(
        "Parallel workers",
        min_value=5,
        max_value=40,
        value=25,
        key="candle_pattern_workers",
    )

    backtest_date = None
    reference_time = None
    selected_slot_label = None
    range_start = None
    range_end = None
    single_symbol = None
    strategy_scope = None

    if strategy_mode:
        strategy_scope = st.radio(
            "Strategy period",
            ["Single date (IST slot)", "Date range"],
            horizontal=True,
            key="strategy_scope",
        )

    use_single_date = mode == "Backtest on date" or (
        strategy_mode and strategy_scope == "Single date (IST slot)"
    )
    use_date_range = mode == "Backtest date range" or (
        strategy_mode and strategy_scope == "Date range"
    )

    if use_single_date:
        bt_col1, bt_col2 = st.columns(2)
        with bt_col1:
            backtest_date = st.date_input(
                "Backtest date (IST)",
                value=datetime.now(IST).date() - timedelta(days=1),
                key="candle_pattern_bt_date",
            )
        with bt_col2:
            if timeframe == "4H":
                selected_slot_label = st.selectbox(
                    "4H candle slot (IST)",
                    options=list(CANDLE_PATTERN_4H_SLOTS_IST.keys()),
                    index=1,
                    key="candle_pattern_4h_slot",
                )
            elif timeframe == "1H":
                selected_slot_label = st.selectbox(
                    "1H candle slot (IST)",
                    options=list(CANDLE_PATTERN_1H_SLOTS_IST.keys()),
                    index=0,
                    key="candle_pattern_1h_slot",
                )

        if timeframe in ("4H", "1H") and selected_slot_label:
            reference_time = candle_pattern_reference_time_ist(
                backtest_date, timeframe, selected_slot_label
            )
            msg = (
                f"**{selected_slot_label}** on **{backtest_date.strftime('%Y-%m-%d')}** "
                f"(closes **{reference_time.strftime('%I:%M %p IST')}**)"
            )
        else:
            msg = f"Last completed **{timeframe}** candle on **{backtest_date.strftime('%Y-%m-%d')}**"
        if strategy_mode:
            st.info(f"Strategy test: {msg} — 5m entry inside C2, SL = C2 extreme, Entry: {entry_type}.")
        else:
            st.info(f"Backtest: {msg}")

    elif use_date_range:
        if timeframe in ("4H", "1H"):
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                range_start = st.date_input(
                    "From",
                    value=datetime.now(IST).date() - timedelta(days=30),
                    key="candle_pattern_range_start",
                )
            with d2:
                range_end = st.date_input(
                    "To",
                    value=datetime.now(IST).date() - timedelta(days=1),
                    key="candle_pattern_range_end",
                )
            with d3:
                if timeframe == "4H":
                    selected_slot_label = st.selectbox(
                        "4H candle slot (IST)",
                        options=["All slots"] + list(CANDLE_PATTERN_4H_SLOTS_IST.keys()),
                        index=0,
                        key="candle_pattern_range_4h_slot",
                    )
                else:
                    selected_slot_label = st.selectbox(
                        "1H candle slot (IST)",
                        options=["All slots"] + list(CANDLE_PATTERN_1H_SLOTS_IST.keys()),
                        index=0,
                        key="candle_pattern_range_1h_slot",
                    )
            with d4:
                coin_options = [f"{s}.P" for s in all_symbols]
                single_pick = st.selectbox(
                    "Single coin (optional)",
                    ["All coins"] + coin_options,
                    index=0,
                    key="candle_pattern_single_coin",
                )
                if single_pick != "All coins":
                    single_symbol = single_pick.replace(".P", "")
        else:
            d1, d2, d3 = st.columns(3)
            with d1:
                range_start = st.date_input(
                    "From",
                    value=datetime.now(IST).date() - timedelta(days=30),
                    key="candle_pattern_range_start",
                )
            with d2:
                range_end = st.date_input(
                    "To",
                    value=datetime.now(IST).date() - timedelta(days=1),
                    key="candle_pattern_range_end",
                )
            with d3:
                coin_options = [f"{s}.P" for s in all_symbols]
                single_pick = st.selectbox(
                    "Single coin (optional)",
                    ["All coins"] + coin_options,
                    index=0,
                    key="candle_pattern_single_coin",
                )
                if single_pick != "All coins":
                    single_symbol = single_pick.replace(".P", "")

        slot_desc = (
            f" (**{selected_slot_label}**)"
            if selected_slot_label and selected_slot_label != "All slots"
            else ""
        )
        if strategy_mode:
            st.info(
                f"Tests every HTF setup in range{slot_desc} with **5m entry**, SL at C2 extreme, TP at selected R:R (1:2–1:10), Entry: {entry_type}."
            )
        else:
            st.info(
                f"Scans every completed candle in the range{slot_desc} and reports forward returns "
                "(+1 / +3 / +5 candles) after each match."
            )

    btn_label = {
        "Live multi-coin scan": "🔍 Scan live pattern",
        "Backtest on date": "🧪 Run backtest scan",
        "Backtest date range": "📊 Run range backtest",
        "Strategy backtest (5m entry)": "🎯 Run strategy backtest",
    }[mode]

    if st.button(btn_label, type="primary", use_container_width=True, key="candle_pattern_btn"):
        if use_date_range and range_start > range_end:
            st.error("Start date must be on or before end date.")
            return

        if use_date_range and single_symbol:
            symbols = [single_symbol]
        elif limit_universe:
            top_volume = get_top_volume_symbols_list(limit=len(all_symbols))
            ranked = [s for s in top_volume if s in set(all_symbols)]
            missing = [s for s in all_symbols if s not in set(ranked)]
            universe = ranked + sorted(missing)
            symbols = universe[: int(scan_n)]
        else:
            symbols = list(all_symbols)

        if strategy_mode:
            slot_info = f" [{selected_slot_label}]" if selected_slot_label and selected_slot_label != "All slots" else ""
            with st.spinner(f"Strategy backtest on {len(symbols)} symbols ({timeframe}{slot_info} + 5m)..."):
                if use_single_date:
                    rows = scan_strategy_backtest_parallel(
                        symbols, timeframe,
                        backtest_date=backtest_date if reference_time is None else None,
                        reference_time=reference_time,
                        rr_ratio=rr_ratio, market_near_pct=market_near_pct,
                        require_second_bullish=require_second_bullish,
                        min_usdt_volume=min_usdt_volume,
                        max_workers=min(parallel_workers, 20),
                        entry_type=entry_type,
                    )
                else:
                    rows = scan_strategy_backtest_parallel(
                        symbols, timeframe,
                        start_date=range_start, end_date=range_end,
                        rr_ratio=rr_ratio, market_near_pct=market_near_pct,
                        require_second_bullish=require_second_bullish,
                        min_usdt_volume=min_usdt_volume,
                        max_workers=min(parallel_workers, 20),
                        selected_slot=selected_slot_label,
                        entry_type=entry_type,
                    )
            if not rows:
                slot_warn = f" for slot {selected_slot_label}" if selected_slot_label and selected_slot_label != "All slots" else ""
                st.warning(f"No strategy results{slot_warn}. Try another date, coin, or timeframe.")
                return
            df = pd.DataFrame(rows)
            _render_strategy_backtest_results(df)
            st.download_button(
                "Download strategy results (CSV)",
                df.to_csv(index=False),
                file_name=f"strategy_backtest_{timeframe}_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="strategy_backtest_csv",
            )
            return

        with st.spinner(f"Scanning {len(symbols)} symbols on {timeframe}..."):
            if mode == "Live multi-coin scan":
                rows = scan_bullish_low_sweep_parallel(
                    symbols,
                    timeframe,
                    require_second_bullish,
                    None,
                    min_usdt_volume,
                    parallel_workers,
                )
            elif mode == "Backtest on date":
                rows = scan_bullish_low_sweep_parallel(
                    symbols,
                    timeframe,
                    require_second_bullish,
                    backtest_date if reference_time is None else None,
                    reference_time,
                    min_usdt_volume,
                    parallel_workers,
                )
            else:
                rows = scan_bullish_low_sweep_history_parallel(
                    symbols,
                    timeframe,
                    range_start,
                    range_end,
                    require_second_bullish,
                    min_usdt_volume,
                    parallel_workers,
                    selected_slot=selected_slot_label,
                )

        if not rows:
            st.warning("No matches found. Try another timeframe, date, or lower min volume.")
            return

        df = pd.DataFrame(rows)
        if mode == "Live multi-coin scan":
            df = df.sort_values(["Sweep Below Low %", "Close Above Prev %"], ascending=False)
        elif mode == "Backtest on date":
            df = df.sort_values("Close Above Prev %", ascending=False)
        else:
            df = df.sort_values("Pattern Date", ascending=False)

        st.success(f"Found **{len(df)}** pattern match(es).")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Matches", len(df))
        with m2:
            st.metric("Avg sweep %", f"{df['Sweep Below Low %'].mean():.2f}%")
        with m3:
            st.metric("Avg close ext %", f"{df['Close Above Prev %'].mean():.2f}%")
        with m4:
            if "Fwd +1 %" in df.columns and df["Fwd +1 %"].notna().any():
                st.metric("Avg fwd +1 candle", f"{df['Fwd +1 %'].dropna().mean():.2f}%")
            else:
                st.metric("Candle 2 bullish", f"{(df['Candle 2 Type'] == 'Bullish').sum()}/{len(df)}")

        display_cols = [
            "Coin",
            "Timeframe",
            "Pattern Date",
            "Prev Close",
            "Cur Close",
            "Sweep Below Low %",
            "Close Above Prev %",
            "Candle 2 Body %",
            "Candle 2 Type",
            "USDT Vol Display",
            "Vol Ratio",
            "Close > Prev High",
            "Spot Match",
            "Scan Mode",
        ]
        if "IST Slot" in df.columns:
            display_cols.insert(3, "IST Slot")
        if "Fwd +1 %" in df.columns:
            display_cols.extend(["Fwd +1 %", "Fwd +3 %", "Fwd +5 %"])

        show = df[display_cols].copy()
        show["Sweep Below Low %"] = show["Sweep Below Low %"].map(lambda x: f"{x:.2f}%")
        show["Close Above Prev %"] = show["Close Above Prev %"].map(lambda x: f"+{x:.2f}%")
        show["Candle 2 Body %"] = show["Candle 2 Body %"].map(lambda x: f"{x:+.2f}%")
        if "Fwd +1 %" in show.columns:
            for col in ["Fwd +1 %", "Fwd +3 %", "Fwd +5 %"]:
                show[col] = show[col].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")

        def color_pattern(val):
            if isinstance(val, str) and val.startswith("+"):
                try:
                    num = float(val.replace("%", "").replace("+", ""))
                    if num >= 5:
                        return "color: #00ff9d; font-weight: bold"
                    if num >= 1:
                        return "color: #ffd700; font-weight: bold"
                except Exception:
                    pass
            return ""

        styled = show.style.map(color_pattern, subset=["Close Above Prev %"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        if mode == "Backtest date range" and df["Fwd +1 %"].notna().any():
            st.subheader("Forward return summary")
            summary = []
            for col, label in [("Fwd +1 %", "+1 candle"), ("Fwd +3 %", "+3 candles"), ("Fwd +5 %", "+5 candles")]:
                s = df[col].dropna()
                if len(s):
                    summary.append({
                        "Horizon": label,
                        "Samples": len(s),
                        "Avg %": round(s.mean(), 2),
                        "Win rate %": round((s > 0).mean() * 100, 1),
                        "Best %": round(s.max(), 2),
                        "Worst %": round(s.min(), 2),
                    })
            if summary:
                st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        st.download_button(
            "Download results (CSV)",
            df.to_csv(index=False),
            file_name=f"bullish_low_sweep_{timeframe}_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="candle_pattern_csv",
        )

# ==============================================
# VOLUME FILTER SCANNER (Breakout tab)
# ==============================================

def classify_price_tier(price):
    if price > 1.0:
        return "> $1"
    if price >= 0.90:
        return "Near $1 (0.90-1.00)"
    if price >= 0.01:
        return "Mid (0.01-0.90)"
    if price >= 0.001:
        return "Small (0.001-0.01)"
    if price >= 0.0001:
        return "Micro (0.0001-0.001)"
    return "Sub-micro (<0.0001)"


def passes_volume_price_filters(
    price, exclude_above_1, use_price_range, min_price, max_price
):
    if exclude_above_1 and price > 1.0:
        return False
    if use_price_range and (price < min_price or price > max_price):
        return False
    return True


def fetch_single_volume_filter_worker(
    symbol,
    num_days,
    min_volume_usdt,
    filter_mode,
    exclude_above_1=False,
    use_price_range=False,
    min_price=0.0,
    max_price=1.0,
):
    """Check last N completed daily candles; return detail dict if matched."""
    try:
        limit = num_days + 3
        params = {"symbol": symbol, "interval": "1d", "limit": limit}
        response = api_get(f"{BINANCE_API_URL}/fapi/v1/klines", params=params, timeout=5)
        data = response.json()

        if not data or len(data) < num_days + 2:
            return None

        completed_candles = data[-(num_days + 1):-1]
        if len(completed_candles) < num_days:
            return None

        last_close = float(completed_candles[-1][4])
        if not passes_volume_price_filters(
            last_close, exclude_above_1, use_price_range, min_price, max_price
        ):
            return None

        daily_rows = []
        daily_volumes = []
        for candle in completed_candles:
            candle_dt = datetime.fromtimestamp(candle[0] / 1000, tz=UTC).astimezone(IST)
            _, usdt_volume = get_candle_volumes(candle)
            daily_volumes.append(usdt_volume)
            daily_rows.append(
                {
                    "date": candle_dt.strftime("%Y-%m-%d"),
                    "volume_m": round(usdt_volume / 1_000_000, 2),
                }
            )

        volume_pass = False
        if filter_mode == "Consistent Volume":
            volume_pass = all(v >= min_volume_usdt for v in daily_volumes)
        else:
            avg_volume = sum(daily_volumes) / len(daily_volumes)
            if filter_mode == "Average volume":
                volume_pass = avg_volume >= min_volume_usdt
            elif filter_mode == "Average + Consistent":
                volume_pass = (
                    avg_volume >= min_volume_usdt
                    and all(v >= min_volume_usdt for v in daily_volumes)
                )

        if not volume_pass:
            return None

        avg_volume_m = round(sum(daily_volumes) / len(daily_volumes) / 1_000_000, 2)
        return {
            "symbol": symbol,
            "price": last_close,
            "price_class": classify_price_tier(last_close),
            "avg_volume_m": avg_volume_m,
            "daily": daily_rows,
        }
    except Exception:
        return None


def build_volume_filter_display_df(results):
    """Wide table: coin info + one column per date with volume in M."""
    if not results:
        return pd.DataFrame()

    date_columns = []
    for row in results[0]["daily"]:
        date_columns.append(row["date"])

    display_rows = []
    for idx, row in enumerate(results, start=1):
        out = {
            "S.No": idx,
            "Coinname": f"{row['symbol']}.P",
            "Price ($)": row["price"],
            "Price Class": row["price_class"],
            "Avg Vol (M)": row["avg_volume_m"],
        }
        daily_by_date = {d["date"]: d["volume_m"] for d in row["daily"]}
        for date_str in date_columns:
            out[f"{date_str} (M)"] = daily_by_date.get(date_str, "")
        display_rows.append(out)

    return pd.DataFrame(display_rows)


def scan_volume_filter_parallel(
    symbols,
    num_days,
    min_volume_m,
    filter_mode,
    max_workers=25,
    exclude_above_1=False,
    use_price_range=False,
    min_price=0.0,
    max_price=1.0,
):
    """Parallel scan for coins meeting daily volume criteria."""
    min_volume_usdt = min_volume_m * 1_000_000
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(
                fetch_single_volume_filter_worker,
                symbol,
                num_days,
                min_volume_usdt,
                filter_mode,
                exclude_above_1,
                use_price_range,
                min_price,
                max_price,
            ): symbol
            for symbol in symbols
        }

        completed = 0
        total = len(symbols)

        for future in as_completed(future_to_symbol):
            completed += 1
            if completed % 10 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(
                    f"Volume scan {completed}/{total} pairs... Found: {len(results)} matches"
                )

            try:
                result = future.result(timeout=5)
                if result:
                    results.append(result)
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()
    results.sort(key=lambda x: x["symbol"])
    return results

# ==============================================
# CLEANED: TIMEFRAME BREAKOUT SCANNER (YOUR LOGIC)
# ==============================================

def fetch_single_timeframe_breakout_worker(symbol, past_timeframe, current_timeframe, scan_mode="both", reference_time=None):
    """
    Worker function for timeframe-based breakout scanning

    Logic:
    - Past Timeframe: Monthly, Weekly, Daily (OPEN price)
    - Current Timeframe: Weekly, Daily, 4H (CLOSE price) - ONLY MOST RECENT candle

    Bullish: Past candle is bearish (close < open) AND current close > past open
    Bearish: Past candle is bullish (close > open) AND current close < past open
    """
    try:
        def get_last_two_completed_volume_ratio(interval, limit):
            """Return latest completed / previous completed USDT volume ratio."""
            candles = get_historical_klines(
                symbol,
                interval,
                end_time=reference_time if reference_time else None,
                limit=limit
            )
            if not candles or len(candles) < 3:
                return np.nan
            latest_completed = candles[-2]
            previous_completed = candles[-3]
            latest_usdt = float(latest_completed[7])
            previous_usdt = float(previous_completed[7])
            if previous_usdt <= 0:
                return np.nan
            return latest_usdt / previous_usdt

        # Map timeframes to Binance intervals
        interval_map = {
            "Monthly": "1M",
            "Weekly": "1w",
            "Daily": "1d",
            "4H": "4h"
        }

        # Get past timeframe data (need at least 2 candles to get previous completed candle)
        past_interval = interval_map.get(past_timeframe)
        if not past_interval:
            return None

        # Determine how many candles needed based on past timeframe
        if past_timeframe == "Monthly":
            past_limit = 3
        elif past_timeframe == "Weekly":
            past_limit = 5
        else:  # Daily
            past_limit = 10

        if reference_time:
            # Backtest mode: evaluate candles completed before reference_time
            past_data = get_historical_klines(symbol, past_interval, end_time=reference_time, limit=past_limit)
        else:
            params_past = {
                'symbol': symbol,
                'interval': past_interval,
                'limit': past_limit
            }
            response_past = api_get(f"{BINANCE_API_URL}/fapi/v1/klines", params=params_past, timeout=5)
            past_data = response_past.json()

        if not past_data or len(past_data) < 2:
            return None

        # Get the LAST COMPLETED past candle
        past_candle = past_data[-2]

        past_open = float(past_candle[1])
        past_close = float(past_candle[4])
        past_high = float(past_candle[2])
        past_low = float(past_candle[3])

        # Determine if past candle is bullish or bearish
        past_is_bullish = past_close > past_open
        past_is_bearish = past_close < past_open
        past_body_size = abs(past_close - past_open)
        past_body_percent = (past_body_size / past_open) * 100 if past_open > 0 else 0

        # Get current timeframe data
        current_interval = interval_map.get(current_timeframe)
        if not current_interval:
            return None

        # For current timeframe, we only need the last 2 candles
        if current_timeframe == "Monthly":
            current_limit = 3
        elif current_timeframe == "Weekly":
            current_limit = 5
        elif current_timeframe == "Daily":
            current_limit = 10
        else:  # 4H
            current_limit = 25

        if reference_time:
            # Backtest mode: current timeframe candle is the latest completed before reference_time
            current_data = get_historical_klines(symbol, current_interval, end_time=reference_time, limit=current_limit)
        else:
            params_current = {
                'symbol': symbol,
                'interval': current_interval,
                'limit': current_limit
            }
            response_current = api_get(f"{BINANCE_API_URL}/fapi/v1/klines", params=params_current, timeout=5)
            current_data = response_current.json()

        if not current_data or len(current_data) < 3:
            return None

        # IMPORTANT: ONLY check the LAST COMPLETED current candle
        # This ensures we only show breakouts from the most recent candle
        current_candle = current_data[-2]
        prev_current_candle = current_data[-3]

        current_close = float(current_candle[4])
        prev_current_close = float(prev_current_candle[4])
        current_volume = float(current_candle[5])
        current_usdt_volume = float(current_candle[7])

        # Candle times for display
        past_candle_time = datetime.fromtimestamp(past_candle[0] / 1000, tz=UTC)
        current_candle_time = datetime.fromtimestamp(current_candle[0] / 1000, tz=UTC)

        # Initialize result dictionary
        result = {
            'Coin Name': symbol,
            'Past Timeframe': past_timeframe,
            'Current Timeframe': current_timeframe,
            'Past Open': round(past_open, 4),
            'Past Close': round(past_close, 4),
            'Past Candle Type': 'Bullish' if past_is_bullish else 'Bearish' if past_is_bearish else 'Doji',
            'Past Body %': round(past_body_percent, 2),
            'Current Close': round(current_close, 4),
            'Price Difference': round(current_close - past_open, 4),
            'Diff %': round(((current_close - past_open) / past_open) * 100, 2),
            'Volume (USDT)': format_large_number(current_usdt_volume),
            'Past Candle Date': past_candle_time.astimezone(IST).strftime('%Y-%m-%d'),
            'Current Candle Date': current_candle_time.astimezone(IST).strftime('%Y-%m-%d %I:%M %p'),
            'Current Candle Time': current_candle_time.astimezone(IST).strftime('%I:%M %p'),
            'Spot Match': 'Yes' if check_spot_match_fast(symbol) else 'No',
            'Scan Mode': 'Backtest' if reference_time else 'Live'
        }

        # Get Open Interest data based on current timeframe (just for display)
        oi_data = get_historical_open_interest(symbol, current_timeframe)
        result['OI Change %'] = oi_data['change_percent']
        result['OI Period'] = oi_data['comparison_text']
        result['Monthly Vol Ratio'] = get_last_two_completed_volume_ratio("1M", 4)
        result['Weekly Vol Ratio'] = get_last_two_completed_volume_ratio("1w", 6)
        result['Daily Vol Ratio'] = get_last_two_completed_volume_ratio("1d", 12)
        result['4H Vol Ratio'] = get_last_two_completed_volume_ratio("4h", 30)

        # Check conditions based on scan mode
        is_bullish_signal = False
        is_bearish_signal = False

        # Fresh bullish breakout:
        # 1) Past candle bearish
        # 2) Current close above past open
        # 3) Previous current candle close was at/below past open (first crossover now)
        if past_is_bearish and current_close > past_open and prev_current_close <= past_open:
            is_bullish_signal = True
            result['Signal Type'] = 'BULLISH 📈'
            result['Condition'] = "Past Bearish | Fresh Close Cross Above Past Open"

        # Fresh bearish breakout:
        # 1) Past candle bullish
        # 2) Current close below past open
        # 3) Previous current candle close was at/above past open (first crossover now)
        elif past_is_bullish and current_close < past_open and prev_current_close >= past_open:
            is_bearish_signal = True
            result['Signal Type'] = 'BEARISH 📉'
            result['Condition'] = "Past Bullish | Fresh Close Cross Below Past Open"

        # No signal
        else:
            return None

        # Filter based on scan mode
        if scan_mode == "bullish" and not is_bullish_signal:
            return None
        if scan_mode == "bearish" and not is_bearish_signal:
            return None

        return result

    except Exception as e:
        return None

def scan_timeframe_breakouts_parallel(symbols, past_timeframe, current_timeframe, scan_mode="both", max_workers=25, reference_time=None):
    """Parallel timeframe-based breakout scanning"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(
                fetch_single_timeframe_breakout_worker,
                symbol,
                past_timeframe,
                current_timeframe,
                scan_mode,
                reference_time
            ): symbol
            for symbol in symbols
        }

        completed = 0
        total = len(symbols)

        for future in as_completed(future_to_symbol):
            completed += 1
            if completed % 10 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(f"Scanning {completed}/{total} pairs... Found: {len(results)} breakouts")

            try:
                result = future.result(timeout=5)
                if result:
                    results.append(result)
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()
    return results

# ==============================================
# CLEANED TAB: TIMEFRAME BREAKOUT SCANNER (NO VOLUME RATIO, NO ADVANCED FILTERS)
# ==============================================

def render_timeframe_breakout_scanner():
    """Render the cleaned timeframe-based breakout scanner tab"""

    st.header("🎯 Timeframe Breakout Scanner")
    st.markdown("""
    **Scan based on Past Timeframe OPEN vs Current Timeframe CLOSE**

    ### 📊 Logic:
    - **Bullish Signal**: Past candle is **BEARISH** (Close < Open) AND Current Close > Past Open
    - **Bearish Signal**: Past candle is **BULLISH** (Close > Open) AND Current Close < Past Open

    ### 🔍 How it works:
    1. Select **Past Timeframe** - uses the **OPEN** price of last completed candle
    2. Select **Current Timeframe** - uses the **CLOSE** price of the **MOST RECENT** completed candle
    3. Scanner checks condition across all Binance Futures pairs
    4. Results show ONLY breakouts from the latest candle
    """)

    # Initialize defaults once for cleaner preset handling
    if "past_tf" not in st.session_state:
        st.session_state["past_tf"] = "Daily"
    if "current_tf" not in st.session_state:
        st.session_state["current_tf"] = "Daily"
    if "scan_mode_tf" not in st.session_state:
        st.session_state["scan_mode_tf"] = "both"
    if "tf_backtest" not in st.session_state:
        st.session_state["tf_backtest"] = False
    if "tf_backtest_date" not in st.session_state:
        st.session_state["tf_backtest_date"] = datetime.now(IST).date() - timedelta(days=1)

    st.subheader("⚙️ Scan Setup")

    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        if st.button("Preset: Weekly -> Daily (Fresh)", use_container_width=True, key="tf_preset_w_d"):
            st.session_state["past_tf"] = "Weekly"
            st.session_state["current_tf"] = "Daily"
            st.session_state["scan_mode_tf"] = "bullish"
            st.session_state["tf_backtest"] = False
            st.rerun()
    with preset_col2:
        if st.button("Preset: Weekly -> 4H (Fresh)", use_container_width=True, key="tf_preset_w_4h"):
            st.session_state["past_tf"] = "Weekly"
            st.session_state["current_tf"] = "4H"
            st.session_state["scan_mode_tf"] = "bullish"
            st.session_state["tf_backtest"] = False
            st.rerun()
    with preset_col3:
        if st.button("Preset: Monthly -> Daily", use_container_width=True, key="tf_preset_m_d"):
            st.session_state["past_tf"] = "Monthly"
            st.session_state["current_tf"] = "Daily"
            st.session_state["scan_mode_tf"] = "both"
            st.session_state["tf_backtest"] = False
            st.rerun()

    col1, col2, col3 = st.columns(3)

    with col1:
        past_timeframe = st.selectbox(
            "📅 Past Timeframe (OPEN price)",
            ["Monthly", "Weekly", "Daily"],
            index=2,
            key="past_tf",
            help="Select timeframe for the PAST candle OPEN price"
        )

    with col2:
        current_timeframe = st.selectbox(
            "⏰ Current Timeframe (CLOSE price)",
            ["Weekly", "Daily", "4H"],
            index=1,
            key="current_tf",
            help="Select timeframe for the CURRENT candle CLOSE price (only most recent candle)"
        )

    with col3:
        scan_mode = st.selectbox(
            "🎯 Scan Mode",
            ["both", "bullish", "bearish"],
            index=0,
            key="scan_mode_tf",
            format_func=lambda x: "Both Bullish & Bearish" if x == "both" else "Bullish Only" if x == "bullish" else "Bearish Only"
        )

    col1, col2 = st.columns(2)

    with col1:
        n_coins = st.number_input(
            "Number of coins to display",
            min_value=5, max_value=100, value=20, step=5,
            key="tf_n_coins"
        )

    with col2:
        priority_scan = st.checkbox(
            "Priority Scan (High volume first)",
            value=True,
            key="tf_priority"
        )

    backtest_mode = st.checkbox("Backtest Mode", value=False, key="tf_backtest")
    selected_4h_slot = None
    if backtest_mode:
        backtest_date = st.date_input(
            "Backtest Date (scan as of this date)",
            value=datetime.now(IST).date() - timedelta(days=1),
            key="tf_backtest_date"
        )
        if current_timeframe == "4H":
            # Binance 4H candles are UTC-aligned; in IST they start at 01:30, 05:30, 09:30, 13:30, 17:30, 21:30
            candle_time_map = {
                "01:30 AM - 05:30 AM": (0, 5, 30),
                "05:30 AM - 09:30 AM": (0, 9, 30),
                "09:30 AM - 01:30 PM": (0, 13, 30),
                "01:30 PM - 05:30 PM": (0, 17, 30),
                "05:30 PM - 09:30 PM": (0, 21, 30),
                "09:30 PM - 01:30 AM": (1, 1, 30),
            }
            selected_4h_slot = st.selectbox(
                "4H Candle to Check (IST, 12-hour format)",
                options=list(candle_time_map.keys()),
                index=1,
                key="tf_backtest_4h_slot"
            )
            close_day_offset, close_hour, close_minute = candle_time_map[selected_4h_slot]
            slot_close_date = backtest_date + timedelta(days=close_day_offset)
            reference_time = datetime.combine(slot_close_date, datetime.min.time(), tzinfo=IST) + timedelta(
                hours=close_hour,
                minutes=close_minute
            )
            st.info(
                f"🧪 Backtest mode: scanning **{selected_4h_slot}** candle on "
                f"**{backtest_date.strftime('%Y-%m-%d')}** ({current_timeframe} close vs {past_timeframe} open)"
            )
        else:
            reference_time = datetime.combine(backtest_date, datetime.min.time(), tzinfo=IST)
            st.info(
                f"🧪 Backtest mode: scanning candles completed before **{backtest_date.strftime('%Y-%m-%d')}** "
                f"({current_timeframe} close vs {past_timeframe} open)"
            )
    else:
        reference_time = None
        st.info(f"🔍 Live mode: scanning ONLY the most recent completed **{current_timeframe}** candle")

    # Compact scan summary bar
    if backtest_mode:
        slot_text = f" | Slot: {selected_4h_slot}" if selected_4h_slot else ""
        st.success(
            f"Backtest | {past_timeframe} Open vs {current_timeframe} Close | "
            f"Date: {backtest_date.strftime('%Y-%m-%d')}{slot_text} | "
            f"Mode: {'Bullish' if scan_mode == 'bullish' else 'Bearish' if scan_mode == 'bearish' else 'Both'}"
        )
    else:
        st.success(
            f"Live | {past_timeframe} Open vs {current_timeframe} Close | "
            f"Mode: {'Bullish' if scan_mode == 'bullish' else 'Bearish' if scan_mode == 'bearish' else 'Both'}"
        )

    parallel_workers = st.slider(
        "Parallel Workers (Faster = More API calls)",
        min_value=5, max_value=50, value=25,
        key="tf_workers"
    )

    scan_btn_label = "🧪 Start Backtest Scan" if backtest_mode else "🔍 Start Live Scan"
    can_scan = not (backtest_mode and not backtest_date)
    if st.button(scan_btn_label, type="primary", use_container_width=True, key="tf_scan_btn", disabled=not can_scan):
        st.subheader("📊 Results")
        start_time = time.time()

        with st.spinner("Loading all perpetual pairs..."):
            all_symbols = get_perpetual_symbols()
            if not all_symbols:
                st.error("Failed to fetch perpetual pairs.")
                st.stop()

            if priority_scan:
                top_volume = get_top_volume_symbols_list(limit=500)
                symbols_to_scan = [s for s in top_volume if s in all_symbols]
                remaining = [s for s in all_symbols if s not in symbols_to_scan]
                symbols_to_scan.extend(remaining)
            else:
                symbols_to_scan = all_symbols

            st.success(f"✅ Loaded {len(all_symbols)} perpetual pairs")
            scan_label = "backtest candles" if backtest_mode else "most recent candle only"
            st.info(
                f"⚡ Scanning {past_timeframe} OPEN vs {current_timeframe} CLOSE "
                f"({scan_label}) with {parallel_workers} parallel workers..."
            )

        # Run the scan
        results = scan_timeframe_breakouts_parallel(
            symbols_to_scan,
            past_timeframe,
            current_timeframe,
            scan_mode,
            parallel_workers,
            reference_time
        )

        if results:
            df_results = pd.DataFrame(results)

            # Sort by difference percentage
            df_results = df_results.sort_values('Diff %', ascending=False).head(n_coins)

            elapsed_time = time.time() - start_time

            if len(df_results) > 0:
                st.success(f"✅ Found {len(df_results)} breakout signals in {elapsed_time:.1f} seconds!")

                # Display results in a nice table
                display_data = []
                for idx, row in df_results.iterrows():
                    signal_emoji = "🟢" if "BULLISH" in row['Signal Type'] else "🔴"

                    display_data.append({
                        'S.No': len(display_data) + 1,
                        'Coin': row['Coin Name'],
                        'Signal': f"{signal_emoji} {row['Signal Type']}",
                        'Past TF': row['Past Timeframe'],
                        'Current TF': row['Current Timeframe'],
                        'Past Open': row['Past Open'],
                        'Current Close': row['Current Close'],
                        'Diff %': f"{row['Diff %']:.2f}%",
                        'Volume': row['Volume (USDT)'],
                        'Monthly Vol Ratio': f"{row['Monthly Vol Ratio']:.2f}x" if pd.notna(row.get('Monthly Vol Ratio')) else "N/A",
                        'Weekly Vol Ratio': f"{row['Weekly Vol Ratio']:.2f}x" if pd.notna(row.get('Weekly Vol Ratio')) else "N/A",
                        'Daily Vol Ratio': f"{row['Daily Vol Ratio']:.2f}x" if pd.notna(row.get('Daily Vol Ratio')) else "N/A",
                        '4H Vol Ratio': f"{row['4H Vol Ratio']:.2f}x" if pd.notna(row.get('4H Vol Ratio')) else "N/A",
                        'OI Change': f"{row['OI Change %']:.2f}%",
                        'Past Candle': f"{row['Past Candle Type']} ({row['Past Body %']:.1f}%)",
                        'Candle Time': row['Current Candle Time'],
                        'Spot': row['Spot Match']
                    })

                display_df = pd.DataFrame(display_data)

                # Color coding for signals
                def color_signals(val):
                    if "BULLISH" in str(val):
                        return 'color: green; font-weight: bold'
                    elif "BEARISH" in str(val):
                        return 'color: red; font-weight: bold'
                    return ''

                def color_oi(val):
                    if isinstance(val, str) and '%' in val:
                        try:
                            num = float(val.replace('%', ''))
                            if num > 20:
                                return 'color: darkgreen; font-weight: bold'
                            elif num > 10:
                                return 'color: green'
                            elif num > 5:
                                return 'color: lightgreen'
                            elif num < -20:
                                return 'color: darkred; font-weight: bold'
                            elif num < -10:
                                return 'color: red'
                            elif num < -5:
                                return 'color: lightcoral'
                        except:
                            pass
                    return ''

                tab_all, tab_bull, tab_bear = st.tabs(["All Signals", "Bullish", "Bearish"])

                with tab_all:
                    st.dataframe(
                        display_df.style.map(color_signals, subset=['Signal']).map(color_oi, subset=['OI Change']),
                        use_container_width=True,
                        hide_index=True
                    )

                with tab_bull:
                    bull_df = display_df[display_df['Signal'].str.contains('BULLISH', na=False)].copy()
                    if bull_df.empty:
                        st.info("No bullish signals in this scan.")
                    else:
                        st.dataframe(
                            bull_df.style.map(color_signals, subset=['Signal']).map(color_oi, subset=['OI Change']),
                            use_container_width=True,
                            hide_index=True
                        )

                with tab_bear:
                    bear_df = display_df[display_df['Signal'].str.contains('BEARISH', na=False)].copy()
                    if bear_df.empty:
                        st.info("No bearish signals in this scan.")
                    else:
                        st.dataframe(
                            bear_df.style.map(color_signals, subset=['Signal']).map(color_oi, subset=['OI Change']),
                            use_container_width=True,
                            hide_index=True
                        )

                # Statistics
                st.subheader("📊 Scan Statistics")
                col1, col2, col3, col4 = st.columns(4)

                bullish_count = len(df_results[df_results['Signal Type'].str.contains('BULLISH', na=False)])
                bearish_count = len(df_results[df_results['Signal Type'].str.contains('BEARISH', na=False)])

                with col1:
                    st.metric("Total Signals", len(df_results))
                with col2:
                    st.metric("Bullish Signals", bullish_count)
                with col3:
                    st.metric("Bearish Signals", bearish_count)
                with col4:
                    avg_diff = df_results['Diff %'].mean()
                    st.metric("Avg Diff %", f"{avg_diff:.2f}%")

                # Download button
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Breakout Results (CSV)",
                    data=csv,
                    file_name=f"timeframe_breakout_{past_timeframe}_vs_{current_timeframe}_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # Show detailed view in expander
                with st.expander("🔍 View Detailed Data"):
                    detail_cols = ['Coin', 'Signal', 'Past Candle', 'Past Open', 'Current Close',
                                 'Diff %', 'Volume', 'Monthly Vol Ratio', 'Weekly Vol Ratio', 'Daily Vol Ratio', '4H Vol Ratio', 'OI Change', 'Candle Time']
                    st.dataframe(display_df[detail_cols], use_container_width=True, hide_index=True)

            else:
                st.warning("No breakout signals match your filters.")
        else:
            st.warning(f"No {past_timeframe} vs {current_timeframe} breakout signals found in the most recent candle.")

# ==============================================
# OPEN INTEREST SCANNER FUNCTIONS - TIMEFRAME AWARE
# ==============================================

def fetch_single_oi_data(symbol, timeframe):
    """Fetch Open Interest data for a single symbol - TIMEFRAME AWARE"""
    try:
        # Get OI change based on selected timeframe
        oi_data = get_historical_open_interest(symbol, timeframe, limit=2)
        oi_change = oi_data['change_percent']
        current_oi = oi_data['current_oi']
        previous_oi = oi_data['previous_oi']

        if current_oi == 0:
            return None

        # Get 24h volume and price change
        try:
            ticker_response = api_get(f"{BINANCE_API_URL}/fapi/v1/ticker/24hr", params={'symbol': symbol}, timeout=3)
            ticker_data = ticker_response.json()
            volume_24h = float(ticker_data.get('quoteVolume', 0))
            price_change = float(ticker_data.get('priceChangePercent', 0))
        except:
            volume_24h = 0
            price_change = 0

        # Determine OI trend and signal based on change percentage
        if oi_change > 20:
            trend = "🚀 Strongly Increasing"
            signal = "Strong Bullish"
        elif oi_change > 10:
            trend = "📈 Increasing"
            signal = "Bullish"
        elif oi_change > 5:
            trend = "↗️ Slightly Increasing"
            signal = "Mildly Bullish"
        elif oi_change > 2:
            trend = "➡️ Moderate Increase"
            signal = "Cautiously Bullish"
        elif oi_change > 0.5:
            trend = "↗️ Minor Increase"
            signal = "Slightly Bullish"
        elif oi_change < -20:
            trend = "💥 Strongly Decreasing"
            signal = "Strong Bearish"
        elif oi_change < -10:
            trend = "📉 Decreasing"
            signal = "Bearish"
        elif oi_change < -5:
            trend = "↘️ Slightly Decreasing"
            signal = "Mildly Bearish"
        elif oi_change < -2:
            trend = "➡️ Moderate Decrease"
            signal = "Cautiously Bearish"
        elif oi_change < -0.5:
            trend = "↘️ Minor Decrease"
            signal = "Slightly Bearish"
        else:
            trend = "➡️ Neutral"
            signal = "Neutral"

        # Add volume trend based on price and OI
        if price_change > 2 and oi_change > 10:
            signal = "🚀 Strong Bullish (Price + OI)"
        elif price_change < -2 and oi_change < -10:
            signal = "💀 Strong Bearish (Price + OI)"
        elif price_change > 2 and oi_change < -10:
            signal = "⚠️ Warning (Price Up, OI Down)"
        elif price_change < -2 and oi_change > 10:
            signal = "⚠️ Warning (Price Down, OI Up)"

        return {
            'Coin Name': symbol,
            'OI_Change_%': oi_change,
            'Current_OI': current_oi,
            'Previous_OI': previous_oi,
            'OI_Trend': trend,
            'Signal': signal,
            '24h_Volume': volume_24h,
            '24h_Volume_Display': format_large_number(volume_24h),
            'Price_Change_%': round(price_change, 2),
            'OI_Period': oi_data['comparison_text'],
            'Timeframe': timeframe
        }

    except Exception as e:
        return None

def scan_open_interest_parallel(symbols, timeframe, max_workers=25, min_oi_change=1.0, oi_direction="Both"):
    """Parallel Open Interest scanner - TIMEFRAME AWARE"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(fetch_single_oi_data, symbol, timeframe): symbol
            for symbol in symbols
        }

        completed = 0
        total = len(symbols)

        for future in as_completed(future_to_symbol):
            completed += 1
            if completed % 10 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(f"Scanning OI {completed}/{total} pairs... Found: {len(results)} OI changes")

            try:
                result = future.result(timeout=5)
                if result:
                    oi_change = result['OI_Change_%']

                    # Apply filters
                    if oi_direction == "Positive (Increasing)" and oi_change <= 0:
                        continue
                    if oi_direction == "Negative (Decreasing)" and oi_change >= 0:
                        continue
                    if abs(oi_change) < min_oi_change:
                        continue

                    results.append(result)
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()
    return results

# ==============================================
# HIGH VOLUME BREAKOUT (previous month peak vol day)
# ==============================================

def get_calendar_month_bounds(year, month):
    """IST start/end datetimes for a calendar month."""
    start = datetime(year, month, 1, tzinfo=IST)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=IST) - timedelta(seconds=1)
    else:
        end = datetime(year, month + 1, 1, tzinfo=IST) - timedelta(seconds=1)
    return start, end


def get_previous_calendar_month(reference_dt=None):
    """Return (year, month, start_dt, end_dt) for the calendar month before reference."""
    reference_dt = reference_dt or datetime.now(IST)
    first_this = reference_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_prev = first_this - timedelta(seconds=1)
    y, m = last_prev.year, last_prev.month
    start, end = get_calendar_month_bounds(y, m)
    return y, m, start, end


def _parse_daily_kline_rows(klines):
    rows = []
    for kline in klines:
        candle_dt = datetime.fromtimestamp(kline[0] / 1000, tz=UTC).astimezone(IST)
        _, usdt_volume = get_candle_volumes(kline)
        rows.append(
            {
                "full_date": candle_dt.strftime("%Y-%m-%d"),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "usdt_volume": usdt_volume,
            }
        )
    return sorted(rows, key=lambda x: x["full_date"])


def fetch_high_volume_breakout_worker(
    symbol, month_start_str, month_end_str, exclude_above_1=False, min_peak_vol_usdt=0, backtest_date=None
):
    """
    TRUTH / AKT-style logic:
    1) Find highest USDT volume day in the selected month (e.g. Apr 29).
    2) Find the FIRST daily candle after that day with bullish breakout + close above peak-day high.
    3) Latest close must still be above peak-day high (setup still active).
    """
    try:
        month_start_dt = datetime.strptime(month_start_str, "%Y-%m-%d").replace(tzinfo=IST)
        fetch_start = month_start_dt - timedelta(days=3)
        fetch_end = datetime.now(IST)
        if backtest_date:
            if not isinstance(backtest_date, datetime):
                backtest_dt = datetime.combine(backtest_date, datetime.max.time()).replace(tzinfo=IST)
            else:
                backtest_dt = backtest_date
            fetch_end = backtest_dt

        klines = get_historical_klines(symbol, "1d", fetch_start, fetch_end, limit=1000)
        if not klines or len(klines) < 4:
            return None

        rows = _parse_daily_kline_rows(klines)

        if backtest_date:
            backtest_str = backtest_date.strftime("%Y-%m-%d")
            completed = [r for r in rows if r["full_date"] <= backtest_str]
        else:
            today_str = datetime.now(IST).strftime("%Y-%m-%d")
            if rows and rows[-1]["full_date"] == today_str:
                completed = rows[:-1]
            else:
                completed = rows

        if len(completed) < 3:
            return None

        month_rows = [
            r for r in completed if month_start_str <= r["full_date"] <= month_end_str
        ]
        if not month_rows:
            return None

        peak_row = max(month_rows, key=lambda x: x["usdt_volume"])
        if peak_row["usdt_volume"] < min_peak_vol_usdt:
            return None

        peak_date = peak_row["full_date"]
        peak_high = peak_row["high"]

        # Find the last index where the close was at or below peak_high (after peak_date)
        last_below_idx = -1
        for i in range(len(completed)):
            if completed[i]["full_date"] > peak_date and completed[i]["close"] <= peak_high:
                last_below_idx = i

        first_breakout = None
        for i in range(1, len(completed)):
            candle = completed[i]
            if candle["full_date"] <= peak_date:
                continue
            if i <= last_below_idx:
                continue
            prev = completed[i - 1]
            bullish_breakout = (
                candle["close"] > prev["high"] and candle["high"] > prev["high"]
            )
            above_peak_high = candle["close"] > peak_high
            if bullish_breakout and above_peak_high:
                first_breakout = candle
                break

        if first_breakout is None:
            return None

        last_candle = completed[-1]
        if last_candle["close"] <= peak_high:
            return None

        if exclude_above_1 and last_candle["close"] > 1.0:
            return None

        breakout_close = first_breakout["close"]
        close_vs_peak_pct = round((breakout_close / peak_high - 1) * 100, 2)
        latest_vs_peak_pct = round((last_candle["close"] / peak_high - 1) * 100, 2)

        # Calculate current calendar month low to high change as of reference date
        curr_month_low_to_high = None
        try:
            now_ist = backtest_date if backtest_date else datetime.now(IST)
            if not isinstance(now_ist, datetime):
                now_ist = datetime.combine(now_ist, datetime.min.time()).replace(tzinfo=IST)
            current_month_start = datetime(now_ist.year, now_ist.month, 1, tzinfo=IST)
            current_klines = get_historical_klines(symbol, "1d", start_time=current_month_start, limit=35)
            if current_klines:
                # filter down to backtest time
                valid_klines = []
                for k in current_klines:
                    k_dt = datetime.fromtimestamp(k[0] / 1000, tz=UTC).astimezone(IST)
                    if k_dt <= now_ist:
                        valid_klines.append(k)
                if valid_klines:
                    lows = [float(k[3]) for k in valid_klines]
                    highs = [float(k[2]) for k in valid_klines]
                    if lows and highs:
                        c_low = min(lows)
                        c_high = max(highs)
                        if c_low > 0:
                            curr_month_low_to_high = round(((c_high - c_low) / c_low) * 100, 2)
        except Exception:
            pass

        return {
            "symbol": symbol,
            "peak_vol_date": peak_date,
            "peak_usdt_vol": peak_row["usdt_volume"],
            "peak_day_high": peak_high,
            "breakout_date": first_breakout["full_date"],
            "close": breakout_close,
            "close_vs_peak_high_%": close_vs_peak_pct,
            "latest_close": last_candle["close"],
            "latest_vs_peak_high_%": latest_vs_peak_pct,
            "curr_month_low_to_high": curr_month_low_to_high,
        }
    except Exception:
        return None


def fetch_yearly_peak_month_breakout_worker(
    symbol, scan_year, exclude_above_1=False, min_peak_vol_usdt=0, only_active=False, backtest_date=None
):
    """
    Finds the highest USDT volume month in the chosen scan_year.
    Checks if any subsequent monthly candle closes above that peak month's high.
    """
    try:
        fetch_start = datetime(scan_year, 1, 1, tzinfo=IST)
        fetch_end = datetime.now(IST)
        if backtest_date:
            if not isinstance(backtest_date, datetime):
                backtest_dt = datetime.combine(backtest_date, datetime.max.time()).replace(tzinfo=IST)
            else:
                backtest_dt = backtest_date
            fetch_end = backtest_dt

        klines = get_historical_klines(symbol, "1M", fetch_start, fetch_end, limit=100)
        if not klines or len(klines) < 1:
            return None

        rows = []
        for kline in klines:
            candle_dt = datetime.fromtimestamp(kline[0] / 1000, tz=UTC).astimezone(IST)
            _, usdt_volume = get_candle_volumes(kline)
            rows.append({
                "year": candle_dt.year,
                "month": candle_dt.month,
                "month_str": candle_dt.strftime("%Y-%m"),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "usdt_volume": usdt_volume,
            })

        # Filter out current calendar month (or backtest month) since it is not completed yet
        ref_time = backtest_date if backtest_date else datetime.now(IST)
        current_year_val = ref_time.year
        current_month_val = ref_time.month

        completed_rows = [
            r for r in rows
            if not (r["year"] == current_year_val and r["month"] == current_month_val)
        ]

        if backtest_date:
            backtest_month_str = backtest_date.strftime("%Y-%m")
            completed_rows = [r for r in completed_rows if r["month_str"] <= backtest_month_str]

        year_rows = [r for r in completed_rows if r["year"] == scan_year]
        if not year_rows:
            return None

        peak_row = max(year_rows, key=lambda x: x["usdt_volume"])
        if peak_row["usdt_volume"] < min_peak_vol_usdt:
            return None

        peak_month_str = peak_row["month_str"]
        peak_high = peak_row["high"]
        peak_low = peak_row["low"]

        first_breakout = None
        for r in completed_rows:
            if r["month_str"] <= peak_month_str:
                continue
            if r["close"] > peak_high:
                first_breakout = r
                break

        if first_breakout is None:
            return None

        last_candle = completed_rows[-1]

        if only_active and last_candle["close"] <= peak_high:
            return None

        if exclude_above_1 and last_candle["close"] > 1.0:
            return None

        close_vs_peak_pct = round((first_breakout["close"] / peak_high - 1) * 100, 2)
        latest_vs_peak_pct = round((last_candle["close"] / peak_high - 1) * 100, 2)

        return {
            "symbol": symbol,
            "peak_month": peak_month_str,
            "peak_usdt_vol": peak_row["usdt_volume"],
            "peak_month_high": peak_high,
            "peak_month_low": peak_low,
            "breakout_month": first_breakout["month_str"],
            "close": first_breakout["close"],
            "close_vs_peak_high_%": close_vs_peak_pct,
            "latest_close": last_candle["close"],
            "latest_vs_peak_high_%": latest_vs_peak_pct,
        }
    except Exception:
        return None


def scan_yearly_peak_month_breakout_parallel(
    symbols,
    scan_year,
    exclude_above_1=False,
    min_peak_vol_m=0,
    only_active=False,
    max_workers=25,
    backtest_date=None,
):
    min_peak_vol_usdt = min_peak_vol_m * 1_000_000
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(
                fetch_yearly_peak_month_breakout_worker,
                symbol,
                scan_year,
                exclude_above_1,
                min_peak_vol_usdt,
                only_active,
                backtest_date,
            ): symbol
            for symbol in symbols
        }

        completed = 0
        total = len(symbols)

        for future in as_completed(future_to_symbol):
            completed += 1
            if completed % 25 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(
                    f"Yearly peak month scan {completed}/{total}... Matches: {len(results)}"
                )
            try:
                row = future.result(timeout=12)
                if row:
                    results.append(row)
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()
    results.sort(key=lambda x: x["peak_usdt_vol"], reverse=True)
    return results


def scan_high_volume_breakout_parallel(
    symbols,
    month_start_str,
    month_end_str,
    exclude_above_1=False,
    min_peak_vol_m=0,
    max_workers=25,
    backtest_date=None,
):
    min_peak_vol_usdt = min_peak_vol_m * 1_000_000
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(
                fetch_high_volume_breakout_worker,
                symbol,
                month_start_str,
                month_end_str,
                exclude_above_1,
                min_peak_vol_usdt,
                backtest_date,
            ): symbol
            for symbol in symbols
        }

        completed = 0
        total = len(symbols)

        for future in as_completed(future_to_symbol):
            completed += 1
            if completed % 25 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(
                    f"HV breakout scan {completed}/{total}... Matches: {len(results)}"
                )
            try:
                row = future.result(timeout=12)
                if row:
                    results.append(row)
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()
    results.sort(key=lambda x: x["latest_vs_peak_high_%"], reverse=True)
    return results


def _render_high_volume_breakout_scanner():
    """Peak USDT volume day/month breakout scan."""
    st.markdown(
        """
**Choose Scan Mode:**
1. **Daily Breakout (Prior Month Peak Day)**: Finds the highest volume day in the selected month and checks daily close breakouts above that day's high.
2. **Monthly Breakout (Prior Year Peak Month)**: Finds the highest volume month in the selected year and checks monthly close breakouts above that month's high.
        """
    )

    scan_type = st.radio(
        "Breakout Type Mode",
        ["Daily Breakout (Prior Month Peak Day)", "Monthly Breakout (Prior Year Peak Month)"],
        horizontal=True,
        key="hvbo_scan_type"
    )

    if scan_type == "Daily Breakout (Prior Month Peak Day)":
        now_ist = datetime.now(IST)
        prev_y, prev_m, prev_start, prev_end = get_previous_calendar_month(now_ist)
        prev_month_label = datetime(prev_y, prev_m, 1).strftime("%B %Y")

        col1, col2, col3 = st.columns(3)
        with col1:
            month_mode = st.radio(
                "Scan month",
                ["Previous month (auto)", "Pick month"],
                horizontal=True,
                key="hvbo_month_mode",
            )
        with col2:
            hvbo_exclude_above_1 = st.checkbox(
                "Exclude coins above $1",
                value=False,
                key="hvbo_exclude_above_1",
            )
        with col3:
            hvbo_min_peak_m = st.number_input(
                "Min peak-day volume (M)",
                min_value=0.0,
                max_value=500.0,
                value=0.0,
                step=0.5,
                key="hvbo_min_peak_m",
                help="e.g. 5 = only coins whose top day in that month had ≥ 5M USDT volume",
            )

        if month_mode == "Previous month (auto)":
            scan_year, scan_month = prev_y, prev_m
            month_start, month_end = prev_start, prev_end
            month_label = prev_month_label
        else:
            month_pick = st.date_input(
                "Select month",
                value=datetime(prev_y, prev_m, 1).date(),
                key="hvbo_month_pick",
            )
            scan_year, scan_month = month_pick.year, month_pick.month
            month_start, month_end = get_calendar_month_bounds(scan_year, scan_month)
            month_label = datetime(scan_year, scan_month, 1).strftime("%B %Y")

        month_start_str = month_start.strftime("%Y-%m-%d")
        month_end_str = month_end.strftime("%Y-%m-%d")

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            hvbo_workers = st.slider(
                "Parallel Workers (Faster = More API calls)",
                min_value=5,
                max_value=50,
                value=25,
                key="hvbo_workers",
            )
        with col_w2:
            hvbo_priority = st.checkbox(
                "Priority Scan (High volume first)",
                value=True,
                key="hvbo_priority",
            )

        # --- Backtest Mode ---
        hvbo_backtest_enabled = st.checkbox(
            "🧪 Backtest Mode (scan as of a past date)",
            value=False,
            key="hvbo_backtest_enabled",
        )
        hvbo_backtest_date = None
        if hvbo_backtest_enabled:
            hvbo_backtest_date = st.date_input(
                "📅 As-Of Date (simulate scan on this date)",
                value=datetime.now(IST).date(),
                key="hvbo_backtest_date",
                help="The scan will only use data available up to this date.",
            )
            st.info(
                f"🧪 **Backtest Mode ON** — Scanning **{month_label}** as of **{hvbo_backtest_date.strftime('%d %b %Y')}**."
            )
        else:
            st.info(
                f"Scanning **{month_label}** for peak USDT volume day, then checking breakout above that day's high."
            )

        if st.button(
            "🔍 Scan High Volume Breakout",
            type="primary",
            use_container_width=True,
            key="hvbo_scan_btn",
        ):
            all_symbols = get_perpetual_symbols()
            if not all_symbols:
                st.error("Failed to load perpetual symbols.")
                return

            symbols_to_scan = all_symbols
            if hvbo_priority:
                top_volume = get_top_volume_symbols_list(limit=500)
                symbols_to_scan = [s for s in top_volume if s in all_symbols]
                symbols_to_scan.extend([s for s in all_symbols if s not in symbols_to_scan])

            with st.spinner(f"Scanning {len(symbols_to_scan)} coins for {month_label}..."):
                matches = scan_high_volume_breakout_parallel(
                    symbols_to_scan,
                    month_start_str,
                    month_end_str,
                    hvbo_exclude_above_1,
                    float(hvbo_min_peak_m),
                    hvbo_workers,
                    backtest_date=hvbo_backtest_date,
                )

            if matches:
                display_rows = []
                for idx, row in enumerate(matches, start=1):
                    display_rows.append(
                        {
                            "S.No": idx,
                            "Coin": f"{row['symbol']}.P",
                            "Peak Vol Date": row["peak_vol_date"],
                            "Peak USDT Vol": format_large_number(row["peak_usdt_vol"]),
                            "Peak Day High": round(row["peak_day_high"], 6),
                            "Breakout Date": row["breakout_date"],
                            "Breakout Close": round(row["close"], 6),
                            "Close vs Peak High %": row["close_vs_peak_high_%"],
                            "Latest Close": round(row["latest_close"], 6),
                            "Latest vs Peak High %": row["latest_vs_peak_high_%"],
                            "Current Month L-H Change %": row.get("curr_month_low_to_high"),
                        }
                    )
                display_df = pd.DataFrame(display_rows)
                st.success(
                    f"✅ Found **{len(display_df)}** coins — peak vol in **{month_label}**, "
                    f"now breaking above that day's high."
                )
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                st.download_button(
                    label="📥 Download High Volume Breakout CSV",
                    data=display_df.to_csv(index=False),
                    file_name=f"hv_breakout_{month_label.replace(' ', '_')}_{datetime.now(IST).strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="hvbo_csv",
                )
            else:
                st.warning(
                    f"No matches for **{month_label}**. Try lowering min peak volume or disabling exclude above $1."
                )

    else:
        now_ist = datetime.now(IST)
        default_year = now_ist.year - 1

        col1, col2, col3 = st.columns(3)
        with col1:
            year_mode = st.radio(
                "Year Mode",
                ["Current Year", "Previous Year"],
                horizontal=True,
                key="hvbo_year_mode"
            )
            if year_mode == "Current Year":
                scan_year = now_ist.year
            else:
                prev_years = list(range(now_ist.year - 1, 2019, -1))
                scan_year = st.selectbox(
                    "Select Previous Year",
                    options=prev_years,
                    index=0,
                    key="hvbo_prev_year_select"
                )
        with col2:
            hvbo_exclude_above_1 = st.checkbox(
                "Exclude coins above $1",
                value=False,
                key="hvbo_exclude_above_1_yearly",
            )
            hvbo_only_active = st.checkbox(
                "Only show active breakouts (Latest close > Peak high)",
                value=False,
                key="hvbo_only_active_yearly",
            )
        with col3:
            hvbo_min_peak_m = st.number_input(
                "Min Peak Month Volume (M)",
                min_value=0.0,
                max_value=10000.0,
                value=0.0,
                step=10.0,
                key="hvbo_min_peak_m_yearly",
                help="e.g. 100 = only coins whose top volume month in that year had ≥ 100M USDT volume",
            )

        col_yw1, col_yw2 = st.columns(2)
        with col_yw1:
            hvbo_workers = st.slider(
                "Parallel Workers (Faster = More API calls)",
                min_value=5,
                max_value=50,
                value=25,
                key="hvbo_workers_yearly",
            )
        with col_yw2:
            hvbo_priority = st.checkbox(
                "Priority Scan (High volume first)",
                value=True,
                key="hvbo_priority_yearly",
            )

        # --- Backtest Mode ---
        hvbo_backtest_enabled_y = st.checkbox(
            "🧪 Backtest Mode (scan as of a past date)",
            value=False,
            key="hvbo_backtest_enabled_yearly",
        )
        hvbo_backtest_date_y = None
        if hvbo_backtest_enabled_y:
            hvbo_backtest_date_y = st.date_input(
                "📅 As-Of Date (simulate scan on this date)",
                value=datetime.now(IST).date(),
                key="hvbo_backtest_date_yearly",
                help="The scan will only use monthly candles available up to this date.",
            )
            st.info(
                f"🧪 **Backtest Mode ON** — Scanning **{scan_year}** peak month breakout as of **{hvbo_backtest_date_y.strftime('%d %b %Y')}**."
            )
        else:
            st.info(
                f"Scanning months of **{scan_year}** for peak USDT volume month, then checking monthly close breakout above that month's high."
            )

        if st.button(
            "🔍 Scan Yearly Peak Month Breakout",
            type="primary",
            use_container_width=True,
            key="hvbo_scan_btn_yearly",
        ):
            all_symbols = get_perpetual_symbols()
            if not all_symbols:
                st.error("Failed to load perpetual symbols.")
                return

            symbols_to_scan = all_symbols
            if hvbo_priority:
                top_volume = get_top_volume_symbols_list(limit=500)
                symbols_to_scan = [s for s in top_volume if s in all_symbols]
                symbols_to_scan.extend([s for s in all_symbols if s not in symbols_to_scan])

            with st.spinner(f"Scanning {len(symbols_to_scan)} coins for {scan_year} peak month breakout..."):
                matches = scan_yearly_peak_month_breakout_parallel(
                    symbols_to_scan,
                    int(scan_year),
                    hvbo_exclude_above_1,
                    float(hvbo_min_peak_m),
                    hvbo_only_active,
                    hvbo_workers,
                    backtest_date=hvbo_backtest_date_y,
                )

            if matches:
                display_rows = []
                for idx, row in enumerate(matches, start=1):
                    display_rows.append(
                        {
                            "S.No": idx,
                            "Coin": f"{row['symbol']}.P",
                            "Peak Month": row["peak_month"],
                            "Peak USDT Month Volume": row["peak_usdt_vol"],
                            "Peak Month High": round(row["peak_month_high"], 6),
                            "Peak Month Low": round(row["peak_month_low"], 6),
                            "Breakout Month": row["breakout_month"],
                            "Breakout Close": round(row["close"], 6),
                            "Close vs Peak High %": row["close_vs_peak_high_%"],
                            "Latest Close": round(row["latest_close"], 6),
                            "Latest vs Peak High %": row["latest_vs_peak_high_%"],
                        }
                    )
                display_df = pd.DataFrame(display_rows)
                st.success(
                    f"✅ Found **{len(display_df)}** coins with monthly breakout above {scan_year} peak month high."
                )

                def format_volume(val):
                    if pd.isna(val) or val is None:
                        return "N/A"
                    if val >= 1_000_000_000:
                        return f"{val / 1_000_000_000:.2f}B"
                    elif val >= 1_000_000:
                        return f"{val / 1_000_000:.2f}M"
                    return f"{val:.2f}"

                styled_df = display_df.style.format({
                    "Peak USDT Month Volume": format_volume
                })
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

                csv_df = display_df.copy()
                csv_df["Peak USDT Month Volume"] = csv_df["Peak USDT Month Volume"].apply(format_volume)

                st.download_button(
                    label="📥 Download Yearly Breakout CSV",
                    data=csv_df.to_csv(index=False),
                    file_name=f"yearly_hv_breakout_{scan_year}_{datetime.now(IST).strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="hvbo_csv_yearly",
                )
            else:
                st.warning(
                    f"No matches for year {scan_year}. Try lowering min peak month volume or changing settings."
                )


# ==============================================
# TAB 1: BREAKOUT SCANNER (WITH TIMEFRAME-BASED OI)
# ==============================================

def _render_breakout_scanner_modes():
    """Breakout detection, OI scanner, and volume filter modes."""
    st.markdown("""
    **Fast parallel scanner for Binance perpetual breakouts, volume & Open Interest**
    - 🔍 **Breakout Detection**: Find bullish/bearish breakouts with volume confirmation
    - 📈 **Volume Filter**: Find coins with consistent or average daily USDT volume over last N days
    - 📊 **Open Interest Scanner**: Find coins with largest OI changes based on your selected timeframe
    - 🎯 **Timeframe-Aware OI**: Compares OI for the SAME timeframe you select (Monthly, Weekly, Daily, 4H, 1H)
    - ⚡ **Ultra-fast parallel processing** (500+ coins in seconds)
    """)

    # ==========================================
    # SCAN MODE SELECTION
    # ==========================================
    scan_mode = st.radio(
        "Select Scan Mode:",
        ["🚀 Breakout Detection", "📊 Open Interest Scanner", "📈 Volume Filter"],
        horizontal=True,
        key="scan_mode_main"
    )

    col1, col2 = st.columns(2)

    with col1:
        timeframe = st.selectbox(
            "Timeframe",
            ["Monthly", "Weekly", "Daily", "4H", "1H"],
            index=2,  # Default to Daily
            key="breakout_tf_main",
            help="Select timeframe for analysis. OI will compare this period vs previous period"
        )

        n_coins = st.number_input(
            "Number of coins to display",
            min_value=5, max_value=100, value=20, step=5,
            key="breakout_n_main"
        )

        priority_scan = st.checkbox(
            "Priority Scan (High volume first)",
            value=True,
            key="breakout_priority_main"
        )

    with col2:
        if scan_mode == "🚀 Breakout Detection":
            min_vol_ratio = st.slider(
                "Minimum Volume Ratio",
                min_value=1, max_value=10, value=2,
                key="breakout_vol_main"
            )

            direction_filter = st.multiselect(
                "Breakout Direction",
                ["Bullish", "Bearish"],
                default=["Bullish", "Bearish"],
                key="breakout_dir_main"
            )

            breakout_type_filter = st.multiselect(
                "Breakout Type",
                ["Normal", "Trap"],
                default=["Normal", "Trap"],
                key="breakout_type_main"
            )

        elif scan_mode == "📈 Volume Filter":
            vol_filter_days = st.number_input(
                "No of Days",
                min_value=1,
                max_value=60,
                value=5,
                step=1,
                key="vol_filter_days",
                help="Check daily USDT volume for the last N completed days",
            )
            vol_filter_m = st.number_input(
                "Volume in M",
                min_value=1.0,
                max_value=5000.0,
                value=30.0,
                step=1.0,
                key="vol_filter_m",
                help="Minimum volume threshold in millions (USDT)",
            )
            vol_filter_type = st.selectbox(
                "Volume Filter Type",
                ["Consistent Volume", "Average volume or Average + Consistent"],
                key="vol_filter_type",
            )
            if vol_filter_type == "Average volume or Average + Consistent":
                vol_filter_subtype = st.selectbox(
                    "Average Mode",
                    ["Average volume", "Average + Consistent"],
                    key="vol_filter_subtype",
                )
            else:
                vol_filter_subtype = "Consistent Volume"
            vol_filter_exclude_above_1 = st.checkbox(
                "Exclude coins above $1 (BTC, ETH, XRP, etc.)",
                value=False,
                key="vol_filter_exclude_above_1",
                help="Skip coins whose latest completed daily close price is above $1",
            )
            vol_filter_use_price_range = st.checkbox(
                "Enable price range filter (optional)",
                value=False,
                key="vol_filter_use_price_range",
            )
            if vol_filter_use_price_range:
                price_col1, price_col2 = st.columns(2)
                with price_col1:
                    vol_filter_min_price = st.number_input(
                        "Min price ($)",
                        min_value=0.0,
                        max_value=100000.0,
                        value=0.01,
                        step=0.001,
                        format="%.6f",
                        key="vol_filter_min_price",
                    )
                with price_col2:
                    vol_filter_max_price = st.number_input(
                        "Max price ($)",
                        min_value=0.0,
                        max_value=100000.0,
                        value=0.90,
                        step=0.001,
                        format="%.6f",
                        key="vol_filter_max_price",
                    )
            else:
                vol_filter_min_price = 0.0
                vol_filter_max_price = 999999.0

            info_parts = [
                f"last **{int(vol_filter_days)}** days",
                f"threshold **{vol_filter_m:.0f}M USDT**",
                "daily volume shown **date-wise (M)** in results",
            ]
            if vol_filter_exclude_above_1:
                info_parts.append("excluding coins above **$1**")
            if vol_filter_use_price_range:
                info_parts.append(
                    f"price range **${vol_filter_min_price:.6f} – ${vol_filter_max_price:.6f}**"
                )
            st.info("Scanning " + " | ".join(info_parts))

        else:  # Open Interest Scanner
            # ==========================================
            # UPDATED: SLIDER RANGE 0.10 TO 25
            # ==========================================
            min_oi_change = st.slider(
                "Minimum OI Change %",
                min_value=0.10, max_value=25.0, value=1.0, step=0.10,
                format="%.2f%%",
                key="min_oi_main",
                help="Filter coins with OI change greater than this percentage (0.10% to 25%)"
            )

            oi_direction = st.radio(
                "OI Direction",
                ["Positive (Increasing)", "Negative (Decreasing)", "Both"],
                horizontal=True,
                key="oi_direction_main"
            )

            # Show timeframe info for OI
            st.info(f"📊 Comparing OI: **Current {timeframe} vs Previous {timeframe}**")

    parallel_workers = st.slider(
        "Parallel Workers (Faster = More API calls)",
        min_value=5, max_value=50, value=25,
        key="breakout_workers_main"
    )

    if st.button("🔍 Start Scan", type="primary", use_container_width=True, key="breakout_btn_main"):
        start_time = time.time()

        with st.spinner("Loading all perpetual pairs (500+)..."):
            all_symbols = get_perpetual_symbols()
            if not all_symbols:
                st.error("Failed to fetch perpetual pairs.")
                st.stop()

            if priority_scan:
                top_volume = get_top_volume_symbols_list(limit=500)
                symbols_to_scan = [s for s in top_volume if s in all_symbols]
                remaining = [s for s in all_symbols if s not in symbols_to_scan]
                symbols_to_scan.extend(remaining)
            else:
                symbols_to_scan = all_symbols

            st.success(f"✅ Loaded {len(all_symbols)} perpetual pairs")
            st.info(f"⚡ Scanning with {parallel_workers} parallel workers...")

        if scan_mode == "🚀 Breakout Detection":
            # ==========================================
            # BREAKOUT DETECTION MODE
            # ==========================================
            results = scan_breakouts_parallel(symbols_to_scan, timeframe, parallel_workers)

            if results:
                df_results = pd.DataFrame(results)
                df_results = df_results[df_results['Volume_Ratio_Num'] >= min_vol_ratio]

                if direction_filter:
                    df_results = df_results[df_results['Direction'].isin(direction_filter)]
                if breakout_type_filter:
                    df_results = df_results[df_results['Breakout Type'].isin(breakout_type_filter)]

                df_results = df_results.sort_values('Volume_Ratio_Num', ascending=False).head(n_coins)

                if len(df_results) > 0:
                    elapsed_time = time.time() - start_time
                    st.success(f"✅ Found {len(df_results)} breakouts in {elapsed_time:.1f} seconds!")

                    display_data = []
                    for idx, row in df_results.iterrows():
                        data_row = {
                            'S.No': len(display_data) + 1,
                            'Coin': row['Coin Name'],
                            'USDT Vol Ratio': row['USDT Volume Ratio'],
                            'Coin Vol Ratio': row['Coin volume Ratio'],
                            'OI Change %': f"{row['Open_Interest_change']}%",
                            'OI Period': row.get('OI_Period', timeframe),
                            'Type': row['Breakout Type'],
                            'Direction': row['Direction'],
                            'Spot': row['Spot Match']
                        }

                        if row['Direction'] == 'Bullish':
                            data_row['Open→High %'] = f"{row['Open to High Change %']}%"
                            data_row['Open→Close %'] = f"{row['Open to close Change %']}%"
                            data_row['Open→Low %'] = '-'
                        else:
                            data_row['Open→Low %'] = f"{row['Open to Low Change %']}%"
                            data_row['Open→Close %'] = f"{row['Open to close Change %']}%"
                            data_row['Open→High %'] = '-'

                        display_data.append(data_row)

                    display_df = pd.DataFrame(display_data)

                    # Color coding for OI Change
                    def color_oi(val):
                        if isinstance(val, str) and '%' in val:
                            try:
                                num = float(val.replace('%', ''))
                                if num > 20:
                                    return 'color: darkgreen; font-weight: bold'
                                elif num > 10:
                                    return 'color: green'
                                elif num > 5:
                                    return 'color: lightgreen'
                                elif num > 2:
                                    return 'color: #90EE90'
                                elif num > 0.5:
                                    return 'color: #E6FFE6'
                                elif num < -20:
                                    return 'color: darkred; font-weight: bold'
                                elif num < -10:
                                    return 'color: red'
                                elif num < -5:
                                    return 'color: lightcoral'
                                elif num < -2:
                                    return 'color: #FFB6C1'
                                elif num < -0.5:
                                    return 'color: #FFE6E6'
                            except:
                                pass
                        return ''

                    st.dataframe(
                        display_df.style.map(color_oi, subset=['OI Change %']),
                        use_container_width=True,
                        hide_index=True
                    )

                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Bullish", len(df_results[df_results['Direction'] == 'Bullish']))
                    with col2:
                        st.metric("Bearish", len(df_results[df_results['Direction'] == 'Bearish']))
                    with col3:
                        st.metric("Normal", len(df_results[df_results['Breakout Type'] == 'Normal']))
                    with col4:
                        st.metric("Trap", len(df_results[df_results['Breakout Type'] == 'Trap']))
                    with col5:
                        avg_oi = df_results['Open_Interest_change'].mean()
                        st.metric(f"Avg OI Change ({timeframe})", f"{avg_oi:.2f}%")

                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Breakout Results (CSV)",
                        data=csv,
                        file_name=f"breakout_{timeframe}_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("No breakouts match your filters.")
            else:
                st.warning("No breakouts found.")

        elif scan_mode == "📈 Volume Filter":
            if vol_filter_type == "Consistent Volume":
                active_filter_mode = "Consistent Volume"
            else:
                active_filter_mode = vol_filter_subtype

            with st.spinner(
                f"Scanning volume for {len(symbols_to_scan)} coins "
                f"({active_filter_mode}, last {int(vol_filter_days)} days)..."
            ):
                volume_results = scan_volume_filter_parallel(
                    symbols_to_scan,
                    int(vol_filter_days),
                    float(vol_filter_m),
                    active_filter_mode,
                    parallel_workers,
                    vol_filter_exclude_above_1,
                    vol_filter_use_price_range,
                    float(vol_filter_min_price),
                    float(vol_filter_max_price),
                )

            if volume_results:
                elapsed_time = time.time() - start_time
                st.success(
                    f"✅ Found {len(volume_results)} coins in {elapsed_time:.1f} seconds!"
                )

                display_df = build_volume_filter_display_df(volume_results)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                st.metric("Matching Coins", len(volume_results))
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Volume Filter Results (CSV)",
                    data=csv,
                    file_name=f"volume_filter_{int(vol_filter_days)}d_{int(vol_filter_m)}M_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="vol_filter_csv",
                )
            else:
                st.warning("No coins matched the volume filter criteria.")

        else:
            # ==========================================
            # OPEN INTEREST SCANNER MODE - TIMEFRAME AWARE
            # ==========================================
            with st.spinner(f"Scanning Open Interest changes for {len(symbols_to_scan)} coins..."):
                oi_results = scan_open_interest_parallel(
                    symbols_to_scan,
                    timeframe,
                    parallel_workers,
                    min_oi_change,
                    oi_direction
                )

                if oi_results:
                    df_oi = pd.DataFrame(oi_results)
                    df_oi = df_oi.sort_values('OI_Change_%', ascending=False).head(n_coins)

                    elapsed_time = time.time() - start_time
                    st.success(f"✅ Found {len(df_oi)} coins with significant OI changes in {elapsed_time:.1f} seconds!")
                    st.info(f"📊 OI Comparison: **Current {timeframe} vs Previous {timeframe}**")

                    # Display Open Interest results
                    display_data = []
                    for idx, row in df_oi.iterrows():
                        display_data.append({
                            'S.No': len(display_data) + 1,
                            'Coin': row['Coin Name'],
                            'OI Change %': f"{row['OI_Change_%']:.2f}%",
                            'Current OI': format_large_number(row['Current_OI']),
                            'Previous OI': format_large_number(row['Previous_OI']),
                            '24h Volume': row['24h_Volume_Display'],
                            'Price Change %': f"{row['Price_Change_%']:.2f}%",
                            'OI Trend': row['OI_Trend'],
                            'Signal': row['Signal']
                        })

                    display_df = pd.DataFrame(display_data)

                    # Color coding for OI Change
                    def color_oi_scan(val):
                        if isinstance(val, str) and '%' in val:
                            try:
                                num = float(val.replace('%', ''))
                                if num > 20:
                                    return 'color: darkgreen; font-weight: bold'
                                elif num > 10:
                                    return 'color: green'
                                elif num > 5:
                                    return 'color: lightgreen'
                                elif num > 2:
                                    return 'color: #90EE90'
                                elif num > 0.5:
                                    return 'color: #E6FFE6'
                                elif num < -20:
                                    return 'color: darkred; font-weight: bold'
                                elif num < -10:
                                    return 'color: red'
                                elif num < -5:
                                    return 'color: lightcoral'
                                elif num < -2:
                                    return 'color: #FFB6C1'
                                elif num < -0.5:
                                    return 'color: #FFE6E6'
                            except:
                                pass
                        return ''

                    st.dataframe(
                        display_df.style.map(color_oi_scan, subset=['OI Change %']),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Statistics for OI scan
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        positive_oi = len(df_oi[df_oi['OI_Change_%'] > 0])
                        st.metric(f"Positive OI ({timeframe})", positive_oi)
                    with col2:
                        negative_oi = len(df_oi[df_oi['OI_Change_%'] < 0])
                        st.metric(f"Negative OI ({timeframe})", negative_oi)
                    with col3:
                        strong_positive = len(df_oi[df_oi['OI_Change_%'] > 20])
                        st.metric("Strong Positive (>20%)", strong_positive)
                    with col4:
                        strong_negative = len(df_oi[df_oi['OI_Change_%'] < -20])
                        st.metric("Strong Negative (<-20%)", strong_negative)

                    # OI Distribution Chart
                    st.subheader(f"📊 OI Change Distribution ({timeframe} Comparison)")
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=df_oi['OI_Change_%'],
                        nbinsx=30,
                        marker_color='#4CAF50',
                        name=f'OI Change % ({timeframe})'
                    ))
                    fig.update_layout(
                        title=f'Distribution of Open Interest Changes ({timeframe} vs Previous {timeframe})',
                        xaxis_title=f'OI Change % ({timeframe})',
                        yaxis_title='Number of Coins',
                        template='plotly_white',
                        height=300,
                        bargap=0.1
                    )
                    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1)
                    st.plotly_chart(fig, use_container_width=True)

                    # Trading Signals based on OI
                    st.subheader("💡 OI Trading Signals")

                    tab1, tab2, tab3 = st.tabs(["🚀 Strong Bullish", "⚠️ Strong Bearish", "📊 OI-Price Divergence"])

                    with tab1:
                        strong_bullish = df_oi[df_oi['OI_Change_%'] > 20].head(10)
                        if not strong_bullish.empty:
                            for _, row in strong_bullish.iterrows():
                                st.markdown(f"""
                                - **{row['Coin Name']}**: OI +{row['OI_Change_%']:.2f}% ({timeframe}) | Price: {row['Price_Change_%']:.2f}% | Vol: {row['24h_Volume_Display']}
                                """)
                        else:
                            st.info(f"No strong bullish signals found for {timeframe} timeframe")

                    with tab2:
                        strong_bearish = df_oi[df_oi['OI_Change_%'] < -20].head(10)
                        if not strong_bearish.empty:
                            for _, row in strong_bearish.iterrows():
                                st.markdown(f"""
                                - **{row['Coin Name']}**: OI {row['OI_Change_%']:.2f}% ({timeframe}) | Price: {row['Price_Change_%']:.2f}% | Vol: {row['24h_Volume_Display']}
                                """)
                        else:
                            st.info(f"No strong bearish signals found for {timeframe} timeframe")

                    with tab3:
                        # Price up, OI down
                        divergence1 = df_oi[(df_oi['Price_Change_%'] > 3) & (df_oi['OI_Change_%'] < -10)].head(5)
                        # Price down, OI up
                        divergence2 = df_oi[(df_oi['Price_Change_%'] < -3) & (df_oi['OI_Change_%'] > 10)].head(5)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**⚠️ Price Up, OI Down (Weakness)**")
                            if not divergence1.empty:
                                for _, row in divergence1.iterrows():
                                    st.markdown(f"- {row['Coin Name']}: +{row['Price_Change_%']:.2f}% price, {row['OI_Change_%']:.2f}% OI ({timeframe})")
                            else:
                                st.info("None found")

                        with col2:
                            st.markdown("**⚠️ Price Down, OI Up (Accumulation)**")
                            if not divergence2.empty:
                                for _, row in divergence2.iterrows():
                                    st.markdown(f"- {row['Coin Name']}: {row['Price_Change_%']:.2f}% price, +{row['OI_Change_%']:.2f}% OI ({timeframe})")
                            else:
                                st.info("None found")

                    # Download OI scan results
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download OI Scan Results (CSV)",
                        data=csv,
                        file_name=f"oi_scan_{timeframe}_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                else:
                    st.warning(f"No significant Open Interest changes found for {timeframe} timeframe. Try adjusting the minimum OI change filter.")


def render_breakout_scanner():
    """Render Tab 1: Breakout Scanner with sub-tabs."""
    st.header("🚀 Breakout Scanner")

    tab_modes, tab_hvbo = st.tabs(["Scanner modes", "🔥 High Volume Breakout"])

    with tab_modes:
        _render_breakout_scanner_modes()

    with tab_hvbo:
        _render_high_volume_breakout_scanner()

# ==============================================
# TAB 2: SMART VOLUME SCANNER (FULL 500+ COINS)
# ==============================================

def analyze_volume_spike(symbol, klines, current_index, lookback=20):
    """Analyze volume spike with proper context"""
    if len(klines) < current_index + 1 or current_index < 1:
        return None, None, None, None

    current_kline = klines[current_index]
    previous_kline = klines[current_index - 1]

    current_trade_vol, current_usdt_vol = get_candle_volumes(current_kline)
    prev_trade_vol, prev_usdt_vol = get_candle_volumes(previous_kline)

    if prev_usdt_vol > 0:
        usdt_volume_ratio = current_usdt_vol / prev_usdt_vol
    else:
        usdt_volume_ratio = float('inf')

    usdt_volumes = []
    start_idx = max(0, current_index - lookback)
    for i in range(start_idx, current_index):
        if i < len(klines):
            _, usdt_vol = get_candle_volumes(klines[i])
            usdt_volumes.append(usdt_vol)

    usdt_volume_percentile = calculate_volume_percentile(current_usdt_vol, usdt_volumes) if usdt_volumes else 0

    return usdt_volume_ratio, usdt_volume_percentile, current_usdt_vol, prev_usdt_vol


def fetch_single_volume_spike_worker(
    symbol,
    timeframe,
    min_usdt_ratio,
    min_usdt_percentile,
    min_usdt_volume,
    backtest_date=None,
):
    """Worker for parallel volume spike scanning."""
    try:
        if backtest_date:
            end_date = backtest_date
            if timeframe == "1H":
                start_date = end_date - timedelta(days=7)
            elif timeframe == "4H":
                start_date = end_date - timedelta(days=14)
            elif timeframe == "Daily":
                start_date = end_date - timedelta(days=60)
            elif timeframe == "Weekly":
                start_date = end_date - timedelta(days=180)
            elif timeframe == "Monthly":
                start_date = end_date - timedelta(days=900)
            else:
                start_date = end_date - timedelta(days=30)

            klines = get_historical_klines(
                symbol, TIMEFRAME_MAP[timeframe], start_date, end_date, limit=100
            )
        else:
            kline_limit = 100 if timeframe != "Monthly" else 36
            klines = get_historical_klines(symbol, TIMEFRAME_MAP[timeframe], limit=kline_limit)

        if not klines or len(klines) < 21:
            return None

        latest_index = len(klines) - 1
        usdt_ratio, usdt_percentile, usdt_volume, _ = analyze_volume_spike(
            symbol, klines, latest_index, lookback=20
        )

        if not all(v is not None for v in [usdt_ratio, usdt_percentile, usdt_volume]):
            return None

        if (
            usdt_ratio >= min_usdt_ratio
            and usdt_percentile >= min_usdt_percentile
            and usdt_volume >= min_usdt_volume
        ):
            latest_kline = klines[latest_index]
            current_price = float(latest_kline[4])
            open_price = float(latest_kline[1])
            price_change = ((current_price - open_price) / open_price) * 100

            candle_timestamp = latest_kline[0]
            candle_date = datetime.fromtimestamp(candle_timestamp / 1000, tz=UTC)
            candle_date_ist = candle_date.astimezone(IST)

            return {
                "symbol": symbol,
                "usdt_volume_ratio": usdt_ratio,
                "usdt_volume_percentile": usdt_percentile,
                "current_usdt_volume": usdt_volume,
                "usdt_volume_display": format_large_number(usdt_volume),
                "price": current_price,
                "price_change_%": price_change,
                "candle_date": candle_date_ist.strftime("%Y-%m-%d %H:%M"),
                "timeframe": timeframe,
            }
        return None
    except Exception:
        return None


def scan_volume_spikes_parallel(
    symbols,
    timeframe,
    min_usdt_ratio,
    min_usdt_percentile,
    min_usdt_volume,
    max_workers=25,
    backtest_date=None,
):
    """Parallel volume spike scanning."""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(
                fetch_single_volume_spike_worker,
                symbol,
                timeframe,
                min_usdt_ratio,
                min_usdt_percentile,
                min_usdt_volume,
                backtest_date,
            ): symbol
            for symbol in symbols
        }

        completed = 0
        total = len(symbols)

        for future in as_completed(future_to_symbol):
            completed += 1
            if completed % 10 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(
                    f"Volume scanning {completed}/{total} pairs... Found: {len(results)} spikes"
                )

            try:
                result = future.result(timeout=10)
                if result:
                    results.append(result)
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()
    return results


def enhanced_volume_scanner_full(timeframe="1H", min_usdt_ratio=2.0, min_usdt_percentile=80,
                               min_usdt_volume=100000, top_n=20, backtest_date=None,
                               specific_coins=None, priority_scan=True, max_workers=25):
    """Enhanced scanner using USDT volume - SCANS ALL COINS"""

    if specific_coins:
        symbols_to_scan = specific_coins
    else:
        all_symbols = get_perpetual_symbols()
        if priority_scan:
            volume_data = get_top_volume_symbols(500)
            symbols_to_scan = [item['symbol'] for item in volume_data if item['symbol'] in all_symbols]
        else:
            symbols_to_scan = all_symbols

    results = scan_volume_spikes_parallel(
        symbols_to_scan,
        timeframe,
        min_usdt_ratio,
        min_usdt_percentile,
        min_usdt_volume,
        max_workers,
        backtest_date,
    )

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('usdt_volume_ratio', ascending=False)
        df = df.head(top_n).reset_index(drop=True)
        df.index = df.index + 1
        df = df.rename_axis('Rank').reset_index()
        return df
    else:
        return pd.DataFrame()

def _render_smart_volume_scanner_content():
    """Render original Smart Volume Scanner - FULL SCAN"""
    st.subheader("💎 Smart Volume Scanner")
    st.markdown("**Advanced volume scanning with USDT volume analysis - SCANS ALL 500+ COINS**")

    col1, col2 = st.columns(2)

    with col1:
        timeframe = st.selectbox(
            "Scan Timeframe:",
            options=["1H", "4H", "Daily", "Weekly", "Monthly"],
            index=1,
            key="volume_tf"
        )
        top_n = st.number_input("Number of results:", min_value=1, max_value=50, value=20, key="volume_n")
        priority_scan = st.checkbox("Priority Scan (High volume first)", value=True, key="volume_priority")

    with col2:
        min_usdt_ratio = st.slider("Min USDT Volume Ratio:", 1.0, 10.0, 2.0, 0.1, key="volume_ratio")
        min_usdt_percentile = st.slider("Min Volume Percentile:", 50, 99, 80, 1, key="volume_percentile")

    min_usdt_volume = st.slider("Min USDT Volume ($):", 1000, 1000000, 100000, 10000, key="volume_min")

    parallel_workers = st.slider(
        "Parallel Workers (Faster = More API calls)",
        min_value=5, max_value=50, value=25,
        key="volume_workers"
    )

    backtest_mode = st.checkbox("Backtest Mode", value=False, key="volume_backtest")

    if backtest_mode:
        backtest_date = st.date_input("Backtest Date:", value=datetime.now(IST).date() - timedelta(days=7))
        backtest_date_dt = datetime.combine(backtest_date, datetime.min.time(), tzinfo=IST)
    else:
        backtest_date_dt = None

    if st.button("🚀 Run Smart Volume Scan (Full 500+ Coins)", type="primary", use_container_width=True, key="volume_btn"):
        start_time = time.time()
        with st.spinner(f"Smart scanning ALL {len(get_perpetual_symbols())} coins..."):
            st.info(f"⚡ Scanning with {parallel_workers} parallel workers...")
            results_df = enhanced_volume_scanner_full(
                timeframe=timeframe,
                min_usdt_ratio=min_usdt_ratio,
                min_usdt_percentile=min_usdt_percentile,
                min_usdt_volume=min_usdt_volume,
                top_n=top_n,
                backtest_date=backtest_date_dt,
                priority_scan=priority_scan,
                max_workers=parallel_workers,
            )

            if not results_df.empty:
                elapsed_time = time.time() - start_time
                st.success(
                    f"✅ Found {len(results_df)} high-quality volume spikes "
                    f"from full scan in {elapsed_time:.1f} seconds!"
                )

                display_cols = ['Rank', 'symbol', 'usdt_volume_ratio', 'usdt_volume_percentile',
                              'usdt_volume_display', 'price', 'price_change_%', 'candle_date']

                st.dataframe(
                    results_df[display_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Rank": st.column_config.NumberColumn("Rank", width="small"),
                        "symbol": st.column_config.TextColumn("Coin", width="medium"),
                        "usdt_volume_ratio": st.column_config.NumberColumn("Vol Ratio", width="small", format="%.1fx"),
                        "usdt_volume_percentile": st.column_config.NumberColumn("Percentile", width="small", format="%.0f%%"),
                        "usdt_volume_display": st.column_config.TextColumn("USDT Vol", width="small"),
                        "price": st.column_config.NumberColumn("Price", width="small", format="%.4f"),
                        "price_change_%": st.column_config.NumberColumn("Change %", width="small", format="%.1f%%"),
                        "candle_date": st.column_config.TextColumn("Date/Time", width="medium")
                    }
                )

                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name=f"volume_scan_full_{timeframe}_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No volume spikes found. Try adjusting parameters.")


# ==============================================================================
# 4HR CANDLE MOVE SCANNER (IST ALIGNED: 6 CANDLES PER 24-HOUR DAY)
# ==============================================================================

FOUR_HOUR_SLOT_DEFS = [
    {
        "id": 0,
        "utc_hour": 0,
        "label": "5.30 TO 9.30",
        "label_detail": "5.30 TO 9.30 (AM)",
        "session": "🌏 Asia / Tokyo Open",
        "short_col": "5.30 TO 9.30",
    },
    {
        "id": 1,
        "utc_hour": 4,
        "label": "9.30 TO 1.30",
        "label_detail": "9.30 TO 1.30 (AM)",
        "session": "🇨🇳 Asia Midday / HK",
        "short_col": "9.30 TO 1.30",
    },
    {
        "id": 2,
        "utc_hour": 8,
        "label": "1.30 TO 5.30",
        "label_detail": "1.30 TO 5.30 (PM)",
        "session": "🇬🇧 London / Europe Open",
        "short_col": "1.30 TO 5.30",
    },
    {
        "id": 3,
        "utc_hour": 12,
        "label": "5.30 TO 9.30",
        "label_detail": "5.30 TO 9.30 (PM)",
        "session": "🇺🇸 US New York Open",
        "short_col": "5.30 TO 9.30 ",  # Trailing space keeps DataFrame column unique while displaying cleanly
    },
    {
        "id": 4,
        "utc_hour": 16,
        "label": "9.30 TO 1.30",
        "label_detail": "9.30 TO 1.30 (PM)",
        "session": "🗽 US Midday / NY Afternoon",
        "short_col": "9.30 TO 1.30 ",
    },
    {
        "id": 5,
        "utc_hour": 20,
        "label": "1.30 TO 5.30",
        "label_detail": "1.30 TO 5.30 (AM)",
        "session": "🌙 US Close / APAC Night",
        "short_col": "1.30 TO 5.30 ",
    },
]


def fetch_coin_4hr_day_candles(symbol, target_date):
    """
    Fetch and calculate 4HR candle moves for a coin on target_date.
    In IST (UTC+5:30), the 24-hour day runs from 05:30 AM on target_date to 05:30 AM on target_date+1.
    In UTC, this corresponds exactly to 00:00:00 to 24:00:00 on target_date.

    Calculation rules:
    - Bull candle (Close >= Open): Low to High % -> +((High - Low) / Low) * 100
    - Bear candle (Close < Open): High to Low % -> -((High - Low) / High) * 100
    - Volume Spike: Compares USDT volume to 20-period 4H average volume (>= 1.5x)
    """
    try:
        start_dt_utc = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=ZoneInfo("UTC"))
        start_ms = int(start_dt_utc.timestamp() * 1000)
        end_ms = start_ms + (24 * 3600 * 1000) - 1
        # Lookback 4 days (24 4h candles) to compute average 4H volume baseline
        lookback_ms = start_ms - (24 * 4 * 3600 * 1000)

        params = {
            'symbol': symbol,
            'interval': '4h',
            'startTime': lookback_ms,
            'endTime': end_ms,
            'limit': 50
        }
        res = api_get(f"{BINANCE_API_URL}/fapi/v1/klines", params=params, timeout=6)
        data = res.json()
        if not isinstance(data, list) or len(data) == 0:
            return None

        # Lookback 4H volume baseline
        lookback_candles = [k for k in data if k[0] < start_ms]
        avg_4h_vol = np.mean([float(k[7]) for k in lookback_candles]) if lookback_candles else 0.0

        now_ms = int(time.time() * 1000)
        slot_results = {}
        all_abs_moves = []
        total_vol = 0.0

        for slot in FOUR_HOUR_SLOT_DEFS:
            idx = slot['id']
            expected_start = start_ms + idx * 4 * 3600 * 1000
            matched = [k for k in data if abs(k[0] - expected_start) < 60000]

            if matched:
                k = matched[0]
                o = float(k[1])
                h = float(k[2])
                l = float(k[3])
                c = float(k[4])
                vol = float(k[7])
                close_time_ms = int(k[6])
                is_live = now_ms < close_time_ms

                is_bull = c >= o
                # Rules:
                # If bull candle -> calculate low to high %
                # If bear candle -> calculate high to low % (negative)
                if is_bull:
                    move_pct = ((h - l) / l * 100.0) if l > 0 else 0.0
                else:
                    move_pct = -((h - l) / h * 100.0) if h > 0 else 0.0

                vol_ratio = (vol / avg_4h_vol) if avg_4h_vol > 0 else 1.0
                is_vol_spike = (vol_ratio >= 1.5 and vol > 15000)

                slot_results[idx] = {
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c,
                    'is_bull': is_bull,
                    'move_pct': round(move_pct, 2),
                    'vol_usdt': vol,
                    'vol_ratio': round(vol_ratio, 2),
                    'is_vol_spike': is_vol_spike,
                    'is_live': is_live
                }
                all_abs_moves.append(abs(move_pct))
                total_vol += vol
            else:
                slot_results[idx] = None

        # Determine best slot for this coin
        best_slot_idx = None
        max_abs_move = -1.0
        for idx, s in slot_results.items():
            if s is not None:
                if abs(s['move_pct']) > max_abs_move:
                    max_abs_move = abs(s['move_pct'])
                    best_slot_idx = idx

        total_range = max_abs_move if max_abs_move >= 0 else 0.0
        best_slot_label = FOUR_HOUR_SLOT_DEFS[best_slot_idx]['label'] if best_slot_idx is not None else "—"

        return {
            'symbol': symbol,
            'display_symbol': f"{symbol}.P",
            'slots': slot_results,
            'total_range': round(total_range, 2),
            'best_slot': best_slot_label,
            'best_slot_idx': best_slot_idx,
            'total_vol': total_vol,
            'avg_4h_vol': avg_4h_vol
        }
    except Exception:
        return None


def scan_multicoin_4hr_day(symbols, target_date, max_workers=30):
    """Scan coins for 4HR moves in parallel with a responsive progress bar"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(symbols)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sym = {
            executor.submit(fetch_coin_4hr_day_candles, sym, target_date): sym
            for sym in symbols
        }
        for future in as_completed(future_to_sym):
            completed += 1
            if completed % 10 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(f"Scanning 4HR candle moves: {completed}/{total} coins...")
            try:
                res = future.result(timeout=10)
                if res:
                    results.append(res)
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()
    return results


def fetch_multicoin_4hr_multiday_agg(symbols, days=14, max_workers=25):
    """
    Fetch multi-day 4H history to compute historical average move % and win rate per slot across coins.
    """
    candles_needed = days * 6 + 10
    hour_to_slot = {0: 0, 4: 1, 8: 2, 12: 3, 16: 4, 20: 5}

    slot_stats = {
        i: {
            'moves': [],
            'bull_moves': [],
            'bear_moves': [],
            'is_bull': [],
            'vols': [],
            'count': 0
        }
        for i in range(6)
    }

    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(symbols)
    completed = 0

    def _fetch_single_sym(sym):
        try:
            params = {
                'symbol': sym,
                'interval': '4h',
                'limit': min(candles_needed, 500)
            }
            res = api_get(f"{BINANCE_API_URL}/fapi/v1/klines", params=params, timeout=5)
            data = res.json()
            return data if isinstance(data, list) else []
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sym = {executor.submit(_fetch_single_sym, s): s for s in symbols}
        for future in as_completed(future_to_sym):
            completed += 1
            if completed % 5 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(f"Aggregating {days}-day 4HR historical potential: {completed}/{total} coins...")
            try:
                klines = future.result(timeout=10)
                for k in klines:
                    dt = datetime.fromtimestamp(k[0] / 1000, tz=ZoneInfo("UTC"))
                    slot_idx = hour_to_slot.get(dt.hour)
                    if slot_idx is not None:
                        o = float(k[1])
                        h = float(k[2])
                        l = float(k[3])
                        c = float(k[4])
                        vol = float(k[7])
                        is_bull = c >= o
                        if is_bull:
                            move = ((h - l) / l * 100.0) if l > 0 else 0.0
                            slot_stats[slot_idx]['bull_moves'].append(move)
                        else:
                            move = -((h - l) / h * 100.0) if h > 0 else 0.0
                            slot_stats[slot_idx]['bear_moves'].append(abs(move))
                        slot_stats[slot_idx]['moves'].append(abs(move))
                        slot_stats[slot_idx]['is_bull'].append(1 if is_bull else 0)
                        slot_stats[slot_idx]['vols'].append(vol)
                        slot_stats[slot_idx]['count'] += 1
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()
    return slot_stats


def fetch_single_coin_history_slots(symbol, days=14, include_volume=False):
    """Fetch day-by-day 6-slot moves for a single coin over the last N days.
    When include_volume=True, also fetches OI history and returns enriched rows
    with per-slot USDT volume, daily totals, OI, and divergence flags.
    """
    candles_needed = days * 6 + 10
    params = {
        'symbol': symbol,
        'interval': '4h',
        'limit': min(candles_needed, 500)
    }
    try:
        res = api_get(f"{BINANCE_API_URL}/fapi/v1/klines", params=params, timeout=6)
        data = res.json()
        if not isinstance(data, list):
            return []

        days_map = {}
        for k in data:
            dt = datetime.fromtimestamp(k[0] / 1000, tz=ZoneInfo("UTC"))
            date_key = dt.date()
            if date_key not in days_map:
                days_map[date_key] = {}

            slot_id = dt.hour // 4
            o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
            vol_base = float(k[5])    # Coin volume (base asset)
            vol_usdt = float(k[7])    # USDT volume (quote asset)
            is_bull = c >= o
            move = ((h - l) / l * 100.0) if is_bull else -((h - l) / h * 100.0)
            days_map[date_key][slot_id] = {
                'move_pct': round(move, 2),
                'is_bull': is_bull,
                'vol_usdt': vol_usdt,
                'vol_base': vol_base,
                'open': o, 'high': h, 'low': l, 'close': c
            }

        # ---------- Fetch OI history (daily) if requested ----------
        oi_by_date = {}
        if include_volume:
            try:
                sorted_dates = sorted(days_map.keys())
                if sorted_dates:
                    oi_start = int(datetime(sorted_dates[0].year, sorted_dates[0].month,
                                            sorted_dates[0].day, tzinfo=ZoneInfo("UTC")).timestamp())
                    oi_end = int(time.time())
                    oi_data = _fetch_oi_hist_chunk(symbol, "1d", oi_start, oi_end)
                    if oi_data and isinstance(oi_data, list):
                        for item in oi_data:
                            ts_ms = int(item.get("timestamp", 0))
                            if ts_ms > 0:
                                # Binance OI timestamp is period-end; subtract 1 day to get the date
                                dt_oi = datetime.fromtimestamp((ts_ms / 1000) - 86400, tz=ZoneInfo("UTC")).date()
                                oi_by_date[dt_oi] = {
                                    'oi_contracts': float(item.get('sumOpenInterest', 0)),
                                    'oi_usdt': float(item.get('sumOpenInterestValue', 0))
                                }
            except Exception:
                pass

        rows = []
        prev_oi_usdt = None
        for d in sorted(days_map.keys(), reverse=True):
            slots = days_map[d]
            row = {'Date': d.strftime('%Y-%m-%d'), '_date_obj': d}
            max_m = 0.0
            best_s = "—"
            day_usdt_vol = 0.0
            day_base_vol = 0.0
            best_vw_score = -1.0       # volume-weighted score
            best_vw_slot = "—"
            day_net_move = 0.0         # sum of signed moves for divergence

            for s_def in FOUR_HOUR_SLOT_DEFS:
                s_id = s_def['id']
                col = s_def['short_col']
                if s_id in slots:
                    s = slots[s_id]
                    m = s['move_pct']
                    v_usdt = s['vol_usdt']
                    v_base = s['vol_base']

                    if include_volume:
                        # Combined cell: move% | $vol
                        vol_fmt = f"${v_usdt/1e6:.1f}M" if v_usdt >= 1e6 else f"${v_usdt/1e3:.0f}K"
                        row[col] = f"{m:+.2f}% | {vol_fmt}"
                    else:
                        row[col] = f"{m:+.2f}%"

                    # Store raw data for sparkline/summary
                    row[f'_raw_{s_id}_move'] = m
                    row[f'_raw_{s_id}_vol'] = v_usdt

                    day_usdt_vol += v_usdt
                    day_base_vol += v_base
                    day_net_move += m

                    if abs(m) > max_m:
                        max_m = abs(m)
                        best_s = s_def['label']

                    # Volume-weighted score: |move%| * vol_usdt
                    vw_score = abs(m) * v_usdt
                    if vw_score > best_vw_score:
                        best_vw_score = vw_score
                        best_vw_slot = s_def['label']
                else:
                    row[col] = "—"
                    row[f'_raw_{s_id}_move'] = None
                    row[f'_raw_{s_id}_vol'] = None

            row['Max Move %'] = f"{max_m:.2f}%" if max_m > 0 else "—"
            row['Best Slot'] = best_s

            if include_volume:
                # Daily USDT Volume
                row['24H USDT Vol'] = f"${day_usdt_vol/1e6:.1f}M" if day_usdt_vol >= 1e6 else f"${day_usdt_vol/1e3:.0f}K"
                row['_raw_usdt_vol'] = day_usdt_vol

                # Daily Coin Volume
                if day_base_vol >= 1e6:
                    row['Coin Vol'] = f"{day_base_vol/1e6:.2f}M"
                elif day_base_vol >= 1e3:
                    row['Coin Vol'] = f"{day_base_vol/1e3:.1f}K"
                else:
                    row['Coin Vol'] = f"{day_base_vol:.1f}"
                row['_raw_coin_vol'] = day_base_vol

                # OI data
                oi_info = oi_by_date.get(d)
                if oi_info:
                    oi_usdt = oi_info['oi_usdt']
                    row['OI (USDT)'] = f"${oi_usdt/1e9:.2f}B" if oi_usdt >= 1e9 else f"${oi_usdt/1e6:.1f}M"
                    row['_raw_oi'] = oi_usdt

                    if prev_oi_usdt and prev_oi_usdt > 0:
                        oi_chg = ((oi_usdt - prev_oi_usdt) / prev_oi_usdt) * 100
                        emoji = "🟢" if oi_chg > 0 else "🔴" if oi_chg < 0 else "⚪"
                        row['OI Chg %'] = f"{oi_chg:+.2f}% {emoji}"
                        row['_raw_oi_chg'] = round(oi_chg, 2)

                        # OI vs Price divergence: price up but OI down, or price down but OI up significantly
                        if (day_net_move > 0 and oi_chg < -2.0) or (day_net_move < 0 and oi_chg > 5.0):
                            row['OI Chg %'] += " ⚠️"
                            row['_divergence'] = True
                        else:
                            row['_divergence'] = False
                    else:
                        row['OI Chg %'] = "—"
                        row['_raw_oi_chg'] = None
                        row['_divergence'] = False
                    prev_oi_usdt = oi_usdt
                else:
                    row['OI (USDT)'] = "—"
                    row['OI Chg %'] = "—"
                    row['_raw_oi'] = None
                    row['_raw_oi_chg'] = None
                    row['_divergence'] = False

                # Volume-weighted best slot
                row['🎯 VW Best'] = best_vw_slot

            rows.append(row)
        return rows
    except Exception:
        return []


def _style_4hr_move_cell(val):
    """Excel-style coloring: Soft green for positive moves, soft red for negative moves"""
    if not isinstance(val, str) or val in ("", "—", "N/A", "-"):
        return "text-align: center; color: #888;"
    clean = val.replace("%", "").replace("+", "").replace("🔥", "").strip()
    try:
        f = float(clean)
        if f > 0:
            return "background-color: #d4edda; color: #155724; font-weight: 700; text-align: center;"
        elif f < 0:
            return "background-color: #f8d7da; color: #721c24; font-weight: 700; text-align: center;"
        else:
            return "text-align: center;"
    except Exception:
        return "text-align: center;"


def generate_excel_4hr_moves(table_rows, slot_metrics, target_date):
    """
    Generate styled Excel (.xlsx) matching the user's exact spreadsheet mockup format:
    - Column A: S.No (1, 2, 3...)
    - Column B: Coin (BTCUSDT.P, XPRUSDT.P...)
    - Columns C-H: 5.30 TO 9.30, 9.30 TO 1.30, 1.30 TO 5.30, 5.30 TO 9.30, 9.30 TO 1.30, 1.30 TO 5.30
    - Soft green fill (#E2EFDA) for Bullish moves
    - Soft red/orange fill (#FCE4D6) for Bearish moves
    - Thin borders around every cell and gridlines explicitly enabled
    - Sheet 2: Detailed Bullish Counts & Maximum Move % Summary by Timeframe
    """
    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    except ImportError:
        return None

    wb = openpyxl.Workbook()

    # --- Sheet 1: 4HR Candle Moves Matrix ---
    ws = wb.active
    ws.title = "4HR Candle Moves"
    ws.views.sheetView[0].showGridLines = True

    headers = ['', 'Coin', '5.30 TO 9.30', '9.30 TO 1.30', '1.30 TO 5.30', '5.30 TO 9.30', '9.30 TO 1.30', '1.30 TO 5.30']
    ws.append(headers)

    header_fill = PatternFill(start_color='F2F2F2', end_color='F2F2F2', fill_type='solid')
    header_font = Font(name='Calibri', size=11, bold=True, color='000000')
    thin_border = Border(
        left=Side(style='thin', color='D9D9D9'),
        right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'),
        bottom=Side(style='thin', color='D9D9D9')
    )
    center_align = Alignment(horizontal='center', vertical='center')
    left_align = Alignment(horizontal='left', vertical='center')

    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_align
        cell.border = thin_border

    # Fills matching user's spreadsheet screenshot
    green_fill = PatternFill(start_color='E2EFDA', end_color='E2EFDA', fill_type='solid')
    green_font = Font(name='Calibri', size=11, color='276A3C', bold=False)
    red_fill = PatternFill(start_color='FCE4D6', end_color='FCE4D6', fill_type='solid')
    red_font = Font(name='Calibri', size=11, color='C00000', bold=False)
    neutral_font = Font(name='Calibri', size=11, color='595959', bold=False)

    slot_col_keys = [
        '5.30 TO 9.30', '9.30 TO 1.30', '1.30 TO 5.30',
        '5.30 TO 9.30 ', '9.30 TO 1.30 ', '1.30 TO 5.30 '
    ]

    for idx, r in enumerate(table_rows, start=1):
        coin = r.get('Coin', '')
        moves_vals = []
        for sk in slot_col_keys:
            v = r.get(sk)
            if v is None:
                v = r.get(sk.strip(), '—')
            moves_vals.append(v)

        row_cells = [idx, coin] + moves_vals
        ws.append(row_cells)
        row_idx = ws.max_row

        # Col A: S.No
        c_sno = ws.cell(row=row_idx, column=1)
        c_sno.alignment = center_align
        c_sno.border = thin_border
        c_sno.font = neutral_font

        # Col B: Coin
        c_coin = ws.cell(row=row_idx, column=2)
        c_coin.alignment = left_align
        c_coin.border = thin_border
        c_coin.font = Font(name='Calibri', size=11, bold=True)

        # Cols C-H: Slots
        for m_i, m_str in enumerate(moves_vals):
            c_slot = ws.cell(row=row_idx, column=3 + m_i)
            c_slot.alignment = center_align
            c_slot.border = thin_border

            clean = str(m_str).replace('%', '').replace('+', '').replace('🔥', '').strip()
            try:
                val_float = float(clean)
                if val_float > 0:
                    c_slot.fill = green_fill
                    c_slot.font = green_font
                elif val_float < 0:
                    c_slot.fill = red_fill
                    c_slot.font = red_font
                else:
                    c_slot.font = neutral_font
            except:
                c_slot.font = neutral_font

    ws.column_dimensions['A'].width = 6
    ws.column_dimensions['B'].width = 16
    for c in ['C', 'D', 'E', 'F', 'G', 'H']:
        ws.column_dimensions[c].width = 15

    # --- Sheet 2: Detailed Bullish & Maximum Move Summary ---
    if slot_metrics:
        ws_sum = wb.create_sheet(title="Bullish & Max Move Summary")
        ws_sum.views.sheetView[0].showGridLines = True
        sum_headers = [
            'Rank', 'Time Slot', 'Session Name', 'Bullish Coins Count',
            'Bearish Coins Count', 'Max Move %', 'Top Mover Coin', 'Avg Move %', 'Total Volume'
        ]
        ws_sum.append(sum_headers)
        for col_idx in range(1, len(sum_headers) + 1):
            cell = ws_sum.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_align
            cell.border = thin_border

        for rank, s in enumerate(slot_metrics, start=1):
            vol_fmt = f"${s['total_vol'] / 1e6:.1f}M" if s['total_vol'] >= 1e6 else f"${s['total_vol']:,.0f}"
            row_data = [
                f"#{rank}",
                s['label'],
                s['session'],
                f"{s.get('bull_count', 0)} coins ({s['bull_pct']:.1f}%)",
                f"{s.get('bear_count', 0)} coins ({s['bear_pct']:.1f}%)",
                f"{s.get('max_move', 0.0):.2f}%",
                s.get('top_mover_coin', '—'),
                f"{s['avg_abs_move']:.2f}%",
                vol_fmt
            ]
            ws_sum.append(row_data)
            r_idx = ws_sum.max_row
            for col_idx in range(1, len(row_data) + 1):
                c = ws_sum.cell(row=r_idx, column=col_idx)
                c.alignment = center_align
                c.border = thin_border

        ws_sum.column_dimensions['A'].width = 8
        ws_sum.column_dimensions['B'].width = 16
        ws_sum.column_dimensions['C'].width = 28
        ws_sum.column_dimensions['D'].width = 22
        ws_sum.column_dimensions['E'].width = 22
        ws_sum.column_dimensions['F'].width = 15
        ws_sum.column_dimensions['G'].width = 16
        ws_sum.column_dimensions['H'].width = 15
        ws_sum.column_dimensions['I'].width = 16

    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def _render_4hr_multicoin_mode():
    """Render Multicoin Mode for 4HR Candle Move Scanner"""
    st.markdown("#### 🌐 Multicoin Market Scan & Potential Summary")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 1.6])

    with ctrl_col1:
        date_preset = st.radio(
            "📅 Backtest Date Selection:",
            options=["Yesterday", "Today (Live)", "Custom Backtest Date"],
            index=0,
            horizontal=False,
            key="scan_4hr_date_preset"
        )
        today_ist = datetime.now(IST).date()
        if date_preset == "Yesterday":
            target_date = today_ist - timedelta(days=1)
        elif date_preset == "Today (Live)":
            target_date = today_ist
        else:
            target_date = st.date_input(
                "Pick Historical Date:",
                value=today_ist - timedelta(days=2),
                max_value=today_ist,
                key="scan_4hr_custom_date"
            )

    with ctrl_col2:
        sub_view = st.radio(
            "📊 View Analysis Mode:",
            options=["📅 Single Day 6-Slot Table", "📈 Multi-Day Potential Aggregator"],
            index=0,
            key="scan_4hr_sub_view"
        )
        scope_option = st.selectbox(
            "Coins Universe Scope:",
            options=["Top 100 Volume Coins", "Top 50 Volume Coins", "All Perpetual Coins (500+)"],
            index=0,
            key="scan_4hr_scope"
        )

    with ctrl_col3:
        show_vol_spike = st.toggle("🔥 Highlight Volume Spikes (>= 1.5x 4H Avg)", value=True, key="scan_4hr_vol_spike_toggle")
        use_detailed_headers = st.toggle("Detailed Headers (with AM/PM & Sessions)", value=False, key="scan_4hr_headers_toggle")
        parallel_workers = st.slider("Parallel Workers (Speed):", min_value=10, max_value=40, value=30, key="scan_4hr_workers")

    st.markdown("---")

    # =========================================================================
    # VIEW 1: SINGLE DAY 6-SLOT TABLE (MATCHING EXCEL SPREADSHEET MOCKUP)
    # =========================================================================
    if sub_view == "📅 Single Day 6-Slot Table":
        col_btn1, col_btn2 = st.columns([2, 5])
        with col_btn1:
            run_scan = st.button(
                f"🚀 Scan 4HR Moves for {target_date.strftime('%d %b %Y')}",
                type="primary",
                use_container_width=True,
                key="scan_4hr_run_btn"
            )

        if run_scan:
            all_symbols = get_perpetual_symbols()
            if scope_option == "Top 50 Volume Coins":
                top_v = get_top_volume_symbols_list(50)
                scan_symbols = [s for s in top_v if s in all_symbols] or all_symbols[:50]
            elif scope_option == "Top 100 Volume Coins":
                top_v = get_top_volume_symbols_list(100)
                scan_symbols = [s for s in top_v if s in all_symbols] or all_symbols[:100]
            else:
                scan_symbols = all_symbols

            start_t = time.time()
            with st.spinner(f"Scanning 4HR candle moves across {len(scan_symbols)} coins for {target_date}..."):
                results = scan_multicoin_4hr_day(scan_symbols, target_date, max_workers=parallel_workers)
                elapsed = time.time() - start_t
                st.session_state['scan_4hr_results'] = results
                st.session_state['scan_4hr_date'] = target_date
                st.session_state['scan_4hr_elapsed'] = elapsed

        if 'scan_4hr_results' in st.session_state and st.session_state.get('scan_4hr_results'):
            results = st.session_state['scan_4hr_results']
            res_date = st.session_state.get('scan_4hr_date', target_date)
            elapsed = st.session_state.get('scan_4hr_elapsed', 0.0)

            st.success(f"✅ Analyzed 4HR candles across **{len(results)} coins** for **{res_date.strftime('%d %b %Y')}** in **{elapsed:.1f}s**!")

            # -----------------------------------------------------------------
            # COMPUTE METRICS: BULLISH NUMBERS & MAXIMUM MOVE %
            # -----------------------------------------------------------------
            slot_metrics = []
            for s_def in FOUR_HOUR_SLOT_DEFS:
                s_id = s_def['id']
                slot_data_list = [
                    (r['display_symbol'], r['slots'][s_id]['move_pct'])
                    for r in results if r['slots'].get(s_id) is not None
                ]
                if slot_data_list:
                    moves = [m for _, m in slot_data_list]
                    abs_moves = [abs(m) for m in moves]
                    avg_abs = float(np.mean(abs_moves))
                    bulls = [m for m in moves if m >= 0]
                    bears = [m for m in moves if m < 0]
                    bull_count = len(bulls)
                    bear_count = len(bears)
                    total_count = len(moves)
                    bull_pct = (bull_count / total_count * 100.0) if total_count > 0 else 0.0
                    bear_pct = (bear_count / total_count * 100.0) if total_count > 0 else 0.0
                    avg_bull = float(np.mean(bulls)) if bulls else 0.0
                    avg_bear = float(np.mean(bears)) if bears else 0.0
                    max_move = max(abs_moves) if abs_moves else 0.0

                    # Find top mover coin for this slot
                    top_mover_coin = "—"
                    top_mover_move = 0.0
                    for c_sym, m in slot_data_list:
                        if abs(m) == max_move:
                            top_mover_coin = c_sym
                            top_mover_move = m
                            break

                    explosive_3 = sum(1 for m in abs_moves if m >= 3.0)
                    total_vol = sum(r['slots'][s_id]['vol_usdt'] for r in results if r['slots'].get(s_id) is not None)
                else:
                    avg_abs = bull_pct = bear_pct = avg_bull = avg_bear = total_vol = 0.0
                    bull_count = bear_count = explosive_3 = 0
                    max_move = 0.0
                    top_mover_coin = "—"
                    top_mover_move = 0.0

                slot_metrics.append({
                    'id': s_id,
                    'label': s_def['label'],
                    'label_detail': s_def['label_detail'],
                    'session': s_def['session'],
                    'avg_abs_move': avg_abs,
                    'bull_count': bull_count,
                    'bear_count': bear_count,
                    'bull_pct': bull_pct,
                    'bear_pct': bear_pct,
                    'avg_bull': avg_bull,
                    'avg_bear': avg_bear,
                    'max_move': max_move,
                    'top_mover_coin': top_mover_coin,
                    'top_mover_move': top_mover_move,
                    'explosive_3': explosive_3,
                    'total_vol': total_vol,
                    'coins_count': len(slot_data_list)
                })

            # Identify leaders
            sorted_by_potential = sorted(slot_metrics, key=lambda x: x['avg_abs_move'], reverse=True)
            top_slot = sorted_by_potential[0]
            most_bull_slot = max(slot_metrics, key=lambda x: x['bull_count'])
            highest_max_move_slot = max(slot_metrics, key=lambda x: x['max_move'])
            most_bear_slot = max(slot_metrics, key=lambda x: x['bear_count'])
            most_explosive = max(slot_metrics, key=lambda x: x['explosive_3'])

            # -----------------------------------------------------------------
            # 1. BULLISH NUMBERS & MAXIMUM MOVE % SUMMARY
            # -----------------------------------------------------------------
            st.markdown("### 🟢 Bullish Moves & Maximum Move % Summary")

            # High-impact Executive Callout Banner
            st.info(
                f"💡 **Executive Summary**: Timeframe **{highest_max_move_slot['label']}** ({highest_max_move_slot['session']}) gave the **Maximum Move % of {highest_max_move_slot['max_move']:.2f}%** "
                f"(Top: `{highest_max_move_slot['top_mover_coin']}`, Average: **{highest_max_move_slot['avg_abs_move']:.2f}%**). "
                f"Timeframe **{most_bull_slot['label']}** ({most_bull_slot['session']}) produced the highest number of bullish moves with **{most_bull_slot['bull_count']} coins** ({most_bull_slot['bull_pct']:.1f}%)."
            )

            b_col1, b_col2 = st.columns(2)

            with b_col1:
                st.markdown("#### 🐂 Bullish Coins by Timeframe")
                st.markdown("> **Total count of coins showing positive/bullish moves in each 4-hour window:**")
                for s in slot_metrics:
                    is_top = (s['id'] == most_bull_slot['id'])
                    trophy = " 🏆 **[HIGHEST BULLISH COUNT]**" if is_top else ""
                    st.markdown(
                        f"- **{s['label']}**: `{s['bull_count']} coins` ({s['bull_pct']:.1f}% of market){trophy}"
                    )

            with b_col2:
                st.markdown("#### 🚀 Maximum Move % by Timeframe")
                st.markdown("> **Peak maximum expansion & average move per 4-hour window:**")
                for s in slot_metrics:
                    is_top_max = (s['id'] == highest_max_move_slot['id'])
                    trophy_max = " 🏆 **[MAXIMUM MOVE %]**" if is_top_max else ""
                    top_c_info = f" *(Top: {s['top_mover_coin']} {s['top_mover_move']:+.2f}%)*" if s['top_mover_coin'] != "—" else ""
                    st.markdown(
                        f"- **{s['label']}**: Max **`{s['max_move']:.2f}%`**{top_c_info} | Avg: `{s['avg_abs_move']:.2f}%`{trophy_max}"
                    )

            st.markdown("---")

            # -----------------------------------------------------------------
            # 2. KPI HERO CARDS
            # -----------------------------------------------------------------
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(
                    "🚀 Maximum Move % Timeframe",
                    f"{highest_max_move_slot['label']}",
                    f"Max: +{highest_max_move_slot['max_move']:.2f}%"
                )
                st.caption(f"Top: {highest_max_move_slot['top_mover_coin']} | Avg: {highest_max_move_slot['avg_abs_move']:.2f}%")

            with c2:
                st.metric(
                    "🐂 Most Bullish Timeframe",
                    f"{most_bull_slot['label']}",
                    f"{most_bull_slot['bull_count']} Coins ({most_bull_slot['bull_pct']:.1f}%)"
                )
                st.caption(f"{most_bull_slot['session']}")

            with c3:
                st.metric(
                    "🐻 Most Bearish Timeframe",
                    f"{most_bear_slot['label']}",
                    f"{most_bear_slot['bear_count']} Coins ({most_bear_slot['bear_pct']:.1f}%)"
                )
                st.caption(f"{most_bear_slot['session']}")

            with c4:
                st.metric(
                    "💥 Explosive Movers (>3%)",
                    f"{most_explosive['label']}",
                    f"{most_explosive['explosive_3']} Coins Made >3% Moves"
                )
                st.caption(f"{most_explosive['session']}")

            # Session Comparison Table
            with st.expander("📊 View All 6 Sessions Potential & Bullish Breakdown Table", expanded=True):
                comp_rows = []
                for rank, s in enumerate(sorted_by_potential, start=1):
                    vol_fmt = f"${s['total_vol'] / 1e6:.1f}M" if s['total_vol'] >= 1e6 else f"${s['total_vol']:,.0f}"
                    top_coin_str = f"{s['top_mover_coin']} ({s['top_mover_move']:+.2f}%)" if s['top_mover_coin'] != "—" else "—"
                    comp_rows.append({
                        "Rank": f"#{rank}",
                        "4HR Time Slot (IST)": s['label_detail'],
                        "Global Market Session": s['session'],
                        "Bullish Coins": f"{s['bull_count']} coins ({s['bull_pct']:.1f}%)",
                        "Bearish Coins": f"{s['bear_count']} coins ({s['bear_pct']:.1f}%)",
                        "Maximum Move %": f"{s['max_move']:.2f}%",
                        "Top Mover Coin": top_coin_str,
                        "Avg Move %": f"{s['avg_abs_move']:.2f}%",
                        "Coins >3% Move": s['explosive_3'],
                        "Total Volume": vol_fmt
                    })
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

            # Plotly Visualization of Sessions
            fig = go.Figure()
            slot_names_list = [s['label_detail'] for s in slot_metrics]
            fig.add_trace(go.Bar(
                x=slot_names_list,
                y=[s['bull_count'] for s in slot_metrics],
                name="Bullish Coins Count",
                marker_color="#28a745"
            ))
            fig.add_trace(go.Bar(
                x=slot_names_list,
                y=[s['bear_count'] for s in slot_metrics],
                name="Bearish Coins Count",
                marker_color="#dc3545"
            ))
            fig.add_trace(go.Scatter(
                x=slot_names_list,
                y=[s['max_move'] for s in slot_metrics],
                name="Maximum Move % (Peak)",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="#ffd700", width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title=f"4HR Session Bullish/Bearish Counts & Peak Move % ({res_date.strftime('%d %b %Y')})",
                barmode='group',
                yaxis_title="Coins Count",
                yaxis2=dict(title="Maximum Move %", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=360,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # -----------------------------------------------------------------
            # FULL COINS TABLE MATCHING EXCEL SPREADSHEET MOCKUP
            # -----------------------------------------------------------------
            st.markdown("### 📋 All Coins 4HR Candle Move Matrix")
            st.markdown(
                "Green indicates Bullish candle (`Low to High %`), Red indicates Bearish candle (`High to Low %`). "
                "Flame icon (🔥) indicates a high-volume breakout surge (>= 1.5x 4H baseline)."
            )

            filter_c1, filter_c2, filter_c3 = st.columns([2, 2, 2])
            with filter_c1:
                search_coin = st.text_input("🔍 Search Coin:", value="", placeholder="e.g. BTC, ETH, SOL", key="scan_4hr_search").upper().strip()
            with filter_c2:
                min_filter_move = st.slider("Min Move % in Any Slot:", min_value=0.0, max_value=15.0, value=0.0, step=0.5, key="scan_4hr_min_move")
            with filter_c3:
                sort_by_col = st.selectbox(
                    "Sort Table By:",
                    options=["Total 24H Range % (High to Low)", "Coin Name (A-Z)", "5.30 TO 9.30 (PM) Move", "1.30 TO 5.30 (PM) Move"],
                    index=0,
                    key="scan_4hr_sort"
                )

            # Build DataFrame rows
            table_rows = []
            for r in results:
                sym = r['symbol']
                disp_sym = r['display_symbol']
                if search_coin and search_coin not in sym:
                    continue
                if min_filter_move > 0 and r['total_range'] < min_filter_move:
                    continue

                row = {
                    'Coin': disp_sym,
                }
                for s_def in FOUR_HOUR_SLOT_DEFS:
                    s_id = s_def['id']
                    col_header = s_def['label_detail'] if use_detailed_headers else s_def['short_col']
                    s_data = r['slots'].get(s_id)
                    if s_data is not None:
                        m = s_data['move_pct']
                        fmt_move = f"{m:+.2f}%"
                        if show_vol_spike and s_data.get('is_vol_spike'):
                            fmt_move += " 🔥"
                        row[col_header] = fmt_move
                    else:
                        row[col_header] = "—"

                row['24H Range %'] = f"{r['total_range']:.2f}%"
                row['Best Slot'] = r['best_slot']
                row['_raw_range'] = r['total_range']
                row['_raw_slot3'] = r['slots'][3]['move_pct'] if r['slots'].get(3) else 0.0
                row['_raw_slot2'] = r['slots'][2]['move_pct'] if r['slots'].get(2) else 0.0
                table_rows.append(row)

            if table_rows:
                df_display = pd.DataFrame(table_rows)

                if sort_by_col == "Total 24H Range % (High to Low)":
                    df_display = df_display.sort_values('_raw_range', ascending=False)
                elif sort_by_col == "Coin Name (A-Z)":
                    df_display = df_display.sort_values('Coin', ascending=True)
                elif sort_by_col == "5.30 TO 9.30 (PM) Move":
                    df_display = df_display.sort_values('_raw_slot3', ascending=False)
                elif sort_by_col == "1.30 TO 5.30 (PM) Move":
                    df_display = df_display.sort_values('_raw_slot2', ascending=False)

                drop_raw = [c for c in ['_raw_range', '_raw_slot3', '_raw_slot2'] if c in df_display.columns]
                clean_df = df_display.drop(columns=drop_raw).reset_index(drop=True)
                clean_df.index = clean_df.index + 1
                clean_df = clean_df.rename_axis('Rank').reset_index()

                slot_cols = [
                    s_def['label_detail'] if use_detailed_headers else s_def['short_col']
                    for s_def in FOUR_HOUR_SLOT_DEFS
                ]

                # Style table with green/red background
                styler = clean_df.style
                if hasattr(styler, 'map'):
                    styled_table = styler.map(_style_4hr_move_cell, subset=slot_cols)
                else:
                    styled_table = styler.map(_style_4hr_move_cell, subset=slot_cols)

                st.dataframe(styled_table, use_container_width=True, hide_index=True)

                # -------------------------------------------------------------
                # DOWNLOAD SECTION: EXCEL (EXACT SPREADSHEET FORMAT) + CSV
                # -------------------------------------------------------------
                st.markdown("#### 📥 Export Scan Data")
                dl_col1, dl_col2 = st.columns(2)

                with dl_col1:
                    excel_bytes = generate_excel_4hr_moves(table_rows, slot_metrics, res_date)
                    if excel_bytes:
                        st.download_button(
                            label="📊 Download Formatted Excel (.xlsx)",
                            data=excel_bytes,
                            file_name=f"4hr_moves_{res_date.strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary",
                            use_container_width=True,
                            key="scan_4hr_download_excel_btn"
                        )
                    else:
                        st.warning("Install openpyxl for Excel download: pip install openpyxl")

                with dl_col2:
                    csv_data = clean_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download Raw CSV",
                        data=csv_data,
                        file_name=f"4hr_candle_moves_{res_date.strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="scan_4hr_download_btn"
                    )
            else:
                st.warning("No coins matched the specified search / filter criteria.")

    # =========================================================================
    # VIEW 2: MULTI-DAY HISTORICAL POTENTIAL AGGREGATOR (7 / 14 / 30 DAYS)
    # =========================================================================
    else:
        st.markdown("##### 📈 Historical Multi-Day 4HR Session Potential (Market-Wide)")
        st.markdown(
            "Answers: *Historically across the crypto market, which 4-hour window consistently generates the highest movement and win-rate?*"
        )

        agg_c1, agg_c2, agg_c3 = st.columns([1.5, 1.5, 2])
        with agg_c1:
            agg_days = st.select_slider("Backtest History Window (Days):", options=[7, 14, 21, 30], value=14, key="scan_4hr_agg_days")
        with agg_c2:
            agg_coins_count = st.selectbox("Market Coins Sample:", options=["Top 50 Volume (Fastest)", "Top 100 Volume", "All 500+ Coins"], index=0, key="scan_4hr_agg_coins")
        with agg_c3:
            st.write("")
            run_agg_btn = st.button(f"🚀 Run {agg_days}-Day Potential Aggregation", type="primary", use_container_width=True, key="scan_4hr_run_agg")

        if run_agg_btn:
            all_symbols = get_perpetual_symbols()
            if agg_coins_count == "Top 50 Volume (Fastest)":
                top_v = get_top_volume_symbols_list(50)
                symbols_to_agg = [s for s in top_v if s in all_symbols] or all_symbols[:50]
            elif agg_coins_count == "Top 100 Volume":
                top_v = get_top_volume_symbols_list(100)
                symbols_to_agg = [s for s in top_v if s in all_symbols] or all_symbols[:100]
            else:
                symbols_to_agg = all_symbols

            t0 = time.time()
            with st.spinner(f"Aggregating {agg_days} days of 4H candles across {len(symbols_to_agg)} coins..."):
                agg_data = fetch_multicoin_4hr_multiday_agg(symbols_to_agg, days=agg_days, max_workers=parallel_workers)
                elapsed_agg = time.time() - t0
                st.session_state['scan_4hr_agg_data'] = agg_data
                st.session_state['scan_4hr_agg_days_val'] = agg_days
                st.session_state['scan_4hr_agg_elapsed'] = elapsed_agg

        if 'scan_4hr_agg_data' in st.session_state:
            agg_data = st.session_state['scan_4hr_agg_data']
            days_used = st.session_state.get('scan_4hr_agg_days_val', 14)
            elapsed_agg = st.session_state.get('scan_4hr_agg_elapsed', 0.0)

            agg_rows = []
            for s_def in FOUR_HOUR_SLOT_DEFS:
                s_id = s_def['id']
                stats = agg_data[s_id]
                m_list = stats['moves']
                if m_list:
                    avg_m = float(np.mean(m_list))
                    wr = float(np.mean(stats['is_bull'])) * 100.0
                    avg_b = float(np.mean(stats['bull_moves'])) if stats['bull_moves'] else 0.0
                    avg_r = float(np.mean(stats['bear_moves'])) if stats['bear_moves'] else 0.0
                    exp = (sum(1 for x in m_list if x >= 3.0) / len(m_list) * 100.0)
                    tot_v = sum(stats['vols'])
                else:
                    avg_m = wr = avg_b = avg_r = exp = tot_v = 0.0

                agg_rows.append({
                    'id': s_id,
                    'label': s_def['label'],
                    'label_detail': s_def['label_detail'],
                    'session': s_def['session'],
                    'avg_move': avg_m,
                    'win_rate': wr,
                    'avg_bull': avg_b,
                    'avg_bear': avg_r,
                    'explosive_rate': exp,
                    'total_vol': tot_v,
                    'samples': len(m_list)
                })

            sorted_agg = sorted(agg_rows, key=lambda x: x['avg_move'], reverse=True)
            best_hist_slot = sorted_agg[0]

            st.success(
                f"🏆 **{days_used}-Day Statistical Leader**: The **{best_hist_slot['label']}** ({best_hist_slot['session']}) "
                f"session has proven to be the **#1 Most Potential 4HR window**, delivering an average movement of **{best_hist_slot['avg_move']:.2f}%** "
                f"and a **{best_hist_slot['explosive_rate']:.1f}% probability of >3% explosive expansion**."
            )

            # Display ranked table
            ranked_agg_table = []
            for rk, row in enumerate(sorted_agg, start=1):
                v_str = f"${row['total_vol'] / 1e9:.2f}B" if row['total_vol'] >= 1e9 else f"${row['total_vol'] / 1e6:.1f}M"
                ranked_agg_table.append({
                    "Rank": f"#{rk}",
                    "4HR Time Slot": row['label_detail'],
                    "Market Session": row['session'],
                    "Avg Move %": f"{row['avg_move']:.2f}%",
                    "Bullish Win Rate %": f"{row['win_rate']:.1f}%",
                    "Avg Bullish Move %": f"+{row['avg_bull']:.2f}%",
                    "Avg Bearish Move %": f"-{row['avg_bear']:.2f}%",
                    ">3% Explosive Move Chance": f"{row['explosive_rate']:.1f}%",
                    "Total USDT Volume": v_str,
                    "Total Candles Analyzed": f"{row['samples']:,}"
                })
            st.dataframe(pd.DataFrame(ranked_agg_table), use_container_width=True, hide_index=True)

            # Visual Bar Chart for Historical
            agg_fig = go.Figure()
            agg_fig.add_trace(go.Bar(
                x=[r['4HR Time Slot'] for r in ranked_agg_table],
                y=[float(r['Avg Move %'].replace('%', '')) for r in ranked_agg_table],
                name="Avg Move %",
                marker_color="#007bff"
            ))
            agg_fig.add_trace(go.Scatter(
                x=[r['4HR Time Slot'] for r in ranked_agg_table],
                y=[float(r['Bullish Win Rate %'].replace('%', '')) for r in ranked_agg_table],
                name="Bullish Win Rate %",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="#28a745", width=3),
                marker=dict(size=8)
            ))
            agg_fig.update_layout(
                title=f"Historical {days_used}-Day 4HR Session Potential Breakdown",
                yaxis=dict(title="Avg Move %"),
                yaxis2=dict(title="Bullish Win Rate %", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=350,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(agg_fig, use_container_width=True)


def _render_4hr_single_coin_mode():
    """Render Single Coin Mode for 4HR Candle Move Scanner"""
    st.markdown("#### 🎯 Single Coin Deep-Dive & Backtest")

    all_symbols = get_perpetual_symbols()
    default_coin_idx = all_symbols.index("BTCUSDT") if "BTCUSDT" in all_symbols else 0

    col_s1, col_s2, col_s3 = st.columns([1.5, 1.5, 1.5])
    with col_s1:
        selected_coin = st.selectbox(
            "Select Coin to Analyze:",
            options=all_symbols,
            index=default_coin_idx,
            key="scan_4hr_single_sym"
        )
    with col_s2:
        single_mode = st.radio(
            "Analysis View:",
            options=["📅 Selected Date 6-Candle Breakdown", "📈 Multi-Day Historical Matrix"],
            index=0,
            key="scan_4hr_single_mode"
        )
    with col_s3:
        today_ist = datetime.now(IST).date()
        single_date = st.date_input(
            "Target Backtest Date:",
            value=today_ist - timedelta(days=1),
            max_value=today_ist,
            key="scan_4hr_single_date"
        )

    st.markdown("---")

    # Mode 1: Detailed 6-candle breakdown for a single date
    if single_mode == "📅 Selected Date 6-Candle Breakdown":
        st.subheader(f"📊 {selected_coin}.P — 4HR Breakdown for {single_date.strftime('%d %b %Y')}")
        with st.spinner(f"Fetching 4HR candles for {selected_coin}..."):
            coin_data = fetch_coin_4hr_day_candles(selected_coin, single_date)

        if coin_data and coin_data.get('slots'):
            slots = coin_data['slots']
            cols = st.columns(6)

            for idx, s_def in enumerate(FOUR_HOUR_SLOT_DEFS):
                s_info = slots.get(idx)
                with cols[idx]:
                    st.markdown(f"**{s_def['label']}**")
                    st.caption(f"{s_def['session']}")

                    if s_info is not None:
                        m = s_info['move_pct']
                        bg = "#d4edda" if m >= 0 else "#f8d7da"
                        color = "#155724" if m >= 0 else "#721c24"
                        sign = "+" if m > 0 else ""
                        spike_badge = " 🔥" if s_info.get('is_vol_spike') else ""
                        live_badge = " *(Live)*" if s_info.get('is_live') else ""

                        st.markdown(
                            f"<div style='background-color:{bg}; color:{color}; padding:8px; border-radius:6px; font-size:18px; font-weight:bold; text-align:center; margin-bottom:8px;'>"
                            f"{sign}{m:.2f}%{spike_badge}{live_badge}</div>",
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f"- **Open**: `{s_info['open']:.4f}`\n"
                            f"- **High**: `{s_info['high']:.4f}`\n"
                            f"- **Low**: `{s_info['low']:.4f}`\n"
                            f"- **Close**: `{s_info['close']:.4f}`\n"
                            f"- **USDT Vol**: `${s_info['vol_usdt']/1e6:.2f}M`\n"
                            f"- **Vol Ratio**: `{s_info['vol_ratio']:.2f}x`"
                        )
                    else:
                        st.markdown("<div style='color:#888; text-align:center; padding:15px;'>— Not Started —</div>", unsafe_allow_html=True)

            # Summary metrics for the coin
            st.markdown("---")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("🏆 Best 4HR Session Today", f"{coin_data['best_slot']}")
            with k2:
                st.metric("💥 Max 4HR Move %", f"{coin_data['total_range']:.2f}%")
            with k3:
                vol_str = f"${coin_data['total_vol'] / 1e6:.2f}M" if coin_data['total_vol'] >= 1e6 else f"${coin_data['total_vol']:,.0f}"
                st.metric("💰 Total 24H USDT Volume", vol_str)
            with k4:
                avg_vol_str = f"${coin_data['avg_4h_vol'] / 1e6:.2f}M" if coin_data['avg_4h_vol'] >= 1e6 else f"${coin_data['avg_4h_vol']:,.0f}"
                st.metric("📊 Baseline 4H Avg Volume", avg_vol_str)

        else:
            st.warning(f"Could not retrieve 4HR candle data for {selected_coin} on {single_date}.")

    # Mode 2: Multi-day historical matrix for this coin
    else:
        st.subheader(f"📈 {selected_coin}.P — Multi-Day 4HR History Matrix")

        ctrl1, ctrl2 = st.columns([2, 2])
        with ctrl1:
            lookback_days = st.slider("Lookback Days:", min_value=7, max_value=30, value=14, key="scan_4hr_single_lookback")
        with ctrl2:
            show_vol_in_cells = st.checkbox("📊 Show Volume in Cells", value=True, key="scan_4hr_show_vol")

        with st.spinner(f"Fetching {lookback_days} days of 4HR history + OI for {selected_coin}..."):
            history_rows = fetch_single_coin_history_slots(
                selected_coin, days=lookback_days, include_volume=show_vol_in_cells
            )

        if history_rows:
            slot_cols = [s_def['short_col'] for s_def in FOUR_HOUR_SLOT_DEFS]

            # ---------- Build display columns (strip internal _raw / _date_obj keys) ----------
            display_cols = ['Date'] + slot_cols + ['Max Move %', 'Best Slot']
            if show_vol_in_cells:
                display_cols += ['24H USDT Vol', 'Coin Vol', 'OI (USDT)', 'OI Chg %', '🎯 VW Best']

            df_hist = pd.DataFrame(history_rows)
            # Keep only display columns that actually exist
            display_cols = [c for c in display_cols if c in df_hist.columns]
            df_display = df_hist[display_cols].copy()

            # ---------- Sparkline summary row ----------
            if show_vol_in_cells and len(history_rows) > 1:
                spark = {'Date': '📊 AVG'}
                for s_def in FOUR_HOUR_SLOT_DEFS:
                    s_id = s_def['id']
                    raw_key = f'_raw_{s_id}_move'
                    vol_key = f'_raw_{s_id}_vol'
                    moves = [r[raw_key] for r in history_rows if r.get(raw_key) is not None]
                    vols = [r[vol_key] for r in history_rows if r.get(vol_key) is not None]
                    avg_m = np.mean(moves) if moves else 0.0
                    avg_v = np.mean(vols) if vols else 0.0
                    bull_count = sum(1 for m in moves if m > 0)
                    bull_pct = (bull_count / len(moves) * 100) if moves else 0
                    vol_fmt = f"${avg_v/1e6:.1f}M" if avg_v >= 1e6 else f"${avg_v/1e3:.0f}K"
                    spark[s_def['short_col']] = f"{avg_m:+.2f}% | {vol_fmt} ({bull_pct:.0f}%🐂)"

                # Aggregate daily columns for sparkline
                raw_vols = [r.get('_raw_usdt_vol', 0) for r in history_rows if r.get('_raw_usdt_vol')]
                raw_cvols = [r.get('_raw_coin_vol', 0) for r in history_rows if r.get('_raw_coin_vol')]
                raw_ois = [r.get('_raw_oi', 0) for r in history_rows if r.get('_raw_oi')]
                raw_oi_chgs = [r.get('_raw_oi_chg', 0) for r in history_rows if r.get('_raw_oi_chg') is not None]

                avg_daily_vol = np.mean(raw_vols) if raw_vols else 0
                spark['24H USDT Vol'] = f"${avg_daily_vol/1e6:.1f}M" if avg_daily_vol >= 1e6 else f"${avg_daily_vol/1e3:.0f}K"
                avg_daily_cvol = np.mean(raw_cvols) if raw_cvols else 0
                if avg_daily_cvol >= 1e6:
                    spark['Coin Vol'] = f"{avg_daily_cvol/1e6:.2f}M"
                elif avg_daily_cvol >= 1e3:
                    spark['Coin Vol'] = f"{avg_daily_cvol/1e3:.1f}K"
                else:
                    spark['Coin Vol'] = f"{avg_daily_cvol:.1f}"
                avg_oi = np.mean(raw_ois) if raw_ois else 0
                spark['OI (USDT)'] = f"${avg_oi/1e9:.2f}B" if avg_oi >= 1e9 else f"${avg_oi/1e6:.1f}M"
                avg_oi_chg = np.mean(raw_oi_chgs) if raw_oi_chgs else 0
                spark['OI Chg %'] = f"{avg_oi_chg:+.2f}% avg"
                spark['Max Move %'] = "—"
                spark['Best Slot'] = "—"
                spark['🎯 VW Best'] = "—"

                spark_filtered = {k: v for k, v in spark.items() if k in display_cols}
                df_spark = pd.DataFrame([spark_filtered])
                df_display = pd.concat([df_display, df_spark], ignore_index=True)

            # ---------- Style the table ----------
            extra_style_cols = []
            if 'OI Chg %' in df_display.columns:
                extra_style_cols.append('OI Chg %')

            all_style_cols = slot_cols + extra_style_cols
            existing_style_cols = [c for c in all_style_cols if c in df_display.columns]

            styler = df_display.style
            if hasattr(styler, 'map'):
                styled_hist = styler.map(_style_4hr_move_cell, subset=existing_style_cols)
            else:
                styled_hist = styler.map(_style_4hr_move_cell, subset=existing_style_cols)

            st.dataframe(styled_hist, use_container_width=True, hide_index=True)

            # ---------- Divergence alerts ----------
            if show_vol_in_cells:
                divg_dates = [r['Date'] for r in history_rows if r.get('_divergence')]
                if divg_dates:
                    st.warning(
                        f"⚠️ **OI vs Price Divergence detected** on: {', '.join(divg_dates)}. "
                        "Price moved in one direction but OI moved contrary — potential squeeze or fake move."
                    )

            # ---------- Export ----------
            # Build clean export DataFrame (without _raw columns)
            df_export = df_hist[[c for c in df_hist.columns if not c.startswith('_')]].copy()
            csv_data = df_export.to_csv(index=False)
            sc_dl1, sc_dl2 = st.columns(2)
            with sc_dl1:
                sc_excel = generate_excel_4hr_moves(history_rows, [], None)
                if sc_excel:
                    st.download_button(
                        label=f"📊 Download {selected_coin} Excel (.xlsx)",
                        data=sc_excel,
                        file_name=f"{selected_coin}_4hr_history.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        use_container_width=True,
                        key="scan_4hr_single_excel_btn"
                    )
            with sc_dl2:
                st.download_button(
                    label=f"📄 Download {selected_coin} CSV",
                    data=csv_data,
                    file_name=f"{selected_coin}_4hr_history.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="scan_4hr_single_csv_btn"
                )
        else:
            st.warning(f"No historical data available for {selected_coin}.")


def render_4hr_candle_move_scanner():
    """Render Tab: 4HR Candle Move Scanner with Multicoin and Single Coin Modes"""
    st.header("⏰ 4HR Candle Move Scanner")
    st.markdown(
        "**Systematic 4-hour candle analysis across all 6 IST trading sessions.** "
        "Calculates low-to-high % for Bullish candles and high-to-low % for Bearish candles, "
        "summarizing which 4-hour window holds the highest trading potential."
    )

    mode = st.radio(
        "Select Scanner Mode:",
        options=[
            "🌐 Multicoin Mode (All 500+ Coins / Market Scan)",
            "🎯 Single Coin Mode (Deep-Dive & History)"
        ],
        horizontal=True,
        key="scan_4hr_main_mode"
    )

    st.markdown("---")

    if mode == "🌐 Multicoin Mode (All 500+ Coins / Market Scan)":
        _render_4hr_multicoin_mode()
    else:
        _render_4hr_single_coin_mode()


# ==============================================================================
# SMART SCANNER SUB-TAB: AVERAGE MOVE SCANNER (IST TIMING)
# ==============================================================================

AVERAGE_MOVE_SLOT_DEFS = [
    {
        "id": 0,
        "utc_hour": 0,
        "label": "5.30 AM - 9.30 AM",
        "full_label": "5.30 AM - 9.30 AM (IST)",
    },
    {
        "id": 1,
        "utc_hour": 4,
        "label": "9.30 AM - 1.30 PM",
        "full_label": "9.30 AM - 1.30 PM (IST)",
    },
    {
        "id": 2,
        "utc_hour": 8,
        "label": "1.30 PM - 5.30 PM",
        "full_label": "1.30 PM - 5.30 PM (IST)",
    },
    {
        "id": 3,
        "utc_hour": 12,
        "label": "5.30 PM - 9.30 PM",
        "full_label": "5.30 PM - 9.30 PM (IST)",
    },
    {
        "id": 4,
        "utc_hour": 16,
        "label": "9.30 PM - 1.30 AM",
        "full_label": "9.30 PM - 1.30 AM (IST)",
    },
    {
        "id": 5,
        "utc_hour": 20,
        "label": "1.30 AM - 5.30 AM",
        "full_label": "1.30 AM - 5.30 AM (IST)",
    },
]


def fetch_coin_average_move_history(symbol, start_date, end_date):
    """
    Fetch 4H klines for a coin between start_date and end_date (inclusive).
    In IST (UTC+5:30), 24-hour day runs 05:30 AM on date D to 05:30 AM on date D+1,
    which corresponds exactly to 00:00:00 UTC to 24:00:00 UTC on date D.

    Calculation rules:
    - Bull candle (Close >= Open): Low to High % -> +((High - Low) / Low) * 100
    - Bear candle (Close < Open): High to Low % -> -((High - Low) / High) * 100
    """
    clean_sym = symbol.strip().upper().replace(".P", "").replace(".p", "")
    if not clean_sym.endswith("USDT"):
        clean_sym += "USDT"

    start_dt_utc = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0, tzinfo=ZoneInfo("UTC"))
    end_dt_utc = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=ZoneInfo("UTC"))
    start_ms = int(start_dt_utc.timestamp() * 1000)
    end_ms = int(end_dt_utc.timestamp() * 1000)

    total_days = max((end_date - start_date).days + 1, 1)
    candles_needed = min(total_days * 6 + 10, 1000)

    params = {
        "symbol": clean_sym,
        "interval": "4h",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": min(candles_needed, 1000),
    }

    try:
        res = api_get(f"{BINANCE_API_URL}/fapi/v1/klines", params=params, timeout=8)
        data = res.json()
        if not isinstance(data, list) or len(data) == 0:
            return {}, clean_sym

        day_slot_map = {}
        for k in data:
            dt_utc = datetime.fromtimestamp(k[0] / 1000, tz=ZoneInfo("UTC"))
            d_key = dt_utc.date()
            if d_key < start_date or d_key > end_date:
                continue
            if d_key not in day_slot_map:
                day_slot_map[d_key] = {}

            slot_id = dt_utc.hour // 4
            o = float(k[1])
            h = float(k[2])
            l = float(k[3])
            c = float(k[4])
            vol_base = float(k[5])
            vol_usdt = float(k[7])

            is_bull = c >= o
            # If bull candle: Low to High % (positive)
            # If bear candle: High to Low % (negative)
            if is_bull:
                move_pct = ((h - l) / l * 100.0) if l > 0 else 0.0
            else:
                move_pct = -((h - l) / h * 100.0) if h > 0 else 0.0

            day_slot_map[d_key][slot_id] = {
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "is_bull": is_bull,
                "move_pct": round(move_pct, 2),
                "vol_usdt": vol_usdt,
                "vol_base": vol_base,
            }

        return day_slot_map, clean_sym
    except Exception:
        return {}, clean_sym


def generate_average_move_excel(clean_sym, days_count, start_date, end_date, slot_label, df_table, summary_stats, all_slots_stats=None):
    """
    Generate Excel file matching the user's exact spreadsheet mockup layout:
    Left: Input panel (Coin, No of days, Date range, Time zone)
    Right: Output table (Date, Time zone, in %, Positive, Negative, Max) + Totals row
    Sheet 2: All 6 Sessions Comparison
    """
    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    except ImportError:
        return None

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Average Move"
    ws.views.sheetView[0].showGridLines = True

    # Styling definitions
    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="000000")
    title_font = Font(name="Calibri", size=12, bold=True, color="000000")
    bold_font = Font(name="Calibri", size=11, bold=True, color="000000")
    regular_font = Font(name="Calibri", size=11, color="000000")
    muted_font = Font(name="Calibri", size=11, color="595959")
    green_font = Font(name="Calibri", size=11, color="276A3C", bold=True)
    red_font = Font(name="Calibri", size=11, color="C00000", bold=True)
    green_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    red_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    total_fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")

    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )
    center_align = Alignment(horizontal="center", vertical="center")
    left_align = Alignment(horizontal="left", vertical="center")

    # --- Left: Input Panel ---
    ws.cell(row=2, column=1, value="Input").font = title_font
    input_items = [
        ("Coin", f"{clean_sym}.p"),
        ("No of days", days_count),
        ("Date range", f"{start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')}"),
        ("Time zone", slot_label),
        ("", "5.30 AM - 9.30 AM"),
        ("", "9.30 AM - 1.30 PM"),
        ("", "1.30 PM - 5.30 PM"),
        ("", "5.30 PM - 9.30 PM"),
        ("", "9.30 PM - 1.30 AM"),
        ("", "1.30 AM - 5.30 AM"),
    ]
    for idx, (lbl, val) in enumerate(input_items, start=4):
        c_lbl = ws.cell(row=idx, column=1, value=lbl)
        c_lbl.font = bold_font if lbl else muted_font
        c_lbl.alignment = left_align
        c_val = ws.cell(row=idx, column=2, value=val)
        c_val.font = regular_font
        c_val.alignment = center_align if isinstance(val, (int, float)) else left_align

    # --- Right: Output Table ---
    ws.cell(row=2, column=4, value="Output").font = title_font
    out_headers = ["Date", "Time zone", "in %", "Postive", "Negative", "Max"]
    for c_idx, h_text in enumerate(out_headers, start=4):
        c = ws.cell(row=3, column=c_idx, value=h_text)
        c.fill = header_fill
        c.font = header_font
        c.alignment = center_align
        c.border = thin_border

    start_row = 4
    for idx, r in df_table.iterrows():
        curr_row = start_row + idx
        ws.cell(row=curr_row, column=4, value=r.get("Date", "")).alignment = center_align
        ws.cell(row=curr_row, column=5, value=r.get("Time zone", "")).alignment = center_align

        # in %
        c_in = ws.cell(row=curr_row, column=6, value=r.get("in %", ""))
        c_in.alignment = center_align

        # Postive
        c_pos = ws.cell(row=curr_row, column=7, value=r.get("Postive", ""))
        c_pos.alignment = center_align

        # Negative
        c_neg = ws.cell(row=curr_row, column=8, value=r.get("Negative", ""))
        c_neg.alignment = center_align

        # Max
        c_max = ws.cell(row=curr_row, column=9, value=r.get("Max", ""))
        c_max.alignment = center_align

        for col_i in range(4, 10):
            ws.cell(row=curr_row, column=col_i).border = thin_border

        # Colors
        val_in = r.get("_raw_move")
        if val_in is not None:
            if val_in > 0:
                c_in.fill = green_fill
                c_in.font = green_font
                c_pos.fill = green_fill
                c_pos.font = green_font
            elif val_in < 0:
                c_in.fill = red_fill
                c_in.font = red_font
                c_neg.fill = red_fill
                c_neg.font = red_font

    # Bottom Summary Row
    summary_row_idx = start_row + len(df_table)
    ws.cell(row=summary_row_idx, column=4, value="TOTAL").font = bold_font
    ws.cell(row=summary_row_idx, column=5, value="").font = bold_font

    c_tot_in = ws.cell(row=summary_row_idx, column=6, value=f"{summary_stats.get('net_move', 0.0):+.2f}%")
    c_tot_in.font = bold_font
    c_tot_in.alignment = center_align

    c_tot_pos = ws.cell(row=summary_row_idx, column=7, value=f"{summary_stats.get('pos_sum', 0.0):.2f}")
    c_tot_pos.font = green_font
    c_tot_pos.alignment = center_align

    c_tot_neg = ws.cell(row=summary_row_idx, column=8, value=f"{summary_stats.get('neg_sum', 0.0):.2f}")
    c_tot_neg.font = red_font
    c_tot_neg.alignment = center_align

    c_tot_max = ws.cell(row=summary_row_idx, column=9, value=f"{summary_stats.get('max_move', 0.0):.2f}%")
    c_tot_max.font = bold_font
    c_tot_max.alignment = center_align

    for col_i in range(4, 10):
        c = ws.cell(row=summary_row_idx, column=col_i)
        c.fill = total_fill
        c.border = thin_border

    # Adjust widths
    ws.column_dimensions["A"].width = 14
    ws.column_dimensions["B"].width = 24
    ws.column_dimensions["C"].width = 4
    ws.column_dimensions["D"].width = 14
    ws.column_dimensions["E"].width = 22
    ws.column_dimensions["F"].width = 12
    ws.column_dimensions["G"].width = 12
    ws.column_dimensions["H"].width = 12
    ws.column_dimensions["I"].width = 12

    # --- Sheet 2: All 6 Sessions Comparison ---
    if all_slots_stats:
        ws2 = wb.create_sheet(title="All 6 Sessions Comparison")
        ws2.views.sheetView[0].showGridLines = True
        s2_headers = ["Rank", "Time Slot (IST)", "Avg Move %", "Bullish Win Rate", "Positive Sum", "Negative Sum", "Net Move %", "Max Move %", "Total Candles"]
        for c_idx, h_text in enumerate(s2_headers, start=1):
            c = ws2.cell(row=1, column=c_idx, value=h_text)
            c.fill = header_fill
            c.font = header_font
            c.alignment = center_align
            c.border = thin_border

        for r_idx, s_row in enumerate(all_slots_stats, start=2):
            vals = [
                f"#{r_idx - 1}",
                s_row["label"],
                f"{s_row['avg_move']:.2f}%",
                f"{s_row['win_rate']:.1f}%",
                f"+{s_row['pos_sum']:.2f}%",
                f"{s_row['neg_sum']:.2f}%",
                f"{s_row['net_move']:+.2f}%",
                f"{s_row['max_move']:.2f}%",
                s_row["count"],
            ]
            for c_idx, val in enumerate(vals, start=1):
                c = ws2.cell(row=r_idx, column=c_idx, value=val)
                c.alignment = center_align
                c.border = thin_border
                c.font = regular_font

        for col_l in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
            ws2.column_dimensions[col_l].width = 18

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def render_average_move_tab():
    """
    Render the 'Average Move' tab under Smart Scanner.
    Matches user requirements:
    - Time zone: IST timing
    - If bull candle: calculate low to high % (+((High - Low) / Low) * 100)
    - If bear candle: calculate high to low % (-((High - Low) / High) * 100)
    - Output format: Date | Time zone | in % | Positive | Negative | Max
    - Summary bottom row: Net %, Positive Sum, Negative Sum, Max %
    """
    st.markdown("### 📊 Average Move Scanner (IST Timing)")
    st.caption(
        "Calculates historical 4HR candle movements across Indian Standard Time (IST) trading sessions. "
        "**Bull candle**: Low to High % (`+((H-L)/L)*100`). "
        "**Bear candle**: High to Low % (`-((H-L)/H)*100`)."
    )

    all_symbols = get_perpetual_symbols()
    default_sym = "LSKUSDT" if "LSKUSDT" in all_symbols else ("BTCUSDT" if "BTCUSDT" in all_symbols else (all_symbols[0] if all_symbols else "BTCUSDT"))
    default_idx = all_symbols.index(default_sym) if default_sym in all_symbols else 0

    col_i1, col_i2, col_i3, col_i4 = st.columns([1.5, 1.2, 1.5, 1.0])

    with col_i1:
        coin_method = st.radio("Coin Input Method:", ["Dropdown", "Type Symbol"], horizontal=True, key="avg_m_coin_method")
        if coin_method == "Dropdown":
            selected_coin = st.selectbox(
                "Select Coin (Perpetual):",
                options=all_symbols,
                index=default_idx,
                key="avg_m_coin_dropdown"
            )
        else:
            selected_coin = st.text_input(
                "Enter Coin Symbol (e.g. LSKUSDT.p, BTCUSDT):",
                value="LSKUSDT.p",
                key="avg_m_coin_text"
            )

    with col_i2:
        date_mode = st.radio("Date Selection:", ["No of days", "Date range"], horizontal=True, key="avg_m_date_mode")
        today_ist = datetime.now(IST).date()
        if date_mode == "No of days":
            no_of_days = st.number_input("No of days:", min_value=1, max_value=90, value=10, step=1, key="avg_m_days")
            end_date = today_ist
            start_date = today_ist - timedelta(days=int(no_of_days) - 1)
        else:
            date_range = st.date_input(
                "Date range (Start - End):",
                value=(today_ist - timedelta(days=9), today_ist),
                max_value=today_ist,
                key="avg_m_daterange"
            )
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                start_date, end_date = date_range[0], date_range[1]
                no_of_days = (end_date - start_date).days + 1
            else:
                start_date = today_ist - timedelta(days=9)
                end_date = today_ist
                no_of_days = 10

    with col_i3:
        slot_options = [s["full_label"] for s in AVERAGE_MOVE_SLOT_DEFS] + ["🌐 All 6 Time Slots (Full Matrix & Ranking)"]
        selected_slot_label = st.selectbox(
            "Time zone (IST):",
            options=slot_options,
            index=0,
            key="avg_m_slot_select"
        )

    with col_i4:
        st.write("")
        st.write("")
        calc_btn = st.button("🚀 Calculate Move", type="primary", use_container_width=True, key="avg_m_calc_btn")

    # Keep data in session state
    state_key = "avg_move_results"
    if calc_btn or state_key not in st.session_state:
        if not selected_coin:
            st.warning("Please specify a coin symbol.")
            return

        with st.spinner(f"Fetching 4H candle data for {selected_coin} from Binance..."):
            day_slot_map, clean_sym = fetch_coin_average_move_history(selected_coin, start_date, end_date)
            st.session_state[state_key] = {
                "day_slot_map": day_slot_map,
                "clean_sym": clean_sym,
                "selected_coin": selected_coin,
                "start_date": start_date,
                "end_date": end_date,
                "no_of_days": no_of_days,
                "selected_slot_label": selected_slot_label
            }

    data_bundle = st.session_state.get(state_key)
    if not data_bundle or not data_bundle.get("day_slot_map"):
        st.warning("No candle data could be retrieved for the specified parameters.")
        return

    day_slot_map = data_bundle["day_slot_map"]
    clean_sym = data_bundle["clean_sym"]
    start_date = data_bundle["start_date"]
    end_date = data_bundle["end_date"]
    no_of_days = data_bundle["no_of_days"]
    selected_slot_label = data_bundle["selected_slot_label"]

    # Determine if single slot or all slots
    is_all_slots = "All 6 Time Slots" in selected_slot_label
    selected_slot_idx = None
    slot_short_name = "5.30-9.30"
    if not is_all_slots:
        for s_def in AVERAGE_MOVE_SLOT_DEFS:
            if s_def["full_label"] == selected_slot_label:
                selected_slot_idx = s_def["id"]
                slot_short_name = s_def["label"]
                break

    sorted_dates = sorted(day_slot_map.keys())

    # Calculate statistics across all 6 slots for comparison
    all_slots_stats = []
    for s_def in AVERAGE_MOVE_SLOT_DEFS:
        s_id = s_def["id"]
        s_moves = []
        s_pos_moves = []
        s_neg_moves = []
        s_vols = []
        for d in sorted_dates:
            c_info = day_slot_map[d].get(s_id)
            if c_info is not None:
                m = c_info["move_pct"]
                s_moves.append(m)
                s_vols.append(c_info.get("vol_usdt", 0))
                if m > 0:
                    s_pos_moves.append(m)
                elif m < 0:
                    s_neg_moves.append(m)

        avg_m = np.mean([abs(x) for x in s_moves]) if s_moves else 0.0
        pos_s = sum(s_pos_moves)
        neg_s = sum(s_neg_moves)
        net_m = pos_s + neg_s
        max_m = max([abs(x) for x in s_moves]) if s_moves else 0.0
        wr = (len(s_pos_moves) / len(s_moves) * 100.0) if s_moves else 0.0

        all_slots_stats.append({
            "id": s_id,
            "label": s_def["label"],
            "avg_move": avg_m,
            "win_rate": wr,
            "pos_sum": pos_s,
            "neg_sum": neg_s,
            "net_move": net_m,
            "max_move": max_m,
            "count": len(s_moves),
            "pos_count": len(s_pos_moves),
            "neg_count": len(s_neg_moves),
            "total_vol": sum(s_vols)
        })

    # Rank slots by Average Move %
    all_slots_stats_sorted = sorted(all_slots_stats, key=lambda x: x["avg_move"], reverse=True)

    def _render_slot_day_wise_breakdown(
        clean_sym,
        slot_id,
        slot_short_name,
        day_slot_map,
        sorted_dates,
        no_of_days,
        start_date,
        end_date,
        all_slots_stats_sorted,
        rank_num=None,
        session_title="",
        key_prefix="single"
    ):
        table_rows = []
        pos_list = []
        neg_list = []
        all_moves = []

        for d in sorted_dates:
            c_info = day_slot_map[d].get(slot_id)
            d_str = d.strftime("%b-%d")
            if c_info is not None:
                m = c_info["move_pct"]
                all_moves.append(m)
                if m > 0:
                    pos_list.append(m)
                    pos_val_str = f"{m:.2f}"
                    neg_val_str = ""
                elif m < 0:
                    neg_list.append(m)
                    pos_val_str = ""
                    neg_val_str = f"{m:.2f}"
                else:
                    pos_val_str = "0.00"
                    neg_val_str = ""

                table_rows.append({
                    "Date": d_str,
                    "Time zone": slot_short_name,
                    "in %": f"{m:+.2f}%" if m != 0 else "0.00%",
                    "Postive": pos_val_str,
                    "Negative": neg_val_str,
                    "Max": "",
                    "_raw_move": m,
                    "_raw_date": d,
                    "open": c_info["open"],
                    "high": c_info["high"],
                    "low": c_info["low"],
                    "close": c_info["close"],
                })

        pos_sum = sum(pos_list)
        neg_sum = sum(neg_list)
        net_move = pos_sum + neg_sum
        max_move = max([abs(x) for x in all_moves]) if all_moves else 0.0
        avg_abs_move = np.mean([abs(x) for x in all_moves]) if all_moves else 0.0
        win_rate = (len(pos_list) / len(all_moves) * 100.0) if all_moves else 0.0

        # Set Max on the first row to match user spreadsheet screenshot (where Max shows on top row)
        if table_rows:
            table_rows[0]["Max"] = f"{max_move:.2f}%"

        df_out = pd.DataFrame(table_rows)

        summary_stats = {
            "pos_sum": pos_sum,
            "neg_sum": neg_sum,
            "net_move": net_move,
            "max_move": max_move,
            "avg_abs_move": avg_abs_move,
            "win_rate": win_rate,
            "pos_count": len(pos_list),
            "neg_count": len(neg_list),
            "total_count": len(all_moves)
        }

        # Header title
        if rank_num is not None:
            st.markdown(f"#### 📊 Day-Wise Breakdown — {clean_sym}.P ({slot_short_name}) [Rank #{rank_num}]")
            st.caption(f"Session **Rank #{rank_num} ({slot_short_name})** delivers an **Avg Move of {avg_abs_move:.2f}%**, **Win Rate of {win_rate:.1f}%**, and **Max Move of {max_move:.2f}%**.")
        else:
            st.markdown(f"#### 📊 Output Table — {clean_sym}.P ({slot_short_name})")

        # Header metrics banner
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric("🟢 Positive Sum", f"+{pos_sum:.2f}%", f"{len(pos_list)} Bull candles")
        with m2:
            st.metric("🔴 Negative Sum", f"{neg_sum:.2f}%", f"{len(neg_list)} Bear candles")
        with m3:
            st.metric("⚖️ Net Move", f"{net_move:+.2f}%", "Positive + Negative")
        with m4:
            st.metric("💥 Max Move", f"{max_move:.2f}%", f"Highest in {no_of_days}d")
        with m5:
            st.metric("📊 Avg Move", f"{avg_abs_move:.2f}%", "Mean absolute move")
        with m6:
            st.metric("🎯 Win Rate", f"{win_rate:.1f}%", f"{len(pos_list)}/{len(all_moves)} Bullish")

        st.markdown("---")

        # Sub-tabs for viewing
        view_tab1, view_tab2, view_tab3 = st.tabs([
            "📋 Output Table (Spreadsheet View)",
            "📊 Side-by-Side Positive vs Negative",
            "📈 Visual Bar Chart"
        ])

        with view_tab1:
            disp_cols = ["Date", "Time zone", "in %", "Postive", "Negative", "Max"]

            summary_row = pd.DataFrame([{
                "Date": "",
                "Time zone": "",
                "in %": f"{net_move:+.2f}%",
                "Postive": f"{pos_sum:.2f}",
                "Negative": f"{neg_sum:.2f}",
                "Max": f"{max_move:.2f}%",
                "_raw_move": None
            }])
            display_df = pd.concat([df_out[disp_cols + ["_raw_move"]], summary_row], ignore_index=True)

            def _style_output_row(row):
                styles = [''] * len(row)
                val = row.get("_raw_move")
                if row.get("Date") == "" and row.get("Time zone") == "":
                    return [
                        'font-weight: bold; background-color: #f2f2f2; text-align: center; border-top: 2px solid #333;'
                    ] * len(row)

                if val is not None:
                    if val > 0:
                        for i, col in enumerate(row.index):
                            if col in ("in %", "Postive"):
                                styles[i] = 'background-color: #E2EFDA; color: #276A3C; font-weight: bold; text-align: center;'
                            else:
                                styles[i] = 'text-align: center;'
                    elif val < 0:
                        for i, col in enumerate(row.index):
                            if col in ("in %", "Negative"):
                                styles[i] = 'background-color: #FCE4D6; color: #C00000; font-weight: bold; text-align: center;'
                            else:
                                styles[i] = 'text-align: center;'
                    else:
                        styles = ['text-align: center;'] * len(row)
                else:
                    styles = ['text-align: center;'] * len(row)
                return styles

            styled_table = display_df[disp_cols + ["_raw_move"]].style.apply(_style_output_row, axis=1)
            st.dataframe(styled_table, use_container_width=True, hide_index=True, column_config={
                "_raw_move": None
            })

            d_col1, d_col2 = st.columns(2)
            with d_col1:
                excel_bytes = generate_average_move_excel(
                    clean_sym=clean_sym,
                    days_count=no_of_days,
                    start_date=start_date,
                    end_date=end_date,
                    slot_label=slot_short_name,
                    df_table=df_out,
                    summary_stats=summary_stats,
                    all_slots_stats=all_slots_stats_sorted
                )
                if excel_bytes:
                    rank_str = f"_rank{rank_num}" if rank_num else ""
                    st.download_button(
                        label="📥 Download Excel (.xlsx) [Exact Spreadsheet Format]",
                        data=excel_bytes,
                        file_name=f"average_move_{clean_sym}_{slot_short_name.replace('.', '_').replace(' ', '_')}{rank_str}_{start_date}_{end_date}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        use_container_width=True,
                        key=f"{key_prefix}_avg_m_dl_excel"
                    )

            with d_col2:
                csv_data = display_df[disp_cols].to_csv(index=False)
                st.download_button(
                    label="📄 Download Raw CSV",
                    data=csv_data,
                    file_name=f"average_move_{clean_sym}_{slot_short_name.replace('.', '_').replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"{key_prefix}_avg_m_dl_csv"
                )

        with view_tab2:
            st.markdown(f"#### 📊 Side-by-Side: Positive Moves vs Negative Moves ({clean_sym}.P)")
            col_p, col_n = st.columns(2)

            with col_p:
                st.markdown("##### 🟢 Bullish (Positive) Moves")
                st.markdown(f"**Total Sum:** `{pos_sum:.2f}%` | **Count:** `{len(pos_list)}` | **Avg:** `{np.mean(pos_list) if pos_list else 0:.2f}%`")
                if pos_list:
                    pos_df_items = [{"#": i+1, "Positive Move %": f"+{v:.2f}%"} for i, v in enumerate(pos_list)]
                    st.dataframe(pd.DataFrame(pos_df_items), use_container_width=True, hide_index=True)
                else:
                    st.info("No positive candles in this period.")

            with col_n:
                st.markdown("##### 🔴 Bearish (Negative) Moves")
                st.markdown(f"**Total Sum:** `{neg_sum:.2f}%` | **Count:** `{len(neg_list)}` | **Avg:** `{np.mean(neg_list) if neg_list else 0:.2f}%`")
                if neg_list:
                    neg_df_items = [{"#": i+1, "Negative Move %": f"{v:.2f}%"} for i, v in enumerate(neg_list)]
                    st.dataframe(pd.DataFrame(neg_df_items), use_container_width=True, hide_index=True)
                else:
                    st.info("No negative candles in this period.")

        with view_tab3:
            st.markdown(f"#### 📈 Daily Candle Move Distribution ({slot_short_name})")
            if table_rows:
                chart_dates = [r["Date"] for r in table_rows]
                chart_moves = [r["_raw_move"] for r in table_rows]
                bar_colors = ["#28a745" if m >= 0 else "#dc3545" for m in chart_moves]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=chart_dates,
                    y=chart_moves,
                    marker_color=bar_colors,
                    name="Move %",
                    text=[f"{m:+.2f}%" for m in chart_moves],
                    textposition="outside"
                ))
                fig.add_hline(y=avg_abs_move, line_dash="dash", line_color="#17a2b8", annotation_text=f"Avg Abs Move ({avg_abs_move:.2f}%)")
                fig.add_hline(y=-avg_abs_move, line_dash="dash", line_color="#17a2b8")
                fig.add_hline(y=0, line_color="#888", line_width=1)

                fig.update_layout(
                    title=f"{clean_sym}.P — {slot_short_name} IST Daily Moves ({start_date.strftime('%d %b')} to {end_date.strftime('%d %b')})",
                    yaxis=dict(title="Move % (Low-to-High / High-to-Low)"),
                    xaxis=dict(title="Date"),
                    height=420,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")

    # If single slot selected, build the exact table matching user mockup
    if not is_all_slots and selected_slot_idx is not None:
        _render_slot_day_wise_breakdown(
            clean_sym=clean_sym,
            slot_id=selected_slot_idx,
            slot_short_name=slot_short_name,
            day_slot_map=day_slot_map,
            sorted_dates=sorted_dates,
            no_of_days=no_of_days,
            start_date=start_date,
            end_date=end_date,
            all_slots_stats_sorted=all_slots_stats_sorted,
            rank_num=None,
            session_title="",
            key_prefix="single"
        )

    # --- Mode: All 6 Sessions Comparison ---
    if is_all_slots or st.checkbox("🔍 Compare All 6 IST Sessions for this Coin", value=is_all_slots, key="avg_m_show_all_sessions"):
        st.markdown("---")
        st.markdown(f"### 🌐 All 6 IST Sessions Breakdown & Ranking ({clean_sym}.P)")
        st.markdown(
            "Ranks each 4-hour window by average volatility/move %, helping you discover **which IST time zone consistently offers the best trading opportunities**."
        )

        best_slot = all_slots_stats_sorted[0]
        st.success(
            f"🏆 **Highest Potential Session**: **{best_slot['label']}** with an average move of **{best_slot['avg_move']:.2f}%**, "
            f"a **{best_slot['win_rate']:.1f}% Bullish win rate**, and a maximum move of **{best_slot['max_move']:.2f}%**."
        )

        comp_table_rows = []
        for rk, s in enumerate(all_slots_stats_sorted, start=1):
            comp_table_rows.append({
                "Rank": f"#{rk}",
                "Time Slot (IST)": s["label"],
                "Avg Move %": f"{s['avg_move']:.2f}%",
                "Bullish Win Rate": f"{s['win_rate']:.1f}% ({s['pos_count']}/{s['count']})",
                "Positive Sum": f"+{s['pos_sum']:.2f}%",
                "Negative Sum": f"{s['neg_sum']:.2f}%",
                "Net Move %": f"{s['net_move']:+.2f}%",
                "Max Move %": f"{s['max_move']:.2f}%",
                "24H USDT Vol": f"${s['total_vol']/1e6:.2f}M" if s['total_vol'] >= 1e6 else f"${s['total_vol']/1e3:.0f}K"
            })

        # Interactive Rank Clickers
        st.markdown("##### 🎯 Click any Rank to view its Day-Wise Breakdown:")
        if "avg_m_selected_rank_idx" not in st.session_state:
            st.session_state["avg_m_selected_rank_idx"] = 0

        rank_cols = st.columns(6)
        for rk, s in enumerate(all_slots_stats_sorted, start=1):
            idx = rk - 1
            is_active = (st.session_state.get("avg_m_selected_rank_idx", 0) == idx)
            btn_type = "primary" if is_active else "secondary"
            btn_prefix = "👉 " if is_active else ""
            with rank_cols[idx]:
                if st.button(
                    f"{btn_prefix}Rank #{rk}\n{s['label']}\n{s['avg_move']:.2f}%",
                    key=f"avg_m_rank_btn_{rk}",
                    type=btn_type,
                    use_container_width=True
                ):
                    st.session_state["avg_m_selected_rank_idx"] = idx
                    st.rerun()

        # Display ranking table with row selection support
        try:
            event = st.dataframe(
                pd.DataFrame(comp_table_rows),
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="avg_m_rank_df_table"
            )
            if event and hasattr(event, "selection") and event.selection and event.selection.rows:
                clicked_idx = event.selection.rows[0]
                if clicked_idx != st.session_state.get("avg_m_selected_rank_idx", 0):
                    st.session_state["avg_m_selected_rank_idx"] = clicked_idx
                    st.rerun()
        except Exception:
            st.dataframe(pd.DataFrame(comp_table_rows), use_container_width=True, hide_index=True)

        # Render Day-Wise Breakdown for the active clicked rank (Directly below ranking table)
        active_idx = st.session_state.get("avg_m_selected_rank_idx", 0)
        active_idx = max(0, min(active_idx, len(all_slots_stats_sorted) - 1))
        active_slot = all_slots_stats_sorted[active_idx]

        st.markdown("---")
        _render_slot_day_wise_breakdown(
            clean_sym=clean_sym,
            slot_id=active_slot["id"],
            slot_short_name=active_slot["label"],
            day_slot_map=day_slot_map,
            sorted_dates=sorted_dates,
            no_of_days=no_of_days,
            start_date=start_date,
            end_date=end_date,
            all_slots_stats_sorted=all_slots_stats_sorted,
            rank_num=active_idx + 1,
            session_title=active_slot["label"],
            key_prefix="rank_active"
        )

        # Comparative Win Rate & Average Move Chart (Moved to the bottom of the page)
        st.markdown("---")
        st.markdown("#### 📈 6 IST Sessions Win Rate & Average Move Comparison")

        # Chronological day cycle (Slot 0 to 5: 5.30 AM to 5.30 AM)
        chrono_slots = sorted(all_slots_stats, key=lambda x: x["id"])

        comp_fig = go.Figure()
        comp_fig.add_trace(go.Bar(
            x=[s["label"] for s in chrono_slots],
            y=[s["avg_move"] for s in chrono_slots],
            name="Avg Move %",
            marker_color="#007bff"
        ))
        comp_fig.add_trace(go.Scatter(
            x=[s["label"] for s in chrono_slots],
            y=[s["win_rate"] for s in chrono_slots],
            name="Bullish Win Rate %",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#28a745", width=3),
            marker=dict(size=8)
        ))
        comp_fig.update_layout(
            title=f"{clean_sym}.P — 6 IST Sessions Win Rate & Average Move Comparison (24-Hour Cycle)",
            yaxis=dict(title="Avg Move %"),
            yaxis2=dict(title="Bullish Win Rate %", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=340,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(comp_fig, use_container_width=True)


def render_volume_scanner():
    """Render Smart Scanner with sub-tabs"""
    st.header("💎 Smart Scanner")

    tab_vol, tab_4hr, tab_avg = st.tabs([
        "💎 Smart Volume Scanner",
        "⏰ 4HR Candle Move Scanner",
        "📊 Average Move"
    ])

    with tab_vol:
        _render_smart_volume_scanner_content()

    with tab_4hr:
        render_4hr_candle_move_scanner()

    with tab_avg:
        render_average_move_tab()


# ==============================================
# TAB 3: INDIVIDUAL COIN CHECKER
# ==============================================

def render_individual_checker():
    """Render Tab 3: Individual Coin Checker"""
    st.header("🎯 Individual Coin Checker")
    st.markdown("**Check specific coins for volume spikes**")

    col1, col2 = st.columns(2)

    with col1:
        all_coins = get_perpetual_symbols()
        selection_method = st.radio("Selection Method:", ["Dropdown", "Manual Input"], horizontal=True, key="ind_radio")

        if selection_method == "Dropdown":
            selected_coins = st.multiselect("Select coins:", options=all_coins,
                                          default=["BTCUSDT", "ETHUSDT"] if all_coins else [],
                                          max_selections=10, key="ind_dropdown")
        else:
            coin_input = st.text_area("Enter coins (one per line):",
                                     value="BTCUSDT\nETHUSDT\nSOLUSDT", height=100, key="ind_text")
            coins = []
            for line in coin_input.split('\n'):
                line = line.strip()
                if ',' in line:
                    coins.extend([c.strip().upper() for c in line.split(',') if c.strip()])
                elif line:
                    coins.append(line.upper())
            selected_coins = list(set(coins))

        if selected_coins:
            st.success(f"✅ Selected {len(selected_coins)} coins")

    with col2:
        individual_timeframe = st.selectbox("Timeframe:", ["1H", "4H", "Daily", "Weekly"], index=1, key="ind_tf")
        individual_min_ratio = st.slider("Min Volume Ratio:", 1.0, 10.0, 2.0, 0.1, key="ind_ratio")

    if st.button("🔍 Check Selected Coins", type="primary", use_container_width=True, key="ind_btn"):
        if not selected_coins:
            st.warning("Please select at least one coin!")
        else:
            with st.spinner(f"Checking {len(selected_coins)} coins..."):
                results_df = enhanced_volume_scanner_full(
                    timeframe=individual_timeframe,
                    min_usdt_ratio=individual_min_ratio,
                    min_usdt_percentile=70,
                    min_usdt_volume=50000,
                    top_n=50,
                    specific_coins=selected_coins,
                    priority_scan=False
                )

                if not results_df.empty:
                    st.success(f"✅ Found volume spikes in {len(results_df)} coins!")
                    display_cols = ['Rank', 'symbol', 'usdt_volume_ratio', 'usdt_volume_percentile',
                                  'usdt_volume_display', 'price_change_%']
                    st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)
                else:
                    st.info("No volume spikes found in selected coins.")

# ==============================================
# TAB 4: VOLUME HISTORY (WITH BTC COMPARISON)
# ==============================================

def get_historical_volume_analysis_with_btc(symbol, start_date, end_date):
    """Get daily USDT, Coin, and BTC volume for a specific coin between start and end dates."""
    if isinstance(start_date, datetime) and start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=IST)
    if isinstance(end_date, datetime) and end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=IST)

    day_span = max((end_date.date() - start_date.date()).days + 3, 7)
    klines = get_historical_klines(symbol, "1d", start_date, end_date, limit=min(day_span + 10, 1500))

    if not klines:
        return pd.DataFrame(), pd.DataFrame()

    btc_days = max((end_date.date() - start_date.date()).days + 5, 7)
    btc_volumes = get_btc_historical_volumes(btc_days)

    results = []
    for kline in klines:
        try:
            timestamp = kline[0]
            candle_date = datetime.fromtimestamp(timestamp / 1000, tz=UTC)
            candle_date_ist = candle_date.astimezone(IST)
            date_str = candle_date_ist.strftime('%b %d %a')
            full_date_str = candle_date_ist.strftime('%Y-%m-%d')

            open_price = float(kline[1])
            high_price = float(kline[2])
            low_price = float(kline[3])
            close_price = float(kline[4])
            trade_volume, usdt_volume = get_candle_volumes(kline)
            price_change = ((close_price - open_price) / open_price) * 100

            # Get BTC volume for this date
            btc_volume = btc_volumes.get(full_date_str, 0)

            results.append({
                'Date': date_str,
                'Full_Date': full_date_str,
                'USDT Volume (USD)': usdt_volume,
                'Coin Volume (USD)': trade_volume,
                'BTC Volume (USD)': btc_volume,
                'USDT_Display': format_large_number(usdt_volume),
                'Coin_Display': format_large_number(trade_volume),
                'BTC_Display': format_large_number(btc_volume),
                'price_change_%': round(price_change, 2)
            })
        except Exception as e:
            continue

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('Full_Date', ascending=True).reset_index(drop=True)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        df = df[(df["Full_Date"] >= start_str) & (df["Full_Date"] <= end_str)].reset_index(drop=True)

        display_df = pd.DataFrame()
        display_df['Date'] = df['Date']
        display_df['USDT Volume (USD)'] = df['USDT_Display']
        display_df['Coin Volume (USD)'] = df['Coin_Display']
        display_df['BTC Volume (USD)'] = df['BTC_Display']

        return display_df, df
    else:
        return pd.DataFrame(), pd.DataFrame()


def get_top_usdt_volume_row(raw_df):
    """Return the row with highest USDT quote volume in the selected period."""
    if raw_df is None or raw_df.empty:
        return None
    idx = raw_df["USDT Volume (USD)"].idxmax()
    return raw_df.loc[idx]


def get_monthwise_top_usdt_volume(raw_df):
    """
    Top USDT volume day per calendar month (YYYY-MM).
    Returns summary DataFrame and set of Full_Date strings to highlight.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(), set(), {}

    df = raw_df.copy()
    df["YearMonth"] = df["Full_Date"].str[:7]

    summary_rows = []
    month_top_dates = set()
    date_to_month_label = {}

    for year_month, group in df.groupby("YearMonth", sort=True):
        idx = group["USDT Volume (USD)"].idxmax()
        row = group.loc[idx]
        month_label = datetime.strptime(year_month, "%Y-%m").strftime("%b %Y")
        summary_rows.append(
            {
                "Month": month_label,
                "Top Date": datetime.strptime(row["Full_Date"], "%Y-%m-%d").strftime("%Y-%m-%d %A"),
                "USDT Volume": row["USDT_Display"],
                "Price Chg %": f"{row['price_change_%']}%",
            }
        )
        month_top_dates.add(row["Full_Date"])
        date_to_month_label[row["Full_Date"]] = month_label

    return pd.DataFrame(summary_rows), month_top_dates, date_to_month_label


def create_usdt_volume_focus_chart(
    volume_df_raw, symbol, month_top_dates=None, overall_top_row=None
):
    """USDT volume chart — gold bar = top USDT volume day for that month."""
    if volume_df_raw.empty:
        return None

    month_top_dates = month_top_dates or set()
    bar_colors = []
    for full_date in volume_df_raw["Full_Date"]:
        bar_colors.append("#f4b400" if full_date in month_top_dates else "#1a73e8")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="USDT Volume",
            x=volume_df_raw["Date"],
            y=volume_df_raw["USDT Volume (USD)"],
            text=volume_df_raw["USDT_Display"],
            textposition="outside",
            marker_color=bar_colors,
            hovertemplate="Date: %{x}<br>USDT Volume: %{text}<extra></extra>",
        )
    )

    n_months = len(month_top_dates)
    top_label = "<br><sup>Gold bar = top USDT volume day for each month</sup>"
    if overall_top_row is not None and n_months <= 1:
        top_label = (
            f"<br><sup>★ Highest: {overall_top_row['Full_Date']} "
            f"({overall_top_row['USDT_Display']})</sup>"
        )

    fig.update_layout(
        title={
            "text": f"{symbol} — USDT Daily Volume{top_label}",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Date",
        yaxis_title="USDT Volume (USD)",
        template="plotly_white",
        height=420,
        hovermode="x unified",
        showlegend=False,
        margin=dict(t=70, b=60, l=60, r=40),
    )
    fig.update_xaxes(tickangle=-45, nticks=min(len(volume_df_raw), 20))
    return fig


def create_volume_bar_chart(volume_df_raw, symbol):
    """Create GROUPED BAR CHART with Coin, BTC, and USDT volumes side by side"""
    if volume_df_raw.empty:
        return None

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Coin Volume',
        x=volume_df_raw['Date'],
        y=volume_df_raw['Coin Volume (USD)'],
        text=volume_df_raw['Coin_Display'],
        textposition='outside',
        marker_color='#2196F3',
        hovertemplate='Date: %{x}<br>Coin Volume: %{text}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        name='BTC Volume',
        x=volume_df_raw['Date'],
        y=volume_df_raw['BTC Volume (USD)'],
        text=volume_df_raw['BTC_Display'],
        textposition='outside',
        marker_color='#FF9800',
        hovertemplate='Date: %{x}<br>BTC Volume: %{text}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        name='USDT Volume',
        x=volume_df_raw['Date'],
        y=volume_df_raw['USDT Volume (USD)'],
        text=volume_df_raw['USDT_Display'],
        textposition='outside',
        marker_color='#4CAF50',
        hovertemplate='Date: %{x}<br>USDT Volume: %{text}<extra></extra>'
    ))

    fig.update_layout(
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        title={
            'text': f'{symbol} - Daily Volume Comparison (Coin vs BTC vs USDT)',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=18, color='#333')
        },
        xaxis_title='Date',
        yaxis_title='Volume (USD)',
        yaxis_type='log',
        template='plotly_white',
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(t=80, b=50, l=60, r=40)
    )

    fig.update_xaxes(tickangle=45, tickfont=dict(size=11))
    fig.update_yaxes(tickfont=dict(size=11), gridcolor='lightgrey')

    return fig

def create_volume_distribution_chart(volume_df_raw, symbol):
    """Create stacked bar chart showing volume distribution percentages"""
    if volume_df_raw.empty:
        return None

    df_copy = volume_df_raw.copy()
    df_copy['Total'] = df_copy['USDT Volume (USD)'] + df_copy['Coin Volume (USD)'] + df_copy['BTC Volume (USD)']
    df_copy['USDT_%'] = (df_copy['USDT Volume (USD)'] / df_copy['Total'] * 100).round(1)
    df_copy['Coin_%'] = (df_copy['Coin Volume (USD)'] / df_copy['Total'] * 100).round(1)
    df_copy['BTC_%'] = (df_copy['BTC Volume (USD)'] / df_copy['Total'] * 100).round(1)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Coin Volume %',
        x=df_copy['Date'],
        y=df_copy['Coin_%'],
        marker_color='#2196F3',
        text=df_copy['Coin_%'].apply(lambda x: f'{x}%'),
        textposition='inside',
        hovertemplate='Date: %{x}<br>Coin Volume: %{y}%<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        name='BTC Volume %',
        x=df_copy['Date'],
        y=df_copy['BTC_%'],
        marker_color='#FF9800',
        text=df_copy['BTC_%'].apply(lambda x: f'{x}%'),
        textposition='inside',
        hovertemplate='Date: %{x}<br>BTC Volume: %{y}%<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        name='USDT Volume %',
        x=df_copy['Date'],
        y=df_copy['USDT_%'],
        marker_color='#4CAF50',
        text=df_copy['USDT_%'].apply(lambda x: f'{x}%'),
        textposition='inside',
        hovertemplate='Date: %{x}<br>USDT Volume: %{y}%<extra></extra>'
    ))

    fig.update_layout(
        barmode='stack',
        title={
            'text': f'{symbol} - Daily Volume Distribution %',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=16, color='#333')
        },
        xaxis_title='Date',
        yaxis_title='Volume Distribution (%)',
        template='plotly_white',
        height=400,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        yaxis=dict(range=[0, 100])
    )

    fig.update_xaxes(tickangle=45, tickfont=dict(size=11))
    fig.update_yaxes(tickfont=dict(size=11), gridcolor='lightgrey')

    return fig

# ==============================================
# ALL ALTCOINS — SINGLE DAY SCAN (Volume History)
# ==============================================

def fetch_single_day_altcoin_stats(symbol, target_date_str, exclude_above_1=False):
    """USDT volume and low-to-high % for one symbol on a specific calendar day."""
    try:
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d").replace(tzinfo=IST)
        window_start = target_dt - timedelta(days=1)
        window_end = target_dt + timedelta(days=2)
        klines = get_historical_klines(symbol, "1d", window_start, window_end, limit=5)
        if not klines:
            return None

        matched = None
        for kline in klines:
            candle_dt = datetime.fromtimestamp(kline[0] / 1000, tz=UTC).astimezone(IST)
            if candle_dt.strftime("%Y-%m-%d") == target_date_str:
                matched = kline
                break

        if not matched:
            return None

        low = float(matched[3])
        high = float(matched[2])
        close = float(matched[4])
        _, usdt_volume = get_candle_volumes(matched)

        if exclude_above_1 and close > 1.0:
            return None

        if low <= 0:
            low_to_high = None
        else:
            low_to_high = round(((high - low) / low) * 100, 2)

        return {
            "symbol": symbol,
            "usdt_volume": usdt_volume,
            "low_to_high_pct": low_to_high,
        }
    except Exception:
        return None


def scan_all_altcoins_on_date(symbols, target_date_str, exclude_above_1=False, max_workers=25):
    """Scan all perpetual symbols for one day's USDT volume and low-to-high %."""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(
                fetch_single_day_altcoin_stats,
                symbol,
                target_date_str,
                exclude_above_1,
            ): symbol
            for symbol in symbols
        }

        completed = 0
        total = len(symbols)

        for future in as_completed(future_to_symbol):
            completed += 1
            if completed % 25 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(
                    f"Day scan {completed}/{total}... Found: {len(results)} coins with data"
                )

            try:
                row = future.result(timeout=8)
                if row and row["usdt_volume"] > 0:
                    results.append(row)
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()
    results.sort(key=lambda x: x["usdt_volume"], reverse=True)
    return results


def build_altcoin_day_scan_display_df(results):
    """Build output table: S.no, Coin, USDT volume, Low to high %."""
    rows = []
    for idx, row in enumerate(results, start=1):
        rows.append(
            {
                "S.no": idx,
                "Coin": f"{row['symbol']}.P",
                "USDT volume": format_large_number(row["usdt_volume"]),
                "Low to high %": (
                    row["low_to_high_pct"]
                    if row["low_to_high_pct"] is not None
                    else "N/A"
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_volume_history_altcoin_day_scan():
    """Scan all USDT perpetual altcoins for one specific date."""
    st.markdown(
        "Scan **all futures altcoins** for a single day. "
        "Lists USDT quote volume and **low → high %** for that daily candle."
    )

    now_ist = datetime.now(IST)
    scan_col1, scan_col2, scan_col3 = st.columns([1, 1, 1])

    with scan_col1:
        scan_date = st.date_input(
            "Select date",
            value=(now_ist - timedelta(days=1)).date(),
            max_value=now_ist.date(),
            key="hist_scan_date",
        )
    with scan_col2:
        scan_exclude_above_1 = st.checkbox(
            "Exclude coins above $1",
            value=True,
            key="hist_scan_exclude_above_1",
        )
    with scan_col3:
        scan_min_vol_m = st.number_input(
            "Min USDT volume (M)",
            min_value=0.0,
            max_value=1000.0,
            value=0.0,
            step=1.0,
            key="hist_scan_min_vol_m",
            help="0 = show all coins with data on that day",
        )

    scan_workers = st.slider(
        "Parallel workers",
        min_value=5,
        max_value=50,
        value=25,
        key="hist_scan_workers",
    )

    target_date_str = scan_date.strftime("%Y-%m-%d")

    if st.button(
        "🔍 Scan all altcoins on date",
        type="primary",
        use_container_width=True,
        key="hist_scan_all_btn",
    ):
        if scan_date > now_ist.date():
            st.error("Cannot scan a future date.")
            return

        all_symbols = get_perpetual_symbols()
        if not all_symbols:
            st.error("Failed to load perpetual symbols.")
            return

        with st.spinner(f"Scanning {len(all_symbols)} coins for {target_date_str}..."):
            scan_results = scan_all_altcoins_on_date(
                all_symbols,
                target_date_str,
                scan_exclude_above_1,
                scan_workers,
            )

        min_vol_usdt = scan_min_vol_m * 1_000_000
        if min_vol_usdt > 0:
            scan_results = [r for r in scan_results if r["usdt_volume"] >= min_vol_usdt]

        if scan_results:
            display_df = build_altcoin_day_scan_display_df(scan_results)
            st.success(
                f"✅ Found **{len(display_df)}** coins on **{target_date_str}**"
                + (" (excluding above $1)" if scan_exclude_above_1 else "")
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.metric("Coins listed", len(display_df))

            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download scan CSV",
                data=csv,
                file_name=f"altcoin_day_scan_{target_date_str}.csv",
                mime="text/csv",
                use_container_width=True,
                key="hist_scan_csv",
            )
        else:
            st.warning(
                f"No coins matched for **{target_date_str}**. "
                "Try another date, lower min volume, or turn off exclude above $1."
            )


def _render_volume_history_single_coin():
    """Single-coin volume history with custom range and month-wise tops."""
    st.markdown(
        "Focus on **USDT quote volume** per day. Pick a date range, see **month-wise tops**, "
        "and review chart + table."
    )

    now_ist = datetime.now(IST)
    col1, col2 = st.columns([2, 1])

    with col1:
        all_coins = get_perpetual_symbols()
        selected_coin = st.selectbox(
            "Select Coin:",
            options=all_coins,
            index=all_coins.index("PLAYUSDT") if "PLAYUSDT" in all_coins else 0,
            key="hist_coin",
        )

        range_mode = st.radio(
            "Date range:",
            ["Last N days", "Custom range", "Select month"],
            horizontal=True,
            key="hist_range_mode",
        )

        if range_mode == "Last N days":
            days_back = st.slider("Days to analyze:", 7, 180, 30, key="hist_days")
            end_dt = now_ist
            start_dt = end_dt - timedelta(days=days_back)
        elif range_mode == "Custom range":
            range_col1, range_col2 = st.columns(2)
            with range_col1:
                start_pick = st.date_input(
                    "Start date",
                    value=(now_ist - timedelta(days=30)).date(),
                    key="hist_start_date",
                )
            with range_col2:
                end_pick = st.date_input(
                    "End date",
                    value=now_ist.date(),
                    key="hist_end_date",
                )
            start_dt = datetime(start_pick.year, start_pick.month, start_pick.day, tzinfo=IST)
            end_dt = datetime(end_pick.year, end_pick.month, end_pick.day, 23, 59, 59, tzinfo=IST)
            if start_dt > end_dt:
                st.error("Start date must be on or before end date.")
                return
        else:
            month_pick = st.date_input(
                "Select month",
                value=now_ist.replace(day=1).date(),
                key="hist_month_pick",
            )
            start_dt = datetime(month_pick.year, month_pick.month, 1, tzinfo=IST)
            if month_pick.month == 12:
                end_dt = datetime(month_pick.year + 1, 1, 1, tzinfo=IST) - timedelta(seconds=1)
            else:
                end_dt = datetime(month_pick.year, month_pick.month + 1, 1, tzinfo=IST) - timedelta(
                    seconds=1
                )
            if end_dt > now_ist:
                end_dt = now_ist

    with col2:
        volume_data = get_top_volume_symbols(50)
        coin_data = next((item for item in volume_data if item["symbol"] == selected_coin), None)
        if coin_data:
            st.metric("24h USDT Volume", format_large_number(coin_data["volume"]))
            st.metric("24h Change", f"{coin_data['priceChangePercent']:.2f}%")

    period_label = f"{start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}"

    if st.button("📊 Analyze Volume", type="primary", use_container_width=True, key="hist_btn"):
        with st.spinner(f"Loading volume for {selected_coin} ({period_label})..."):
            display_df, raw_df = get_historical_volume_analysis_with_btc(
                selected_coin, start_dt, end_dt
            )

            if not display_df.empty:
                top_row = get_top_usdt_volume_row(raw_df)
                month_tops_df, month_top_dates, date_to_month_label = get_monthwise_top_usdt_volume(
                    raw_df
                )
                st.success(f"✅ {len(display_df)} days loaded · {period_label}")

                st.subheader("📅 Month-wise top USDT volume")
                if not month_tops_df.empty:
                    st.dataframe(month_tops_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No month-wise data in this range.")

                st.subheader("🏆 Overall (full selected period)")
                t1, t2, t3, t4 = st.columns(4)
                with t1:
                    if top_row is not None:
                        st.metric(
                            "Highest day (overall)",
                            top_row["Full_Date"],
                            help="Single highest USDT volume day across entire range",
                        )
                with t2:
                    if top_row is not None:
                        st.metric("Top USDT volume", top_row["USDT_Display"])
                with t3:
                    avg_usdt = raw_df["USDT Volume (USD)"].mean()
                    st.metric("Average USDT vol", format_large_number(avg_usdt))
                with t4:
                    if top_row is not None and avg_usdt > 0:
                        vs_avg = (top_row["USDT Volume (USD)"] / avg_usdt - 1) * 100
                        st.metric("Top vs average", f"+{vs_avg:.0f}%")

                st.subheader("📊 USDT Volume by Day")
                st.caption(
                    "Gold bar = **top USDT volume day for that calendar month** "
                    "(one highlight per month in range)."
                )
                fig_main = create_usdt_volume_focus_chart(
                    raw_df, selected_coin, month_top_dates, top_row
                )
                if fig_main:
                    st.plotly_chart(fig_main, use_container_width=True)

                with st.expander("📊 Advanced: Coin vs BTC vs USDT (grouped — can look crowded)"):
                    fig1 = create_volume_bar_chart(raw_df, selected_coin)
                    if fig1:
                        st.plotly_chart(fig1, use_container_width=True)
                    fig2 = create_volume_distribution_chart(raw_df, selected_coin)
                    if fig2:
                        st.plotly_chart(fig2, use_container_width=True)

                st.subheader("📋 Daily Volume Data")

                st.markdown("""
                <style>
                .volume-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-family: 'Courier New', monospace;
                    font-size: 14px;
                }
                .volume-table th {
                    background-color: #4CAF50;
                    color: white;
                    padding: 12px;
                    text-align: left;
                    font-weight: bold;
                }
                .volume-table td {
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                }
                .volume-table tr:hover {
                    background-color: #f5f5f5;
                }
                </style>
                """, unsafe_allow_html=True)

                html_table = "<table class='volume-table'><tr>"
                html_table += "<th>Date</th><th>USDT Volume (USD)</th><th>Coin Volume (USD)</th><th>BTC Volume (USD)</th></tr>"

                for _, raw_row in raw_df.iterrows():
                    full_date = raw_row["Full_Date"]
                    is_month_top = full_date in month_top_dates
                    is_overall_top = top_row is not None and full_date == top_row["Full_Date"]
                    row_style = (
                        ' style="background-color:#fff8e1;font-weight:bold;"'
                        if is_month_top
                        else ""
                    )
                    star = ""
                    if is_month_top:
                        month_lbl = date_to_month_label.get(full_date, "")
                        star = f" ★ TOP {month_lbl}"
                    if is_overall_top:
                        star = (star + " · overall") if is_month_top else " ★ TOP (overall)"
                    html_table += f"<tr{row_style}>"
                    html_table += f"<td>{raw_row['Date']}{star}</td>"
                    html_table += f"<td>{raw_row['USDT_Display']}</td>"
                    html_table += f"<td>{raw_row['Coin_Display']}</td>"
                    html_table += f"<td>{raw_row['BTC_Display']}</td>"
                    html_table += f"</tr>"

                html_table += "</table>"
                st.markdown(html_table, unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Table as CSV",
                        data=csv,
                        file_name=f"volume_{selected_coin}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col2:
                    if fig_main:
                        st.download_button(
                            label="📥 Download Chart as HTML",
                            data=fig_main.to_html(),
                            file_name=f"chart_{selected_coin}_{start_dt.strftime('%Y%m%d')}.html",
                            mime="text/html",
                            use_container_width=True
                        )

                with st.expander("📈 View Raw Data with Price Changes"):
                    raw_display = raw_df[['Date', 'USDT Volume (USD)', 'Coin Volume (USD)',
                                        'BTC Volume (USD)', 'price_change_%']].copy()
                    raw_display['USDT Volume (USD)'] = raw_display['USDT Volume (USD)'].apply(lambda x: format_large_number(x))
                    raw_display['Coin Volume (USD)'] = raw_display['Coin Volume (USD)'].apply(lambda x: format_large_number(x))
                    raw_display['BTC Volume (USD)'] = raw_display['BTC Volume (USD)'].apply(lambda x: format_large_number(x))
                    raw_display['price_change_%'] = raw_display['price_change_%'].apply(lambda x: f"{x}%")

                    st.dataframe(raw_display, use_container_width=True, hide_index=True)

            else:
                st.warning(f"No historical data found for {selected_coin}.")


def _render_volume_history_monthly_data():
    """Monthly USDT volume history for a single coin — bar chart + sortable table."""
    try:
        import plotly.graph_objects as go

        st.markdown(
            "View **monthly USDT volume** per candle. Identify the peak volume month, "
            "track open/high/low/close price and month-over-month % change."
        )

        now_ist = datetime.now(IST)

        # --- Coin selector ---
        all_coins = get_perpetual_symbols()
        if not all_coins:
            st.error("Failed to load perpetual symbols. Please refresh the page.")
            return

        default_idx = all_coins.index("BTCUSDT") if "BTCUSDT" in all_coins else 0
        selected_coin_m = st.selectbox(
            "Select Coin:",
            options=all_coins,
            index=default_idx,
            key="monthly_hist_coin",
        )

        # --- Year range selector ---
        current_year = now_ist.year
        year_options = list(range(current_year, 2017, -1))  # newest first for end_year
        year_options_asc = list(range(2018, current_year + 1))  # oldest first for start_year

        col_y1, col_y2 = st.columns(2)
        with col_y1:
            start_year = st.selectbox(
                "From Year",
                options=year_options_asc,
                index=max(0, len(year_options_asc) - 3),  # default: 3 years ago
                key="monthly_hist_start_year",
            )
        with col_y2:
            end_year = st.selectbox(
                "To Year",
                options=year_options,
                index=0,  # default: current year
                key="monthly_hist_end_year",
            )

        if start_year > end_year:
            st.warning("⚠️ From Year must be ≤ To Year.")
            return

        st.info(
            f"Fetching monthly USDT candles for **{selected_coin_m}** from **{start_year}** to **{end_year}**."
        )

        if st.button("📅 Load Monthly Volume", type="primary", use_container_width=True, key="monthly_hist_btn"):
            fetch_start = datetime(start_year, 1, 1, tzinfo=IST)
            fetch_end = datetime(end_year, 12, 31, 23, 59, 59, tzinfo=IST)
            if fetch_end > now_ist:
                fetch_end = now_ist

            with st.spinner(f"Loading monthly candles for {selected_coin_m} ({start_year}–{end_year})..."):
                klines = get_historical_klines(
                    selected_coin_m, "1M", fetch_start, fetch_end, limit=200
                )

            if not klines:
                st.warning(f"No monthly data found for **{selected_coin_m}**. The coin may be too new or the date range has no data.")
                return

            # Build rows — skip current incomplete month
            monthly_rows = []
            for kline in klines:
                candle_dt = datetime.fromtimestamp(kline[0] / 1000, tz=UTC).astimezone(IST)
                if candle_dt.year == now_ist.year and candle_dt.month == now_ist.month:
                    continue
                _, usdt_vol = get_candle_volumes(kline)
                monthly_rows.append({
                    "month_dt": candle_dt,
                    "Month": candle_dt.strftime("%b %Y"),
                    "month_sort": candle_dt.strftime("%Y-%m"),
                    "Open": float(kline[1]),
                    "High": float(kline[2]),
                    "Low": float(kline[3]),
                    "Close": float(kline[4]),
                    "USDT Volume": usdt_vol,
                })

            if not monthly_rows:
                st.warning("No completed monthly candles in this range.")
                return

            # Month-over-month close % change
            for i, row in enumerate(monthly_rows):
                if i == 0:
                    row["Close Change %"] = None
                else:
                    prev_close = monthly_rows[i - 1]["Close"]
                    if prev_close and prev_close > 0:
                        row["Close Change %"] = round((row["Close"] / prev_close - 1) * 100, 2)
                    else:
                        row["Close Change %"] = None

            # Peak volume month
            peak_row_m = max(monthly_rows, key=lambda x: x["USDT Volume"])
            peak_month_str = peak_row_m["month_sort"]

            st.success(
                f"\u2705 **{len(monthly_rows)} months** loaded \u00b7 "
                f"Peak volume: **{peak_row_m['Month']}** "
                f"({format_large_number(peak_row_m['USDT Volume'])})"
            )

            # --- Bar Chart ---
            months_labels = [r["Month"] for r in monthly_rows]
            usdt_vols = [r["USDT Volume"] for r in monthly_rows]
            bar_colors = [
                "#FFD700" if r["month_sort"] == peak_month_str else "#4C8BF5"
                for r in monthly_rows
            ]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=months_labels,
                y=usdt_vols,
                marker_color=bar_colors,
                text=[format_large_number(v) for v in usdt_vols],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>USDT Volume: %{text}<extra></extra>",
                name="USDT Volume",
            ))
            fig.update_layout(
                title=f"\U0001f4c5 {selected_coin_m} \u2014 Monthly USDT Volume ({start_year}\u2013{end_year})",
                xaxis_title="Month",
                yaxis_title="USDT Volume",
                yaxis=dict(tickformat=".2s"),
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(color="white"),
                xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
                height=480,
                margin=dict(t=60, b=120),
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Table ---
            st.subheader("\U0001f4cb Monthly Volume Table")

            def fmt_vol(val):
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return "N/A"
                if val >= 1_000_000_000:
                    return f"{val / 1_000_000_000:.2f}B"
                elif val >= 1_000_000:
                    return f"{val / 1_000_000:.2f}M"
                return f"{val:.2f}"

            def fmt_chg(val):
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return "\u2014"
                return f"{'+' if val >= 0 else ''}{val:.2f}%"

            display_rows = []
            for idx, row in enumerate(monthly_rows, start=1):
                is_peak = row["month_sort"] == peak_month_str
                display_rows.append({
                    "S.No": idx,
                    "Month": ("\U0001f3c6 " if is_peak else "") + row["Month"],
                    "Open": round(row["Open"], 6),
                    "High": round(row["High"], 6),
                    "Low": round(row["Low"], 6),
                    "Close": round(row["Close"], 6),
                    "Close Change %": fmt_chg(row["Close Change %"]),
                    "USDT Volume": row["USDT Volume"],
                })

            display_df_m = pd.DataFrame(display_rows)
            styled_m = display_df_m.style.format({"USDT Volume": fmt_vol})
            st.dataframe(styled_m, use_container_width=True, hide_index=True)

            csv_df_m = display_df_m.copy()
            csv_df_m["USDT Volume"] = csv_df_m["USDT Volume"].apply(fmt_vol)
            st.download_button(
                label="\U0001f4e5 Download Monthly CSV",
                data=csv_df_m.to_csv(index=False),
                file_name=f"monthly_volume_{selected_coin_m}_{start_year}_{end_year}.csv",
                mime="text/csv",
                use_container_width=True,
                key="monthly_hist_csv",
            )

    except Exception as e:
        st.error(f"Error in Monthly Data tab: {e}")
        st.exception(e)


def render_volume_history():
    """Render Tab 4: Volume History — single coin, monthly, or all-altcoin day scan."""
    st.header("📈 Historical Volume Analysis")

    tab_single, tab_monthly, tab_day_scan = st.tabs(
        ["📊 Single coin (Daily)", "📅 Monthly Data", "🔍 All altcoins on date"]
    )

    with tab_single:
        _render_volume_history_single_coin()

    with tab_monthly:
        _render_volume_history_monthly_data()

    with tab_day_scan:
        _render_volume_history_altcoin_day_scan()

# ==============================================
# TAB 5: TOP GAINERS/LOSERS (FULL SCAN)
# ==============================================

def _gainers_period_params(timeframe, period_date):
    """Return start_date, end_date, interval, limit for top gainers scan."""
    if timeframe == "Monthly":
        start_date = datetime(period_date.year, period_date.month, 1, tzinfo=IST)
        if period_date.month == 12:
            end_date = datetime(period_date.year + 1, 1, 1, tzinfo=IST) - timedelta(seconds=1)
        else:
            end_date = datetime(period_date.year, period_date.month + 1, 1, tzinfo=IST) - timedelta(
                seconds=1
            )
        return start_date, end_date, "1d", 31
    if timeframe == "Weekly":
        start_date = period_date - timedelta(days=period_date.weekday())
        start_date = datetime(
            start_date.year, start_date.month, start_date.day, 0, 0, 0, tzinfo=IST
        )
        end_date = start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
        return start_date, end_date, "4h", 42
    start_date = datetime(
        period_date.year, period_date.month, period_date.day, 0, 0, 0, tzinfo=IST
    )
    end_date = datetime(
        period_date.year, period_date.month, period_date.day, 23, 59, 59, tzinfo=IST
    )
    return start_date, end_date, "1h", 24


def fetch_single_gainer_stats(symbol, timeframe, period_date):
    """Worker: low-to-high % and open-to-close % for one symbol in the period."""
    try:
        start_date, end_date, interval, limit = _gainers_period_params(timeframe, period_date)
        klines = get_historical_klines(symbol, interval, start_date, end_date, limit=limit)

        if not klines or len(klines) < 2:
            return None

        low_prices = [float(kline[3]) for kline in klines]
        high_prices = [float(kline[2]) for kline in klines]
        period_low = min(low_prices)
        period_high = max(high_prices)

        if period_low <= 0:
            return None

        percent_change = ((period_high - period_low) / period_low) * 100
        open_price = float(klines[0][1])
        close_price = float(klines[-1][4])
        open_close_change = ((close_price - open_price) / open_price) * 100

        return {
            "symbol": symbol,
            "low_to_high_%": percent_change,
            "open_close_%": open_close_change,
        }
    except Exception:
        return None


def get_top_gainers_full(timeframe, period_date, top_n=10, max_workers=25):
    """Find top gainers/losers candidates — parallel scan of all coins."""
    symbols = get_perpetual_symbols()
    if not symbols:
        return pd.DataFrame()

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(fetch_single_gainer_stats, symbol, timeframe, period_date): symbol
            for symbol in symbols
        }

        completed = 0
        total = len(symbols)

        for future in as_completed(future_to_symbol):
            completed += 1
            if completed % 25 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(
                    f"Scanning {completed}/{total} pairs... Collected: {len(results)}"
                )

            try:
                row = future.result(timeout=10)
                if row:
                    results.append(row)
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("low_to_high_%", ascending=False).head(top_n).reset_index(drop=True)
        df.index = df.index + 1
        df = df.rename_axis("Rank").reset_index()
        return df
    return pd.DataFrame()

def render_top_gainers():
    """Render Tab 5: Top Gainers/Losers - FULL SCAN"""
    st.header("🏆 Top Gainers & Losers Finder")
    st.markdown("**Find the biggest gainers and losers for any period - SCANS ALL 500+ COINS**")

    col1, col2 = st.columns(2)

    with col1:
        timeframe = st.selectbox("Timeframe:", ["Monthly", "Weekly", "Daily"], index=0, key="gainers_tf")
        analysis_type = st.radio("Analysis Type:", ["Gainers Only", "Losers Only", "Both"], horizontal=True, key="gainers_type")

    with col2:
        top_n = st.number_input("Number to fetch:", 1, 50, 10, key="gainers_n")
        now_ist = datetime.now(IST)

        if timeframe == "Monthly":
            selected_month = st.date_input("Select Month:", value=now_ist - relativedelta(months=1), key="gainers_month")
            period_date = datetime(selected_month.year, selected_month.month, 1, tzinfo=IST)
        elif timeframe == "Weekly":
            selected_week = st.date_input("Select any day in week:", value=now_ist - timedelta(days=7), key="gainers_week")
            period_date = datetime(selected_week.year, selected_week.month, selected_week.day, tzinfo=IST)
        else:
            selected_day = st.date_input("Select Day:", value=now_ist - timedelta(days=1), key="gainers_day")
            period_date = datetime(selected_day.year, selected_day.month, selected_day.day, tzinfo=IST)

    gainers_workers = st.slider(
        "Parallel Workers (Faster = More API calls)",
        min_value=5,
        max_value=50,
        value=25,
        key="gainers_workers",
    )

    if st.button("🚀 Find Top Performers (Full Scan)", type="primary", use_container_width=True, key="gainers_btn"):
        symbol_count = len(get_perpetual_symbols())
        with st.spinner(
            f"Finding top performers from ALL {symbol_count} coins "
            f"({gainers_workers} workers)..."
        ):
            df = get_top_gainers_full(
                timeframe, period_date, top_n * 2, gainers_workers
            )

            if not df.empty:
                gainers_df = df[df['open_close_%'] > 0].copy().head(top_n)
                losers_df = df[df['open_close_%'] < 0].copy().head(top_n)

                if not losers_df.empty:
                    losers_df['high_to_low_%'] = abs(losers_df['low_to_high_%']) * -1

                if analysis_type in ["Gainers Only", "Both"] and not gainers_df.empty:
                    st.subheader(f"🏆 Top {len(gainers_df)} Gainers")
                    st.dataframe(
                        gainers_df[['Rank', 'symbol', 'open_close_%', 'low_to_high_%']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "open_close_%": st.column_config.NumberColumn("Open→Close %", format="%.2f%%"),
                            "low_to_high_%": st.column_config.NumberColumn("Low→High %", format="%.2f%%")
                        }
                    )

                if analysis_type in ["Losers Only", "Both"] and not losers_df.empty:
                    st.subheader(f"📉 Top {len(losers_df)} Losers")
                    st.dataframe(
                        losers_df[['Rank', 'symbol', 'open_close_%', 'high_to_low_%']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "open_close_%": st.column_config.NumberColumn("Open→Close %", format="%.2f%%"),
                            "high_to_low_%": st.column_config.NumberColumn("High→Low %", format="%.2f%%")
                        }
                    )

                col1, col2 = st.columns(2)
                with col1:
                    if not gainers_df.empty:
                        csv = gainers_df.to_csv(index=False)
                        st.download_button("📥 Download Gainers", csv, f"gainers_{period_date.strftime('%Y%m%d')}.csv", use_container_width=True)
                with col2:
                    if not losers_df.empty:
                        csv = losers_df.to_csv(index=False)
                        st.download_button("📥 Download Losers", csv, f"losers_{period_date.strftime('%Y%m%d')}.csv", use_container_width=True)
            else:
                st.warning("No data found for selected period.")

# ==============================================
# TAB 6: DAY PATTERN ANALYZER (WITH CHARTS)
# ==============================================

def analyze_day_patterns_full(symbol, day_of_week, timeframe, lookback_days=30, min_volume_usdt=100000, price_threshold=0.5):
    """Analyze patterns for a specific day of week - WITH CHARTS.

    Returns (pattern_df, daily_series_df). For timeframe 'Daily', daily_series_df holds
    every 1d candle in the lookback for charting (true daily x-axis); pattern_df remains
    rows for the selected weekday only (table, metrics, pie). For 4H/1H, daily_series_df is None.
    """
    end_date = datetime.now(IST)
    start_date = end_date - timedelta(days=lookback_days)

    binance_interval = '1d' if timeframe == 'Daily' else ('4h' if timeframe == '4H' else '1h')
    candles_per_day = 1 if timeframe == 'Daily' else (6 if timeframe == '4H' else 24)

    klines = get_historical_klines(symbol, binance_interval, start_date, end_date,
                                  limit=lookback_days * candles_per_day + 10)

    if not klines:
        return pd.DataFrame(), None

    results = []
    all_daily_chart = []
    day_name_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                    4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    for kline in klines:
        try:
            timestamp = kline[0]
            candle_date = datetime.fromtimestamp(timestamp / 1000, tz=UTC)
            candle_date_ist = candle_date.astimezone(IST)
            candle_day = day_name_map[candle_date_ist.weekday()]

            open_price = float(kline[1])
            high_price = float(kline[2])
            low_price = float(kline[3])
            close_price = float(kline[4])
            trade_volume, usdt_volume = get_candle_volumes(kline)
            price_change_pct = ((close_price - open_price) / open_price) * 100

            if price_change_pct > price_threshold:
                status = 'Bullish'
            elif price_change_pct < -price_threshold:
                status = 'Bearish'
            else:
                status = 'Neutral'

            if timeframe == 'Daily':
                all_daily_chart.append({
                    'date': candle_date_ist.strftime('%Y-%m-%d'),
                    'day': candle_day,
                    'price_change_%': price_change_pct,
                    'usdt_volume': usdt_volume,
                    'usdt_volume_display': format_large_number(usdt_volume),
                    'status': status,
                    'is_pattern_day': candle_day == day_of_week,
                })

            if candle_day != day_of_week:
                continue

            if timeframe in ['4H', '1H']:
                time_period = candle_date_ist.strftime('%H:%M')
                if timeframe == '4H':
                    end_time = (candle_date_ist + timedelta(hours=4)).strftime('%H:%M')
                    time_period = f"{time_period}-{end_time}"
            else:
                time_period = 'Daily'

            results.append({
                'date': candle_date_ist.strftime('%Y-%m-%d'),
                'day': candle_day,
                'time_period': time_period,
                'status': status,
                'price_change_%': price_change_pct,
                'usdt_volume': usdt_volume,
                'usdt_volume_display': format_large_number(usdt_volume),
                'coin_volume': trade_volume,
                'coin_volume_display': format_large_number(trade_volume),
                'open_price': open_price,
                'close_price': close_price,
                'high_price': high_price,
                'low_price': low_price,
                'volume_spike': 'Yes' if usdt_volume >= min_volume_usdt else 'No'
            })
        except:
            continue

    daily_for_chart = None
    if timeframe == 'Daily' and all_daily_chart:
        daily_for_chart = pd.DataFrame(all_daily_chart).drop_duplicates(subset=['date'], keep='last')
        daily_for_chart = daily_for_chart.sort_values('date')

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('date', ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df = df.rename_axis('No.').reset_index()
        return df, daily_for_chart

    return pd.DataFrame(), daily_for_chart


def _classify_daily_status(price_change_pct, price_threshold=0.5):
    if price_change_pct > price_threshold:
        return "Bullish"
    if price_change_pct < -price_threshold:
        return "Bearish"
    return "Neutral"


def fetch_single_coin_day_stats(
    symbol,
    lookback_days=20,
    below_1_only=False,
    min_volume_usdt=0,
    price_threshold=0.5,
    day_of_week=None,
):
    """Count bullish/bearish/neutral daily candles over lookback for one symbol.

    day_of_week: if set (e.g. 'Monday'), only count candles on that weekday.
    """
    day_name_map = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    try:
        end_date = datetime.now(IST)
        start_date = end_date - timedelta(days=lookback_days)
        klines = get_historical_klines(
            symbol, "1d", start_date, end_date, limit=lookback_days + 5
        )
        if not klines or len(klines) < 2:
            return None

        latest_close = float(klines[-1][4])
        if below_1_only and latest_close > 1.0:
            return None

        bullish = bearish = neutral = 0
        usdt_volumes = []
        lh_pcts = []
        oc_pcts = []
        lh_vol_products = []
        total_matched_volume = 0

        for kline in klines[-lookback_days:]:
            candle_date = datetime.fromtimestamp(kline[0] / 1000, tz=UTC).astimezone(IST)
            candle_day = day_name_map[candle_date.weekday()]
            if day_of_week and candle_day != day_of_week:
                continue

            open_price = float(kline[1])
            high_price = float(kline[2])
            low_price = float(kline[3])
            close_price = float(kline[4])
            if open_price <= 0 or low_price <= 0:
                continue
            _, usdt_volume = get_candle_volumes(kline)
            usdt_volumes.append(usdt_volume)

            price_change_pct = ((close_price - open_price) / open_price) * 100
            oc_pcts.append(abs(price_change_pct))

            lh_pct = ((high_price - low_price) / low_price) * 100
            lh_pcts.append(lh_pct)
            lh_vol_products.append(lh_pct * usdt_volume)
            total_matched_volume += usdt_volume

            status = _classify_daily_status(price_change_pct, price_threshold)
            if status == "Bullish":
                bullish += 1
            elif status == "Bearish":
                bearish += 1
            else:
                neutral += 1

        total_days = bullish + bearish + neutral
        if total_days == 0:
            return None

        avg_volume = sum(usdt_volumes) / len(usdt_volumes) if usdt_volumes else 0
        if min_volume_usdt > 0 and avg_volume < min_volume_usdt:
            return None

        score = round(sum(lh_pcts) / len(lh_pcts)) if lh_pcts else 0
        vw_score = round(sum(lh_vol_products) / total_matched_volume) if total_matched_volume > 0 else 0
        oc_score = round(sum(oc_pcts) / len(oc_pcts)) if oc_pcts else 0
        volatility = round(float(np.std(lh_pcts)), 2) if len(lh_pcts) > 1 else 0.0

        return {
            "symbol": symbol,
            "total_days": total_days,
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
            "bullish_pct": round(bullish / total_days * 100, 1),
            "bearish_pct": round(bearish / total_days * 100, 1),
            "avg_usdt_volume": avg_volume,
            "score": score,
            "vw_score": vw_score,
            "oc_score": oc_score,
            "volatility": volatility
        }
    except Exception:
        return None


def scan_all_coins_day_patterns_parallel(
    symbols,
    lookback_days=20,
    below_1_only=False,
    min_volume_usdt=0,
    max_workers=25,
    price_threshold=0.5,
    day_of_week=None,
):
    """Parallel scan: rank all coins by daily bullish/bearish counts."""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(
                fetch_single_coin_day_stats,
                symbol,
                lookback_days,
                below_1_only,
                min_volume_usdt,
                price_threshold,
                day_of_week,
            ): symbol
            for symbol in symbols
        }

        completed = 0
        total = len(symbols)

        for future in as_completed(future_to_symbol):
            completed += 1
            if completed % 25 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(
                    f"Scanning day patterns {completed}/{total} pairs... "
                    f"Matched: {len(results)}"
                )

            try:
                result = future.result(timeout=10)
                if result:
                    results.append(result)
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()
    return results


def create_scan_all_day_pattern_charts(positive_df, negative_df, lookback_days, day_of_week=None):
    """Grouped bar charts for top positive and negative day-pattern coins."""
    if positive_df.empty and negative_df.empty:
        return None

    day_label = f"{day_of_week}s only" if day_of_week else "all days"
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Top Bullish ({day_label}, {lookback_days}d lookback)",
            f"Top Bearish ({day_label}, {lookback_days}d lookback)",
        ),
        horizontal_spacing=0.12,
    )

    if not positive_df.empty:
        pos = positive_df.head(10)
        fig.add_trace(
            go.Bar(
                x=pos["Coin"],
                y=pos["Bullish"],
                name="Bullish",
                marker_color="#4CAF50",
                text=pos["Bullish"],
                textposition="outside",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=pos["Coin"],
                y=pos["Bearish"],
                name="Bearish (positive table)",
                marker_color="#F44336",
                text=pos["Bearish"],
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    if not negative_df.empty:
        neg = negative_df.head(10)
        fig.add_trace(
            go.Bar(
                x=neg["Coin"],
                y=neg["Bearish"],
                name="Bearish",
                marker_color="#F44336",
                text=neg["Bearish"],
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=neg["Coin"],
                y=neg["Bullish"],
                name="Bullish (negative table)",
                marker_color="#4CAF50",
                text=neg["Bullish"],
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        height=500,
        barmode="group",
        template="plotly_white",
        title=(
            f"Day Pattern Rankings — {day_label}, last {lookback_days} calendar days"
        ),
        title_x=0.5,
    )
    fig.update_yaxes(title_text="Days", row=1, col=1)
    fig.update_yaxes(title_text="Days", row=1, col=2)

    return fig


def create_day_pattern_chart(pattern_df, symbol, day_of_week, daily_series_df=None):
    """Create interactive chart for day patterns.

    When daily_series_df is provided (Daily timeframe), bar charts use every calendar day
    (each bar = one 1d candle); vivid colors mark the selected weekday, muted bars are other days.
    """
    if pattern_df.empty:
        return None

    use_daily_context = daily_series_df is not None and not daily_series_df.empty
    chart_df = daily_series_df.sort_values('date') if use_daily_context else pattern_df.sort_values('date')
    x_plot = pd.to_datetime(chart_df['date'])

    if use_daily_context:
        title_price = f'{symbol} - Daily price change % (vivid = {day_of_week})'
        title_vol = f'{symbol} - Daily USDT volume (vivid = {day_of_week})'
        colors_price = [
            ('#4CAF50' if s == 'Bullish' else '#F44336' if s == 'Bearish' else '#FFC107')
            if is_p else '#CFD8DC'
            for s, is_p in zip(chart_df['status'], chart_df['is_pattern_day'])
        ]
        colors_vol = ['#2196F3' if is_p else '#B0BEC5' for is_p in chart_df['is_pattern_day']]
        text_price = chart_df['status'].where(chart_df['is_pattern_day'], '').fillna('')
        text_vol = chart_df['usdt_volume_display'].where(chart_df['is_pattern_day'], '').fillna('')
        customdata = list(zip(chart_df['day'], chart_df['status']))
        hover_price = (
            'Weekday: %{customdata[0]}<br>Date: %{x|%Y-%m-%d}<br>Change: %{y:.2f}%<br>Status: %{customdata[1]}<extra></extra>'
        )
        hover_vol = (
            'Weekday: %{customdata[0]}<br>Date: %{x|%Y-%m-%d}<br>Volume: %{customdata[1]}<extra></extra>'
        )
        vol_custom = list(zip(chart_df['day'], chart_df['usdt_volume_display']))
    else:
        title_price = f'{symbol} - {day_of_week} Price Change Pattern'
        title_vol = f'{symbol} - {day_of_week} USDT Volume'
        colors_price = [
            '#4CAF50' if x == 'Bullish' else '#F44336' if x == 'Bearish' else '#FFC107'
            for x in chart_df['status']
        ]
        colors_vol = ['#2196F3'] * len(chart_df)
        text_price = chart_df['status']
        text_vol = chart_df['usdt_volume_display']
        customdata = None
        hover_price = 'Date: %{x}<br>Change: %{y:.2f}%<br>Status: %{text}<extra></extra>'
        hover_vol = None
        vol_custom = None

    fig = make_subplots(
        rows=3, cols=1,
        specs=[
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "domain"}],
        ],
        subplot_titles=(
            title_price,
            title_vol,
            f'{symbol} - {day_of_week} Performance Distribution'
        ),
        vertical_spacing=0.15,
        row_heights=[0.4, 0.3, 0.3]
    )

    fig.add_trace(
        go.Bar(
            x=x_plot,
            y=chart_df['price_change_%'],
            name='Price Change %',
            marker_color=colors_price,
            text=text_price,
            textposition='outside',
            customdata=customdata,
            hovertemplate=hover_price
        ),
        row=1, col=1
    )

    vol_trace_kwargs = dict(
        x=x_plot,
        y=chart_df['usdt_volume'],
        name='USDT Volume',
        marker_color=colors_vol,
        text=text_vol,
        textposition='outside',
    )
    if use_daily_context:
        vol_trace_kwargs['customdata'] = vol_custom
        vol_trace_kwargs['hovertemplate'] = hover_vol
    fig.add_trace(go.Bar(**vol_trace_kwargs), row=2, col=1)

    status_counts = pattern_df['status'].value_counts()
    colors_pie = {'Bullish': '#4CAF50', 'Bearish': '#F44336', 'Neutral': '#FFC107'}
    pie_colors = [colors_pie.get(status, '#9E9E9E') for status in status_counts.index]

    fig.add_trace(
        go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            name='Distribution',
            marker_colors=pie_colors,
            textinfo='label+percent',
            hovertemplate='%{label}: %{value} occurrences (%{percent})<extra></extra>'
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=900,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified',
        title=f"{symbol} - {day_of_week} Pattern Analysis",
        title_x=0.5
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price Change %", row=1, col=1)
    fig.update_yaxes(title_text="USDT Volume", row=2, col=1)

    return fig

def render_day_patterns():
    """Render Tab 6: Day Pattern Analyzer with Charts"""
    st.header("📅 Day Pattern Analyzer")
    st.markdown(
        "**Specific coin:** day-of-week patterns with charts. "
        "**Scan all coins:** rank coins by bullish/bearish daily candles."
    )

    scan_mode = st.radio(
        "Scan Mode:",
        ["Specific coin", "Scan all coins"],
        horizontal=True,
        key="pattern_scan_mode",
    )

    col1, col2 = st.columns(2)
    with col1:
        n_coins = st.number_input(
            "No of coins to display",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="pattern_n_coins",
        )
    with col2:
        lookback_days = st.slider(
            "Lookback Days:",
            10,
            365,
            20 if scan_mode == "Scan all coins" else 90,
            key="pattern_lookback",
        )

    all_coins = get_perpetual_symbols()

    if scan_mode == "Specific coin":
        col1, col2 = st.columns(2)

        with col1:
            selected_coins = st.multiselect(
                "Select Coin(s) or Paste:",
                options=all_coins,
                default=["BTCUSDT"] if "BTCUSDT" in all_coins else [all_coins[0]] if all_coins else [],
                key="pattern_coin",
            )

        with col2:
            st.metric("Coins Selected", len(selected_coins))

        col1, col2, col3 = st.columns(3)

        with col1:
            day_of_week = st.selectbox(
                "Day of Week:",
                [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ],
                key="pattern_day",
            )

        with col2:
            timeframe = st.selectbox(
                "Timeframe:",
                ["Daily", "4H", "1H"],
                index=0,
                key="pattern_tf",
                help="Daily: charts plot every calendar day (each bar = one 1d candle). "
                "Vivid bars highlight the weekday you selected; muted bars are other days.",
            )

        with col3:
            min_volume = st.slider(
                "Min USDT Volume:",
                1000,
                10000000,
                100000,
                10000,
                key="pattern_volume",
            )

        if st.button(
            "🔍 Analyze Day Patterns",
            type="primary",
            use_container_width=True,
            key="pattern_btn",
        ):
            if not selected_coins:
                st.warning("Please select at least one coin to analyze.")
            else:
                all_patterns = []
                summary_data = []

                with st.spinner(f"Analyzing {day_of_week} patterns for {len(selected_coins)} coins..."):
                    for selected_coin in selected_coins:
                        pattern_df, _ = analyze_day_patterns_full(
                            selected_coin,
                            day_of_week,
                            timeframe,
                            lookback_days,
                            min_volume,
                            0.5,
                        )

                        if not pattern_df.empty:
                            bullish_count = len(pattern_df[pattern_df["status"] == "Bullish"])
                            bearish_count = len(pattern_df[pattern_df["status"] == "Bearish"])
                            total = len(pattern_df)

                            bullish_pct = (bullish_count / total * 100) if total > 0 else 0
                            bearish_pct = (bearish_count / total * 100) if total > 0 else 0

                            summary_data.append({
                                "Coin": selected_coin,
                                "Total Days": total,
                                "Bullish": bullish_count,
                                "Bearish": bearish_count,
                                "Bullish %": f"{bullish_pct:.1f}%",
                                "Bearish %": f"{bearish_pct:.1f}%"
                            })

                            # Add Coin column at the beginning for the detailed table
                            pattern_df.insert(0, "Coin", selected_coin)
                            all_patterns.append(pattern_df)

                if summary_data:
                    st.success(f"✅ Found {day_of_week} occurrences for {len(summary_data)} coins!")

                    st.subheader("📊 Summary Results")
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

                    st.subheader("📋 Combined Detailed Pattern Data")
                    combined_df = pd.concat(all_patterns, ignore_index=True)

                    display_cols = [
                        "Coin",
                        "date",
                        "time_period",
                        "status",
                        "price_change_%",
                        "usdt_volume_display",
                        "volume_spike",
                    ]

                    st.dataframe(
                        combined_df[display_cols],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "price_change_%": st.column_config.NumberColumn(
                                "Change %", format="%.2f%%"
                            )
                        },
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Summary (CSV)",
                            data=summary_df.to_csv(index=False),
                            file_name=f"summary_{day_of_week}_patterns.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    with col2:
                        st.download_button(
                            label="📥 Download Detailed Data (CSV)",
                            data=combined_df.to_csv(index=False),
                            file_name=f"detailed_{day_of_week}_patterns.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.warning(f"No {day_of_week} data found for the selected coins.")

    else:
        col1, col2 = st.columns(2)
        with col1:
            day_filter_mode = st.radio(
                "Day Filter:",
                ["All day", "Specific day"],
                horizontal=True,
                key="pattern_scan_day_mode",
            )
        with col2:
            scan_day_of_week = None
            if day_filter_mode == "Specific day":
                scan_day_of_week = st.selectbox(
                    "Day of Week:",
                    [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday",
                    ],
                    key="pattern_scan_day",
                    help="Use a longer lookback (e.g. 90+ days) for more weekday samples.",
                )

        col1, col2, col3 = st.columns(3)

        with col1:
            below_1_only = st.checkbox(
                "Below $1 coins only",
                value=False,
                key="pattern_below_1",
                help="Skip coins whose latest daily close is above $1",
            )

        with col2:
            use_volume_filter = st.checkbox(
                "Volume filter (optional)",
                value=False,
                key="pattern_use_vol_filter",
            )

        with col3:
            parallel_workers = st.slider(
                "Parallel Workers (Faster = More API calls)",
                min_value=5,
                max_value=50,
                value=25,
                key="pattern_workers",
            )

        min_volume_scan = 0
        if use_volume_filter:
            min_volume_scan = st.slider(
                "Min avg daily USDT volume ($):",
                1000,
                10000000,
                100000,
                10000,
                key="pattern_scan_min_volume",
            )

        priority_scan = st.checkbox(
            "Priority Scan (High volume first)",
            value=True,
            key="pattern_priority",
        )

        with st.expander("⚙️ Advanced Scoring Options"):
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                adv_use_vw_score = st.checkbox("Show Volume-Weighted Score", value=False, help="Weight the daily moves by USDT volume")
                adv_show_oc_score = st.checkbox("Show Open-to-Close Score", value=False, help="Show the average of the absolute Open-to-Close percentage")
            with col_adv2:
                adv_show_volatility = st.checkbox("Show Volatility (Consistency)", value=False, help="Show the standard deviation of the daily moves. Lower is more consistent.")
                adv_min_score = st.slider("Minimum Base Score Filter", 0, 100, 0, 1, help="Only show coins with a base Score greater than or equal to this")

        filter_parts = [f"last **{lookback_days}** calendar days"]
        if day_filter_mode == "Specific day" and scan_day_of_week:
            filter_parts.append(f"**{scan_day_of_week}s** only")
        if below_1_only:
            filter_parts.append("below **$1** only")
        if use_volume_filter:
            filter_parts.append(
                f"avg volume ≥ **{format_large_number(min_volume_scan)}**"
            )
        st.info("Scanning " + " | ".join(filter_parts))

        if st.button(
            "🔍 Scan All Coins",
            type="primary",
            use_container_width=True,
            key="pattern_scan_all_btn",
        ):
            start_time = time.time()

            with st.spinner("Loading perpetual pairs..."):
                if not all_coins:
                    st.error("Failed to fetch perpetual pairs.")
                    st.stop()

                if priority_scan:
                    top_volume = get_top_volume_symbols(500)
                    symbols_to_scan = [
                        item["symbol"]
                        for item in top_volume
                        if item["symbol"] in all_coins
                    ]
                    remaining = [s for s in all_coins if s not in symbols_to_scan]
                    symbols_to_scan.extend(remaining)
                else:
                    symbols_to_scan = all_coins

            st.info(f"⚡ Scanning {len(symbols_to_scan)} coins with {parallel_workers} workers...")

            results = scan_all_coins_day_patterns_parallel(
                symbols_to_scan,
                lookback_days,
                below_1_only,
                min_volume_scan,
                parallel_workers,
                day_of_week=scan_day_of_week,
            )

            if not results:
                st.warning("No coins matched the scan criteria. Try adjusting filters.")
            else:
                df = pd.DataFrame(results)

                if adv_min_score > 0:
                    df = df[df['score'] >= adv_min_score]
                    if df.empty:
                        st.warning(f"No coins matched the minimum score filter of {adv_min_score}.")
                        st.stop()

                elapsed = time.time() - start_time

                rename_cols = {
                    "symbol": "Coin",
                    "total_days": "No of Day",
                    "bullish": "Bullish",
                    "bearish": "Bearish",
                    "score": "Score",
                    "vw_score": "VW Score",
                    "oc_score": "O-to-C Score",
                    "volatility": "Volatility"
                }

                pos_display_cols = ["Coin", "No of Day", "Bullish", "Bearish", "Score"]
                if adv_use_vw_score: pos_display_cols.append("VW Score")
                if adv_show_oc_score: pos_display_cols.append("O-to-C Score")
                if adv_show_volatility: pos_display_cols.append("Volatility")

                positive_df = (
                    df.sort_values(["bullish", "bullish_pct"], ascending=False)
                    .head(n_coins)
                    .reset_index(drop=True)
                )
                positive_df.index = positive_df.index + 1
                positive_display = positive_df.rename(columns=rename_cols)[pos_display_cols]
                positive_display = positive_display.rename_axis("#").reset_index()

                neg_display_cols = ["Coin", "No of Day", "Bearish", "Bullish", "Score"]
                if adv_use_vw_score: neg_display_cols.append("VW Score")
                if adv_show_oc_score: neg_display_cols.append("O-to-C Score")
                if adv_show_volatility: neg_display_cols.append("Volatility")

                negative_df = (
                    df.sort_values(["bearish", "bearish_pct"], ascending=False)
                    .head(n_coins)
                    .reset_index(drop=True)
                )
                negative_df.index = negative_df.index + 1
                negative_display = negative_df.rename(columns=rename_cols)[neg_display_cols]
                negative_display = negative_display.rename_axis("#").reset_index()

                st.success(
                    f"✅ Scanned {len(df)} qualifying coins in {elapsed:.1f} seconds "
                    f"(from {len(symbols_to_scan)} pairs)"
                )

                chart_fig = create_scan_all_day_pattern_charts(
                    positive_display, negative_display, lookback_days, scan_day_of_week
                )
                if chart_fig:
                    st.plotly_chart(chart_fig, use_container_width=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📈 Positive Days")
                    day_caption = (
                        f"Coins with the most bullish {scan_day_of_week} candles"
                        if scan_day_of_week
                        else "Coins with the most bullish daily candles"
                    )
                    st.caption(day_caption)
                    st.dataframe(positive_display, use_container_width=True, hide_index=True)

                with col2:
                    st.subheader("📉 Negative Days")
                    day_caption = (
                        f"Coins with the most bearish {scan_day_of_week} candles"
                        if scan_day_of_week
                        else "Coins with the most bearish daily candles"
                    )
                    st.caption(day_caption)
                    st.dataframe(negative_display, use_container_width=True, hide_index=True)

                day_suffix = f"_{scan_day_of_week.lower()}" if scan_day_of_week else ""
                combined_csv = pd.concat(
                    [
                        positive_display.assign(Table="Positive Days"),
                        negative_display.assign(Table="Negative Days"),
                    ],
                    ignore_index=True,
                ).to_csv(index=False)
                st.download_button(
                    label="📥 Download Scan Results (CSV)",
                    data=combined_csv,
                    file_name=f"day_patterns{day_suffix}_scan_{lookback_days}d_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# ==============================================
# TAB 7: MONTHLY REPORT GENERATOR (FULL SCAN)
# ==============================================

def analyze_single_day_for_gainers_full(date, coins_per_day=20):
    """Analyze a single day for top gainers and losers - FULL SCAN"""
    try:
        start_date = datetime(date.year, date.month, date.day, 0, 0, 0, tzinfo=IST)
        end_date = datetime(date.year, date.month, date.day, 23, 59, 59, tzinfo=IST)

        all_symbols = get_perpetual_symbols()
        results = []

        for symbol in all_symbols:
            try:
                day_klines = get_historical_klines(symbol, "1d", start_date, end_date, limit=2)
                if day_klines and len(day_klines) >= 1:
                    kline = day_klines[0]
                    open_price = float(kline[1])
                    high_price = float(kline[2])
                    low_price = float(kline[3])
                    close_price = float(kline[4])

                    open_close_pct = ((close_price - open_price) / open_price) * 100
                    low_high_pct = ((high_price - low_price) / low_price) * 100

                    results.append({
                        'symbol': symbol,
                        'open_close_%': open_close_pct,
                        'low_high_%': low_high_pct,
                        'date': date.strftime('%Y-%m-%d')
                    })
            except:
                continue

        if not results:
            return pd.DataFrame(), pd.DataFrame()

        df = pd.DataFrame(results)
        gainers = df[df['open_close_%'] > 0].copy().sort_values('low_high_%', ascending=False).head(coins_per_day)
        losers = df[df['open_close_%'] < 0].copy().sort_values('low_high_%', ascending=False).head(coins_per_day)

        if not gainers.empty:
            gainers = gainers[['date', 'symbol', 'open_close_%', 'low_high_%']]
            gainers.columns = ['Date', 'Coin', 'Open to Close %', 'Low to High %']
            gainers['Rank'] = range(1, len(gainers) + 1)

        if not losers.empty:
            losers = losers[['date', 'symbol', 'open_close_%', 'low_high_%']]
            losers.columns = ['Date', 'Coin', 'Open to Close %', 'Low to High %']
            losers['Rank'] = range(1, len(losers) + 1)

        return gainers, losers

    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()

def generate_monthly_report_full(selected_month, coins_per_day=20, max_workers=5):
    """Generate monthly report for each day - FULL SCAN"""
    try:
        year = selected_month.year
        month = selected_month.month

        if month == 12:
            next_month = datetime(year + 1, 1, 1, tzinfo=IST)
        else:
            next_month = datetime(year, month + 1, 1, tzinfo=IST)

        last_day = next_month - timedelta(days=1)
        days_in_month = last_day.day

        days = [datetime(year, month, day, tzinfo=IST) for day in range(1, days_in_month + 1)]
        today = datetime.now(IST)
        days = [day for day in days if day.date() <= today.date()]

        if not days:
            return {}, {}

        all_gainers = {}
        all_losers = {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_date = {executor.submit(analyze_single_day_for_gainers_full, day, coins_per_day): day for day in days}
            completed = 0
            total = len(future_to_date)

            for future in as_completed(future_to_date):
                date = future_to_date[future]
                date_str = date.strftime('%Y-%m-%d')

                try:
                    gainers_df, losers_df = future.result(timeout=60)
                    if not gainers_df.empty:
                        all_gainers[date_str] = gainers_df
                    if not losers_df.empty:
                        all_losers[date_str] = losers_df

                    completed += 1
                    progress_bar.progress(completed / total)
                    status_text.text(f"✅ Processed {date_str} ({completed}/{total})")
                except Exception:
                    completed += 1
                    progress_bar.progress(completed / total)

        status_text.empty()
        return all_gainers, all_losers

    except Exception as e:
        return {}, {}

def render_monthly_report():
    """Render Tab 7: Monthly Report Generator - FULL SCAN"""
    st.header("📊 Monthly Report Generator")
    st.markdown("**Generate comprehensive daily reports for an entire month - SCANS ALL 500+ COINS**")

    col1, col2, col3 = st.columns(3)

    with col1:
        now_ist = datetime.now(IST)
        available_months = []
        month_descriptions = []

        for i in range(12):
            month_date = now_ist - relativedelta(months=i)
            month_str = month_date.strftime("%B %Y")

            if month_date.month < now_ist.month or month_date.year < now_ist.year:
                available_months.append(month_date)
                month_descriptions.append(f"{month_str} (Complete)")
            elif i == 0:
                available_months.append(month_date)
                month_descriptions.append(f"{month_str} (Current - Up to {now_ist.day})")

        if available_months:
            selected_month_idx = st.selectbox("Select Month:", range(len(available_months)),
                                            format_func=lambda x: month_descriptions[x], index=1,
                                            key="report_month")
            selected_month = available_months[selected_month_idx]
        else:
            st.error("No valid months available.")
            return

    with col2:
        coins_per_day = st.number_input("Coins per day:", 5, 50, 20, 5, key="report_coins")

    with col3:
        max_workers = st.slider("Parallel workers:", 1, 10, 5, key="report_workers")

    if st.button("🚀 Generate Monthly Report (Full Scan)", type="primary", use_container_width=True, key="report_btn"):
        if selected_month > datetime.now(IST):
            st.error("Cannot generate report for future dates.")
            return

        total_coins = len(get_perpetual_symbols())
        with st.spinner(f"Generating report for {selected_month.strftime('%B %Y')} - Scanning ALL {total_coins} coins..."):
            all_gainers, all_losers = generate_monthly_report_full(selected_month, coins_per_day, max_workers)

            if all_gainers or all_losers:
                st.success(f"✅ Report generated! {len(all_gainers)} days with gainers, {len(all_losers)} days with losers")

                gainers_list = []
                for date_str in sorted(all_gainers.keys()):
                    df = all_gainers[date_str].copy()
                    gainers_list.append(df)

                losers_list = []
                for date_str in sorted(all_losers.keys()):
                    df = all_losers[date_str].copy()
                    losers_list.append(df)

                st.subheader("📊 Monthly Summary")
                summary_data = []
                for date_str in sorted(all_gainers.keys()):
                    gainers_df = all_gainers[date_str]
                    losers_df = all_losers.get(date_str, pd.DataFrame())

                    summary_data.append({
                        'Date': date_str,
                        'Gainers': len(gainers_df),
                        'Losers': len(losers_df),
                        'Top Gainer': gainers_df.iloc[0]['Coin'] if not gainers_df.empty else 'N/A',
                        'Top Gainer %': f"{gainers_df.iloc[0]['Open to Close %']:.2f}%" if not gainers_df.empty else 'N/A',
                        'Top Loser': losers_df.iloc[0]['Coin'] if not losers_df.empty else 'N/A',
                        'Top Loser %': f"{losers_df.iloc[0]['Open to Close %']:.2f}%" if not losers_df.empty else 'N/A'
                    })

                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                col1, col2 = st.columns(2)
                with col1:
                    if gainers_list:
                        consolidated_gainers = pd.concat(gainers_list, ignore_index=True)
                        csv = consolidated_gainers.to_csv(index=False)
                        st.download_button(
                            label="📥 Download All Gainers CSV",
                            data=csv,
                            file_name=f"gainers_{selected_month.strftime('%Y%m')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                with col2:
                    if losers_list:
                        consolidated_losers = pd.concat(losers_list, ignore_index=True)
                        csv = consolidated_losers.to_csv(index=False)
                        st.download_button(
                            label="📥 Download All Losers CSV",
                            data=csv,
                            file_name=f"losers_{selected_month.strftime('%Y%m')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                st.subheader("📅 Sample Daily Reports")
                sample_dates = sorted(all_gainers.keys())[:3]
                for date_str in sample_dates:
                    with st.expander(f"📊 {date_str} - Daily Report"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**🏆 Top Gainers**")
                            st.dataframe(all_gainers[date_str][['Rank', 'Coin', 'Open to Close %', 'Low to High %']],
                                       use_container_width=True, hide_index=True)
                        with col2:
                            if date_str in all_losers:
                                st.markdown("**📉 Top Losers**")
                                st.dataframe(all_losers[date_str][['Rank', 'Coin', 'Open to Close %', 'Low to High %']],
                                           use_container_width=True, hide_index=True)
            else:
                st.error("No data available for selected month.")

def create_vmc_trend_chart(df, symbol):
    """Price vs V/MC, plus Newton force when the columns exist."""
    if df.empty:
        return None

    has_force = "Force" in df.columns and df["Force"].notna().any()
    if has_force:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.40, 0.30, 0.30],
            subplot_titles=(
                f"{symbol} — Price",
                f"{symbol} — Momentum (Volume / Market Cap %)",
                f"{symbol} — Newton Force (A × Volume/MCap)",
            ),
        )
    else:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.55, 0.45],
            subplot_titles=(
                f"{symbol} — Daily Price",
                f"{symbol} — Momentum (Volume / Market Cap %)"
            )
        )

    # Price line (top)
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Price"],
            mode="lines+markers",
            name="Price (USD)",
            line=dict(color="#00d4ff", width=2),
            marker=dict(size=5),
            hovertemplate="Date: %{x}<br>Price: $%{y:,.4f}<extra></extra>"
        ),
        row=1, col=1
    )

    # V/MC bar colours: >50% vivid green, >25% gold, else blue
    bar_colors = []
    for v in df["V/MC"]:
        if v >= 50:
            bar_colors.append("#00ff9d")
        elif v >= 25:
            bar_colors.append("#ffd700")
        else:
            bar_colors.append("#1a73e8")

    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["V/MC"],
            name="Momentum %",
            marker_color=bar_colors,
            text=df["V/MC"].apply(lambda x: f"{x:.2f}%"),
            textposition="outside",
            hovertemplate="Date: %{x}<br>Momentum: %{text}<extra></extra>"
        ),
        row=2, col=1
    )

    if has_force:
        force_colors = ["#00ff9d" if (pd.notna(f) and f > 0) else "#ff6b6b" for f in df["Force"]]
        fig.add_trace(
            go.Bar(
                x=df["Date"],
                y=df["Force"],
                name="Newton Force",
                marker_color=force_colors,
                hovertemplate="Date: %{x}<br>Force: %{y:.6f}<extra></extra>",
            ),
            row=3, col=1,
        )

    fig.update_layout(
        height=780 if has_force else 600,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=60, r=40),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="V/MC %", row=2, col=1)
    if has_force:
        fig.update_yaxes(title_text="Force", row=3, col=1)
        fig.update_xaxes(tickangle=-45, row=3, col=1)
    else:
        fig.update_xaxes(tickangle=-45, row=2, col=1)

    return fig


def render_vmc_analysis():
    st.header("📊 Volume / Market Cap Activity")
    st.markdown("""
    Multi-coin scan uses **Binance USDT volume** and **Market Cap = Binance close × circulating supply**.
    Momentum is heat, not a buy. Confirmation = Vol/MCap + volume expansion + rising market cap + close holding.
    """)
    mode = st.radio(
        "Mode",
        ["Multi-coin activity scan", "Single-coin history", "Relative Volume (RVOL)"],
        horizontal=True,
        key="vmc_mode_radio",
    )
    if mode == "Multi-coin activity scan":
        render_vmc_multi_scan()
    elif mode == "Relative Volume (RVOL)":
        render_rvol_scan()
    else:
        render_vmc_history()


def render_rvol_scan():
    st.markdown("""
    **Relative Volume (RVOL)** = Volume on your chosen date ÷ average volume of the prior **N** daily candles.

    Uses **Binance futures USDT volume** (daily UTC candles, same as the V/MC history table).
    - **RVOL > 2** — unusually high activity
    - **RVOL 1–2** — above normal
    - **RVOL < 1** — below average
    """)

    all_symbols = get_perpetual_symbols()
    if not all_symbols:
        st.error("Failed to load symbols from Binance.")
        return

    scan_mode = st.radio(
        "RVOL scan type",
        ["Single coin", "Multi-coin scan"],
        horizontal=True,
        key="rvol_scan_type",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        target_date = st.date_input(
            "📅 Volume date (UTC daily candle)",
            value=datetime.now(UTC).date(),
            key="rvol_target_date",
            help="The day whose volume is the numerator (today volume). Matches UTC dates in V/MC daily table.",
        )
    with c2:
        avg_days = st.number_input(
            "📊 Average volume — prior N days",
            min_value=1,
            max_value=365,
            value=20,
            step=1,
            key="rvol_avg_days",
            help="Average of the N completed daily volumes BEFORE the volume date.",
        )
    with c3:
        min_rvol = st.number_input(
            "Min RVOL (multi-coin only)",
            min_value=0.0,
            max_value=20.0,
            value=1.5,
            step=0.1,
            key="rvol_min_filter",
            disabled=(scan_mode == "Single coin"),
        )

    if scan_mode == "Single coin":
        coin_options = [f"{s}.P" for s in all_symbols]
        selected = st.selectbox(
            "🪙 Coin",
            coin_options,
            index=coin_options.index("BTCUSDT.P") if "BTCUSDT.P" in coin_options else 0,
            key="rvol_single_coin",
        )
        if st.button("Calculate RVOL", type="primary", use_container_width=True, key="rvol_single_btn"):
            symbol = selected.replace(".P", "")
            with st.spinner(f"Calculating RVOL for {selected} on {target_date}..."):
                row = compute_rvol(symbol, target_date, int(avg_days))
            if not row:
                st.warning(
                    f"No daily volume found for **{selected}** on **{target_date}** (UTC). "
                    "Pick an earlier date or check the symbol."
                )
                return

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Target Volume", format_large_number(row["Target Volume"]))
            with m2:
                st.metric(
                    f"Avg Volume ({row['Avg Days Used']}d)",
                    format_large_number(row["Avg Volume"]),
                )
            with m3:
                st.metric("RVOL", f"{row['RVOL']:.2f}x")
            with m4:
                st.metric("Signal", row["Signal"])

            st.success(
                f"**{selected}** on **{row['Date']}**: "
                f"RVOL = **{row['RVOL']:.2f}x** "
                f"({format_large_number(row['Target Volume'])} ÷ "
                f"{format_large_number(row['Avg Volume'])} over prior "
                f"{row['Avg Days Used']} days)"
            )
            if row["Avg Days Used"] < row["Avg Days Requested"]:
                st.info(
                    f"Only **{row['Avg Days Used']}** prior days available "
                    f"(requested {row['Avg Days Requested']})."
                )
        return

    # Multi-coin scan
    c4, c5 = st.columns(2)
    with c4:
        scan_n = st.selectbox("Universe (by Binance volume)", [50, 100, 200, 300, 500], index=2, key="rvol_scan_n")
    with c5:
        sort_by = st.selectbox("Sort by", ["RVOL", "Target Volume"], index=0, key="rvol_sort_by")

    if st.button("Scan RVOL", type="primary", use_container_width=True, key="rvol_multi_btn"):
        top_volume = get_top_volume_symbols_list(limit=500)
        universe = [s for s in top_volume if s in set(all_symbols)] or all_symbols
        symbols = universe[: int(scan_n)]

        with st.spinner(f"Scanning RVOL for {len(symbols)} coins ({target_date}, avg {avg_days}d)..."):
            rows = scan_rvol_parallel(symbols, target_date, int(avg_days))

        if not rows:
            st.warning("No RVOL data returned. Try a different date.")
            return

        df = pd.DataFrame(rows)
        df = df[df["RVOL"] >= float(min_rvol)]
        sort_col = "RVOL" if sort_by == "RVOL" else "Target Volume"
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

        if df.empty:
            st.info(f"No coins with RVOL ≥ {min_rvol:.1f}x on {target_date}.")
            return

        high = int((df["RVOL"] >= 2).sum())
        very_high = int((df["RVOL"] >= 3).sum())
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Coins listed", len(df))
        with m2:
            st.metric("RVOL ≥ 2x", high)
        with m3:
            st.metric("RVOL ≥ 3x", very_high)

        show = pd.DataFrame()
        show["Coin"] = df["Coin"]
        show["Date"] = df["Date"]
        show["Price"] = df["Price"].map(
            lambda x: f"${x:,.6f}" if pd.notna(x) and x < 1 else (f"${x:,.4f}" if pd.notna(x) else "N/A")
        )
        show["Target Volume"] = df["Target Volume"].map(format_large_number)
        show["Avg Volume"] = df["Avg Volume"].map(format_large_number)
        show["Avg Days"] = df["Avg Days Used"]
        show["RVOL"] = df["RVOL"].map(lambda x: f"{x:.2f}x")
        show["Signal"] = df["Signal"]

        def color_rvol_signal(val):
            if val == "Very High":
                return "color: #00ff9d; font-weight: bold"
            if val == "High":
                return "color: #ffd700; font-weight: bold"
            if val == "Elevated":
                return "color: #1a73e8; font-weight: bold"
            if val == "Low":
                return "color: #888888"
            return ""

        styled = show.style.map(color_rvol_signal, subset=["Signal"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.download_button(
            "Download RVOL scan (CSV)",
            df.to_csv(index=False),
            file_name=f"rvol_{target_date}_{datetime.now().strftime('%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="rvol_multi_csv",
        )


def render_vmc_multi_scan():
    all_symbols = get_perpetual_symbols()
    if not all_symbols:
        st.error("Failed to load symbols from Binance.")
        return

    top_volume = get_top_volume_symbols_list(limit=500)
    universe = [s for s in top_volume if s in set(all_symbols)] or all_symbols

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        scan_n = st.selectbox("Universe (by Binance volume)", [50, 100, 200, 300, 500], index=2, key="vmc_scan_n")
    with c2:
        max_mc = st.selectbox(
            "Max market cap (low-cap filter)",
            ["All", "50M", "100M", "250M", "500M"],
            index=3,
            key="vmc_max_mc",
        )
    with c3:
        min_score = st.slider("Min activity score", 0, 12, 4, key="vmc_min_score")
    with c4:
        use_live = st.checkbox("Include today's live candle", value=True, key="vmc_use_live")

    regime_filter = st.multiselect(
        "Show regimes",
        [
            "Activity expansion",
            "Climax — do not chase",
            "Dump / shakeout",
            "Distribution (dying volume)",
            "Hot — needs confirmation",
            "Watch",
            "Quiet",
        ],
        default=["Activity expansion", "Climax — do not chase", "Dump / shakeout", "Hot — needs confirmation"],
        key="vmc_regime_filter",
    )

    st.caption(
        "Activity expansion = Vol/MCap > 100% AND volume +100% AND market cap +10% AND price up. "
        "Climax = same but daily price +50% or more — watch, do not chase the close. "
        "Newton Force = acceleration × (volume / market cap); score is −3 to +3."
    )

    if st.button("Scan activity", type="primary", use_container_width=True, key="vmc_multi_scan_btn"):
        gecko = get_coingecko_markets_by_symbol(pages=4)
        if not gecko:
            st.error("Could not load circulating supply from CoinGecko. Wait a minute and retry (rate limit).")
            return

        pairs = []
        for symbol in universe[: int(scan_n)]:
            base = symbol[:-4] if symbol.endswith("USDT") else symbol
            info = gecko.get(base)
            if not info or not info.get("circ"):
                continue
            pairs.append((symbol, float(info["circ"])))

        if not pairs:
            st.warning("No overlapping Binance perps with CoinGecko circulating supply.")
            return

        with st.spinner(f"Scanning {len(pairs)} coins..."):
            rows = scan_vmc_activity_parallel(pairs, use_live)

        if not rows:
            st.warning("No rows returned.")
            return

        df = pd.DataFrame(rows)
        max_mc_map = {"50M": 50e6, "100M": 100e6, "250M": 250e6, "500M": 500e6}
        if max_mc != "All":
            df = df[df["Market Cap"] <= max_mc_map[max_mc]]
        df = df[df["Score"] >= min_score]
        if regime_filter:
            df = df[df["Regime"].isin(regime_filter)]
        df = df.sort_values(["Score", "Vol/MCap %"], ascending=False).reset_index(drop=True)

        n_exp = int((df["Expansion"] == "Yes").sum()) if not df.empty else 0
        n_climax = int((df["Regime"] == "Climax — do not chase").sum()) if not df.empty else 0
        n_dump = int((df["Regime"] == "Dump / shakeout").sum()) if not df.empty else 0

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Listed", len(df))
        with m2:
            st.metric("Activity expansion", n_exp)
        with m3:
            st.metric("Climax (do not chase)", n_climax)
        with m4:
            st.metric("Dump / shakeout", n_dump)

        if df.empty:
            st.info("No coins passed the filters. Lower min score or widen regimes.")
            return

        show = pd.DataFrame()
        show["Coin"] = df["Coin"]
        show["Price"] = df["Price"].map(lambda x: f"${x:.6f}" if x < 1 else f"${x:.4f}")
        show["Price %"] = df["Price %"].map(lambda x: f"{x:+.2f}%")
        show["Coin Volume"] = df["Coin Volume"].map(format_large_number)
        show["Vol %"] = df["Vol %"].map(lambda x: f"{x:+.1f}%")
        show["Market Cap"] = df["Market Cap"].map(format_large_number)
        show["MC %"] = df["MC %"].map(lambda x: f"{x:+.2f}%")
        show["Vol/MCap %"] = df["Vol/MCap %"].map(lambda x: f"{x:.1f}%")
        show["Score"] = df["Score"]
        show["Band"] = df["Band"]
        show["Regime"] = df["Regime"]
        show["Newton Score"] = df["Newton Score"].map(
            lambda x: newton_score_label(int(x)) if pd.notna(x) else "—"
        )
        show["Newton Regime"] = df["Newton Regime"]
        show["Close > Prior High"] = df["Close > Prior High"]

        def color_regime(val):
            if val == "Activity expansion":
                return "color: #00ff9d; font-weight: bold"
            if val == "Climax — do not chase":
                return "color: #ff6b6b; font-weight: bold"
            if val == "Dump / shakeout":
                return "color: #ffd700; font-weight: bold"
            if val == "Distribution (dying volume)":
                return "color: #ff8c42; font-weight: bold"
            return ""

        def color_newton_regime(val):
            if val == "Strong bullish force":
                return "color: #00ff9d; font-weight: bold"
            if val == "Strong bearish force":
                return "color: #ff6b6b; font-weight: bold"
            if val == "Bullish but losing force" or val == "Selling pressure weakening":
                return "color: #ffd700; font-weight: bold"
            if val == "Potentially weak rally" or val == "Weak bearish move":
                return "color: #ff8c42; font-weight: bold"
            return ""

        styled = show.style.map(color_regime, subset=["Regime"]).map(
            color_newton_regime, subset=["Newton Regime"]
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.download_button(
            "Download activity scan (CSV)",
            df.to_csv(index=False),
            file_name=f"vmc_activity_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="vmc_multi_csv",
        )


def render_vmc_history():
    st.markdown("""
    **Single-coin history:** Choose **Binance**, **CoinGecko**, or **CMC** data source.
    **Binance mode:** Volume + Price from Binance futures; Market Cap from CoinGecko.
    **CoinGecko mode:** Volume + Market Cap from CoinGecko (API timestamps are **UTC**).
    **CMC mode:** Volume, Market Cap, Momentum from CoinMarketCap.
    Momentum = Volume / Market Cap × 100.
    **Newton candle (analogy):** Velocity = close return, Acceleration = change in velocity,
    Force = Acceleration × (Volume / Market Cap). Needs at least 3 candles. Score is −3 to +3.
    **UTC day** = 00:00 UTC → next 00:00 UTC. **IST day** = 05:30 AM IST → next 05:30 AM IST (same instant as UTC midnight).
    """)

    # Check if symbols are loaded
    all_symbols = get_perpetual_symbols()
    if not all_symbols:
        st.error("Failed to load symbols from Binance.")
        return

    # Format options with .P suffix to match the requested Coinname format (e.g. RENZUSDT.P)
    coin_options = [f"{s}.P" for s in all_symbols]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        selected_coin = st.selectbox(
            "🪙 Select Coin (Coinname):",
            options=coin_options,
            index=coin_options.index("BTCUSDT.P") if "BTCUSDT.P" in coin_options else 0,
            key="vmc_coin_select"
        )
    with col2:
        vmc_data_source = st.selectbox(
            "Data Source",
            ["Binance", "CoinGecko", "CMC"],
            index=0,
            key="vmc_data_source",
            help=(
                "Binance = futures volume + CoinGecko market cap. "
                "CoinGecko = volume + market cap from CoinGecko. "
                "CMC = all fields from CoinMarketCap API."
            ),
        )
    with col3:
        vmc_hist_timeframe = st.selectbox(
            "Time Frame",
            ["Daily", "1H", "4H"],
            index=0,
            key="vmc_hist_timeframe",
            help="UTC uses 00:00 UTC candles. IST uses 05:30 AM IST (same instants as UTC).",
        )
    with col4:
        tz_choice = st.radio(
            "Timezone",
            options=["UTC Time", "IST Time"],
            index=1,
            horizontal=True,
            key="vmc_timezone",
            help="CoinGecko is UTC-only. IST Time labels the same timestamps as 05:30 Asia/Kolkata.",
        )
        vmc_timezone = "UTC" if tz_choice == "UTC Time" else "IST"
        tz_cfg = vmc_tz_config(vmc_timezone)
        tz_label = tz_cfg["label"]
    with col5:
        range_mode = st.radio(
            "Period",
            options=["Today", "Last N days", "Date range"],
            horizontal=True,
            key="vmc_range_mode"
        )

    no_of_days = 10
    now_local = datetime.now(tz_cfg["zone"])
    start_date = now_local.date() - timedelta(days=90)
    end_date = now_local.date()
    if range_mode == "Today":
        crypto_start = get_crypto_day_start(timezone_name=vmc_timezone)
        if vmc_timezone == "UTC":
            start_txt = crypto_start.strftime("%d-%b-%Y %H:%M UTC")
        else:
            start_txt = crypto_start.strftime("%d-%b-%Y %I:%M %p IST")
        st.info(f"📅 **Today** = current crypto day from **{start_txt}** up to now.")
    elif range_mode == "Last N days":
        max_days = {"Daily": 3650, "1H": 30, "4H": 90}.get(vmc_hist_timeframe, 3650)
        default_days = {"Daily": 10, "1H": 3, "4H": 7}.get(vmc_hist_timeframe, 10)
        day_span = (
            "00:00 UTC → 00:00 UTC"
            if vmc_timezone == "UTC"
            else "05:30 AM IST → 05:30 AM IST"
        )
        no_of_days = st.number_input(
            f"📅 Number of crypto days ({day_span}):",
            min_value=1,
            max_value=max_days,
            value=min(default_days, max_days),
            step=1,
            key="vmc_days_input",
            help=f"Max {max_days} days for {vmc_hist_timeframe} (API limits).",
        )
    else:
        d1, d2 = st.columns(2)
        with d1:
            start_date = st.date_input(f"From ({tz_label})", value=start_date, key="vmc_start_date")
        with d2:
            end_date = st.date_input(f"To ({tz_label})", value=end_date, key="vmc_end_date")

    # Add a checkbox to show raw vs formatted values
    show_raw = st.checkbox("Show Raw Numbers (e.g. full digits for Volume and Market Cap)", value=False, key="vmc_show_raw")

    # Custom CoinGecko / CMC overrides
    with st.expander("⚙️ Advanced Settings"):
        auto_id = get_coingecko_id(selected_coin)
        custom_id = st.text_input(
            "CoinGecko API ID (Binance / CoinGecko mode)",
            value=auto_id,
            help="Used in Binance and CoinGecko modes. Example: 'bitcoin' for BTC.",
            disabled=(vmc_data_source == "CMC"),
        )
        cmc_symbol_default = symbol_to_cmc_base(selected_coin)
        custom_cmc_symbol = st.text_input(
            "CMC Symbol",
            value=cmc_symbol_default,
            help="CoinMarketCap symbol (e.g. BTC). Used in CMC mode.",
            disabled=(vmc_data_source != "CMC"),
        )
        saved_cmc_key = get_cmc_api_key() or ""
        cmc_api_key_input = st.text_input(
            "CoinMarketCap API Key",
            value=saved_cmc_key,
            type="password",
            help="Required for CMC mode. You can also set CMC_API_KEY env var or Streamlit secrets.",
            disabled=(vmc_data_source != "CMC"),
        )
        if cmc_api_key_input:
            st.session_state["cmc_api_key"] = cmc_api_key_input.strip()

    if st.button("📊 Fetch V/MC Data", type="primary", use_container_width=True, key="vmc_fetch_btn"):
        coin_to_fetch = custom_id if custom_id else auto_id
        cmc_symbol = (custom_cmc_symbol or cmc_symbol_default).upper().strip()
        cmc_api_key = get_cmc_api_key() if vmc_data_source == "CMC" else None
        if vmc_data_source == "CMC" and not cmc_api_key:
            st.error("Please enter your CoinMarketCap API key in Advanced Settings.")
            return

        if range_mode == "Date range" and start_date > end_date:
            st.error("Start date must be on or before end date.")
            return

        range_start, range_end = get_crypto_range_bounds(
            range_mode, start_date, end_date, no_of_days, timezone_name=vmc_timezone
        )
        start_ts = int(range_start.timestamp())
        end_ts = int(range_end.timestamp())
        tf_label = vmc_hist_timeframe
        source_label = vmc_data_source
        if vmc_timezone == "UTC":
            range_txt = (
                f"{range_start.strftime('%d-%b-%Y %H:%M')} → "
                f"{range_end.strftime('%d-%b-%Y %H:%M')} UTC"
            )
        else:
            range_txt = (
                f"{range_start.strftime('%d-%b-%Y %H:%M')} → "
                f"{range_end.strftime('%d-%b-%Y %H:%M')} IST"
            )
        spinner_msg = (
            f"Fetching {source_label} {tf_label} data for {selected_coin} "
            f"({range_txt})..."
        )

        with st.spinner(spinner_msg):
            details = {}
            try:
                if vmc_data_source == "CMC":
                    rows = build_cmc_vmc_rows(
                        selected_coin, start_ts, end_ts, vmc_hist_timeframe, cmc_api_key,
                        timezone_name=vmc_timezone,
                    )
                    details = get_cmc_coin_details(cmc_symbol, cmc_api_key)
                elif vmc_data_source == "CoinGecko":
                    details = get_coingecko_coin_details(coin_to_fetch)
                    rows = build_coingecko_vmc_rows(
                        selected_coin, coin_to_fetch, details, start_ts, end_ts, vmc_hist_timeframe,
                        timezone_name=vmc_timezone,
                    )
                elif vmc_hist_timeframe == "Daily":
                    cg_rows = get_coingecko_historical_data(coin_to_fetch, start_ts=start_ts, end_ts=end_ts)
                    details = get_coingecko_coin_details(coin_to_fetch)
                    rows = build_binance_vmc_rows(
                        selected_coin, cg_rows, details, start_ts, end_ts,
                        timezone_name=vmc_timezone,
                    )
                else:
                    cg_points = get_coingecko_historical_points(
                        coin_to_fetch, start_ts=start_ts, end_ts=end_ts
                    )
                    details = get_coingecko_coin_details(coin_to_fetch)
                    rows = build_binance_vmc_rows_interval(
                        selected_coin, cg_points, details, start_ts, end_ts, vmc_hist_timeframe,
                        timezone_name=vmc_timezone,
                    )
            except RuntimeError as exc:
                st.error(str(exc))
                return

            if not rows:
                hint = ""
                if vmc_data_source == "CMC":
                    hint = (
                        " CMC may not list this coin or your plan may not include historical data — "
                        "try **Binance** or **CoinGecko** mode with the correct CoinGecko ID."
                    )
                elif vmc_data_source == "CoinGecko":
                    hint = (
                        " Check the CoinGecko API ID in Advanced Settings "
                        "(e.g. 'bitcoin' for BTC)."
                    )
                st.warning(
                    f"No {source_label} {tf_label} data found for {selected_coin}."
                    f"{hint} Check the symbol/API key, or try a shorter date range."
                )
                return

            # Convert to DataFrame
            df = pd.DataFrame(rows)

            if range_mode == "Last N days" and vmc_hist_timeframe == "Daily":
                df = df.tail(no_of_days).reset_index(drop=True)

            df = attach_newton_to_df(df)

            # Add Coinname column
            df["Coinname"] = selected_coin

            # ── Supply info from details ──
            total_supply = details.get("total_supply")
            max_supply = details.get("max_supply")
            supply_denominator = max_supply if max_supply else total_supply
            if vmc_data_source == "CMC" and details.get("circulating_supply"):
                static_circ = details.get("circulating_supply")
                if supply_denominator and supply_denominator > 0:
                    df["Circ Supply %"] = (static_circ / supply_denominator * 100)
                else:
                    df["Circ Supply %"] = 0

            # Calculate FDV per row: Price × Max Supply (fallback to Total Supply)
            if supply_denominator and supply_denominator > 0:
                df["FDV"] = df["Price"] * supply_denominator
                if "Circ Supply %" not in df.columns:
                    df["Circ Supply %"] = (df["Circulating Supply"] / supply_denominator * 100)
            else:
                df["FDV"] = 0
                if "Circ Supply %" not in df.columns:
                    df["Circ Supply %"] = 0

            # ── Metric cards ──
            if vmc_data_source == "CMC":
                source_note = (
                    "Market Cap from CoinMarketCap quotes/historical. "
                    "Volume uses CMC 24h volume at each snapshot. "
                    "Momentum = Volume / Market Cap × 100"
                )
            elif vmc_data_source == "CoinGecko":
                source_note = (
                    "Volume and Market Cap from CoinGecko. "
                    "Momentum = Volume / Market Cap × 100"
                )
            else:
                source_note = (
                    "Volume + Price from Binance futures; Market Cap from CoinGecko "
                    "(Binance has no market cap API). "
                    "Momentum = Volume / Market Cap × 100"
                )
            if vmc_timezone == "UTC":
                day_note = "00:00 UTC"
            else:
                day_note = "05:30 AM IST"
            period_note = f"today's crypto day ({day_note})" if range_mode == "Today" else (
                f"hourly periods from {day_note}" if vmc_hist_timeframe == "1H" else (
                    f"4H blocks from {day_note}" if vmc_hist_timeframe == "4H" else f"daily candles ({day_note})"
                )
            )
            st.success(
                f"✅ {selected_coin} [{source_label}]: {tf_label} — {period_note}. {source_note}"
            )

            avg_vmc = df["V/MC"].mean()
            peak_vmc = df["V/MC"].max()
            peak_vmc_date = df.loc[df["V/MC"].idxmax(), "Date"] if not df.empty else "N/A"
            latest_score = df["Newton Score"].dropna().iloc[-1] if df["Newton Score"].notna().any() else None
            latest_regime = (
                df["Newton Regime"].dropna().iloc[-1] if df["Newton Regime"].notna().any() else "—"
            )

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            with mc1:
                st.metric("Avg Momentum", f"{avg_vmc:.2f}%")
            with mc2:
                st.metric("Peak Momentum", f"{peak_vmc:.2f}%", help=f"On {peak_vmc_date}")
            with mc3:
                ts_label = format_large_number(total_supply) if total_supply else "N/A"
                st.metric("Total Supply", ts_label)
            with mc4:
                ms_label = format_large_number(max_supply) if max_supply else "N/A"
                st.metric("Max Supply", ms_label)
            with mc5:
                st.metric(
                    "Newton Score (last)",
                    newton_score_label(int(latest_score)) if latest_score is not None else "—",
                    help=latest_regime,
                )

            # ── Trend chart ──
            st.subheader("📈 Price vs V/MC vs Newton Force")
            fig = create_vmc_trend_chart(df, selected_coin)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # ── Build display table (CMC column order) ──
            st.subheader("📋 Detailed Data")
            display_df = pd.DataFrame()
            date_col = f"Period ({tz_label})" if vmc_hist_timeframe in ("1H", "4H") else "Date"
            display_df[date_col] = df["Date"]

            if show_raw:
                display_df["Price"] = df["Price"].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
                display_df["Coin Volume"] = df["Volume"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "N/A")
                display_df["Market Cap"] = df["Market Cap"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "N/A")
                display_df["Momentum"] = df["V/MC"].apply(lambda x: f"{x:.8f}" if pd.notna(x) else "N/A")
                display_df["Velocity"] = df["Velocity"].apply(lambda x: f"{x:.8f}" if pd.notna(x) else "N/A")
                display_df["Acceleration"] = df["Acceleration"].apply(lambda x: f"{x:.8f}" if pd.notna(x) else "N/A")
                display_df["Force"] = df["Force"].apply(lambda x: f"{x:.8f}" if pd.notna(x) else "N/A")
                display_df["Newton Score"] = df["Newton Score"].apply(
                    lambda x: newton_score_label(int(x)) if pd.notna(x) else "N/A"
                )
                display_df["Newton Regime"] = df["Newton Regime"].fillna("N/A")
                display_df["FDV"] = df["FDV"].apply(lambda x: f"{int(x)}" if pd.notna(x) and x > 0 else "N/A")
                display_df["Open Interest"] = df["Open Interest"].apply(lambda x: f"{int(x)}" if pd.notna(x) and x > 0 else "N/A")
                display_df["MOI"] = df["MOI"].apply(lambda x: f"{x:.8f}" if pd.notna(x) else "N/A")
                display_df["VOI"] = df["VOI"].apply(lambda x: f"{x:.8f}" if pd.notna(x) else "N/A")
            else:
                display_df["Price"] = df["Price"].apply(lambda x: f"${x:,.4f}" if pd.notna(x) else "N/A")
                display_df["Coin Volume"] = df["Volume"].apply(format_large_number)
                display_df["Market Cap"] = df["Market Cap"].apply(format_large_number)
                display_df["Momentum"] = df["V/MC"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                display_df["Velocity"] = df["Velocity"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
                display_df["Acceleration"] = df["Acceleration"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")
                display_df["Force"] = df["Force"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                display_df["Newton Score"] = df["Newton Score"].apply(
                    lambda x: newton_score_label(int(x)) if pd.notna(x) else "N/A"
                )
                display_df["Newton Regime"] = df["Newton Regime"].fillna("N/A")
                display_df["FDV"] = df["FDV"].apply(lambda x: format_large_number(x) if pd.notna(x) and x > 0 else "N/A")
                display_df["Open Interest"] = df["Open Interest"].apply(lambda x: format_large_number(x) if pd.notna(x) and x > 0 else "N/A")
                display_df["MOI"] = df["MOI"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                display_df["VOI"] = df["VOI"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")

            # ── Color-code V/MC column ──
            def color_vmc(val):
                if not isinstance(val, str):
                    return ''
                try:
                    num = float(val.replace('%', ''))
                    if num >= 50:
                        return 'color: #00ff9d; font-weight: bold'
                    elif num >= 25:
                        return 'color: #ffd700; font-weight: bold'
                except Exception:
                    pass
                return ''

            def color_newton_regime_hist(val):
                if val == "Strong bullish force":
                    return "color: #00ff9d; font-weight: bold"
                if val == "Strong bearish force":
                    return "color: #ff6b6b; font-weight: bold"
                if val in ("Bullish but losing force", "Selling pressure weakening"):
                    return "color: #ffd700; font-weight: bold"
                if val in ("Potentially weak rally", "Weak bearish move"):
                    return "color: #ff8c42; font-weight: bold"
                return ""

            styled = display_df.style.map(color_vmc, subset=["Momentum"]).map(
                color_newton_regime_hist, subset=["Newton Regime"]
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            oi_ok = int((df["Open Interest"] > 0).sum()) if "Open Interest" in df.columns else 0
            if oi_ok == 0:
                st.warning(
                    "Open Interest came back empty. Binance only stores ~30 days of OI history, "
                    "and some new/low-liquidity perps have no OI stats. Try a range inside the last 30 days."
                )
            elif oi_ok < len(df):
                st.info(
                    f"Open Interest filled for **{oi_ok}/{len(df)}** rows. "
                    "Binance OI statistics only cover the latest ~30 days, so older dates show N/A."
                )

            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download V/MC Data (CSV)",
                data=csv,
                file_name=f"vmc_{selected_coin}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="vmc_download_btn"
            )

# ==============================================
# MAIN APP
# ==============================================

# ==============================================
# data.binance.vision STATIC FALLBACK
# REST calls now retry/backoff in api_get() and fall back to
# data.binance.vision static ZIP files (no IP rate limits) when Binance
# rate-limits/bans this server. See _maybe_vision_klines() and
# _vision_klines_fallback() below.
# ==============================================
import csv as _csv
import zipfile as _zipfile

VISION_BASE = "https://data.binance.vision/data/futures/um"


def _vision_klines_fallback(symbol, interval, start_ms=None, end_ms=None, limit=500):
    """Download USD-M futures klines from data.binance.vision static ZIP files.

    Returns rows in the exact same 12-element list format as /fapi/v1/klines.
    Static file host => no IP rate limits / bans. Uses daily ZIPs for small
    intervals and monthly ZIPs otherwise, filtered to [start_ms, end_ms]."""
    if not symbol or not interval:
        return []
    symbol = str(symbol).upper().replace(".P", "")
    iv = "1mo" if interval == "1M" else interval
    try:
        start_ms = int(start_ms) if start_ms else None
        end_ms = int(end_ms) if end_ms else None
    except (TypeError, ValueError):
        start_ms = end_ms = None
    try:
        if start_ms and end_ms:
            s_dt = datetime.fromtimestamp(start_ms / 1000, tz=UTC)
            e_dt = datetime.fromtimestamp(end_ms / 1000, tz=UTC)
        else:
            now = datetime.now(UTC)
            s_dt, e_dt = now - timedelta(days=40), now
    except Exception:
        now = datetime.now(UTC)
        s_dt, e_dt = now - timedelta(days=40), now

    urls = []
    if iv in ("1s", "1m", "3m", "5m"):
        d = s_dt.date()
        while d <= e_dt.date() and len(urls) < 45:
            urls.append(f"{VISION_BASE}/daily/klines/{symbol}/{iv}/{symbol}-{iv}-{d.isoformat()}.zip")
            d += timedelta(days=1)
        urls.append(f"{VISION_BASE}/monthly/klines/{symbol}/{iv}/{symbol}-{iv}-{e_dt.year}-{e_dt.month:02d}.zip")
    else:
        y, m = s_dt.year, s_dt.month
        while (y, m) <= (e_dt.year, e_dt.month) and len(urls) < 40:
            urls.append(f"{VISION_BASE}/monthly/klines/{symbol}/{iv}/{symbol}-{iv}-{y}-{m:02d}.zip")
            m += 1
            if m > 12:
                y, m = y + 1, 1

    seen = {}
    for u in urls:
        try:
            r = api_get(u, timeout=60)
            if getattr(r, "status_code", 200) == 404:
                continue
            r.raise_for_status()
            zf = _zipfile.ZipFile(io.BytesIO(r.content))
            csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            text = zf.read(csv_name).decode("utf-8", errors="replace")
            for row in _csv.reader(text.splitlines()):
                if len(row) < 12 or not row[0].strip().isdigit():
                    continue
                t = int(row[0])
                if start_ms and t < start_ms:
                    continue
                if end_ms and t > end_ms:
                    continue
                seen[t] = [t, row[1], row[2], row[3], row[4], row[5],
                           int(row[6]), row[7], int(row[8]), row[9], row[10], row[11]]
        except Exception:
            continue
    out = [seen[k] for k in sorted(seen)]
    if limit:
        out = out[-int(limit):]
    return out


def main():
    st.title("🚀 Binance Scanner Pro - Breakout + Volume Analysis")
    st.markdown("---")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "🚀 Breakout Scanner",
        "🎯 TF Breakout",
        "💎 Smart Scanner",
        "🎯 Individual Coin",
        "📈 Volume History",
        "🏆 Top Gainers",
        "📅 Day Patterns",
        "📊 Monthly Report",
        "📊 V/MC Activity",
        "🕯️ Candle Pattern",
    ])

    with tab1:
        render_breakout_scanner()          # Original

    with tab2:                             # NEW TAB
        render_timeframe_breakout_scanner()

    with tab3:
        render_volume_scanner()

    with tab4:
        render_individual_checker()

    with tab5:
        render_volume_history()

    with tab6:
        render_top_gainers()

    with tab7:
        render_day_patterns()

    with tab8:
        render_monthly_report()

    with tab9:
        render_vmc_analysis()

    with tab10:
        render_candle_pattern_scanner()

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Binance Scanner Pro - Enhanced Edition**

        🚀 **New Feature Added:**
        - ✅ **Average Move Scanner** (under Smart Scanner) - Historical IST 4HR candle movements, Low to High % (bull), High to Low % (bear), Positive vs Negative distribution & Max move
        - ✅ **4HR Candle Move Scanner** (under Smart Scanner) - 6 IST sessions move %, volume spikes (🔥), and potential ranking
        - ✅ **Timeframe Breakout Scanner** - Compare Past OPEN vs Current CLOSE
        - ✅ **Bullish Signal**: Past Bearish Candle + Current Close > Past Open
        - ✅ **Bearish Signal**: Past Bullish Candle + Current Close < Past Open
        - ✅ **Multiple Timeframe Combinations**: Monthly/Daily, Weekly/4H, etc.
        - ✅ **Only Most Recent Candle**: Shows ONLY breakouts from the latest candle

        **📊 Timeframe-Based OI Calculation:**
        - **Monthly**: Current month vs last month
        - **Weekly**: This week vs last week
        - **Daily**: Today vs yesterday
        - **4H**: Current 4h vs previous 4h

        **📈 OI Change Interpretation:**
        - **>20%**: Strongly Increasing (🚀)
        - **10-20%**: Increasing (📈)
        - **5-10%**: Slightly Increasing (↗️)
        - **2-5%**: Moderate Increase (➡️)
        - **0.5-2%**: Minor Increase (↗️)
        - **-0.5 to 0.5%**: Neutral (➡️)
        - **-2 to -0.5%**: Minor Decrease (↘️)
        - **-5 to -2%**: Moderate Decrease (➡️)
        - **-10 to -5%**: Slightly Decreasing (↘️)
        - **-20 to -10%**: Decreasing (📉)
        - **<-20%**: Strongly Decreasing (💥)

        **Performance Tips:**
        - Full scans take 1-2 minutes for 500+ coins
        - Use priority scan for faster results
        - First scan may be slower due to caching
        """)

        # Show stats
        all_symbols = get_perpetual_symbols()
        if all_symbols:
            st.metric("Total Perpetual Pairs", len(all_symbols))

        volume_data = get_top_volume_symbols(5)
        if volume_data:
            st.subheader("🏆 Top Volume Coins")
            for item in volume_data[:5]:
                st.text(f"{item['symbol']}: {format_large_number(item['volume'])}")

        # Clear cache button
        if st.button("🧹 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared! Data will refresh on next scan.")

        st.divider()
        st.subheader("🌐 Proxy Settings")
        proxy_mode = st.radio(
            "Proxy Mode",
            ["None", "Single", "Multiple"],
            index=0,
            key="proxy_mode",
            help="None = direct connection. Single = one IP for all requests. Multiple = round-robin across multiple IPs.",
        )
        proxy_input = st.text_area(
            "Proxy URLs (one per line)",
            value="",
            height=100,
            key="proxy_input",
            help="Format: http://user:pass@ip:port or http://ip:port. One per line.",
        )

        # Sync proxy manager from UI
        if proxy_mode == "None":
            _proxy_manager.set_proxies("none", [])
        elif proxy_mode == "Single":
            proxy_list = [p.strip() for p in proxy_input.splitlines() if p.strip()]
            _proxy_manager.set_proxies("single", proxy_list[:1] if proxy_list else [])
        else:
            proxy_list = [p.strip() for p in proxy_input.splitlines() if p.strip()]
            _proxy_manager.set_proxies("multiple", proxy_list)

        st.caption(_proxy_manager.get_proxy_display())

if __name__ == "__main__":
    main()
