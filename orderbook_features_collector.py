import asyncio
import json
import csv
import os
from datetime import datetime, timezone
from typing import Optional, Tuple
import zipfile
from collections import deque

import numpy as np
# pip install websockets numpy


# ====== CONFIG ======
DEPTH_LEVELS = 10          # orderbook depth we use
INTERVAL_SEC = 5.0         # feature row every 5 seconds

RESULT_DIR = "./result"    # all files stored here
EXCHANGE = "binance"       # "binance" or "bitget"
SYMBOL = "BTCUSDT"         # market, e.g. BTCUSDT

MAX_CSV_MB = 100           # rotate+zip when CSV > 100 MB


def make_output_csv_path() -> str:
    os.makedirs(RESULT_DIR, exist_ok=True)
    basename = f"{EXCHANGE.lower()}_{SYMBOL}_orderbook_features_5s.csv"
    return os.path.join(RESULT_DIR, basename)


OUTPUT_CSV = make_output_csv_path()


def make_orderbook_ws_url(exchange: str, symbol: str) -> str:
    ex = exchange.lower()
    if ex == "binance":
        # Binance futures partial depth 10 stream
        return f"wss://fstream.binance.com/ws/{symbol.lower()}@depth10@100ms"
    elif ex == "bitget":
        # Bitget public WS
        return "wss://ws.bitget.com/v2/ws/public"
    else:
        raise ValueError(f"Unsupported EXCHANGE={exchange}")


def make_kline_ws_url(exchange: str, symbol: str) -> str:
    ex = exchange.lower()
    if ex == "binance":
        # Binance futures 5m kline stream
        return f"wss://fstream.binance.com/ws/{symbol.lower()}@kline_5m"
    elif ex == "bitget":
        # Bitget uses same WS host, different channel (candle5m)
        return "wss://ws.bitget.com/v2/ws/public"
    else:
        raise ValueError(f"Unsupported EXCHANGE={exchange}")


def make_trade_ws_url(exchange: str, symbol: str) -> str:
    ex = exchange.lower()
    if ex == "binance":
        # Binance futures trades
        return f"wss://fstream.binance.com/ws/{symbol.lower()}@trade"
    elif ex == "bitget":
        # Bitget public WS, trade channel
        return "wss://ws.bitget.com/v2/ws/public"
    else:
        raise ValueError(f"Unsupported EXCHANGE={exchange}")


ORDERBOOK_WS_URL = make_orderbook_ws_url(EXCHANGE, SYMBOL)
KLINE_WS_URL = make_kline_ws_url(EXCHANGE, SYMBOL)
TRADE_WS_URL = make_trade_ws_url(EXCHANGE, SYMBOL)


# ====== FEATURE EXTRACTION (ORDERBOOK) ======

def extract_ob_features(bids: np.ndarray, asks: np.ndarray, eps: float = 1e-9):
    """
    bids, asks: np.array shape (N, 2) -> [ [price, size], ... ] best to worst
    returns: dict of features for top DEPTH_LEVELS levels
    """
    bids = np.asarray(bids, dtype=float).reshape(-1, 2)
    asks = np.asarray(asks, dtype=float).reshape(-1, 2)

    # sort: bids descending, asks ascending
    if bids.size > 0:
        bids = bids[np.argsort(bids[:, 0])[::-1]]
    if asks.size > 0:
        asks = asks[np.argsort(asks[:, 0])]

    bids = bids[:DEPTH_LEVELS]
    asks = asks[:DEPTH_LEVELS]

    if bids.shape[0] == 0 or asks.shape[0] == 0:
        raise ValueError("Need at least one bid and one ask level")

    b_prices, b_sizes = bids[:, 0], bids[:, 1]
    a_prices, a_sizes = asks[:, 0], asks[:, 1]

    best_bid = b_prices[0]
    best_ask = a_prices[0]
    mid = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid

    feats = {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": spread,
        "spread_rel": spread / (mid + eps),
    }

    # depth per side
    depth_bid_10 = b_sizes.sum()
    depth_ask_10 = a_sizes.sum()
    feats["depth_bid_10"] = depth_bid_10
    feats["depth_ask_10"] = depth_ask_10
    feats["depth_ratio_10"] = depth_bid_10 / (depth_bid_10 + depth_ask_10 + eps)

    # top-k depths and imbalances
    for k in (1, 3, 5, 10):
        k_eff = min(k, DEPTH_LEVELS)
        B_k = b_sizes[:k_eff].sum()
        A_k = a_sizes[:k_eff].sum()
        feats[f"depth_bid_{k_eff}"] = B_k
        feats[f"depth_ask_{k_eff}"] = A_k
        feats[f"imb_{k_eff}"] = (B_k - A_k) / (B_k + A_k + eps)

    # microprice using level 1 liquidity
    microprice = (best_ask * b_sizes[0] + best_bid * a_sizes[0]) / (b_sizes[0] + a_sizes[0] + eps)
    feats["microprice"] = microprice
    feats["microprice_delta"] = (microprice - mid) / (mid + eps)

    # top heaviness
    feats["top_heaviness_bid"] = b_sizes[0] / (depth_bid_10 + eps)
    feats["top_heaviness_ask"] = a_sizes[0] / (depth_ask_10 + eps)

    # VWAPs
    def vwap(prices, sizes, k):
        k_eff = min(k, len(prices))
        if k_eff == 0:
            return 0.0
        p = prices[:k_eff]
        s = sizes[:k_eff]
        denom = s.sum()
        if denom <= eps:
            return 0.0
        return float((p * s).sum() / denom)

    feats["vwap_bid_5"] = vwap(b_prices, b_sizes, 5)
    feats["vwap_ask_5"] = vwap(a_prices, a_sizes, 5)
    feats["vwap_bid_10"] = vwap(b_prices, b_sizes, 10)
    feats["vwap_ask_10"] = vwap(a_prices, a_sizes, 10)

    # convexity proxies
    avg_depth_bid = depth_bid_10 / max(1, bids.shape[0])
    avg_depth_ask = depth_ask_10 / max(1, asks.shape[0])
    feats["convex_bid"] = (b_sizes[0] + eps) / (avg_depth_bid + eps)
    feats["convex_ask"] = (a_sizes[0] + eps) / (avg_depth_ask + eps)

    # price ranges and gaps
    if bids.shape[0] > 1:
        feats["price_range_bid_10"] = b_prices[0] - b_prices[-1]
        gaps_bid = -np.diff(b_prices)  # bids descending
        feats["avg_gap_bid_10"] = float(gaps_bid.mean())
    else:
        feats["price_range_bid_10"] = 0.0
        feats["avg_gap_bid_10"] = 0.0

    if asks.shape[0] > 1:
        feats["price_range_ask_10"] = a_prices[-1] - a_prices[0]
        gaps_ask = np.diff(a_prices)
        feats["avg_gap_ask_10"] = float(gaps_ask.mean())
    else:
        feats["price_range_ask_10"] = 0.0
        feats["avg_gap_ask_10"] = 0.0

    return feats


# ====== SHARED STATES ======

class OrderBookState:
    """
    Latest orderbook snapshot (top DEPTH_LEVELS).
    """

    def __init__(self):
        self._bids: Optional[np.ndarray] = None
        self._asks: Optional[np.ndarray] = None
        self._lock = asyncio.Lock()

    async def update(self, bids, asks):
        async with self._lock:
            self._bids = np.asarray(bids, dtype=float).reshape(-1, 2)
            self._asks = np.asarray(asks, dtype=float).reshape(-1, 2)

    async def get_snapshot(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        async with self._lock:
            if self._bids is None or self._asks is None:
                return None
            return self._bids.copy(), self._asks.copy()


class Kline5mState:
    """
    Last CLOSED 5m OHLCV bar (from kline/candle WS).
    """

    def __init__(self):
        self._bar = None
        self._lock = asyncio.Lock()

    async def update_from_binance_kline(self, msg: dict):
        # Payload is either raw kline or {"stream": "...", "data": {...}}
        data = msg.get("data", msg)
        k = data.get("k")
        if not isinstance(k, dict):
            return

        # "x" == True means bar is closed/final
        if not k.get("x"):
            return

        try:
            o = float(k["o"])
            h = float(k["h"])
            l = float(k["l"])
            c = float(k["c"])
            v = float(k["v"])
            start_time = int(k["t"])
            end_time = int(k["T"])
        except (KeyError, ValueError, TypeError):
            return

        bar = {
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "start_time": start_time,
            "end_time": end_time,
        }

        async with self._lock:
            self._bar = bar

    async def update_from_bitget_candle(self, msg: dict):
        # Bitget candle message shape:
        # { "arg": {...}, "data": [ { "open": "...", "high": "...", ... } ] }
        data = msg.get("data")
        if not isinstance(data, list) or not data:
            return

        d0 = data[-1]
        try:
            o = float(d0.get("open") or d0.get("o"))
            h = float(d0.get("high") or d0.get("h"))
            l = float(d0.get("low") or d0.get("l"))
            c = float(d0.get("close") or d0.get("c"))
            v = float(d0.get("volume") or d0.get("v") or 0.0)
            ts = int(d0.get("ts") or d0.get("t") or 0)
        except (ValueError, TypeError):
            return

        bar = {
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "start_time": ts,
            "end_time": ts + 5 * 60 * 1000,
        }

        async with self._lock:
            self._bar = bar

    async def get_last_bar(self):
        async with self._lock:
            if self._bar is None:
                return None
            return dict(self._bar)


class TradeState:
    """
    Stores recent trades and computes CVD/OFI features.
    Trades are kept for at most max_window_ms (default 1h).
    """

    def __init__(self, max_window_ms: int = 3600 * 1000):
        self._trades = deque()  # each: (ts_ms, side, size)
        self._lock = asyncio.Lock()
        self._max_window_ms = max_window_ms

    async def add_trade(self, ts_ms: int, side: str, size: float):
        if size <= 0:
            return
        if side not in ("buy", "sell"):
            return
        async with self._lock:
            self._trades.append((ts_ms, side, size))
            cutoff = ts_ms - self._max_window_ms
            while self._trades and self._trades[0][0] < cutoff:
                self._trades.popleft()

    async def get_features(self,
                           now_ms: int,
                           window_5s_ms: int,
                           bar_start_ms: Optional[int],
                           bar_end_ms: Optional[int],
                           eps: float = 1e-9):
        async with self._lock:
            trades = list(self._trades)

        # ---- 5s rolling window ----
        w_start = now_ms - window_5s_ms
        buy_5s = sell_5s = 0.0
        for ts, side, size in trades:
            if ts >= w_start:
                if side == "buy":
                    buy_5s += size
                else:
                    sell_5s += size

        total_5s = buy_5s + sell_5s
        cvd_5s = buy_5s - sell_5s
        imb_5s = cvd_5s / (total_5s + eps) if total_5s > 0 else 0.0

        # ---- 5m window aligned with last closed bar ----
        buy_5m = sell_5m = 0.0
        if bar_start_ms is not None and bar_end_ms is not None:
            for ts, side, size in trades:
                if bar_start_ms <= ts < bar_end_ms:
                    if side == "buy":
                        buy_5m += size
                    else:
                        sell_5m += size

        total_5m = buy_5m + sell_5m
        cvd_5m = buy_5m - sell_5m
        imb_5m = cvd_5m / (total_5m + eps) if total_5m > 0 else 0.0

        return {
            # 5s window
            "buy_vol_5s": buy_5s,
            "sell_vol_5s": sell_5s,
            "total_vol_5s": total_5s,
            "cvd_5s": cvd_5s,
            "ofi_5s": cvd_5s,           # trade-based OFI ~= CVD
            "imbalance_5s": imb_5s,

            # 5m window aligned to last closed 5m bar
            "buy_vol_5m": buy_5m,
            "sell_vol_5m": sell_5m,
            "total_vol_5m": total_5m,
            "cvd_5m": cvd_5m,
            "ofi_5m": cvd_5m,           # same idea
            "imbalance_5m": imb_5m,
        }


# ====== CSV + ROTATION HELPERS ======

def ensure_csv_header(path: str, fieldnames):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    exists = os.path.exists(path)
    if not exists:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def rotate_csv_if_needed(path: str, max_mb: float):
    if not os.path.exists(path):
        return

    size_bytes = os.path.getsize(path)
    if size_bytes < max_mb * 1024 * 1024:
        return

    base, ext = os.path.splitext(path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rotated_csv = f"{base}_{ts}{ext}"
    zip_path = f"{base}_{ts}.zip"

    os.rename(path, rotated_csv)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(rotated_csv, arcname=os.path.basename(rotated_csv))

    os.remove(rotated_csv)

    print(f"[rotate] Rotated and zipped {path} -> {zip_path}")


# ====== FEATURE WRITER LOOP (EVERY 5s) ======

async def feature_writer_loop(ob_state: OrderBookState,
                              kline_state: Kline5mState,
                              trade_state: TradeState,
                              output_path: str,
                              interval_sec: float,
                              exchange: str,
                              symbol: str):
    """
    Every 'interval_sec':
      - take current OB snapshot -> OB features
      - read last closed 5m OHLCV bar
      - compute trade-based CVD/OFI (5s rolling + 5m aligned)
      - write one row in CSV
    """
    # build feature names from dummy OB
    dummy_bids = np.array([[100.0 - i * 0.5, 1.0 + i] for i in range(DEPTH_LEVELS)])
    dummy_asks = np.array([[100.5 + i * 0.5, 1.0 + i] for i in range(DEPTH_LEVELS)])
    base_feats = extract_ob_features(dummy_bids, dummy_asks)

    trade_cols = [
        "buy_vol_5s", "sell_vol_5s", "total_vol_5s",
        "cvd_5s", "ofi_5s", "imbalance_5s",
        "buy_vol_5m", "sell_vol_5m", "total_vol_5m",
        "cvd_5m", "ofi_5m", "imbalance_5m",
    ]

    fieldnames = (
        ["ts_iso", "exchange", "symbol"] +
        sorted(base_feats.keys()) +
        ["open_5m", "high_5m", "low_5m", "close_5m", "volume_5m"] +
        trade_cols
    )

    ensure_csv_header(output_path, fieldnames)

    window_5s_ms = int(interval_sec * 1000)

    while True:
        await asyncio.sleep(interval_sec)

        snapshot = await ob_state.get_snapshot()
        if snapshot is None:
            continue

        bids, asks = snapshot
        try:
            feats = extract_ob_features(bids, asks)
        except Exception as e:
            print(f"[feature_writer] skip snapshot: {e}")
            continue

        rotate_csv_if_needed(output_path, MAX_CSV_MB)
        ensure_csv_header(output_path, fieldnames)

        bar = await kline_state.get_last_bar()
        if bar is None:
            o = h = l = c = v = ""
            bar_start_ms = bar_end_ms = None
        else:
            o = bar["open"]
            h = bar["high"]
            l = bar["low"]
            c = bar["close"]
            v = bar["volume"]
            bar_start_ms = bar["start_time"]
            bar_end_ms = bar["end_time"]

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        trade_feats = await trade_state.get_features(
            now_ms=now_ms,
            window_5s_ms=window_5s_ms,
            bar_start_ms=bar_start_ms,
            bar_end_ms=bar_end_ms,
        )

        row = {
            "ts_iso": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange.lower(),
            "symbol": symbol,
            "open_5m": o,
            "high_5m": h,
            "low_5m": l,
            "close_5m": c,
            "volume_5m": v,
        }
        row.update(feats)
        row.update(trade_feats)

        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)


# ====== MESSAGE PARSING (ORDERBOOK) ======

def _generic_snapshot_from_message(msg: dict):
    if "bids" in msg and "asks" in msg:
        bids = [[float(p), float(q)] for p, q in msg["bids"]][:DEPTH_LEVELS]
        asks = [[float(p), float(q)] for p, q in msg["asks"]][:DEPTH_LEVELS]
        return bids, asks

    data = msg.get("data", {})
    if isinstance(data, dict):
        if "bids" in data and "asks" in data:
            bids = [[float(p), float(q)] for p, q in data["bids"]][:DEPTH_LEVELS]
            asks = [[float(p), float(q)] for p, q in data["asks"]][:DEPTH_LEVELS]
            return bids, asks
        if "b" in data and "a" in data:
            bids = [[float(p), float(q)] for p, q in data["b"]][:DEPTH_LEVELS]
            asks = [[float(p), float(q)] for p, q in data["a"]][:DEPTH_LEVELS]
            return bids, asks

    if "b" in msg and "a" in msg:
        bids = [[float(p), float(q)] for p, q in msg["b"]][:DEPTH_LEVELS]
        asks = [[float(p), float(q)] for p, q in msg["a"]][:DEPTH_LEVELS]
        return bids, asks

    raise ValueError(f"Don't know how to parse message: keys={list(msg.keys())}")


def extract_snapshot_from_message(msg: dict, exchange: str):
    """
    Normalize WS message into (bids, asks) for:
      - Binance depth10 stream
      - Bitget books15 depth channel
    """
    ex = exchange.lower()

    if ex == "binance":
        data = msg.get("data", msg)
        if "b" in data and "a" in data:
            bids = [[float(p), float(q)] for p, q in data["b"]][:DEPTH_LEVELS]
            asks = [[float(p), float(q)] for p, q in data["a"]][:DEPTH_LEVELS]
            return bids, asks
        return _generic_snapshot_from_message(msg)

    if ex == "bitget":
        data = msg.get("data")
        if isinstance(data, list) and data:
            d0 = data[0]
            if "bids" in d0 and "asks" in d0:
                bids = [[float(p), float(q)] for p, q in d0["bids"]][:DEPTH_LEVELS]
                asks = [[float(p), float(q)] for p, q in d0["asks"]][:DEPTH_LEVELS]
                return bids, asks
        return _generic_snapshot_from_message(msg)

    return _generic_snapshot_from_message(msg)


# ====== WS LOOPS ======

async def ws_orderbook_loop(ob_state: OrderBookState):
    import websockets  # type: ignore

    while True:
        try:
            print(f"[ws-ob] connecting to {EXCHANGE} at {ORDERBOOK_WS_URL} ...")
            async with websockets.connect(ORDERBOOK_WS_URL, ping_interval=20, ping_timeout=20) as ws:
                print("[ws-ob] connected")

                # Bitget requires subscribe; Binance stream is in the URL
                if EXCHANGE.lower() == "bitget":
                    sub_msg = {
                        "op": "subscribe",
                        "args": [{
                            "instType": "USDT-FUTURES",
                            "channel": "books15",
                            "instId": SYMBOL
                        }]
                    }
                    await ws.send(json.dumps(sub_msg))
                    print(f"[ws-ob] sent subscribe: {sub_msg}")

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    try:
                        bids, asks = extract_snapshot_from_message(msg, EXCHANGE)
                    except Exception:
                        # uncomment to debug:
                        # print(f"[ws-ob] parse error: {e}, msg={msg}")
                        continue

                    await ob_state.update(bids, asks)

        except Exception as e:
            print(f"[ws-ob] error: {e}, reconnect in 3s")
            await asyncio.sleep(3.0)


async def ws_kline_loop(kline_state: Kline5mState):
    import websockets  # type: ignore

    while True:
        try:
            print(f"[ws-kl] connecting to {EXCHANGE} at {KLINE_WS_URL} ...")
            async with websockets.connect(KLINE_WS_URL, ping_interval=20, ping_timeout=20) as ws:
                print("[ws-kl] connected")

                # Bitget needs candle subscription; Binance kline is in URL
                if EXCHANGE.lower() == "bitget":
                    sub_msg = {
                        "op": "subscribe",
                        "args": [{
                            "instType": "USDT-FUTURES",
                            "channel": "candle5m",
                            "instId": SYMBOL
                        }]
                    }
                    await ws.send(json.dumps(sub_msg))
                    print(f"[ws-kl] sent subscribe: {sub_msg}")

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    try:
                        if EXCHANGE.lower() == "binance":
                            await kline_state.update_from_binance_kline(msg)
                        else:
                            await kline_state.update_from_bitget_candle(msg)
                    except Exception:
                        continue

        except Exception as e:
            print(f"[ws-kl] error: {e}, reconnect in 3s")
            await asyncio.sleep(3.0)


async def ws_trade_loop(trade_state: TradeState):
    import websockets  # type: ignore

    while True:
        try:
            print(f"[ws-tr] connecting to {EXCHANGE} at {TRADE_WS_URL} ...")
            async with websockets.connect(TRADE_WS_URL, ping_interval=20, ping_timeout=20) as ws:
                print("[ws-tr] connected")

                # Bitget needs trade subscription; Binance trade stream is in URL
                if EXCHANGE.lower() == "bitget":
                    sub_msg = {
                        "op": "subscribe",
                        "args": [{
                            "instType": "USDT-FUTURES",
                            "channel": "trade",
                            "instId": SYMBOL
                        }]
                    }
                    await ws.send(json.dumps(sub_msg))
                    print(f"[ws-tr] sent subscribe: {sub_msg}")

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    try:
                        if EXCHANGE.lower() == "binance":
                            await _handle_binance_trade(msg, trade_state)
                        else:
                            await _handle_bitget_trade(msg, trade_state)
                    except Exception:
                        continue

        except Exception as e:
            print(f"[ws-tr] error: {e}, reconnect in 3s")
            await asyncio.sleep(3.0)


async def _handle_binance_trade(msg: dict, trade_state: TradeState):
    # Payload is either raw trade or {"stream": "...", "data": {...}}
    data = msg.get("data", msg)
    # For @trade stream: data["e"] == "trade"
    try:
        ts_ms = int(data.get("T") or data.get("E"))
        size = float(data["q"])
        m_flag = data["m"]  # is buyer the market maker
    except (KeyError, TypeError, ValueError):
        return

    # aggressor side: if buyer is maker -> seller is taker -> sell-initiated
    side = "sell" if m_flag else "buy"
    await trade_state.add_trade(ts_ms, side, size)


async def _handle_bitget_trade(msg: dict, trade_state: TradeState):
    # Expect shape:
    # { "arg": {..., "channel":"trade", "instId":...}, "data":[{...}, ...] }
    arg = msg.get("arg", {})
    if arg.get("channel") != "trade":
        return
    data_list = msg.get("data")
    if not isinstance(data_list, list):
        return

    for d in data_list:
        side = d.get("side")
        sz = d.get("sz") or d.get("size") or d.get("vol") or d.get("volume")
        ts = d.get("ts") or d.get("time") or d.get("t")
        if side not in ("buy", "sell") or sz is None or ts is None:
            continue
        try:
            size = float(sz)
            ts_ms = int(ts)
        except (ValueError, TypeError):
            continue
        await trade_state.add_trade(ts_ms, side, size)


# ====== MAIN ======

async def main():
    ob_state = OrderBookState()
    kl_state = Kline5mState()
    tr_state = TradeState()

    writer = asyncio.create_task(
        feature_writer_loop(ob_state, kl_state, tr_state,
                            OUTPUT_CSV, INTERVAL_SEC, EXCHANGE, SYMBOL)
    )
    ob_consumer = asyncio.create_task(ws_orderbook_loop(ob_state))
    kl_consumer = asyncio.create_task(ws_kline_loop(kl_state))
    tr_consumer = asyncio.create_task(ws_trade_loop(tr_state))

    try:
        await asyncio.gather(writer, ob_consumer, kl_consumer, tr_consumer)
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    # Install deps first:
    #   pip install websockets numpy
    asyncio.run(main())
