# -*- coding: utf-8 -*-
"""
半自動股票分析 Web App
======================
功能：
1. 分頁一【自選股觀察與診斷】：手動管理自選股清單，按下「一鍵更新與診斷」才會呼叫 yfinance
   抓取資料並計算 60MA / 10MA / KD(9,3,3)，依規則標示 買進 / 停利 / 停損 狀態。
2. 分頁二【市場強勢標的推薦】：對預設的市場清單（可自行編輯）進行條件篩選，
   找出「股價 > 60MA 且 KD 在低檔(<30)出現黃金交叉」的標的，並給出建議進場價與預估停損價。

執行方式：
    pip install streamlit yfinance pandas numpy
    streamlit run stock_analyzer_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ============================================================
# 基本頁面設定
# ============================================================
st.set_page_config(
    page_title="股市分析平台",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# Session State 初始化
# ============================================================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["2330.TW", "2454.TW", "2317.TW"]
if "holdings" not in st.session_state:
    st.session_state.holdings = {}
if "watchlist_result" not in st.session_state:
    st.session_state.watchlist_result = None
if "market_scan_result" not in st.session_state:
    st.session_state.market_scan_result = None

DEFAULT_MARKET_LIST = [
    "2330.TW", "2317.TW", "2454.TW", "2412.TW", "2308.TW", "2882.TW", "1303.TW",
    "1301.TW", "2891.TW", "2303.TW", "2881.TW", "2886.TW", "2884.TW", "3711.TW",
    "2892.TW", "5880.TW", "2885.TW", "2002.TW", "2887.TW", "1216.TW", "2801.TW",
    "2880.TW", "3008.TW", "2382.TW", "2883.TW", "2890.TW", "1101.TW", "2327.TW",
    "6505.TW", "5876.TW", "2207.TW", "9910.TW", "2379.TW", "2395.TW", "4938.TW",
    "2357.TW", "3045.TW", "4904.TW", "2609.TW", "2603.TW", "2615.TW", "1326.TW",
    "9945.TW", "2912.TW", "1102.TW", "2377.TW", "2408.TW", "3034.TW", "6446.TW",
    "2474.TW",
]


# ============================================================
# 技術指標計算函式
# ============================================================
def compute_kd(df: pd.DataFrame, n: int = 9, k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
    """
    計算 KD 指標（台式平滑公式，等同 RSV 以 1/3, 2/3 權重遞迴平滑）
    K = 前一日K * (2/3) + 今日RSV * (1/3)
    D = 前一日D * (2/3) + 今日K  * (1/3)
    起始 K = D = 50
    """
    low_min = df["Low"].rolling(window=n, min_periods=n).min()
    high_max = df["High"].rolling(window=n, min_periods=n).max()

    rsv = (df["Close"] - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50)

    k_values, d_values = [], []
    k_prev, d_prev = 50.0, 50.0
    for val in rsv:
        k_curr = k_prev * (k_smooth - 1) / k_smooth + val * (1 / k_smooth)
        d_curr = d_prev * (d_smooth - 1) / d_smooth + k_curr * (1 / d_smooth)
        k_values.append(k_curr)
        d_values.append(d_curr)
        k_prev, d_prev = k_curr, d_curr

    df["K"] = k_values
    df["D"] = d_values
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算 10MA / 60MA / KD，回傳新增欄位後的 DataFrame"""
    df = df.copy()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df = compute_kd(df)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def fetch_data(ticker: str, period: str = "9mo") -> pd.DataFrame | None:
    """
    透過 yfinance 抓取歷史資料。
    - 使用 9 個月區間，確保 60MA / KD(9,3,3) 有足夠的回看資料。
    - auto_adjust=False：技術分析慣例使用未還原股價計算均線與 KD。
    - 加上 cache（10 分鐘）避免同一次 session 中重複呼叫 API。
    """
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["Close", "High", "Low"])
        return df
    except Exception:
        return None


# ============================================================
# 分頁一：自選股診斷邏輯
# ============================================================
def diagnose_stock(ticker: str, df: pd.DataFrame | None) -> dict:
    """
    對單一自選股計算指標並依規則判斷狀態：
      買進：股價 > 60MA 且 KD 低檔(<30)出現黃金交叉
      停利：收盤價跌破 10MA
      停損：跌破「進場日」的最低點
    使用 st.session_state.holdings 追蹤「是否已進場」與「進場日低點」。
    """
    if df is None or len(df) < 61:
        return {"股票代號": ticker, "狀態": "⚠️ 資料不足或抓取失敗"}

    df = compute_indicators(df)
    latest, prev = df.iloc[-1], df.iloc[-2]

    price = float(latest["Close"])
    ma60 = float(latest["MA60"]) if pd.notna(latest["MA60"]) else np.nan
    ma10 = float(latest["MA10"]) if pd.notna(latest["MA10"]) else np.nan
    k, d = float(latest["K"]), float(latest["D"])
    k_prev, d_prev = float(prev["K"]), float(prev["D"])
    trade_date = df.index[-1].strftime("%Y-%m-%d")

    golden_cross = (k_prev < d_prev) and (k > d)
    low_zone = (k < 30) and (d < 30)
    buy_signal = (not np.isnan(ma60)) and (price > ma60) and golden_cross and low_zone

    holding = st.session_state.holdings.get(ticker)
    status = "⚪ 觀察中"

    if buy_signal and not holding:
        st.session_state.holdings[ticker] = {
            "entry_date": trade_date,
            "entry_low": float(latest["Low"]),
            "entry_price": price,
        }
        status = "🟢 買進"
    elif holding:
        entry_low = holding["entry_low"]
        if price < entry_low:
            status = "🔴 停損"
            st.session_state.holdings.pop(ticker, None)
        elif not np.isnan(ma10) and price < ma10:
            status = "🟠 停利"
            st.session_state.holdings.pop(ticker, None)
        else:
            status = "🔵 持有中"

    return {
        "股票代號": ticker,
        "最新收盤": round(price, 2),
        "60MA": round(ma60, 2) if not np.isnan(ma60) else None,
        "10MA": round(ma10, 2) if not np.isnan(ma10) else None,
        "K值": round(k, 2),
        "D值": round(d, 2),
        "狀態": status,
        "進場參考低點": round(holding["entry_low"], 2) if (ticker in st.session_state.holdings) else None,
        "更新日期": trade_date,
    }


def style_watchlist(df: pd.DataFrame):
    """依「狀態」欄位替表格套上顏色標籤"""
    def color_status(val):
        if "買進" in str(val):
            return "background-color:#d4edda; color:#155724; font-weight:bold;"
        elif "停利" in str(val):
            return "background-color:#fff3cd; color:#856404; font-weight:bold;"
        elif "停損" in str(val):
            return "background-color:#f8d7da; color:#721c24; font-weight:bold;"
        elif "持有" in str(val):
            return "background-color:#d1ecf1; color:#0c5460; font-weight:bold;"
        elif "觀察" in str(val):
            return "background-color:#f0f0f0; color:#555555;"
        else:
            return "background-color:#f5c6cb; color:#721c24;"
    styler = df.style
    if hasattr(styler, "map"):
        return styler.map(color_status, subset=["狀態"])
    return styler.applymap(color_status, subset=["狀態"])


# ============================================================
# 分頁二：市場強勢標的篩選邏輯
# ============================================================
def scan_stock(ticker: str, df: pd.DataFrame | None) -> dict | None:
    """
    篩選條件：股價 > 60MA 且 KD 在 30 以下出現黃金交叉
    符合條件則回傳建議進場價（以當日收盤為基準）與預估停損價（當日低點）
    """
    if df is None or len(df) < 61:
        return None

    df = compute_indicators(df)
    latest, prev = df.iloc[-1], df.iloc[-2]

    price = float(latest["Close"])
    ma60 = float(latest["MA60"]) if pd.notna(latest["MA60"]) else np.nan
    k, d = float(latest["K"]), float(latest["D"])
    k_prev, d_prev = float(prev["K"]), float(prev["D"])

    golden_cross = (k_prev < d_prev) and (k > d)
    low_zone = (k < 30) and (d < 30)

    if np.isnan(ma60) or not (price > ma60 and golden_cross and low_zone):
        return None

    entry_price = round(price, 2)
    stop_loss = round(float(latest["Low"]), 2)

    return {
        "股票代號": ticker,
        "收盤價": entry_price,
        "60MA": round(ma60, 2),
        "K值": round(k, 2),
        "D值": round(d, 2),
        "建議進場價": entry_price,
        "預估停損價": stop_loss,
        "停損風險(%)": round((entry_price - stop_loss) / entry_price * 100, 2) if entry_price else None,
        "更新日期": df.index[-1].strftime("%Y-%m-%d"),
    }


# ============================================================
# UI：頁首
# ============================================================
st.title("📊 股市分析平台")
st.caption("資料來源：Yahoo Finance（yfinance）｜僅供技術分析參考，非投資建議")

with st.sidebar:
    st.header("⚙️ 使用說明")
    st.markdown(
        """
        **半自動設計理念**
        本工具不會自動連續抓取資料，所有網路請求皆須由使用者
        點擊「一鍵更新與診斷」或「開始市場掃描」按鈕才會觸發，
        避免不必要的 API 呼叫與過度交


        
