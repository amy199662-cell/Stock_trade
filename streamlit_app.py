# -*- coding: utf-8 -*-
"""
半自動股票分析 Web App
======================
分頁一：自選股觀察與診斷（買進 / 停利 / 停損）
分頁二：市場強勢標的掃描（60MA + KD 低檔黃金交叉）

執行方式：
    pip install streamlit yfinance pandas numpy
    streamlit run stock_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ------------------------------------------------------------------
# 全域設定
# ------------------------------------------------------------------
st.set_page_config(
    page_title="半自動股票分析工具",
    page_icon="📈",
    layout="wide",
)

# 台股前 30 大權值股（預設市場觀察清單）
DEFAULT_MARKET_LIST = [
    "2330.TW", "2317.TW", "2454.TW", "2412.TW", "2882.TW",
    "2881.TW", "1303.TW", "1301.TW", "2308.TW", "2891.TW",
    "2886.TW", "3711.TW", "2884.TW", "5880.TW", "2892.TW",
    "2303.TW", "1216.TW", "2002.TW", "2885.TW", "5871.TW",
    "3008.TW", "2880.TW", "2887.TW", "6505.TW", "2801.TW",
    "2382.TW", "4938.TW", "9910.TW", "2603.TW", "1101.TW",
]

# ------------------------------------------------------------------
# Session State 初始化
# ------------------------------------------------------------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["2330.TW", "2454.TW"]

if "diagnosis_df" not in st.session_state:
    st.session_state.diagnosis_df = pd.DataFrame()

if "scan_df" not in st.session_state:
    st.session_state.scan_df = pd.DataFrame()

if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = None

if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None


# ------------------------------------------------------------------
# 資料抓取（加上快取，避免同一 ticker 重複打 API）
# ------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    """抓取單一股票的歷史 OHLCV 資料"""
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


# ------------------------------------------------------------------
# 技術指標計算
# ------------------------------------------------------------------
def calc_indicators(df: pd.DataFrame, kd_n: int = 9, kd_m1: int = 3, kd_m2: int = 3) -> pd.DataFrame:
    """計算 MA10 / MA60 / KD(9,3,3)"""
    df = df.copy()

    # --- 移動平均線 ---
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA60"] = df["Close"].rolling(window=60).mean()

    # --- KD 指標（台股慣用平滑公式）---
    low_n = df["Low"].rolling(window=kd_n).min()
    high_n = df["High"].rolling(window=kd_n).max()
    rsv = (df["Close"] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)  # 前 n 天資料不足時，用中性值 50 帶入

    k_values, d_values = [], []
    prev_k, prev_d = 50.0, 50.0
    for val in rsv:
        k = prev_k * (kd_m1 - 1) / kd_m1 + val / kd_m1
        d = prev_d * (kd_m2 - 1) / kd_m2 + k / kd_m2
        k_values.append(k)
        d_values.append(d)
        prev_k, prev_d = k, d

    df["K"] = k_values
    df["D"] = d_values

    # 黃金交叉判斷：前一天 K<=D，今天 K>D
    df["KD_GOLDEN_CROSS"] = (df["K"].shift(1) <= df["D"].shift(1)) & (df["K"] > df["D"])
    return df


# ------------------------------------------------------------------
# 分頁一：個股診斷邏輯
# ------------------------------------------------------------------
def diagnose_stock(df: pd.DataFrame):
    """
    判斷邏輯：
    1. 買進：股價 > MA60 且 KD 在低檔(<30)出現黃金交叉
    2. 停利：收盤價跌破 MA10（持有中才觸發）
    3. 停損：跌破「進場日」的最低點（持有中才觸發）
    4. 若無有效訊號則為「觀望」，若買進後尚未觸發停利/停損則為「持有中」
    """
    if len(df) < 70:
        return "資料不足", None, None

    latest = df.iloc[-1]

    # 找出符合「低檔黃金交叉 + 站上 60MA」條件的所有歷史買進訊號日
    buy_mask = (
        (df["Close"] > df["MA60"])
        & (df["K"] < 30)
        & (df["D"] < 30)
        & df["KD_GOLDEN_CROSS"]
    )
    buy_dates = df.index[buy_mask]

    if len(buy_dates) == 0:
        return "觀望", None, None

    # 取最近一次的買進訊號日，作為「進場日」
    entry_date = buy_dates[-1]
    entry_low = df.loc[entry_date, "Low"]

    # 檢查「進場日」之後、今天之前，是否已經觸發過停利或停損（代表這筆訊號已出場）
    after_entry = df.loc[entry_date:]
    exited = False
    if len(after_entry) > 2:
        between = after_entry.iloc[1:-1]  # 進場日隔天 ~ 昨天
        if (between["Close"] < between["MA10"]).any() or (between["Close"] < entry_low).any():
            exited = True

    if not exited:
        # 部位仍視為持有中，逐一檢查今天的出場條件
        if latest["Close"] < entry_low:
            return "停損", entry_date, entry_low
        elif latest["Close"] < latest["MA10"]:
            return "停利", entry_date, entry_low
        elif bool(buy_mask.iloc[-1]):
            return "買進", entry_date, entry_low
        else:
            return "持有中", entry_date, entry_low
    else:
        # 舊訊號已出場，檢查今天是否是新的買進訊號
        if bool(buy_mask.iloc[-1]):
            return "買進", df.index[-1], latest["Low"]
        return "觀望", None, None


# ------------------------------------------------------------------
# 表格色彩樣式
# ------------------------------------------------------------------
STATUS_STYLE = {
    "買進":   "background-color:#d4f7dc; color:#1a7431; font-weight:700;",
    "停利":   "background-color:#fff3cd; color:#8a6100; font-weight:700;",
    "停損":   "background-color:#f8d7da; color:#a31621; font-weight:700;",
    "持有中": "background-color:#e2e3ff; color:#3949ab; font-weight:600;",
    "觀望":   "background-color:#f0f0f0; color:#666666;",
    "資料不足": "background-color:#f0f0f0; color:#999999;",
}


def style_status_col(val):
    return STATUS_STYLE.get(val, "")


# ==================================================================
# UI 主體
# ==================================================================
st.title("📈 半自動股票分析工具")
st.caption("以 60MA 站上趨勢 + KD 低檔交叉為核心，輔助買進 / 停利 / 停損判斷（僅供研究參考，非投資建議）")

tab1, tab2 = st.tabs(["🔍 自選股觀察與診斷", "🚀 市場強勢標的推薦"])

# ------------------------------------------------------------------
# 分頁一
# ------------------------------------------------------------------
with tab1:
    st.subheader("自選股清單管理")

    col_input, col_add = st.columns([4, 1])
    with col_input:
        new_ticker = st.text_input(
            "輸入股票代號後按新增（例如：2330.TW）",
            key="new_ticker_input",
            label_visibility="collapsed",
            placeholder="輸入股票代號，例如 2330.TW",
        )
    with col_add:
        if st.button("➕ 新增自選股", use_container_width=True):
            t = new_ticker.strip().upper()
            if t and t not in st.session_state.watchlist:
                st.session_state.watchlist.append(t)
                st.rerun()
            elif t in st.session_state.watchlist:
                st.warning(f"{t} 已在清單中")

    # 目前自選股（chip 形式 + 刪除按鈕）
    if st.session_state.watchlist:
        st.write("目前自選股：")
        chip_cols = st.columns(min(len(st.session_state.watchlist), 8) or 1)
        for i, ticker in enumerate(st.session_state.watchlist):
            with chip_cols[i % len(chip_cols)]:
                st.markdown(f"**{ticker}**")
                if st.button("🗑️ 刪除", key=f"del_{ticker}"):
                    st.session_state.watchlist.remove(ticker)
                    st.rerun()
    else:
        st.info("目前尚無自選股，請先新增。")

    st.divider()

    # 大按鈕：一鍵更新與診斷
    update_clicked = st.button(
        "🚀 一鍵更新與診斷", type="primary", use_container_width=True,
        disabled=(len(st.session_state.watchlist) == 0),
    )

    if update_clicked:
        results = []
        progress = st.progress(0, text="準備開始抓取資料...")
        total = len(st.session_state.watchlist)

        for idx, ticker in enumerate(st.session_state.watchlist):
            progress.progress((idx) / total, text=f"正在處理 {ticker} ...")
            df = fetch_stock_data(ticker)

            if df is None or len(df) < 70:
                results.append({
                    "股票代號": ticker, "收盤價": "-", "MA10": "-", "MA60": "-",
                    "K": "-", "D": "-", "診斷狀態": "資料不足", "進場日低點": "-",
                })
                continue

            df = calc_indicators(df)
            status, entry_date, entry_low = diagnose_stock(df)
            latest = df.iloc[-1]

            results.append({
                "股票代號": ticker,
                "收盤價": round(float(latest["Close"]), 2),
                "MA10": round(float(latest["MA10"]), 2) if not pd.isna(latest["MA10"]) else "-",
                "MA60": round(float(latest["MA60"]), 2) if not pd.isna(latest["MA60"]) else "-",
                "K": round(float(latest["K"]), 1),
                "D": round(float(latest["D"]), 1),
                "診斷狀態": status,
                "進場日低點": round(float(entry_low), 2) if entry_low is not None else "-",
            })

        progress.progress(1.0, text="完成！")
        progress.empty()

        st.session_state.diagnosis_df = pd.DataFrame(results)
        st.session_state.last_update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 顯示結果表格
    if not st.session_state.diagnosis_df.empty:
        st.caption(f"最後更新時間：{st.session_state.last_update_time}")
        styled = (
            st.session_state.diagnosis_df.style
            .applymap(style_status_col, subset=["診斷狀態"])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        with st.expander("📖 判斷邏輯說明"):
            st.markdown(
                "- **買進**：收盤價 > 60MA，且 KD 在低檔（K、D 皆 < 30）出現黃金交叉\n"
                "- **持有中**：已觸發買進訊號，尚未跌破 10MA 或進場日低點\n"
                "- **停利**：收盤價跌破 10MA\n"
                "- **停損**：收盤價跌破「進場日」當天的最低點\n"
                "- **觀望**：目前無有效買進訊號"
            )
    else:
        st.info("請點擊上方「一鍵更新與診斷」按鈕以取得最新診斷結果。")


# ------------------------------------------------------------------
# 分頁二
# ------------------------------------------------------------------
with tab2:
    st.subheader("市場強勢標的掃描")
    st.caption("預設清單：台股前 30 大權值股，可自行調整。")

    market_list = st.multiselect(
        "掃描標的清單",
        options=DEFAULT_MARKET_LIST,
        default=DEFAULT_MARKET_LIST,
    )

    scan_clicked = st.button(
        "🚀 開始市場掃描", type="primary", use_container_width=True,
        disabled=(len(market_list) == 0),
    )

    if scan_clicked:
        matched = []
        progress = st.progress(0, text="準備開始掃描...")
        total = len(market_list)

        for idx, ticker in enumerate(market_list):
            progress.progress(idx / total, text=f"正在掃描 {ticker} ...")
            df = fetch_stock_data(ticker)

            if df is None or len(df) < 70:
                continue

            df = calc_indicators(df)
            latest = df.iloc[-1]

            # 篩選條件：股價 > 60MA，且 KD 低檔（<30）出現黃金交叉
            cond_above_ma60 = latest["Close"] > latest["MA60"]
            cond_low_zone = (latest["K"] < 30) and (latest["D"] < 30)
            cond_golden_cross = bool(latest["KD_GOLDEN_CROSS"])

            if cond_above_ma60 and cond_low_zone and cond_golden_cross:
                suggested_entry = round(float(latest["Close"]), 2)
                suggested_stop = round(float(latest["Low"]), 2)
                risk_pct = round((suggested_entry - suggested_stop) / suggested_entry * 100, 2)

                matched.append({
                    "股票代號": ticker,
                    "收盤價": round(float(latest["Close"]), 2),
                    "MA60": round(float(latest["MA60"]), 2),
                    "K": round(float(latest["K"]), 1),
                    "D": round(float(latest["D"]), 1),
                    "建議進場價": suggested_entry,
                    "預估停損價": suggested_stop,
                    "潛在風險(%)": risk_pct,
                })

        progress.progress(1.0, text="掃描完成！")
        progress.empty()

        st.session_state.scan_df = pd.DataFrame(matched)
        st.session_state.last_scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not st.session_state.scan_df.empty:
        st.caption(f"最後掃描時間：{st.session_state.last_scan_time}")
        st.success(f"共篩選出 {len(st.session_state.scan_df)} 檔符合條件的強勢標的")
        st.dataframe(
            st.session_state.scan_df.style.background_gradient(
                subset=["潛在風險(%)"], cmap="RdYlGn_r"
            ),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("📖 篩選邏輯說明"):
            st.markdown(
                "- 篩選條件：**收盤價 > 60MA**，且 **KD 在低檔（K、D 皆 < 30）出現黃金交叉**\n"
                "- **建議進場價**：以當日收盤價為參考\n"
                "- **預估停損價**：以當日最低點為參考\n"
                "- 此為量化篩選結果，仍建議搭配基本面與大盤環境綜合判斷"
            )
    elif st.session_state.last_scan_time is not None:
        st.warning("本次掃描沒有符合條件的標的。")
    else:
        st.info("請點擊上方「開始市場掃描」按鈕以取得符合條件的強勢標的。")

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.divider()
st.caption("⚠️ 本工具僅供技術分析教學與研究參考，不構成投資建議。股市有風險，投資請謹慎評估並自負盈虧。")
