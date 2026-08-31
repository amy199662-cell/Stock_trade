# -*- coding: utf-8 -*-
"""
股市分析平台
======================
功能：
1. 分頁一【自選股觀察與診斷】：手動管理自選股清單，按下「一鍵更新與診斷」才會呼叫 yfinance
   抓取資料並計算 60MA / 10MA / KD(9,3,3)，依規則標示 買進 / 停利 / 停損 狀態。
2. 分頁二【市場強勢標的推薦】：提供台股市值代表性清單（可自行編輯 / 擴充至完整前200大），
   使用者可從 10 種技術指標中最多挑選 5 項作為篩選依據（每列可留空、可複選），
   按下「開始市場掃描」後列出符合條件的標的，並給出建議進場價與預估停損價。

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
# watchlist:      自選股代號清單
# holdings:       {ticker: {"entry_date", "entry_low", "entry_price"}} 追蹤買進後的進場低點
# watchlist_result / market_scan_result：暫存上次按下按鈕後的計算結果，避免每次互動都重新整頁時清空
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["2330.TW", "2454.TW", "2317.TW"]
if "holdings" not in st.session_state:
    st.session_state.holdings = {}
if "watchlist_result" not in st.session_state:
    st.session_state.watchlist_result = None
if "market_scan_result" not in st.session_state:
    st.session_state.market_scan_result = None

# ------------------------------------------------------------
# 預設市場觀察清單
# ------------------------------------------------------------
# 注意：股票市值排名每日都會隨股價變動，無法用「寫死的清單」保證永遠等於即時前200大排名。
# 以下提供約 140 檔具代表性的台股中大型市值股（涵蓋半導體、電子、金融、傳產、航運、生技等主要族群）
# 作為掃描起點，使用者可依需求自行編輯／貼上最新的前200大市值清單（可參考 Goodinfo 或證交所公告）。
DEFAULT_MARKET_LIST = [
    # 權值龍頭 / 電子科技
    "2330.TW", "2317.TW", "2454.TW", "2412.TW", "2308.TW", "2382.TW", "3711.TW",
    "2357.TW", "4938.TW", "2379.TW", "2395.TW", "3045.TW", "4904.TW", "2377.TW",
    "2408.TW", "3034.TW", "2474.TW", "2324.TW", "2356.TW", "6669.TW", "2301.TW",
    "3231.TW", "1503.TW", "6770.TW", "2337.TW", "2344.TW", "2360.TW", "3702.TW",
    "2347.TW", "6176.TW", "2385.TW", "3044.TW", "6213.TW", "2439.TW", "2059.TW",
    "3653.TW", "2049.TW", "1590.TW", "6274.TW", "8299.TW", "3529.TW", "5269.TW",
    "2338.TW", "2481.TW", "4919.TW", "5347.TW", "3532.TW", "6488.TW", "3707.TW",
    "3596.TW", "2451.TW", "3037.TW", "2313.TW", "3105.TW", "8086.TW", "6414.TW",
    "3406.TW", "6116.TW", "2409.TW", "3481.TW", "2492.TW", "2404.TW", "8046.TW",
    "3008.TW",
    # 金融
    "2882.TW", "2891.TW", "2881.TW", "2886.TW", "2884.TW", "2892.TW", "5880.TW",
    "2885.TW", "2887.TW", "2801.TW", "2880.TW", "2883.TW", "2890.TW", "5876.TW",
    "2809.TW", "2812.TW", "2834.TW", "2867.TW", "2823.TW", "2820.TW", "5871.TW",
    "2845.TW", "2849.TW",
    # 傳產 / 石化 / 鋼鐵 / 水泥
    "1301.TW", "1303.TW", "1326.TW", "6505.TW", "1101.TW", "1102.TW", "2002.TW",
    "2027.TW", "9958.TW", "2006.TW", "1904.TW", "1907.TW", "1402.TW", "1440.TW",
    "1802.TW",
    # 食品 / 內需 / 零售
    "1216.TW", "1210.TW", "1229.TW", "1224.TW", "2912.TW", "2903.TW", "9945.TW",
    "9910.TW", "9921.TW", "9914.TW", "1476.TW", "2731.TW", "2723.TW", "5904.TW",
    "2915.TW",
    # 航運 / 運輸
    "2603.TW", "2609.TW", "2615.TW", "2606.TW", "2610.TW", "2618.TW", "2207.TW",
    "1319.TW",
    # 生技醫療
    "1795.TW", "6547.TW", "6446.TW", "6491.TW", "1720.TW",
    # 營建
    "2542.TW", "2547.TW", "5522.TW", "2534.TW", "1015.TW",
]
DEFAULT_MARKET_LIST = sorted(set(DEFAULT_MARKET_LIST))  # 去除重複代號

# ------------------------------------------------------------
# 股票代號 -> 中文名稱 對照表
# ------------------------------------------------------------
# 涵蓋預設市場清單中的常見大型股，避免每次掃描都要額外呼叫 API 查名稱（拖慢速度）。
# 若使用者自行新增清單中沒有的代號，get_stock_name() 會自動 fallback 呼叫 yfinance 即時查詢。
TICKER_NAME_MAP = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2412.TW": "中華電",
    "2308.TW": "台達電", "2382.TW": "廣達", "3711.TW": "日月光投控", "2357.TW": "華碩",
    "4938.TW": "和碩", "2379.TW": "瑞昱", "2395.TW": "研華", "3045.TW": "台灣大",
    "4904.TW": "遠傳", "2377.TW": "微星", "2408.TW": "南亞科", "3034.TW": "聯詠",
    "2474.TW": "可成", "2324.TW": "仁寶", "2356.TW": "英業達", "6669.TW": "緯穎",
    "2301.TW": "光寶科", "3231.TW": "緯創", "1503.TW": "士電", "6770.TW": "力積電",
    "2337.TW": "旺宏", "2344.TW": "華邦電", "2360.TW": "致茂", "3702.TW": "大聯大",
    "2347.TW": "聯強", "6176.TW": "瑞儀", "2385.TW": "群光", "3044.TW": "健鼎",
    "6213.TW": "聯茂", "2439.TW": "美律", "2059.TW": "川湖", "3653.TW": "健策",
    "2049.TW": "上銀", "1590.TW": "亞德客-KY", "6274.TW": "台燿", "8299.TW": "群聯",
    "3529.TW": "力旺", "5269.TW": "祥碩", "2338.TW": "光罩", "2481.TW": "強茂",
    "4919.TW": "新唐", "5347.TW": "世界先進", "3532.TW": "台勝科", "6488.TW": "環球晶",
    "3707.TW": "漢磊", "3596.TW": "智易", "2451.TW": "創見", "3037.TW": "欣興",
    "2313.TW": "華通", "3105.TW": "穩懋", "8086.TW": "宏捷科", "6414.TW": "樺漢",
    "3406.TW": "玉晶光", "6116.TW": "彩晶", "2409.TW": "友達", "3481.TW": "群創",
    "2492.TW": "華新科", "2404.TW": "漢唐", "8046.TW": "南電", "3008.TW": "大立光",
    "2882.TW": "國泰金", "2891.TW": "中信金", "2881.TW": "富邦金", "2886.TW": "兆豐金",
    "2884.TW": "玉山金", "2892.TW": "第一金", "5880.TW": "合庫金", "2885.TW": "元大金",
    "2887.TW": "台新金", "2801.TW": "彰銀", "2880.TW": "華南金", "2883.TW": "開發金",
    "2890.TW": "永豐金", "5876.TW": "上海商銀", "2809.TW": "京城銀", "2812.TW": "台中銀",
    "2834.TW": "臺企銀", "2867.TW": "三商壽", "2823.TW": "中壽", "2820.TW": "華票",
    "5871.TW": "中租-KY", "2845.TW": "遠東銀", "2849.TW": "安泰銀",
    "1301.TW": "台塑", "1303.TW": "南亞", "1326.TW": "台化", "6505.TW": "台塑化",
    "1101.TW": "台泥", "1102.TW": "亞泥", "2002.TW": "中鋼", "2027.TW": "大成鋼",
    "9958.TW": "世紀鋼", "2006.TW": "東和鋼鐵", "1904.TW": "正隆", "1907.TW": "永豐餘",
    "1402.TW": "遠東新", "1440.TW": "南紡", "1802.TW": "台玻",
    "1216.TW": "統一", "1210.TW": "大成", "1229.TW": "聯華", "1224.TW": "幸福",
    "2912.TW": "統一超", "2903.TW": "遠百", "9945.TW": "潤泰新", "9910.TW": "豐泰",
    "9921.TW": "巨大", "9914.TW": "美利達", "1476.TW": "儒鴻", "2731.TW": "雄獅",
    "2723.TW": "美食-KY", "5904.TW": "寶雅", "2915.TW": "潤泰全",
    "2603.TW": "長榮", "2609.TW": "陽明", "2615.TW": "萬海", "2606.TW": "裕民",
    "2610.TW": "華航", "2618.TW": "長榮航", "2207.TW": "和泰車", "1319.TW": "東陽",
    "1795.TW": "美時", "6547.TW": "高端疫苗", "6446.TW": "藥華藥", "6491.TW": "晶碩",
    "1720.TW": "生達",
    "2542.TW": "興富發", "2547.TW": "日勝生", "5522.TW": "遠雄", "2534.TW": "宏盛",
}


@st.cache_data(ttl=86400, show_spinner=False)
def get_stock_name(ticker: str) -> str:
    """
    取得股票中文名稱。
    優先查內建對照表（不需額外呼叫 API，速度快）；
    若清單外的代號（如使用者自行新增），才即時向 yfinance 查詢並快取 24 小時；
    查詢失敗則退回顯示代號本身，確保不會因查無名稱而中斷流程。
    """
    if ticker in TICKER_NAME_MAP:
        return TICKER_NAME_MAP[ticker]
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName")
        if name:
            return name
    except Exception:
        pass
    return ticker


# ============================================================
# 技術指標計算函式（分頁一：MA / KD）
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
    rsv = rsv.fillna(50)  # 資料不足時以中性值 50 帶入，避免 NaN 造成後續計算中斷

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
    """計算 10MA / 60MA / KD，回傳新增欄位後的 DataFrame（分頁一使用）"""
    df = df.copy()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df = compute_kd(df)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def fetch_data(ticker: str, period: str = "9mo") -> pd.DataFrame | None:
    """
    透過 yfinance 抓取歷史資料。
    - 使用 9 個月區間，確保 60MA / KD(9,3,3) / MACD 等指標有足夠的回看資料。
    - auto_adjust=False：技術分析慣例使用未還原股價計算均線與各項指標。
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
        return {"股票代號": ticker, "股票名稱": get_stock_name(ticker), "狀態": "⚠️ 資料不足或抓取失敗"}

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
        # 首次觸發買進訊號 -> 記錄進場日低點，作為未來停損判斷基準
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
            st.session_state.holdings.pop(ticker, None)  # 出場後清除持股記錄
        elif not np.isnan(ma10) and price < ma10:
            status = "🟠 停利"
            st.session_state.holdings.pop(ticker, None)  # 出場後清除持股記錄
        else:
            status = "🔵 持有中"

    return {
        "股票代號": ticker,
        "股票名稱": get_stock_name(ticker),
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
    # pandas >= 2.1 將 Styler.applymap 更名為 Styler.map；為相容新舊版本先嘗試 map，失敗再退回 applymap
    if hasattr(styler, "map"):
        return styler.map(color_status, subset=["狀態"])
    return styler.applymap(color_status, subset=["狀態"])


# ============================================================
# 分頁二：多指標技術分析引擎
# ============================================================
# 每個指標代碼對應的下拉選單顯示文字（含判斷邏輯說明，直接取自使用者需求表）
INDICATOR_OPTIONS = {
    "MA":   "MA - 移動平均線 (黃金交叉買進 / MA > 長天期MA)",
    "KD":   "KD - 隨機指標 (K < 20 超賣買進 / K 向上穿越 D)",
    "RSI":  "RSI - 相對強弱指標 (RSI < 30 超賣買進 / 低檔背離)",
    "MACD": "MACD - 指數平滑異同移動平均線 (OSC 由綠翻紅 / DIF 向上穿越 MACD)",
    "BB":   "BB - 布林通道 (股價觸及下軌買進 / 突破上軌加碼)",
    "VOL":  "VOL - 成交量 (放量突破買進 / 價漲量增)",
    "BIAS": "BIAS - 乖離率 (BIAS < -5% 負乖離過大買進)",
    "DMI":  "DMI - 動向指標 (+DI 向上穿越 -DI 買進)",
    "WR":   "WR - 威廉指標 (WR < -80 超賣買進)",
    "OBV":  "OBV - 能量潮指標 (OBV 創新高 / 突破壓力線買進)",
}
BLANK_OPTION = "（不使用）"
LABEL_TO_CODE = {v: k for k, v in INDICATOR_OPTIONS.items()}


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI：以 Wilder's 平滑法計算相對強弱指標"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD：計算 DIF（快慢EMA差）、訊號線（DIF的EMA）與 OSC 柱狀圖"""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    macd_signal = dif.ewm(span=signal, adjust=False).mean()
    df["DIF"] = dif
    df["MACD_SIGNAL"] = macd_signal
    df["OSC"] = dif - macd_signal
    return df


def compute_bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """布林通道：中軌(MA20) ± 2倍標準差"""
    ma = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    df["BB_MID"] = ma
    df["BB_UPPER"] = ma + num_std * std
    df["BB_LOWER"] = ma - num_std * std
    return df


def compute_dmi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """DMI：Wilder's 平滑法計算 +DI / -DI"""
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr

    df["PLUS_DI"] = plus_di.fillna(0)
    df["MINUS_DI"] = minus_di.fillna(0)
    return df


def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """威廉指標：範圍 -100 ~ 0"""
    highest_high = df["High"].rolling(period).max()
    lowest_low = df["Low"].rolling(period).min()
    rng = (highest_high - lowest_low).replace(0, np.nan)
    wr = (highest_high - df["Close"]) / rng * -100
    return wr.fillna(-50)


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """能量潮：依收盤漲跌方向累加成交量"""
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def compute_full_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """分頁二使用：一次計算全部 10 種技術指標所需欄位"""
    df = df.copy()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df = compute_kd(df)
    df["RSI14"] = compute_rsi(df["Close"], 14)
    df = compute_macd(df)
    df = compute_bollinger(df, 20, 2)
    df["VolMA20"] = df["Volume"].rolling(20).mean()
    ma20_price = df["Close"].rolling(20).mean()
    df["BIAS20"] = (df["Close"] - ma20_price) / ma20_price * 100
    df = compute_dmi(df, 14)
    df["WR14"] = compute_williams_r(df, 14)
    df["OBV"] = compute_obv(df)
    return df


def evaluate_indicator(code: str, df: pd.DataFrame) -> tuple[bool, bool, str]:
    """
    對單一指標在最新一筆資料上判斷 買進(buy) / 賣出(sell) 訊號是否成立，
    並回傳文字說明(detail)，供結果表格顯示判斷依據。
    """
    latest, prev = df.iloc[-1], df.iloc[-2]
    buy, sell, detail = False, False, "-"

    if code == "MA":
        golden = prev["MA10"] < prev["MA60"] and latest["MA10"] > latest["MA60"]
        death = prev["MA10"] > prev["MA60"] and latest["MA10"] < latest["MA60"]
        buy = bool(golden or (latest["MA10"] > latest["MA60"]))
        sell = bool(death or (latest["MA10"] < latest["MA60"]))
        detail = f"MA10={latest['MA10']:.2f}/MA60={latest['MA60']:.2f}"

    elif code == "KD":
        cross_up = prev["K"] < prev["D"] and latest["K"] > latest["D"]
        cross_down = prev["K"] > prev["D"] and latest["K"] < latest["D"]
        buy = bool((latest["K"] < 20) or cross_up)
        sell = bool((latest["K"] > 80) or cross_down)
        detail = f"K={latest['K']:.1f}/D={latest['D']:.1f}"

    elif code == "RSI":
        buy = bool(latest["RSI14"] < 30)
        sell = bool(latest["RSI14"] > 70)
        detail = f"RSI={latest['RSI14']:.1f}"

    elif code == "MACD":
        golden = prev["DIF"] < prev["MACD_SIGNAL"] and latest["DIF"] > latest["MACD_SIGNAL"]
        death = prev["DIF"] > prev["MACD_SIGNAL"] and latest["DIF"] < latest["MACD_SIGNAL"]
        buy = bool(golden)
        sell = bool(death)
        detail = f"DIF={latest['DIF']:.2f}/SIG={latest['MACD_SIGNAL']:.2f}/OSC={latest['OSC']:.2f}"

    elif code == "BB":
        buy = bool((latest["Close"] <= latest["BB_LOWER"]) or (latest["Close"] > latest["BB_UPPER"]))
        sell = bool((latest["Close"] >= latest["BB_UPPER"]) or (latest["Close"] < latest["BB_MID"]))
        detail = f"價格={latest['Close']:.2f}/上軌={latest['BB_UPPER']:.2f}/下軌={latest['BB_LOWER']:.2f}"

    elif code == "VOL":
        vol_ma = latest["VolMA20"]
        vol_surge = bool(pd.notna(vol_ma) and vol_ma > 0 and latest["Volume"] > 1.5 * vol_ma)
        price_up = bool(latest["Close"] > prev["Close"])
        buy = bool(vol_surge and price_up)
        sell = bool(vol_surge and (not price_up))
        detail = f"量={latest['Volume']:.0f}/均量={vol_ma:.0f}" if pd.notna(vol_ma) else "資料不足"

    elif code == "BIAS":
        buy = bool(latest["BIAS20"] < -5)
        sell = bool(latest["BIAS20"] > 5)
        detail = f"BIAS={latest['BIAS20']:.2f}%"

    elif code == "DMI":
        cross_up = prev["PLUS_DI"] < prev["MINUS_DI"] and latest["PLUS_DI"] > latest["MINUS_DI"]
        cross_down = prev["PLUS_DI"] > prev["MINUS_DI"] and latest["PLUS_DI"] < latest["MINUS_DI"]
        buy = bool(cross_up)
        sell = bool(cross_down)
        detail = f"+DI={latest['PLUS_DI']:.1f}/-DI={latest['MINUS_DI']:.1f}"

    elif code == "WR":
        buy = bool(latest["WR14"] < -80)
        sell = bool(latest["WR14"] > -20)
        detail = f"WR={latest['WR14']:.1f}"

    elif code == "OBV":
        lookback = df["OBV"].tail(20)
        buy = bool(latest["OBV"] >= lookback.max())
        sell = bool(latest["OBV"] <= lookback.min())
        detail = f"OBV={latest['OBV']:.0f}"

    return buy, sell, detail


def scan_stock_multi(ticker: str, df: pd.DataFrame | None, selected_codes: list, logic: str = "AND") -> dict | None:
    """
    依使用者勾選的多項指標篩選股票：
      logic="AND"：所有已選指標都需同時出現買進訊號才符合
      logic="OR" ：任一已選指標出現買進訊號即符合
    符合條件才回傳結果 dict，否則回傳 None（不列入清單）
    """
    if df is None or len(df) < 60 or not selected_codes:
        return None

    df2 = compute_full_indicators(df)
    if len(df2) < 60:
        return None

    signals = {code: evaluate_indicator(code, df2) for code in selected_codes}
    buy_flags = [signals[c][0] for c in selected_codes]

    is_buy = all(buy_flags) if logic == "AND" else any(buy_flags)
    if not is_buy:
        return None

    latest = df2.iloc[-1]
    entry_price = round(float(latest["Close"]), 2)
    stop_loss = round(float(latest["Low"]), 2)
    triggered = "、".join([c for c in selected_codes if signals[c][0]])
    detail_str = " | ".join([f"{c}:{signals[c][2]}" for c in selected_codes])

    return {
        "股票代號": ticker,
        "股票名稱": get_stock_name(ticker),
        "收盤價": entry_price,
        "建議進場價": entry_price,
        "預估停損價": stop_loss,
        "停損風險(%)": round((entry_price - stop_loss) / entry_price * 100, 2) if entry_price else None,
        "符合條件": triggered,
        "指標明細": detail_str,
        "更新日期": df2.index[-1].strftime("%Y-%m-%d"),
    }


def style_risk_column(df: pd.DataFrame):
    """
    依「停損風險(%)」數值上色（風險越低越綠、越高越紅）。
    不依賴 matplotlib 的 background_gradient，改用純 CSS 手動上色，避免部署環境缺少 matplotlib 而噴錯。
    """
    def color_risk(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if v <= 2:
            return "background-color:#d4edda; color:#155724;"
        elif v <= 4:
            return "background-color:#fff3cd; color:#856404;"
        else:
            return "background-color:#f8d7da; color:#721c24;"
    styler = df.style
    if hasattr(styler, "map"):
        return styler.map(color_risk, subset=["停損風險(%)"])
    return styler.applymap(color_risk, subset=["停損風險(%)"])


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
        避免不必要的 API 呼叫與過度交易訊號干擾。

        **分頁一訊號定義**
        - 🟢 買進：股價 > 60MA 且 KD 低檔(<30)黃金交叉
        - 🟠 停利：收盤價跌破 10MA
        - 🔴 停損：跌破進場日最低點
        - 🔵 持有中：已進場但尚未觸及停利/停損

        **分頁二可選指標**
        MA、KD、RSI、MACD、BB、VOL、BIAS、DMI、WR、OBV，
        最多可選 5 項組合，並可切換 AND / OR 篩選邏輯。
        """
    )
    st.divider()
    st.caption("⚠️ 本工具僅為技術指標運算輔助，不構成任何投資建議。")

tab1, tab2 = st.tabs(["📌 自選股觀察與診斷", "🚀 市場強勢標的推薦"])

# ============================================================
# 分頁一 UI
# ============================================================
with tab1:
    st.subheader("自選股清單管理")

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        new_ticker = st.text_input(
            "輸入股票代號後按下新增（例如 2330.TW、2454.TW）",
            key="new_ticker_input",
            label_visibility="collapsed",
            placeholder="例如：2330.TW",
        )
    with col_btn:
        if st.button("➕ 新增至自選股", use_container_width=True):
            t = new_ticker.strip().upper()
            if not t:
                st.warning("請輸入股票代號")
            elif t in st.session_state.watchlist:
                st.warning(f"{t} 已在清單中")
            else:
                st.session_state.watchlist.append(t)
                st.success(f"已新增 {t}")
                st.rerun()

    if st.session_state.watchlist:
        st.write("**目前自選股：**", "、".join(st.session_state.watchlist))
        to_remove = st.multiselect("選擇要刪除的股票", st.session_state.watchlist, key="remove_select")
        if st.button("🗑️ 刪除選定股票"):
            for t in to_remove:
                st.session_state.watchlist.remove(t)
                st.session_state.holdings.pop(t, None)
            st.rerun()
    else:
        st.info("目前尚無自選股，請於上方新增。")

    st.divider()

    # ------- 一鍵更新與診斷 -------
    if st.button("🔄 一鍵更新與診斷", type="primary", use_container_width=True):
        if not st.session_state.watchlist:
            st.warning("請先新增至少一檔自選股")
        else:
            results = []
            progress = st.progress(0, text="準備抓取資料...")
            total = len(st.session_state.watchlist)
            for i, ticker in enumerate(st.session_state.watchlist):
                progress.progress((i + 1) / total, text=f"處理中：{ticker}")
                df = fetch_data(ticker)
                results.append(diagnose_stock(ticker, df))
            progress.empty()
            st.session_state.watchlist_result = pd.DataFrame(results)
            st.success(f"更新完成，共處理 {total} 檔股票")

    # ------- 結果表格 -------
    if st.session_state.watchlist_result is not None and not st.session_state.watchlist_result.empty:
        st.dataframe(style_watchlist(st.session_state.watchlist_result), use_container_width=True)

        if st.session_state.holdings:
            with st.expander("⚙️ 手動管理持股記錄（如需強制出場或修正進場低點）"):
                reset_ticker = st.selectbox("選擇股票", list(st.session_state.holdings.keys()))
                if st.button("清除此股票的持股記錄"):
                    st.session_state.holdings.pop(reset_ticker, None)
                    st.success(f"已清除 {reset_ticker} 的持股記錄")
                    st.rerun()
    else:
        st.info("請點擊上方「一鍵更新與診斷」以取得最新分析結果。")

# ============================================================
# 分頁二 UI
# ============================================================
with tab2:
    st.subheader("市場觀察清單設定")
    st.caption(
        "預設為台股市值代表性大型股清單（約140檔，涵蓋主要產業族群）。"
        "市值排名每日隨股價變動，如需精確對應「前200大」，"
        "建議至 Goodinfo／證交所查詢最新排名後貼上覆蓋下方清單。"
    )

    market_list_str = st.text_area(
        "市場觀察清單",
        value=",".join(DEFAULT_MARKET_LIST),
        height=120,
        label_visibility="collapsed",
    )
    market_list = [t.strip().upper() for t in market_list_str.split(",") if t.strip()]
    st.caption(f"目前清單共 {len(market_list)} 檔股票")

    st.divider()

    # ------- 多指標篩選條件設定 -------
    st.subheader("篩選條件設定")
    st.caption("最多可選擇 5 項技術指標作為篩選依據，每一列皆可選擇「（不使用）」留空。")

    dropdown_choices = [BLANK_OPTION] + list(INDICATOR_OPTIONS.values())
    selected_labels = []
    for i in range(5):
        choice = st.selectbox(f"條件 {i + 1}", dropdown_choices, key=f"cond_{i}")
        selected_labels.append(choice)

    # 轉換回指標代碼，並排除留白與重複選擇
    selected_indicators = []
    for label in selected_labels:
        if label != BLANK_OPTION:
            code = LABEL_TO_CODE[label]
            if code not in selected_indicators:
                selected_indicators.append(code)

    logic_mode = st.radio(
        "篩選邏輯",
        ["需同時符合所有已選條件（AND）", "符合任一已選條件即可（OR）"],
        horizontal=True,
    )
    logic = "AND" if "AND" in logic_mode else "OR"

    if selected_indicators:
        st.info(f"目前已選擇指標：{'、'.join(selected_indicators)}（邏輯：{logic}）")
    else:
        st.warning("尚未選擇任何指標，請至少選擇 1 項才能進行掃描。")

    st.divider()

    # ------- 開始市場掃描 -------
    if st.button("🚀 開始市場掃描", type="primary", use_container_width=True):
        if not selected_indicators:
            st.warning("請至少選擇一項篩選指標")
        elif not market_list:
            st.warning("市場觀察清單為空，請至少輸入一檔股票代號")
        else:
            results = []
            progress = st.progress(0, text="準備開始掃描...")
            total = len(market_list)
            for i, ticker in enumerate(market_list):
                progress.progress((i + 1) / total, text=f"掃描中：{ticker}")
                df = fetch_data(ticker)
                res = scan_stock_multi(ticker, df, selected_indicators, logic)
                if res:
                    results.append(res)
            progress.empty()
            st.session_state.market_scan_result = pd.DataFrame(results)
            st.success(f"掃描完成，共檢視 {total} 檔股票")

    # ------- 結果表格 -------
    if st.session_state.market_scan_result is not None:
        result_df = st.session_state.market_scan_result
        if result_df.empty:
            st.info("目前無符合篩選條件之標的，可嘗試調整指標組合或切換為 OR 邏輯。")
        else:
            st.success(f"🎯 共篩選出 {len(result_df)} 檔強勢標的")
            st.dataframe(style_risk_column(result_df), use_container_width=True)
    else:
        st.info("請設定篩選條件後，點擊上方「開始市場掃描」以取得符合條件的標的。")
