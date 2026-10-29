
import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(
    page_title="AI Trading Agent",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Trading Agent")
st.caption("Analysis + Paper Trading Only | Real orders disabled")

st.warning(
    "यह learning prototype है। Signals की गारंटी नहीं है। "
    "कोई broker connection या real order execution नहीं है।"
)

MARKETS = {
    "Gold Futures": "GC=F",
    "EUR/USD Forex": "EURUSD=X",
    "Reliance (NSE)": "RELIANCE.NS",
    "Nifty 50": "^NSEI",
}

market = st.selectbox("Market चुनें", list(MARKETS.keys()))
interval_label = st.selectbox(
    "Timeframe चुनें",
    ["5 मिनट", "15 मिनट", "1 घंटा", "Daily"]
)

interval_map = {
    "5 मिनट": "5m",
    "15 मिनट": "15m",
    "1 घंटा": "60m",
    "Daily": "1d",
}
interval = interval_map[interval_label]
symbol = MARKETS[market]

@st.cache_data(ttl=60)
def load_data(ticker, timeframe):
    period = "5d" if timeframe == "5m" else (
        "60d" if timeframe in ["15m", "60m"] else "1y"
    )
    df = yf.download(
        ticker,
        period=period,
        interval=timeframe,
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Newer yfinance versions may return MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.loc[:, ~df.columns.duplicated()].copy()
    required = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).copy()
    if len(df) < 55:
        return pd.DataFrame()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    change = df["Close"].diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI"] = 100 - (100 / (1 + rs))

    previous_close = df["Close"].shift(1)
    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - previous_close).abs(),
        (df["Low"] - previous_close).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = true_range.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    return df.dropna(subset=["EMA20", "EMA50", "RSI", "ATR"])


if "paper_trades" not in st.session_state:
    st.session_state.paper_trades = []

df = load_data(symbol, interval)

if df.empty:
    st.error(
        "इस market/timeframe का पर्याप्त data नहीं मिला। "
        "Daily या दूसरा market चुनकर फिर कोशिश करें।"
    )
    st.stop()

last = df.iloc[-1]
price = float(last["Close"])
ema20 = float(last["EMA20"])
ema50 = float(last["EMA50"])
rsi = float(last["RSI"])
atr = float(last["ATR"])

# Simple rule-based signal, not a trained AI model
signal = "WAIT"
reason = "सभी नियम एक दिशा में स्पष्ट नहीं हैं।"

if ema20 > ema50 and 50 <= rsi <= 65:
    signal = "BUY"
    reason = "EMA20 > EMA50 और RSI 50–65 के बीच है।"
elif ema20 < ema50 and 35 <= rsi <= 50:
    signal = "SELL"
    reason = "EMA20 < EMA50 और RSI 35–50 के बीच है।"

if signal == "BUY":
    stop_loss = price - 1.5 * atr
    target = price + 3 * atr
elif signal == "SELL":
    stop_loss = price + 1.5 * atr
    target = price - 3 * atr
else:
    stop_loss = None
    target = None

st.subheader(f"{market} — {symbol}")

a, b, c = st.columns(3)
a.metric("Last available price", f"{price:,.4f}")
b.metric("Rule-based signal", signal)
c.metric("RSI (14)", f"{rsi:.2f}")

st.caption(
    f"EMA20: {ema20:.4f} | EMA50: {ema50:.4f} | ATR: {atr:.4f}"
)
st.write("**Analysis:**", reason)

if stop_loss is not None:
    x, y = st.columns(2)
    x.metric("Illustrative Stop Loss", f"{stop_loss:,.4f}")
    y.metric("Illustrative Target", f"{target:,.4f}")

st.subheader("Price chart")
st.line_chart(df[["Close", "EMA20", "EMA50"]].tail(150))

st.caption(
    "Data Yahoo Finance से आता है; यह broker का live executable price "
    "नहीं माना जाना चाहिए। आखिरी candle अधूरी हो सकती है।"
)

st.divider()
st.subheader("🧪 Paper Trading")
st.write("यहाँ दर्ज trades केवल virtual हैं; कोई असली order नहीं जाएगा।")

quantity = st.number_input(
    "Virtual quantity / units",
    min_value=0.01,
    value=1.0,
    step=1.0
)

if signal in ["BUY", "SELL"]:
    if st.button(f"Open PAPER {signal} trade"):
        st.session_state.paper_trades.append({
            "Market": market,
            "Symbol": symbol,
            "Side": signal,
            "Entry": price,
            "Quantity": float(quantity),
            "Status": "OPEN",
            "Exit": None,
            "P/L": None
        })
        st.success("Virtual trade दर्ज हो गया। Real order नहीं भेजा गया।")
else:
    st.info("अभी signal WAIT है; नया paper trade नहीं खोला जा सकता।")

trades = st.session_state.paper_trades

if trades:
    st.subheader("Paper trade history")
    st.dataframe(pd.DataFrame(trades), use_container_width=True)

    open_indices = [
        i for i, trade in enumerate(trades)
        if trade["Status"] == "OPEN"
    ]

    if open_indices:
        labels = {
            i: (
                f"#{i+1} {trades[i]['Symbol']} "
                f"{trades[i]['Side']} @ {trades[i]['Entry']:.4f}"
            )
            for i in open_indices
        }
        chosen = st.selectbox(
            "Close करने के लिए open virtual trade चुनें",
            open_indices,
            format_func=lambda i: labels[i]
        )

        if st.button("Close selected PAPER trade at displayed price"):
            trade = trades[chosen]
            exit_price = price
            if trade["Side"] == "BUY":
                pnl = (exit_price - trade["Entry"]) * trade["Quantity"]
            else:
                pnl = (trade["Entry"] - exit_price) * trade["Quantity"]

            trade["Exit"] = exit_price
            trade["P/L"] = round(pnl, 4)
            trade["Status"] = "CLOSED"
            st.success(f"Virtual trade बंद हुआ। Simulated P/L: {pnl:.4f}")
            st.rerun()

    total_closed_pnl = sum(
        t["P/L"] for t in trades
        if t["Status"] == "CLOSED" and t["P/L"] is not None
    )
    st.metric("Closed paper trades: simulated P/L", f"{total_closed_pnl:.4f}")
else:
    st.info("अभी कोई paper trade दर्ज नहीं है।")

st.caption(
    "नोट: Paper trade history इस app session में रखी जाती है; "
    "app restart होने पर history मिट सकती है। P/L में fees, spread और slippage शामिल नहीं हैं।"
)
