import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime
import streamlit as st

# ----------------------------
# Black-Scholes
# ----------------------------
def black_scholes_call(S, K, r, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, r, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ----------------------------
# Data fetch helpers (cached)
# ----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_expirations(ticker_symbol: str):
    t = yf.Ticker(ticker_symbol)
    return t.options  # list of "YYYY-MM-DD"

@st.cache_data(ttl=3600, show_spinner=False)
def get_history_1y(ticker_symbol: str):
    t = yf.Ticker(ticker_symbol)
    hist = t.history(period="1y")
    return hist

@st.cache_data(ttl=3600, show_spinner=False)
def get_option_chain(ticker_symbol: str, expiration: str):
    t = yf.Ticker(ticker_symbol)
    oc = t.option_chain(expiration)
    return oc.calls, oc.puts

def compute_hist_vol(hist: pd.DataFrame):
    # annualized stdev of daily log returns
    close = hist["Close"].dropna()
    log_returns = np.log(close / close.shift(1)).dropna()
    return float(log_returns.std() * np.sqrt(250))

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Option Hedging", layout="wide")
st.title("Option Hedging")
st.caption("Inserisci un ticker, scegli una scadenza, e confronta prezzi market vs Black-Scholes.")

with st.sidebar:
    st.header("Input")
    ticker_symbol = st.text_input("Ticker (es. MSTR, AAPL, TSLA)", value="MSTR").strip().upper()
    r = st.number_input("Risk-free rate (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.005, format="%.3f")
    hist_period_note = st.caption("Volatilità storica stimata su 1 anno (log-returns).")

    st.divider()
    st.subheader("Range di strike per Grafico 2 ")
    range_low = st.slider("Min strike (% di S0)", min_value=10, max_value=95, value=70, step=5)
    range_high = st.slider("Max strike (% di S0)", min_value=105, max_value=300, value=130, step=5)
    n_points = st.slider("Numero punti strike", min_value=10, max_value=200, value=60, step=10)

# Validate ticker & load expirations
if not ticker_symbol:
    st.stop()

try:
    expirations = get_expirations(ticker_symbol)
except Exception as e:
    st.error(f"Errore nel recupero scadenze per {ticker_symbol}: {e}")
    st.stop()

if not expirations:
    st.warning("Nessuna scadenza opzioni trovata su Yahoo Finance per questo ticker.")
    st.stop()

expiration = st.selectbox("Seleziona scadenza (Yahoo Finance)", options=expirations, index=0)

# Load history and compute S0, sigma
try:
    hist = get_history_1y(ticker_symbol)
    if hist.empty:
        st.warning("Storico prezzi vuoto. Prova un altro ticker.")
        st.stop()

    S0 = float(hist["Close"].iloc[-1])
    sigma_annual = compute_hist_vol(hist)
except Exception as e:
    st.error(f"Errore nel calcolo storico/volatilità: {e}")
    st.stop()

# Time to expiration
today = datetime.now()
expiration_date = datetime.strptime(expiration, "%Y-%m-%d")
T = (expiration_date - today).days / 365.0

colA, colB, colC, colD = st.columns(4)
colA.metric("S0 (ultimo close)", f"{S0:,.2f}")
colB.metric("σ annual (storica)", f"{sigma_annual:.4f}")
colC.metric("T (anni)", f"{T:.4f}")
colD.metric("Scadenza", expiration)

if T <= 0:
    st.warning("La scadenza selezionata risulta nel passato (o oggi). Scegli una scadenza futura.")
    st.stop()

# Load option chain
try:
    calls, puts = get_option_chain(ticker_symbol, expiration)
except Exception as e:
    st.error(f"Errore nel recupero option chain: {e}")
    st.stop()

if calls.empty and puts.empty:
    st.warning("Option chain vuota.")
    st.stop()

# ----------------------------
# TABLE 1: union strikes, market vs BS (hist sigma)
# ----------------------------
st.subheader("Tabella: Market vs Black-Scholes (σ storica)")
strike_union = np.union1d(calls["strike"].values, puts["strike"].values)

results = []
for K in strike_union:
    call_bs = black_scholes_call(S0, K, r, T, sigma_annual)
    put_bs  = black_scholes_put(S0, K, r, T, sigma_annual)

    call_mkt = calls.loc[calls["strike"] == K, "lastPrice"]
    put_mkt  = puts.loc[puts["strike"] == K, "lastPrice"]

    results.append({
        "Strike": float(K),
        "BS Call (Hist σ)": float(call_bs) if pd.notna(call_bs) else np.nan,
        "Market Call": float(call_mkt.values[0]) if not call_mkt.empty else np.nan,
        "BS Put (Hist σ)": float(put_bs) if pd.notna(put_bs) else np.nan,
        "Market Put": float(put_mkt.values[0]) if not put_mkt.empty else np.nan,
    })

df1 = pd.DataFrame(results).sort_values("Strike").reset_index(drop=True)
st.dataframe(df1, use_container_width=True)

# ----------------------------
# PLOT 1: differences Market - BS
# ----------------------------
st.subheader("Grafico 1: Differenza (Market − Black-Scholes) usando σ storica")

df_plot = df1.dropna(subset=["Market Call"]).copy()
df_plot["Call_Diff"] = df_plot["Market Call"] - df_plot["BS Call (Hist σ)"]
df_plot["Put_Diff"]  = df_plot["Market Put"]  - df_plot["BS Put (Hist σ)"]

fig1 = plt.figure(figsize=(14, 6))
plt.bar(df_plot["Strike"], df_plot["Call_Diff"], width=2.0, label="Call (Market - BS)")
plt.bar(df_plot["Strike"], df_plot["Put_Diff"],  width=2.0, label="Put (Market - BS)")
plt.axhline(0, linestyle="--")
plt.title("Market vs Black-Scholes — Price Difference")
plt.xlabel("Strike")
plt.ylabel("Difference (Market - BS)")
plt.legend()
plt.tight_layout()
st.pyplot(fig1)

# ----------------------------
# IV average + Plot 2: market vs BS (hist vs IV avg) on custom strike range
# ----------------------------
st.subheader("Grafico 2: Market vs BS (σ storica vs IV media) su range di strike")

# implied volatility average (calls + puts)
iv_series = pd.concat([calls.get("impliedVolatility", pd.Series(dtype=float)),
                       puts.get("impliedVolatility", pd.Series(dtype=float))], axis=0)
iv_media = float(iv_series.dropna().mean()) if not iv_series.dropna().empty else np.nan

st.write(f"IV media (calls+puts): **{iv_media:.4f}**" if pd.notna(iv_media) else "IV media non disponibile (NaN)")

low = S0 * (range_low / 100.0)
high = S0 * (range_high / 100.0)
strike_range = np.linspace(low, high, n_points)

results2 = []
# per ogni strike “target” troviamo la riga più vicina in calls/puts
for K in strike_range:
    # se calls/puts vuote, skip
    if calls.empty or puts.empty:
        continue

    call_idx = (calls["strike"] - K).abs().idxmin()
    put_idx  = (puts["strike"]  - K).abs().idxmin()

    row_call = calls.loc[call_idx]
    row_put  = puts.loc[put_idx]

    bs_call_hist = black_scholes_call(S0, K, r, T, sigma_annual)
    bs_put_hist  = black_scholes_put(S0, K, r, T, sigma_annual)

    bs_call_iv = black_scholes_call(S0, K, r, T, iv_media) if pd.notna(iv_media) else np.nan
    bs_put_iv  = black_scholes_put(S0, K, r, T, iv_media)  if pd.notna(iv_media) else np.nan

    results2.append({
        "Strike": float(K),
        "Market_Call": float(row_call.get("lastPrice", np.nan)),
        "BS_Call_Hist": float(bs_call_hist) if pd.notna(bs_call_hist) else np.nan,
        "BS_Call_IVavg": float(bs_call_iv) if pd.notna(bs_call_iv) else np.nan,
        "Market_Put": float(row_put.get("lastPrice", np.nan)),
        "BS_Put_Hist": float(bs_put_hist) if pd.notna(bs_put_hist) else np.nan,
        "BS_Put_IVavg": float(bs_put_iv) if pd.notna(bs_put_iv) else np.nan,
    })

df2 = pd.DataFrame(results2)

if df2.empty:
    st.warning("Impossibile costruire df2 (range strike o chain non validi).")
else:
    fig2 = plt.figure(figsize=(14, 7))
    plt.plot(df2["Strike"], df2["Market_Call"], marker="o", label="Market Call")
    plt.plot(df2["Strike"], df2["BS_Call_IVavg"], linestyle="--", label="BS Call (IV avg)")
    plt.plot(df2["Strike"], df2["BS_Call_Hist"], linestyle="dotted", label="BS Call (Hist σ)")

    plt.plot(df2["Strike"], df2["Market_Put"], marker="o", label="Market Put")
    plt.plot(df2["Strike"], df2["BS_Put_IVavg"], linestyle="--", label="BS Put (IV avg)")
    plt.plot(df2["Strike"], df2["BS_Put_Hist"], linestyle="dotted", label="BS Put (Hist σ)")

    plt.title("Market vs Black-Scholes Option Prices")
    plt.xlabel("Strike")
    plt.ylabel("Option Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    with st.expander("Mostra tabella df2"):
        st.dataframe(df2, use_container_width=True)

st.caption("Nota: i dati opzioni Yahoo possono avere lastPrice obsoleti o spread elevati; per confronti seri, spesso conviene usare mid-price (bid/ask).")
