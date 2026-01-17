
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime
import re

# ===============================
# Black–Scholes
# ===============================
def bs_call(S, K, r, T, sigma):
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, r, T, sigma):
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ===============================
# Helpers
# ===============================
def is_valid_date(s):
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", (s or "").strip()))

def hist_vol(hist):
    lr = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    return lr.std() * np.sqrt(252)

def mid_price(row):
    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
        return (bid + ask) / 2
    return np.nan

def tradable(row, max_spread=0.25, min_oi=10):
    bid = row.get("bid", 0)
    ask = row.get("ask", 0)
    if bid <= 0 or ask <= 0:
        return False
    spread = (ask - bid) / ((ask + bid) / 2)
    if spread > max_spread:
        return False
    if row.get("openInterest", 0) < min_oi:
        return False
    return True

def payoff(opt_type, ST, K):
    return max(ST - K, 0) if opt_type == "Call" else max(K - ST, 0)

# ===============================
# UI
# ===============================
st.set_page_config("Options Analytics", layout="wide")
st.title("Options Analytics – Market vs Black-Scholes")

with st.sidebar:
    ticker = st.text_input("Ticker (SPY, QQQ, IWM)", "SPY").upper()
    r = st.number_input("Risk-free rate", 0.0, 1.0, 0.05, 0.005)
    max_spread = st.slider("Max bid-ask spread %", 5, 50, 25) / 100
    min_oi = st.number_input("Min Open Interest", 1, 500, 10)

    load = st.button("Carica scadenze")
    if "expirations" not in st.session_state:
        st.session_state.expirations = []

    if load and ticker:
        st.session_state.expirations = yf.Ticker(ticker).options

    if st.session_state.expirations:
        expiration = st.selectbox("Scadenza", st.session_state.expirations)
    else:
        expiration = st.text_input("Scadenza (YYYY-MM-DD)", "")

    run = st.button("Esegui analisi", type="primary")

if not run or not is_valid_date(expiration):
    st.stop()

# ===============================
# Data
# ===============================
t = yf.Ticker(ticker)
hist = t.history(period="1y")
S0 = hist["Close"].iloc[-1]
sigma_hist = hist_vol(hist)

T = (datetime.strptime(expiration, "%Y-%m-%d") - datetime.now()).days / 365
calls, puts = t.option_chain(expiration)

calls = calls[calls.apply(tradable, axis=1, max_spread=max_spread, min_oi=min_oi)]
puts  = puts[puts.apply(tradable, axis=1, max_spread=max_spread, min_oi=min_oi)]

calls["mid"] = calls.apply(mid_price, axis=1)
puts["mid"]  = puts.apply(mid_price, axis=1)

# ===============================
# Market vs BS normalized
# ===============================
strikes = np.union1d(calls["strike"], puts["strike"])
rows = []

for K in strikes:
    c = calls[calls["strike"] == K]
    p = puts[puts["strike"] == K]

    if not c.empty:
        ivc = c.iloc[0]["impliedVolatility"]
        mktc = c.iloc[0]["mid"]
        bsc = bs_call(S0, K, r, T, ivc)
        ndc = (mktc - bsc) / bsc if bsc > 0 else np.nan
    else:
        ndc = np.nan

    if not p.empty:
        ivp = p.iloc[0]["impliedVolatility"]
        mktp = p.iloc[0]["mid"]
        bsp = bs_put(S0, K, r, T, ivp)
        ndp = (mktp - bsp) / bsp if bsp > 0 else np.nan
    else:
        ndp = np.nan

    rows.append([K, ndc, ndp])

df = pd.DataFrame(rows, columns=["Strike", "Call_norm", "Put_norm"])

fig = plt.figure(figsize=(14, 6))
plt.plot(df["Strike"], df["Call_norm"], label="Call")
plt.plot(df["Strike"], df["Put_norm"], label="Put")
plt.axhline(0, ls="--")
plt.axvline(S0, ls="--")
plt.text(S0, plt.ylim()[1] * 0.9, f"S0={S0:.2f}", rotation=90)
plt.grid(True)
plt.legend()
plt.title("Normalized (Market − BS) / BS")
st.pyplot(fig)

# ===============================
# Simulation
# ===============================
st.divider()
st.subheader("Simulazione operazione")

col1, col2, col3, col4 = st.columns(4)
opt_type = col1.selectbox("Tipo", ["Call", "Put"])
K_sel = col2.selectbox("Strike", df["Strike"])
qty = col3.number_input("Contratti", 1, 100, 1)
mult = col4.number_input("Moltiplicatore", 100)

ST = st.number_input("Prezzo a scadenza ST", value=float(S0))

row = calls[calls["strike"] == K_sel] if opt_type == "Call" else puts[puts["strike"] == K_sel]
premium = row.iloc[0]["mid"]

pay = payoff(opt_type, ST, K_sel)
pnl = (pay - premium) * qty * mult
notional_now = S0 * qty * mult

a, b, c = st.columns(3)
a.metric("Costo premio", f"{premium * qty * mult:,.2f}")
b.metric("PnL a scadenza", f"{pnl:,.2f}")
c.metric("Nozionale equivalente", f"{notional_now:,.2f}")

ST_grid = np.linspace(S0 * 0.5, S0 * 1.5, 120)
curve = [(payoff(opt_type, x, K_sel) - premium) * qty * mult for x in ST_grid]

fig2 = plt.figure(figsize=(14, 4))
plt.plot(ST_grid, curve)
plt.axhline(0, ls="--")
plt.axvline(S0, ls="--")
plt.axvline(K_sel, ls=":")
plt.grid(True)
plt.title("PnL a scadenza")
st.pyplot(fig2)
