import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime
import re

# ======================================================
# Black–Scholes
# ======================================================
def bs_call(S: float, K: float, r: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

def bs_put(S: float, K: float, r: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

# ======================================================
# Helpers
# ======================================================
def is_valid_date(s: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", (s or "").strip()))

def compute_hist_vol(hist: pd.DataFrame) -> float:
    close = hist["Close"].dropna()
    log_returns = np.log(close / close.shift(1)).dropna()
    return float(log_returns.std() * np.sqrt(252))

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["strike", "bid", "ask", "lastPrice", "impliedVolatility", "openInterest", "volume"]
    out = df.copy()
    for c in needed:
        if c not in out.columns:
            out[c] = np.nan
    return out

def mid_price_row(row: pd.Series) -> float:
    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    last = row.get("lastPrice", np.nan)
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0 and ask >= bid:
        return float((bid + ask) / 2.0)
    return float(last) if pd.notna(last) else np.nan

def rel_spread(row: pd.Series) -> float:
    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0:
        return np.nan
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return np.nan
    return float((ask - bid) / mid)

def safe_norm(mkt: float, bs: float, eps: float = 1e-8) -> float:
    if pd.isna(mkt) or pd.isna(bs):
        return np.nan
    if abs(bs) < eps:
        return np.nan
    return float((mkt - bs) / bs)

def payoff_at_expiry(opt_type: str, ST: float, K: float) -> float:
    if opt_type.lower() == "call":
        return max(ST - K, 0.0)
    if opt_type.lower() == "put":
        return max(K - ST, 0.0)
    return np.nan

# ======================================================
# Cached fetchers (helps rate-limits)
# ======================================================
@st.cache_data(ttl=1800, show_spinner=False)  # 30 min
def fetch_expirations(ticker: str):
    return list(yf.Ticker(ticker).options)

@st.cache_data(ttl=1800, show_spinner=False)  # 30 min
def fetch_history_1y(ticker: str):
    return yf.Ticker(ticker).history(period="1y")

@st.cache_data(ttl=900, show_spinner=False)  # 15 min
def fetch_option_chain(ticker: str, expiration: str):
    oc = yf.Ticker(ticker).option_chain(expiration)
    return oc.calls, oc.puts

# ======================================================
# UI
# ======================================================
st.set_page_config(page_title="Options Analytics", layout="wide")
st.title("Options Analytics – Market vs Black–Scholes")
st.caption("Base: mid-price (bid/ask) + IV media. Grafico 1 con sola linea S0. Grafico 2 invariato. Simulazioni Buy/Sell senza ricalcolare i grafici.")

with st.sidebar:
    st.header("Input")
    ticker = st.text_input("Ticker (es. SPY, QQQ, IWM, AAPL)", value="SPY").strip().upper()
    r = st.number_input("Risk-free rate (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.005, format="%.3f")

    st.divider()
    st.subheader("Filtro tradabilità")
    max_spread_pct = st.slider("Max spread relativo (%)", min_value=1, max_value=50, value=25, step=1)
    min_oi = st.number_input("Min Open Interest", min_value=0, max_value=500000, value=10, step=1)
    min_vol = st.number_input("Min Volume (opzionale)", min_value=0, max_value=500000, value=0, step=1)

    st.divider()
    st.subheader("Scadenza")
    if "expirations" not in st.session_state:
        st.session_state["expirations"] = []

    if st.button("Carica scadenze", use_container_width=True, disabled=(not ticker)):
        try:
            st.session_state["expirations"] = fetch_expirations(ticker)
            if not st.session_state["expirations"]:
                st.warning("Nessuna scadenza opzioni trovata su Yahoo per questo ticker.")
        except Exception as e:
            st.session_state["expirations"] = []
            st.warning("Errore nel caricamento scadenze (rate limit / ticker non valido).")
            st.caption(f"Dettaglio: {e}")

    if st.session_state["expirations"]:
        expiration = st.selectbox("Seleziona scadenza", st.session_state["expirations"], index=0)
    else:
        expiration = st.text_input("Oppure inserisci scadenza (YYYY-MM-DD)", value="")

    exp_ok = is_valid_date(expiration)

    st.divider()
    st.subheader("Range strike (Grafico 2)")
    range_low = st.slider("Min strike (% di S0)", min_value=10, max_value=95, value=70, step=5)
    range_high = st.slider("Max strike (% di S0)", min_value=105, max_value=300, value=130, step=5)
    n_points = st.slider("Numero punti strike", min_value=10, max_value=200, value=60, step=10)

    st.divider()
    run = st.button("Esegui analisi", type="primary", use_container_width=True, disabled=(not ticker or not exp_ok))

# ======================================================
# Analysis step (only when button pressed)
# Store results in session_state so simulations won't trigger data reload
# ======================================================
def run_analysis():
    hist = fetch_history_1y(ticker)
    if hist.empty:
        raise ValueError("Storico prezzi vuoto.")
    S0 = float(hist["Close"].iloc[-1])
    sigma_hist = compute_hist_vol(hist)

    expiration_date = datetime.strptime(expiration, "%Y-%m-%d")
    T = (expiration_date - datetime.now()).days / 365.0
    if T <= 0:
        raise ValueError("Scadenza nel passato (o oggi).")

    # option chain with fallback
    calls = puts = None
    err_first = None
    exp_effective = expiration

    try:
        calls, puts = fetch_option_chain(ticker, expiration)
    except Exception as e:
        err_first = e
        calls = puts = None

    if calls is None or puts is None:
        exps = st.session_state.get("expirations", []) or []
        for exp_try in exps:
            if exp_try == expiration:
                continue
            try:
                calls, puts = fetch_option_chain(ticker, exp_try)
                exp_effective = exp_try
                expiration_date = datetime.strptime(exp_effective, "%Y-%m-%d")
                T = (expiration_date - datetime.now()).days / 365.0
                if T <= 0:
                    continue
                break
            except Exception:
                continue

    if calls is None or puts is None:
        raise ValueError(f"Impossibile recuperare option chain. Primo errore: {err_first}")

    calls = ensure_columns(calls)
    puts = ensure_columns(puts)

    # tradability + mid
    max_spread = max_spread_pct / 100.0

    def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        out["mid"] = out.apply(mid_price_row, axis=1)
        out["rel_spread"] = out.apply(rel_spread, axis=1)

        cond = (
            (out["bid"] > 0) &
            (out["ask"] > 0) &
            (out["ask"] >= out["bid"]) &
            (out["rel_spread"].notna()) &
            (out["rel_spread"] <= max_spread) &
            (out["openInterest"].fillna(0) >= min_oi)
        )
        if min_vol and min_vol > 0:
            cond = cond & (out["volume"].fillna(0) >= min_vol)

        out = out.loc[cond].copy()
        out = out.dropna(subset=["mid"])
        return out

    calls_f = apply_filters(calls)
    puts_f = apply_filters(puts)

    if calls_f.empty and puts_f.empty:
        raise ValueError("Dopo i filtri di tradabilità non rimane nessuna opzione. Riduci spread/OI/volume.")

    # IV media (separata call/put) sulla chain filtrata
    iv_call_mean = float(calls_f["impliedVolatility"].dropna().mean()) if not calls_f["impliedVolatility"].dropna().empty else np.nan
    iv_put_mean = float(puts_f["impliedVolatility"].dropna().mean()) if not puts_f["impliedVolatility"].dropna().empty else np.nan

    # build df for graph1 + simulator: union strikes (filtrati)
    calls_by_strike = calls_f.drop_duplicates(subset=["strike"]).set_index("strike")
    puts_by_strike = puts_f.drop_duplicates(subset=["strike"]).set_index("strike")
    strike_union = np.union1d(calls_by_strike.index.values, puts_by_strike.index.values)

    rows = []
    for K in strike_union:
        K = float(K)

        # call
        if K in calls_by_strike.index:
            rc = calls_by_strike.loc[K]
            mkt_c = float(rc["mid"])
            bs_c = bs_call(S0, K, r, T, iv_call_mean) if pd.notna(iv_call_mean) and iv_call_mean > 0 else np.nan
            norm_c = safe_norm(mkt_c, bs_c)
        else:
            mkt_c = bs_c = norm_c = np.nan

        # put
        if K in puts_by_strike.index:
            rp = puts_by_strike.loc[K]
            mkt_p = float(rp["mid"])
            bs_p = bs_put(S0, K, r, T, iv_put_mean) if pd.notna(iv_put_mean) and iv_put_mean > 0 else np.nan
            norm_p = safe_norm(mkt_p, bs_p)
        else:
            mkt_p = bs_p = norm_p = np.nan

        rows.append({
            "Strike": K,
            "Market Call (mid)": mkt_c,
            "BS Call (IV mean)": bs_c,
            "Norm (C)": norm_c,
            "Market Put (mid)": mkt_p,
            "BS Put (IV mean)": bs_p,
            "Norm (P)": norm_p,
        })

    df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)

    # Graph2 data (kept as: Market(mid) vs BS (hist sigma vs IV mean) on strike range)
    low = S0 * (range_low / 100.0)
    high = S0 * (range_high / 100.0)
    strike_range = np.linspace(low, high, n_points)

    df2_rows = []
    if not calls_f.empty and not puts_f.empty:
        for K in strike_range:
            K = float(K)

            call_idx = (calls_f["strike"] - K).abs().idxmin()
            put_idx = (puts_f["strike"] - K).abs().idxmin()
            row_c = calls_f.loc[call_idx]
            row_p = puts_f.loc[put_idx]

            mkt_call = float(row_c["mid"])
            mkt_put = float(row_p["mid"])

            bs_call_ivmean = bs_call(S0, K, r, T, iv_call_mean) if pd.notna(iv_call_mean) and iv_call_mean > 0 else np.nan
            bs_put_ivmean = bs_put(S0, K, r, T, iv_put_mean) if pd.notna(iv_put_mean) and iv_put_mean > 0 else np.nan

            bs_call_hist = bs_call(S0, K, r, T, sigma_hist)
            bs_put_hist = bs_put(S0, K, r, T, sigma_hist)

            df2_rows.append({
                "Strike": K,
                "Market Call (mid)": mkt_call,
                "BS Call (IV mean)": bs_call_ivmean,
                "BS Call (Hist σ)": bs_call_hist,
                "Market Put (mid)": mkt_put,
                "BS Put (IV mean)": bs_put_ivmean,
                "BS Put (Hist σ)": bs_put_hist,
            })

    df2 = pd.DataFrame(df2_rows)

    return {
        "ticker": ticker,
        "expiration": exp_effective,
        "S0": S0,
        "sigma_hist": sigma_hist,
        "T": T,
        "df": df,
        "df2": df2,
        "iv_call_mean": iv_call_mean,
        "iv_put_mean": iv_put_mean,
        "strikes_call": df.loc[df["Market Call (mid)"].notna(), "Strike"].values,
        "strikes_put": df.loc[df["Market Put (mid)"].notna(), "Strike"].values,
        "filters": {"max_spread_pct": max_spread_pct, "min_oi": min_oi, "min_vol": min_vol},
    }

if run:
    try:
        st.session_state["analysis_data"] = run_analysis()
        st.session_state["analysis_ready"] = True
    except Exception as e:
        st.session_state["analysis_ready"] = False
        st.error(str(e))

if not st.session_state.get("analysis_ready", False):
    st.info("Esegui l'analisi per generare i grafici. La sezione simulazione funzionerà senza ricalcolare i grafici.")
    st.stop()

A = st.session_state["analysis_data"]

# ======================================================
# Header metrics
# ======================================================
colA, colB, colC, colD = st.columns(4)
colA.metric("S0 (ultimo close)", f"{A['S0']:,.2f}")
colB.metric("σ storica (1y)", f"{A['sigma_hist']:.4f}")
colC.metric("T (anni)", f"{A['T']:.4f}")
colD.metric("Scadenza effettiva", A["expiration"])

st.caption(
    f"IV media: Call={A['iv_call_mean']:.4f} | Put={A['iv_put_mean']:.4f}  |  "
    f"Filtro tradabilità: spread≤{A['filters']['max_spread_pct']}%, OI≥{A['filters']['min_oi']}"
    + (f", volume≥{A['filters']['min_vol']}" if A['filters']['min_vol'] and A['filters']['min_vol'] > 0 else "")
    + "."
)

# ======================================================
# Table
# ======================================================
st.subheader("Tabella: Market(mid) vs BS (IV media) + Normalizzazione (Market−BS)/BS")
st.dataframe(A["df"], use_container_width=True)

# ======================================================
# Graph 1: ONLY S0 vertical marker
# ======================================================
st.subheader("Grafico 1: Normalizzato (Market − BS) / BS (IV media)")
df_plot = A["df"]

fig1 = plt.figure(figsize=(14, 6))
plt.plot(df_plot["Strike"], df_plot["Norm (C)"], marker="o", linewidth=1, label="Call: (M-BS)/BS")
plt.plot(df_plot["Strike"], df_plot["Norm (P)"], marker="o", linewidth=1, label="Put:  (M-BS)/BS")
plt.axhline(0, linestyle="--")
plt.axvline(A["S0"], linestyle="--", linewidth=1, label="S0")
plt.title("Normalized Mispricing vs Black–Scholes (IV mean)")
plt.xlabel("Strike")
plt.ylabel("(Market - BS) / BS")
plt.grid(True)
plt.legend()
plt.tight_layout()
st.pyplot(fig1)

# ======================================================
# Graph 2: kept intact (Market vs BS: Hist σ vs IV mean)
# ======================================================
st.subheader("Grafico 2: Market vs BS (σ storica vs IV media) su range di strike")
df2 = A["df2"]

if df2 is None or df2.empty:
    st.warning("Impossibile costruire il Grafico 2 (range strike o chain non validi dopo filtri).")
else:
    fig2 = plt.figure(figsize=(14, 7))
    plt.plot(df2["Strike"], df2["Market Call (mid)"], marker="o", label="Market Call (mid)")
    plt.plot(df2["Strike"], df2["BS Call (IV mean)"], linestyle="--", label="BS Call (IV mean)")
    plt.plot(df2["Strike"], df2["BS Call (Hist σ)"], linestyle="dotted", label="BS Call (Hist σ)")

    plt.plot(df2["Strike"], df2["Market Put (mid)"], marker="o", label="Market Put (mid)")
    plt.plot(df2["Strike"], df2["BS Put (IV mean)"], linestyle="--", label="BS Put (IV mean)")
    plt.plot(df2["Strike"], df2["BS Put (Hist σ)"], linestyle="dotted", label="BS Put (Hist σ)")

    plt.title("Market(mid) vs Black–Scholes Option Prices")
    plt.xlabel("Strike")
    plt.ylabel("Option Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    with st.expander("Mostra tabella Grafico 2"):
        st.dataframe(df2, use_container_width=True)

# ======================================================
# Simulation section (Buy/Sell Call/Put) WITHOUT new analysis
# ======================================================
st.divider()
st.subheader("Simulazioni (Buy/Sell Call/Put) — senza ricalcolare i grafici")

sim_col1, sim_col2, sim_col3, sim_col4 = st.columns([1.2, 1.1, 1.3, 1.2])
side = sim_col1.selectbox("Operazione", ["Buy", "Sell"], index=0, key="sim_side")
opt_type = sim_col2.selectbox("Tipo", ["Call", "Put"], index=0, key="sim_type")

strikes_sim = A["strikes_call"] if opt_type == "Call" else A["strikes_put"]
if len(strikes_sim) == 0:
    st.warning("Nessuno strike disponibile (dopo filtri) per il tipo selezionato.")
    st.stop()

default_idx = int(np.argmin(np.abs(strikes_sim - A["S0"])))
K_sel = float(sim_col3.selectbox("Strike", strikes_sim, index=default_idx, key="sim_strike"))
qty = int(sim_col4.number_input("Quantità contratti", min_value=1, value=1, step=1, key="sim_qty"))

colm1, colm2 = st.columns([1.2, 1.0])
mult = int(colm1.number_input("Moltiplicatore (shares/contratto)", min_value=1, value=100, step=1, key="sim_mult"))
ST = float(colm2.number_input("Prezzo sottostante a scadenza (ST)", min_value=0.0, value=float(A["S0"]), step=float(max(1.0, A["S0"] * 0.01)), key="sim_ST"))

row_sel = A["df"].loc[A["df"]["Strike"] == K_sel].iloc[0]
premium = float(row_sel["Market Call (mid)"] if opt_type == "Call" else row_sel["Market Put (mid)"])
pay_share = payoff_at_expiry(opt_type, ST, K_sel)

if side == "Buy":
    pnl_share = pay_share - premium
    premium_cashflow = -premium
else:
    pnl_share = premium - pay_share
    premium_cashflow = premium

pnl_total = pnl_share * mult * qty
premium_total = premium_cashflow * mult * qty
pay_total = pay_share * mult * qty

notional_now = A["S0"] * mult * qty
notional_ST = ST * mult * qty

mA, mB, mC, mD = st.columns(4)
mA.metric("Premium per opzione (mid)", f"{premium:,.4f}")
mB.metric("Cashflow premio (t0)", f"{premium_total:,.2f}")
mC.metric("Payoff totale (scadenza)", f"{pay_total:,.2f}")
mD.metric("PnL totale (scadenza)", f"{pnl_total:,.2f}")

nA, nB = st.columns(2)
nA.metric("Nozionale equivalente (oggi)", f"{notional_now:,.2f}")
nB.metric("Nozionale equivalente (a scadenza)", f"{notional_ST:,.2f}")

st.caption("Curva PnL totale a scadenza al variare di ST (strike e quantità fissi).")

st_min = max(0.0, A["S0"] * 0.5)
st_max = A["S0"] * 1.5 if A["S0"] > 0 else 1.0
ST_grid = np.linspace(st_min, st_max, 160)

pnl_curve = []
for stx in ST_grid:
    pay = payoff_at_expiry(opt_type, float(stx), K_sel)
    if side == "Buy":
        pnl_curve.append((pay - premium) * mult * qty)
    else:
        pnl_curve.append((premium - pay) * mult * qty)

fig3 = plt.figure(figsize=(14, 5))
plt.plot(ST_grid, pnl_curve)
plt.axhline(0, linestyle="--")
plt.axvline(A["S0"], linestyle="--", linewidth=1, label="S0")
plt.axvline(K_sel, linestyle=":", linewidth=1, label="Strike")
plt.title("PnL totale a scadenza vs ST")
plt.xlabel("ST (prezzo sottostante a scadenza)")
plt.ylabel("PnL totale")
plt.grid(True)
plt.legend()
plt.tight_layout()
st.pyplot(fig3)

st.caption(
    "Nota: Streamlit rilancia lo script a ogni modifica dei controlli, "
    "ma qui non rifacciamo le chiamate a Yahoo né ricalcoliamo i grafici: "
    "riusiamo i risultati salvati in session_state (analisi)."
)
