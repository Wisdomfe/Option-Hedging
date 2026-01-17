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

def mid_price_row(row: pd.Series) -> float:
    """
    Prefer bid/ask mid if usable; otherwise fallback to lastPrice.
    """
    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    last = row.get("lastPrice", np.nan)

    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0 and ask >= bid:
        return float((bid + ask) / 2.0)

    # fallback
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

def payoff_at_expiry(opt_type: str, ST: float, K: float) -> float:
    if opt_type.lower() == "call":
        return max(ST - K, 0.0)
    if opt_type.lower() == "put":
        return max(K - ST, 0.0)
    return np.nan

def safe_norm(mkt: float, bs: float, eps: float = 1e-8) -> float:
    """
    (Market - BS) / BS, guarded when BS ~ 0.
    """
    if pd.isna(mkt) or pd.isna(bs):
        return np.nan
    if abs(bs) < eps:
        return np.nan
    return float((mkt - bs) / bs)

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yahoo/yfinance sometimes returns missing columns for some tickers/expiries.
    Make sure required columns exist.
    """
    needed = ["strike", "bid", "ask", "lastPrice", "impliedVolatility", "openInterest", "volume"]
    out = df.copy()
    for c in needed:
        if c not in out.columns:
            out[c] = np.nan
    return out

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
st.caption("Mid-price (bid/ask), IV strike-specifica, normalizzazione (Market−BS)/BS, filtro tradabilità e simulatore PnL.")

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

    load_exp = st.button("Carica scadenze", use_container_width=True, disabled=(not ticker))

    if load_exp and ticker:
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

    if expiration and not is_valid_date(expiration):
        st.error("Formato scadenza non valido. Usa YYYY-MM-DD.")

    st.divider()
    st.subheader("Simulazione")
    default_mult = st.number_input("Moltiplicatore contratto (shares/contratto)", min_value=1, value=100, step=1)

    st.divider()
    run = st.button("Esegui analisi", type="primary", use_container_width=True, disabled=(not ticker or not is_valid_date(expiration)))

if not run:
    st.info("1) Inserisci ticker  2) Carica scadenze  3) Seleziona scadenza  4) Esegui analisi")
    st.stop()

# ======================================================
# Load history + S0 + sigma_hist
# ======================================================
try:
    hist = fetch_history_1y(ticker)
    if hist.empty:
        st.error("Storico prezzi vuoto. Prova un altro ticker.")
        st.stop()

    S0 = float(hist["Close"].iloc[-1])
    sigma_hist = compute_hist_vol(hist)
except Exception as e:
    st.error(f"Errore nel recupero storico/volatilità: {e}")
    st.stop()

# Time to expiry
try:
    expiration_date = datetime.strptime(expiration, "%Y-%m-%d")
except Exception:
    st.error("Scadenza non parsabile. Usa YYYY-MM-DD.")
    st.stop()

T = (expiration_date - datetime.now()).days / 365.0
if T <= 0:
    st.error("Scadenza nel passato (o oggi). Scegli una scadenza futura.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("S0 (ultimo close)", f"{S0:,.2f}")
m2.metric("σ storica (1y)", f"{sigma_hist:.4f}")
m3.metric("T (anni)", f"{T:.4f}")
m4.metric("Scadenza", expiration)

# ======================================================
# Load option chain (robusto: gestisce scadenze 'non interrogabili')
# ======================================================
calls = puts = None
err_first = None

# 1) try selected expiration
try:
    calls, puts = fetch_option_chain(ticker, expiration)
except Exception as e:
    err_first = e
    calls = puts = None

# 2) fallback: try other expirations if available
if calls is None or puts is None:
    tried = [expiration]
    exps = st.session_state.get("expirations", []) or []
    for exp_try in exps:
        if exp_try in tried:
            continue
        try:
            calls, puts = fetch_option_chain(ticker, exp_try)
            expiration = exp_try  # effective
            expiration_date = datetime.strptime(expiration, "%Y-%m-%d")
            T = (expiration_date - datetime.now()).days / 365.0
            if T <= 0:
                continue
            st.warning(f"Scadenza selezionata non disponibile. Uso fallback: {expiration}")
            break
        except Exception:
            continue

if calls is None or puts is None:
    st.error("Impossibile recuperare option chain da Yahoo (scadenza non disponibile o rate limit).")
    if err_first is not None:
        st.caption(f"Dettaglio primo errore: {err_first}")
    st.stop()

calls = ensure_columns(calls)
puts = ensure_columns(puts)

if calls.empty and puts.empty:
    st.error("Option chain vuota.")
    st.stop()

# ======================================================
# Tradability filter + mid + spread
# ======================================================
max_spread = max_spread_pct / 100.0

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["mid"] = out.apply(mid_price_row, axis=1)
    out["rel_spread"] = out.apply(rel_spread, axis=1)

    # tradable conditions
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
    # drop rows where mid is NaN (shouldn't happen if bid/ask ok, but keep safe)
    out = out.dropna(subset=["mid"])
    return out

calls_f = apply_filters(calls)
puts_f = apply_filters(puts)

st.caption(
    f"Filtro tradabilità: spread≤{max_spread_pct}%, OI≥{min_oi}"
    + (f", volume≥{min_vol}" if min_vol and min_vol > 0 else "")
    + "."
)

if calls_f.empty and puts_f.empty:
    st.warning("Dopo i filtri di tradabilità non rimane nessuna opzione. Riduci i filtri (spread/OI/volume).")
    st.stop()

# ======================================================
# TABLE: union strikes, strike-specific IV + normalized mispricing
# ======================================================
st.subheader("Tabella: Market(mid) vs BS (IV strike-specifica) + Normalizzazione (Market−BS)/BS")

# build dict by strike for fast lookup
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
        iv_c = float(rc["impliedVolatility"]) if pd.notna(rc["impliedVolatility"]) else np.nan
        bs_c = bs_call(S0, K, r, T, iv_c) if pd.notna(iv_c) and iv_c > 0 else np.nan
        norm_c = safe_norm(mkt_c, bs_c)
        spread_c = float(rc.get("rel_spread", np.nan))
        oi_c = float(rc.get("openInterest", np.nan))
        vol_c = float(rc.get("volume", np.nan))
    else:
        mkt_c = iv_c = bs_c = norm_c = spread_c = oi_c = vol_c = np.nan

    # put
    if K in puts_by_strike.index:
        rp = puts_by_strike.loc[K]
        mkt_p = float(rp["mid"])
        iv_p = float(rp["impliedVolatility"]) if pd.notna(rp["impliedVolatility"]) else np.nan
        bs_p = bs_put(S0, K, r, T, iv_p) if pd.notna(iv_p) and iv_p > 0 else np.nan
        norm_p = safe_norm(mkt_p, bs_p)
        spread_p = float(rp.get("rel_spread", np.nan))
        oi_p = float(rp.get("openInterest", np.nan))
        vol_p = float(rp.get("volume", np.nan))
    else:
        mkt_p = iv_p = bs_p = norm_p = spread_p = oi_p = vol_p = np.nan

    rows.append({
        "Strike": K,
        "Market Call (mid)": mkt_c,
        "IV Call": iv_c,
        "BS Call (IV)": bs_c,
        "Norm (C)": norm_c,
        "Spread% (C)": spread_c,
        "OI (C)": oi_c,
        "Vol (C)": vol_c,
        "Market Put (mid)": mkt_p,
        "IV Put": iv_p,
        "BS Put (IV)": bs_p,
        "Norm (P)": norm_p,
        "Spread% (P)": spread_p,
        "OI (P)": oi_p,
        "Vol (P)": vol_p,
    })

df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
st.dataframe(df, use_container_width=True)

# ======================================================
# PLOT: Normalized mispricing + S0 vertical line
# ======================================================
st.subheader("Grafico: Normalizzato (Market − BS) / BS (IV strike-specifica)")
df_plot = df.copy()

fig = plt.figure(figsize=(14, 6))
plt.plot(df_plot["Strike"], df_plot["Norm (C)"], marker="o", linewidth=1, label="Call: (M-BS)/BS")
plt.plot(df_plot["Strike"], df_plot["Norm (P)"], marker="o", linewidth=1, label="Put:  (M-BS)/BS")
plt.axhline(0, linestyle="--")
plt.axvline(S0, linestyle="--", linewidth=1)
ylim = plt.ylim()
plt.text(S0, ylim[1] * 0.95, f"S0={S0:.2f}", rotation=90, va="top")
plt.title("Normalized Mispricing vs Black–Scholes (IV strike-specific)")
plt.xlabel("Strike")
plt.ylabel("(Market - BS) / BS")
plt.grid(True)
plt.legend()
plt.tight_layout()
st.pyplot(fig)

# ======================================================
# SIMULATOR
# ======================================================
st.divider()
st.subheader("Simulazione operazione (ST a scadenza + qty + strike) + notional equivalente")

# strike list for simulator: only where chosen option type exists
strike_calls = df.loc[df["Market Call (mid)"].notna(), "Strike"].values
strike_puts  = df.loc[df["Market Put (mid)"].notna(), "Strike"].values

col1, col2, col3, col4 = st.columns([1.1, 1.3, 1.1, 1.2])
opt_type = col1.selectbox("Tipo", ["Call", "Put"], index=0)

if opt_type == "Call":
    strikes_sim = strike_calls
else:
    strikes_sim = strike_puts

if len(strikes_sim) == 0:
    st.warning("Nessuno strike disponibile (dopo i filtri) per il tipo selezionato.")
    st.stop()

default_idx = int(np.argmin(np.abs(strikes_sim - S0)))
K_sel = float(col2.selectbox("Strike", strikes_sim, index=default_idx))
qty = int(col3.number_input("Quantità contratti", min_value=1, value=1, step=1))
mult = int(col4.number_input("Moltiplicatore", min_value=1, value=int(default_mult), step=1))

ST = st.number_input("Prezzo sottostante a scadenza (ST)", min_value=0.0, value=float(S0), step=float(max(1.0, S0 * 0.01)))

# get premium mid for selected
row_sel = df.loc[df["Strike"] == K_sel].iloc[0]
premium = float(row_sel["Market Call (mid)"] if opt_type == "Call" else row_sel["Market Put (mid)"])
iv_sel = float(row_sel["IV Call"] if opt_type == "Call" else row_sel["IV Put"])

pay_share = payoff_at_expiry(opt_type, ST, K_sel)
pnl_share = pay_share - premium

premium_total = premium * mult * qty
pay_total = pay_share * mult * qty
pnl_total = pnl_share * mult * qty

# Notional equivalent: underlying portfolio value corresponding to shares controlled
notional_now = S0 * mult * qty
notional_ST = ST * mult * qty

a, b, c, d = st.columns(4)
a.metric("Premium (mid) per opzione", f"{premium:,.4f}")
b.metric("Costo premio totale", f"{premium_total:,.2f}")
c.metric("Payoff totale a scadenza", f"{pay_total:,.2f}")
d.metric("PnL totale a scadenza", f"{pnl_total:,.2f}")

e, f, g = st.columns(3)
e.metric("Nozionale equivalente (oggi)", f"{notional_now:,.2f}")
f.metric("Nozionale equivalente (a scadenza)", f"{notional_ST:,.2f}")
g.metric("IV strike (info)", f"{iv_sel:.4f}" if pd.notna(iv_sel) else "NaN")

st.caption("Curva PnL totale a scadenza al variare di ST (strike e quantità fissi).")

st_min = max(0.0, S0 * 0.5)
st_max = S0 * 1.5 if S0 > 0 else 1.0
ST_grid = np.linspace(st_min, st_max, 160)

pnl_curve = []
for stx in ST_grid:
    px = payoff_at_expiry(opt_type, float(stx), K_sel) - premium
    pnl_curve.append(px * mult * qty)

fig2 = plt.figure(figsize=(14, 5))
plt.plot(ST_grid, pnl_curve)
plt.axhline(0, linestyle="--")
plt.axvline(S0, linestyle="--", linewidth=1)
plt.axvline(K_sel, linestyle=":", linewidth=1)
plt.title("PnL totale a scadenza vs ST")
plt.xlabel("ST (prezzo sottostante a scadenza)")
plt.ylabel("PnL totale")
plt.grid(True)
plt.tight_layout()
st.pyplot(fig2)

st.caption(
    "Nota: dati Yahoo Finance possono essere incompleti. "
    "Questa app usa mid-price quando possibile e filtra strike non tradabili (spread/OI/volume)."
)
