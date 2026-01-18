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
    # fallback if no bid/ask
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
st.caption("Grafico 1: differenza (Market(mid) − BS) + S0. Grafico 2 invariato. Strategia multi-leg Call/Put (Buy/Sell) senza ricalcolare i grafici.")

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
    st.subheader("Range strike (Grafico 2) – invariato")
    range_low = st.slider("Min strike (% di S0)", min_value=10, max_value=95, value=70, step=5)
    range_high = st.slider("Max strike (% di S0)", min_value=105, max_value=300, value=130, step=5)
    n_points = st.slider("Numero punti strike", min_value=10, max_value=200, value=60, step=10)

    st.divider()
    run = st.button("Esegui analisi", type="primary", use_container_width=True, disabled=(not ticker or not exp_ok))

# ======================================================
# Analysis step (only when button pressed)
# Store results in session_state so strategy changes won't refetch/recompute graphs
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

    # option chain (robusto: fallback su altre scadenze se quella selezionata è 'non interrogabile')
    calls = puts = None
    exp_effective = expiration
    err_first = None

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
        raise ValueError(f"Impossibile recuperare option chain da Yahoo. Primo errore: {err_first}")

    calls = ensure_columns(calls)
    puts = ensure_columns(puts)

    # tradability filter + mid
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

    # build df for graph1 + strategy pricing
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
            diff_c = (mkt_c - bs_c) if pd.notna(mkt_c) and pd.notna(bs_c) else np.nan
        else:
            mkt_c = iv_c = bs_c = diff_c = np.nan

        # put
        if K in puts_by_strike.index:
            rp = puts_by_strike.loc[K]
            mkt_p = float(rp["mid"])
            iv_p = float(rp["impliedVolatility"]) if pd.notna(rp["impliedVolatility"]) else np.nan
            bs_p = bs_put(S0, K, r, T, iv_p) if pd.notna(iv_p) and iv_p > 0 else np.nan
            diff_p = (mkt_p - bs_p) if pd.notna(mkt_p) and pd.notna(bs_p) else np.nan
        else:
            mkt_p = iv_p = bs_p = diff_p = np.nan

        rows.append({
            "Strike": K,
            "Market Call (mid)": mkt_c, "IV Call": iv_c, "BS Call (IV)": bs_c, "Diff (C) = M-BS": diff_c,
            "Market Put (mid)": mkt_p,  "IV Put": iv_p,  "BS Put (IV)": bs_p,  "Diff (P) = M-BS": diff_p,
        })

    df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)

    # df2 for graph2 (kept "intatto"): Market(mid) vs BS (Hist σ vs IV strike) on strike range
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

            iv_call = float(row_c["impliedVolatility"]) if pd.notna(row_c["impliedVolatility"]) else np.nan
            iv_put = float(row_p["impliedVolatility"]) if pd.notna(row_p["impliedVolatility"]) else np.nan

            bs_call_iv = bs_call(S0, K, r, T, iv_call) if pd.notna(iv_call) and iv_call > 0 else np.nan
            bs_put_iv = bs_put(S0, K, r, T, iv_put) if pd.notna(iv_put) and iv_put > 0 else np.nan

            bs_call_hist = bs_call(S0, K, r, T, sigma_hist)
            bs_put_hist = bs_put(S0, K, r, T, sigma_hist)

            df2_rows.append({
                "Strike": K,
                "Market Call (mid)": mkt_call,
                "BS Call (IV strike)": bs_call_iv,
                "BS Call (Hist σ)": bs_call_hist,
                "Market Put (mid)": mkt_put,
                "BS Put (IV strike)": bs_put_iv,
                "BS Put (Hist σ)": bs_put_hist,
            })

    df2 = pd.DataFrame(df2_rows)

    strikes_call = df.loc[df["Market Call (mid)"].notna(), "Strike"].values
    strikes_put = df.loc[df["Market Put (mid)"].notna(), "Strike"].values

    return {
        "ticker": ticker,
        "expiration": exp_effective,
        "S0": S0,
        "sigma_hist": sigma_hist,
        "T": T,
        "df": df,
        "df2": df2,
        "strikes_call": strikes_call,
        "strikes_put": strikes_put,
        "filters": {"max_spread_pct": max_spread_pct, "min_oi": min_oi, "min_vol": min_vol},
    }

if run:
    try:
        st.session_state["analysis_data"] = run_analysis()
        st.session_state["analysis_ready"] = True
        # reset legs when analysis changes
        st.session_state.setdefault("strategy_legs", [])
    except Exception as e:
        st.session_state["analysis_ready"] = False
        st.error(str(e))

if not st.session_state.get("analysis_ready", False):
    st.info("Esegui l'analisi per generare i grafici. Poi potrai gestire la strategia multi-leg sotto senza rifare i grafici.")
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
    f"Filtro tradabilità: spread≤{A['filters']['max_spread_pct']}%, OI≥{A['filters']['min_oi']}"
    + (f", volume≥{A['filters']['min_vol']}" if A['filters']['min_vol'] and A['filters']['min_vol'] > 0 else "")
    + "."
)

# ======================================================
# Table
# ======================================================
st.subheader("Tabella: Market(mid) vs BS (IV strike-specifica) + Differenza (Market−BS)")
st.dataframe(A["df"], use_container_width=True)

# ======================================================
# Graph 1: ABSOLUTE diff (Market − BS) + ONLY S0 marker (as requested)
# ======================================================
st.subheader("Grafico 1: Differenza (Market − BS) usando mid-price (IV strike-specifica)")
df_plot = A["df"]

fig1 = plt.figure(figsize=(14, 6))
plt.plot(df_plot["Strike"], df_plot["Diff (C) = M-BS"], marker="o", linewidth=1, label="Call: Market(mid) − BS")
plt.plot(df_plot["Strike"], df_plot["Diff (P) = M-BS"], marker="o", linewidth=1, label="Put:  Market(mid) − BS")
plt.axhline(0, linestyle="--")
plt.axvline(A["S0"], linestyle="--", linewidth=1, label="S0")
plt.title("Market(mid) − Black–Scholes (IV strike-specific)")
plt.xlabel("Strike")
plt.ylabel("Difference (Market - BS)")
plt.grid(True)
plt.legend()
plt.tight_layout()
st.pyplot(fig1)

# ======================================================
# Graph 2 (kept intact)
# ======================================================
st.subheader("Grafico 2: Market vs BS (σ storica vs IV strike-specifica) su range di strike")
df2 = A["df2"]

if df2 is None or df2.empty:
    st.warning("Impossibile costruire il Grafico 2 (range strike o chain non validi dopo filtri).")
else:
    fig2 = plt.figure(figsize=(14, 7))
    plt.plot(df2["Strike"], df2["Market Call (mid)"], marker="o", label="Market Call (mid)")
    plt.plot(df2["Strike"], df2["BS Call (IV strike)"], linestyle="--", label="BS Call (IV strike)")
    plt.plot(df2["Strike"], df2["BS Call (Hist σ)"], linestyle="dotted", label="BS Call (Hist σ)")

    plt.plot(df2["Strike"], df2["Market Put (mid)"], marker="o", label="Market Put (mid)")
    plt.plot(df2["Strike"], df2["BS Put (IV strike)"], linestyle="--", label="BS Put (IV strike)")
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
# Strategy builder (multi-leg CALL + PUT, BUY + SELL) – NO third chart
# ======================================================
st.divider()
st.subheader("Strategie multi-leg (Call/Put, Buy/Sell) — senza ricalcolare i grafici")

if "strategy_legs" not in st.session_state:
    st.session_state["strategy_legs"] = []

# controls
c1, c2, c3, c4, c5 = st.columns([1.0, 1.0, 1.3, 1.0, 1.0])
side = c1.selectbox("Side", ["Buy", "Sell"], key="leg_side")
opt_type = c2.selectbox("Tipo", ["Call", "Put"], key="leg_type")

strikes_avail = A["strikes_call"] if opt_type == "Call" else A["strikes_put"]
if len(strikes_avail) == 0:
    st.warning("Nessuno strike disponibile (dopo filtri) per il tipo selezionato.")
    st.stop()

default_idx = int(np.argmin(np.abs(strikes_avail - A["S0"])))
K_sel = float(c3.selectbox("Strike", strikes_avail, index=default_idx, key="leg_strike"))
qty = int(c4.number_input("Qty contratti", min_value=1, value=1, step=1, key="leg_qty"))
mult = int(c5.number_input("Moltiplicatore", min_value=1, value=100, step=1, key="leg_mult"))

ST = st.number_input("Prezzo sottostante a scadenza (ST) per valutazione strategia", min_value=0.0, value=float(A["S0"]), step=float(max(1.0, A["S0"] * 0.01)), key="strategy_ST")

btn1, btn2, btn3 = st.columns([1.1, 1.1, 1.6])
if btn1.button("Aggiungi gamba", type="primary"):
    st.session_state["strategy_legs"].append({
        "Side": side, "Type": opt_type, "Strike": K_sel, "Qty": qty, "Mult": mult
    })

if btn2.button("Rimuovi ultima gamba"):
    if st.session_state["strategy_legs"]:
        st.session_state["strategy_legs"].pop()

if btn3.button("Svuota strategia"):
    st.session_state["strategy_legs"] = []

legs = st.session_state["strategy_legs"]

if not legs:
    st.info("Aggiungi una o più gambe (Call/Put, Buy/Sell). Poi vedrai cashflow, payoff e PnL a scadenza (ST).")
    st.stop()

# price lookup
df_price = A["df"].set_index("Strike")

leg_rows = []
cashflow0_total = 0.0
payoff_total = 0.0
pnl_total = 0.0
gross_contracts = 0

for i, leg in enumerate(legs, start=1):
    side_i = leg["Side"]
    typ_i = leg["Type"]
    K_i = float(leg["Strike"])
    q_i = int(leg["Qty"])
    m_i = int(leg["Mult"])

    gross_contracts += abs(q_i)

    row = df_price.loc[K_i]

    premium = float(row["Market Call (mid)"] if typ_i == "Call" else row["Market Put (mid)"])
    pay_share = payoff_at_expiry(typ_i, ST, K_i)

    # cashflow at t0 and position payoff
    if side_i == "Buy":
        cash0 = -premium * q_i * m_i
        payoff_pos = +pay_share * q_i * m_i
        pnl = cash0 + payoff_pos
    else:  # Sell
        cash0 = +premium * q_i * m_i
        payoff_pos = -pay_share * q_i * m_i
        pnl = cash0 + payoff_pos

    cashflow0_total += cash0
    payoff_total += payoff_pos
    pnl_total += pnl

    leg_rows.append({
        "#": i,
        "Side": side_i,
        "Type": typ_i,
        "Strike": K_i,
        "Qty": q_i,
        "Mult": m_i,
        "Premium(mid) per share": premium,
        "Cashflow t0": cash0,
        "Payoff (position) @ST": payoff_pos,
        "PnL @ST": pnl
    })

legs_df = pd.DataFrame(leg_rows)
st.dataframe(legs_df, use_container_width=True)

# notional equivalent (simple gross-equivalent in underlying value)
notional_equiv = A["S0"] * 100 * gross_contracts  # assumes standard 100; still show separate:
notional_equiv_custom = A["S0"] * sum(int(l["Mult"]) * abs(int(l["Qty"])) for l in legs)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Cashflow totale t0", f"{cashflow0_total:,.2f}")
m2.metric("Payoff posizione totale @ST", f"{payoff_total:,.2f}")
m3.metric("PnL totale @ST", f"{pnl_total:,.2f}")
m4.metric("Nozionale equivalente (oggi)", f"{notional_equiv_custom:,.2f}")

st.caption(
    "Nota: lo script Streamlit viene rieseguito quando cambi i controlli, "
    "ma i grafici sopra non vengono ricalcolati: usano i dati salvati in session_state dopo 'Esegui analisi'."
)
