import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime
import re

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
# Helpers
# ----------------------------
def is_valid_date(s: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", (s or "").strip()))

def compute_hist_vol(hist: pd.DataFrame):
    close = hist["Close"].dropna()
    log_returns = np.log(close / close.shift(1)).dropna()
    return float(log_returns.std() * np.sqrt(250))
    
def option_payoff_at_expiry(option_type: str, ST: float, K: float) -> float:
    if option_type.lower() == "call":
        return max(ST - K, 0.0)
    if option_type.lower() == "put":
        return max(K - ST, 0.0)
    return np.nan

def safe_mid_price(row: pd.Series) -> float:
    """
    Prefer mid-price if bid/ask are usable, otherwise fallback to lastPrice.
    """
    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    last = row.get("lastPrice", np.nan)

    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0 and ask >= bid:
        return float((bid + ask) / 2.0)

    if pd.notna(last):
        return float(last)

    return np.nan

def safe_norm_diff(mkt: float, bs: float, eps: float = 1e-8) -> float:
    """
    Normalized difference: (Market - BS) / BS
    If BS is near zero, return NaN to avoid blow-ups.
    """
    if pd.isna(mkt) or pd.isna(bs):
        return np.nan
    if abs(bs) < eps:
        return np.nan
    return float((mkt - bs) / bs)

# ----------------------------
# Data fetch helpers (cached)
# ----------------------------
@st.cache_data(ttl=1800, show_spinner=False)  # 30 minuti
def get_expirations(ticker_symbol: str):
    t = yf.Ticker(ticker_symbol)
    return list(t.options)  # list of "YYYY-MM-DD"

@st.cache_data(ttl=1800, show_spinner=False)  # 30 minuti
def get_history_1y(ticker_symbol: str):
    t = yf.Ticker(ticker_symbol)
    return t.history(period="1y")

@st.cache_data(ttl=900, show_spinner=False)  # 15 minuti
def get_option_chain(ticker_symbol: str, expiration: str):
    t = yf.Ticker(ticker_symbol)
    oc = t.option_chain(expiration)
    return oc.calls, oc.puts

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Option Hedging", layout="wide")
st.title("Option Hedging")
st.caption("Inserisci un ticker, scegli una scadenza, e confronta prezzi market vs Black-Scholes.")

with st.sidebar:
    st.header("Input")

    ticker_symbol = st.text_input("Ticker (es. SPY, QQQ, IWM, AAPL, TSLA)", value="SPY").strip().upper()
    r = st.number_input("Risk-free rate (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.005, format="%.3f")
    st.caption("Volatilità storica stimata su 1 anno (log-returns).")

    st.divider()
    st.subheader("Scadenza opzioni")

    if "expirations" not in st.session_state:
        st.session_state["expirations"] = []

    load_exp = st.button("Carica scadenze da Yahoo", use_container_width=True, disabled=(not ticker_symbol))

    if load_exp and ticker_symbol:
        try:
            st.session_state["expirations"] = get_expirations(ticker_symbol)
            if not st.session_state["expirations"]:
                st.warning("Nessuna scadenza opzioni trovata per questo ticker su Yahoo.")
        except Exception as e:
            st.session_state["expirations"] = []
            st.warning(
                "Yahoo Finance sta limitando le richieste (rate limit). "
                "Riprova tra qualche minuto oppure inserisci manualmente la scadenza."
            )
            st.caption(f"Dettaglio errore: {e}")

    if st.session_state["expirations"]:
        expiration = st.selectbox("Seleziona scadenza (Yahoo Finance)", options=st.session_state["expirations"], index=0)
    else:
        expiration = st.text_input("Oppure inserisci scadenza (YYYY-MM-DD)", value="2026-03-20")

    exp_ok = is_valid_date(expiration)
    if not exp_ok:
        st.error("Formato scadenza non valido. Usa YYYY-MM-DD.")

    st.divider()
    st.subheader("Range di strike per Grafico 2")
    range_low = st.slider("Min strike (% di S0)", min_value=10, max_value=95, value=70, step=5)
    range_high = st.slider("Max strike (% di S0)", min_value=105, max_value=300, value=130, step=5)
    n_points = st.slider("Numero punti strike", min_value=10, max_value=200, value=60, step=10)

    st.divider()
    run = st.button("Esegui analisi", type="primary", use_container_width=True, disabled=(not ticker_symbol or not exp_ok))

if not run:
    st.info("1) Inserisci ticker  2) (opzionale) carica scadenze  3) scegli/inserisci scadenza  4) Esegui analisi")
    st.stop()

# ----------------------------
# Load history and compute S0, sigma
# ----------------------------
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
try:
    expiration_date = datetime.strptime(expiration, "%Y-%m-%d")
except Exception:
    st.error("Scadenza non parsabile. Usa YYYY-MM-DD.")
    st.stop()

T = (expiration_date - today).days / 365.0
if T <= 0:
    st.warning("La scadenza selezionata risulta nel passato (o oggi). Scegli una scadenza futura.")
    st.stop()

colA, colB, colC, colD = st.columns(4)
colA.metric("S0 (ultimo close)", f"{S0:,.2f}")
colB.metric("σ annual (storica)", f"{sigma_annual:.4f}")
colC.metric("T (anni)", f"{T:.4f}")
colD.metric("Scadenza", expiration)

# ----------------------------
# Load option chain
# ----------------------------
try:
    calls, puts = get_option_chain(ticker_symbol, expiration)
except Exception as e:
    st.warning(
        "Errore nel recupero option chain (possibile rate limit Yahoo). "
        "Riprova tra qualche minuto."
    )
    st.caption(f"Dettaglio errore: {e}")
    st.stop()

if calls.empty and puts.empty:
    st.warning("Option chain vuota.")
    st.stop()

# Build quick lookup by strike (keep first match)
calls_by_strike = calls.drop_duplicates(subset=["strike"]).set_index("strike")
puts_by_strike = puts.drop_duplicates(subset=["strike"]).set_index("strike")

# ----------------------------
# TABLE 1: union strikes, market(mid) vs BS (hist sigma + IV strike-specific)
# ----------------------------
st.subheader("Tabella: Market(mid) vs Black-Scholes (σ storica + IV strike-specifica)")

strike_union = np.union1d(calls["strike"].values, puts["strike"].values)

results = []
for K in strike_union:
    K = float(K)

    # Market mid + IV per strike
    row_call = calls_by_strike.loc[K] if K in calls_by_strike.index else None
    row_put  = puts_by_strike.loc[K]  if K in puts_by_strike.index  else None

    mkt_call = safe_mid_price(row_call) if row_call is not None else np.nan
    mkt_put  = safe_mid_price(row_put)  if row_put  is not None else np.nan

    iv_call = float(row_call.get("impliedVolatility", np.nan)) if row_call is not None else np.nan
    iv_put  = float(row_put.get("impliedVolatility", np.nan))  if row_put  is not None else np.nan

    # BS with historical sigma
    call_bs_hist = black_scholes_call(S0, K, r, T, sigma_annual)
    put_bs_hist  = black_scholes_put(S0, K, r, T, sigma_annual)

    # BS with strike-specific IV (if available)
    call_bs_iv = black_scholes_call(S0, K, r, T, iv_call) if pd.notna(iv_call) and iv_call > 0 else np.nan
    put_bs_iv  = black_scholes_put(S0, K, r, T, iv_put)  if pd.notna(iv_put)  and iv_put  > 0 else np.nan

    # Normalized diffs (Market - BS)/BS
    call_norm_hist = safe_norm_diff(mkt_call, call_bs_hist)
    put_norm_hist  = safe_norm_diff(mkt_put,  put_bs_hist)

    call_norm_iv = safe_norm_diff(mkt_call, call_bs_iv)
    put_norm_iv  = safe_norm_diff(mkt_put,  put_bs_iv)

    results.append({
        "Strike": K,

        "Market Call (mid)": mkt_call,
        "BS Call (Hist σ)": float(call_bs_hist) if pd.notna(call_bs_hist) else np.nan,
        "IV Call (strike)": iv_call,
        "BS Call (IV strike)": float(call_bs_iv) if pd.notna(call_bs_iv) else np.nan,
        "NormDiff Call vs Hist": call_norm_hist,
        "NormDiff Call vs IV": call_norm_iv,

        "Market Put (mid)": mkt_put,
        "BS Put (Hist σ)": float(put_bs_hist) if pd.notna(put_bs_hist) else np.nan,
        "IV Put (strike)": iv_put,
        "BS Put (IV strike)": float(put_bs_iv) if pd.notna(put_bs_iv) else np.nan,
        "NormDiff Put vs Hist": put_norm_hist,
        "NormDiff Put vs IV": put_norm_iv,
    })

df1 = pd.DataFrame(results).sort_values("Strike").reset_index(drop=True)
st.dataframe(df1, use_container_width=True)

# ----------------------------
# PLOT 1: normalized differences (Market - BS)/BS
# ----------------------------
st.subheader("Grafico 1: Normalizzato (Market − BS) / BS")

df_plot = df1.copy()

fig1 = plt.figure(figsize=(14, 6))

# Hist sigma normalized
plt.plot(df_plot["Strike"], df_plot["NormDiff Call vs Hist"], marker="o", linewidth=1, label="Call: (M-BS_hist)/BS_hist")
plt.plot(df_plot["Strike"], df_plot["NormDiff Put vs Hist"],  marker="o", linewidth=1, label="Put:  (M-BS_hist)/BS_hist")

# IV strike-specific normalized
plt.plot(df_plot["Strike"], df_plot["NormDiff Call vs IV"], marker=".", linewidth=1, linestyle="--", label="Call: (M-BS_IV)/BS_IV")
plt.plot(df_plot["Strike"], df_plot["NormDiff Put vs IV"],  marker=".", linewidth=1, linestyle="--", label="Put:  (M-BS_IV)/BS_IV")

plt.axhline(0, linestyle="--")
plt.title("Normalized Mispricing vs Black-Scholes")
plt.xlabel("Strike")
plt.ylabel("(Market - BS) / BS")
plt.grid(True)
plt.legend()
plt.tight_layout()
st.pyplot(fig1)

with st.expander("Mostra anche la versione NON normalizzata (Market − BS)"):
    df_abs = df1.copy()
    df_abs["Call_Diff_abs_hist"] = df_abs["Market Call (mid)"] - df_abs["BS Call (Hist σ)"]
    df_abs["Put_Diff_abs_hist"]  = df_abs["Market Put (mid)"]  - df_abs["BS Put (Hist σ)"]
    df_abs["Call_Diff_abs_iv"]   = df_abs["Market Call (mid)"] - df_abs["BS Call (IV strike)"]
    df_abs["Put_Diff_abs_iv"]    = df_abs["Market Put (mid)"]  - df_abs["BS Put (IV strike)"]

    fig1b = plt.figure(figsize=(14, 6))
    plt.plot(df_abs["Strike"], df_abs["Call_Diff_abs_hist"], marker="o", linewidth=1, label="Call: M-BS_hist")
    plt.plot(df_abs["Strike"], df_abs["Put_Diff_abs_hist"],  marker="o", linewidth=1, label="Put:  M-BS_hist")
    plt.plot(df_abs["Strike"], df_abs["Call_Diff_abs_iv"], marker=".", linewidth=1, linestyle="--", label="Call: M-BS_IV")
    plt.plot(df_abs["Strike"], df_abs["Put_Diff_abs_iv"],  marker=".", linewidth=1, linestyle="--", label="Put:  M-BS_IV")
    plt.axhline(0, linestyle="--")
    plt.title("Absolute Price Difference")
    plt.xlabel("Strike")
    plt.ylabel("Market - BS")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig1b)
st.divider()
st.subheader("Simulazione operazione (payoff/PnL + nozionale equivalente)")

with st.expander("Apri simulatore", expanded=True):



# ----------------------------
# PLOT 2: market(mid) vs BS (hist vs IV strike-specific) on custom strike range
# ----------------------------
st.subheader("Grafico 2: Market(mid) vs BS (σ storica vs IV strike-specifica) su range di strike")

low = S0 * (range_low / 100.0)
high = S0 * (range_high / 100.0)
strike_range = np.linspace(low, high, n_points)

results2 = []
if not calls.empty and not puts.empty:
    for K in strike_range:
        K = float(K)

        # nearest strike rows
        call_idx = (calls["strike"] - K).abs().idxmin()
        put_idx  = (puts["strike"]  - K).abs().idxmin()

        row_call = calls.loc[call_idx]
        row_put  = puts.loc[put_idx]

        mkt_call = safe_mid_price(row_call)
        mkt_put  = safe_mid_price(row_put)

        iv_call = float(row_call.get("impliedVolatility", np.nan))
        iv_put  = float(row_put.get("impliedVolatility", np.nan))

        bs_call_hist = black_scholes_call(S0, K, r, T, sigma_annual)
        bs_put_hist  = black_scholes_put(S0, K, r, T, sigma_annual)

        bs_call_iv = black_scholes_call(S0, K, r, T, iv_call) if pd.notna(iv_call) and iv_call > 0 else np.nan
        bs_put_iv  = black_scholes_put(S0, K, r, T, iv_put)  if pd.notna(iv_put)  and iv_put  > 0 else np.nan

        results2.append({
            "Strike": K,
            "Market_Call(mid)": mkt_call,
            "BS_Call_Hist": float(bs_call_hist) if pd.notna(bs_call_hist) else np.nan,
            "IV_Call(strike)": iv_call,
            "BS_Call_IVstrike": float(bs_call_iv) if pd.notna(bs_call_iv) else np.nan,

            "Market_Put(mid)": mkt_put,
            "BS_Put_Hist": float(bs_put_hist) if pd.notna(bs_put_hist) else np.nan,
            "IV_Put(strike)": iv_put,
            "BS_Put_IVstrike": float(bs_put_iv) if pd.notna(bs_put_iv) else np.nan,
        })

df2 = pd.DataFrame(results2)

if df2.empty:
    st.warning("Impossibile costruire df2 (range strike o chain non validi).")
else:
    fig2 = plt.figure(figsize=(14, 7))

    plt.plot(df2["Strike"], df2["Market_Call(mid)"], marker="o", label="Market Call (mid)")
    plt.plot(df2["Strike"], df2["BS_Call_IVstrike"], linestyle="--", label="BS Call (IV strike)")
    plt.plot(df2["Strike"], df2["BS_Call_Hist"], linestyle="dotted", label="BS Call (Hist σ)")

    plt.plot(df2["Strike"], df2["Market_Put(mid)"], marker="o", label="Market Put (mid)")
    plt.plot(df2["Strike"], df2["BS_Put_IVstrike"], linestyle="--", label="BS Put (IV strike)")
    plt.plot(df2["Strike"], df2["BS_Put_Hist"], linestyle="dotted", label="BS Put (Hist σ)")

    plt.title("Market(mid) vs Black-Scholes Option Prices")
    plt.xlabel("Strike")
    plt.ylabel("Option Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    with st.expander("Mostra tabella df2"):
        st.dataframe(df2, use_container_width=True)

st.caption(
    "Nota: Yahoo può avere dati bid/ask mancanti o spread molto larghi. "
    "Se bid/ask non sono utilizzabili, il codice fa fallback a lastPrice."
)
    # --- Parametri base
    col1, col2, col3, col4 = st.columns([1.1, 1.1, 1.1, 1.4])

    option_type = col1.selectbox("Tipo opzione", ["Call", "Put"], index=0)

    # strike disponibili (solo quelli con market mid non-NaN per il tipo scelto)
    if option_type == "Call":
        strikes_ok = df1.loc[df1["Market Call (mid)"].notna(), "Strike"].values
    else:
        strikes_ok = df1.loc[df1["Market Put (mid)"].notna(), "Strike"].values

    if len(strikes_ok) == 0:
        st.warning("Nessuno strike con prezzo di mercato disponibile per questo tipo opzione.")
        st.stop()

    # default strike vicino a S0
    default_idx = int(np.argmin(np.abs(strikes_ok - S0)))
    K_sel = col2.selectbox("Strike", strikes_ok, index=default_idx)

    qty = col3.number_input("Quantità contratti", min_value=1, value=1, step=1)

    # moltiplicatore (per equity/ETF di solito 100)
    multiplier = col4.number_input("Moltiplicatore contratto (shares/contratto)", min_value=1, value=100, step=1)

    st.caption("Inserisci ST = prezzo del sottostante a scadenza per simulare payoff e PnL.")

    ST = st.number_input("Prezzo sottostante a scadenza (ST)", min_value=0.0, value=float(S0), step=float(max(1.0, S0*0.01)))

    # --- Recupero premio (mid) e IV strike (per info)
    row_sel = df1.loc[df1["Strike"] == float(K_sel)].iloc[0]

    if option_type == "Call":
        premium = float(row_sel["Market Call (mid)"])
        iv_used = float(row_sel["IV Call (strike)"]) if pd.notna(row_sel["IV Call (strike)"]) else np.nan
    else:
        premium = float(row_sel["Market Put (mid)"])
        iv_used = float(row_sel["IV Put (strike)"]) if pd.notna(row_sel["IV Put (strike)"]) else np.nan

    # --- Payoff/PnL per share
    payoff_per_share = option_payoff_at_expiry(option_type, ST, float(K_sel))
    pnl_per_share = payoff_per_share - premium

    # --- Totali (per contratti)
    premium_total = premium * multiplier * qty
    payoff_total = payoff_per_share * multiplier * qty
    pnl_total = pnl_per_share * multiplier * qty

    # --- Nozionale equivalente (portafoglio sottostante “equivalente”)
    # interpretazione semplice: esposizione in controvalore se muovessi lo stesso numero di shares del moltiplicatore
    notional_now = S0 * multiplier * qty
    notional_at_expiry = ST * multiplier * qty

    # --- Output sintetico
    a, b, c, d = st.columns(4)
    a.metric("Premio (mid) per opzione", f"{premium:,.4f}")
    b.metric("Costo totale premio", f"{premium_total:,.2f}")
    c.metric("Payoff totale a scadenza", f"{payoff_total:,.2f}")
    d.metric("PnL totale a scadenza", f"{pnl_total:,.2f}")

    e, f, g = st.columns(3)
    e.metric("Nozionale equivalente (oggi)", f"{notional_now:,.2f}")
    f.metric("Nozionale equivalente (a scadenza)", f"{notional_at_expiry:,.2f}")
    g.metric("IV strike (info)", f"{iv_used:.4f}" if pd.notna(iv_used) else "NaN")

    st.write(
        f"**Dettaglio:** {option_type} K={float(K_sel):.2f}, Qty={qty}, Mult={multiplier}  |  "
        f"Payoff/share={payoff_per_share:.4f}  |  PnL/share={pnl_per_share:.4f}"
    )

    # --- Mini-grafico payoff vs ST (curva)
    st.caption("Curva PnL a scadenza al variare di ST (tenendo fissi strike e quantità).")
    st_min = max(0.0, S0 * 0.5)
    st_max = S0 * 1.5 if S0 > 0 else 1.0
    ST_grid = np.linspace(st_min, st_max, 120)

    pnl_curve = []
    for stx in ST_grid:
        px = option_payoff_at_expiry(option_type, stx, float(K_sel)) - premium
        pnl_curve.append(px * multiplier * qty)

    fig_sim = plt.figure(figsize=(14, 5))
    plt.plot(ST_grid, pnl_curve, marker=None)
    plt.axhline(0, linestyle="--")
    plt.axvline(S0, linestyle="--", linewidth=1)
    plt.axvline(float(K_sel), linestyle=":", linewidth=1)
    plt.title("PnL a scadenza (totale) vs ST")
    plt.xlabel("ST (prezzo sottostante a scadenza)")
    plt.ylabel("PnL totale")
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(fig_sim)
