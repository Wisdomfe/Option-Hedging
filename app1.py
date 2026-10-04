import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from datetime import datetime
import re


# ======================================================
# BLACK-SCHOLES
# ======================================================

def bs_call(
    S: float,
    K: float,
    r: float,
    T: float,
    sigma: float
) -> float:

    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return float(
        S * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )


def bs_put(
    S: float,
    K: float,
    r: float,
    T: float,
    sigma: float
) -> float:

    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return float(
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S * norm.cdf(-d1)
    )


# ======================================================
# HELPERS
# ======================================================

def is_valid_date(s: str) -> bool:

    return bool(
        re.fullmatch(
            r"\d{4}-\d{2}-\d{2}",
            (s or "").strip()
        )
    )


def compute_hist_vol(hist: pd.DataFrame) -> float:

    close = hist["Close"].dropna()

    log_returns = np.log(
        close / close.shift(1)
    ).dropna()

    return float(
        log_returns.std() * np.sqrt(252)
    )


def ensure_columns(
    df: pd.DataFrame
) -> pd.DataFrame:

    needed = [
        "strike",
        "bid",
        "ask",
        "lastPrice",
        "impliedVolatility",
        "openInterest",
        "volume",
    ]

    out = df.copy()

    for c in needed:

        if c not in out.columns:
            out[c] = np.nan

    return out


def mid_price_row(
    row: pd.Series
) -> float:

    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    last = row.get("lastPrice", np.nan)

    # ----------------------------------------------
    # Caso normale: bid/ask disponibili
    # ----------------------------------------------

    if (
        pd.notna(bid)
        and pd.notna(ask)
        and bid > 0
        and ask > 0
        and ask >= bid
    ):

        return float(
            (bid + ask) / 2.0
        )

    # ----------------------------------------------
    # Fallback:
    # se bid/ask non disponibili usa lastPrice
    # ----------------------------------------------

    if (
        pd.notna(last)
        and last > 0
    ):

        return float(last)

    return np.nan


def payoff_at_expiry(
    opt_type: str,
    ST: float,
    K: float
) -> float:

    if opt_type.lower() == "call":
        return max(ST - K, 0.0)

    if opt_type.lower() == "put":
        return max(K - ST, 0.0)

    return np.nan


# ======================================================
# YAHOO / YFINANCE FETCHERS
# ======================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def fetch_expirations(ticker: str):

    """
    Recupera le scadenze delle opzioni
    disponibili tramite yfinance.

    Restituisce:

    expirations
    diagnostic
    """

    ticker = (
        ticker
        .strip()
        .upper()
    )

    diagnostic = {

        "ticker": ticker,

        "status": "unknown",

        "message": "",

        "error_type": "",
    }

    try:

        tk = yf.Ticker(ticker)

        expirations_raw = tk.options

        # ------------------------------------------
        # Yahoo restituisce None
        # ------------------------------------------

        if expirations_raw is None:

            diagnostic["status"] = "empty"

            diagnostic["message"] = (
                "Yahoo/yfinance ha restituito None "
                "per l'elenco delle scadenze."
            )

            return [], diagnostic

        # ------------------------------------------
        # Converte tuple in lista
        # ------------------------------------------

        expirations = list(
            expirations_raw
        )

        # ------------------------------------------
        # Lista vuota
        # ------------------------------------------

        if len(expirations) == 0:

            diagnostic["status"] = "empty"

            diagnostic["message"] = (
                f"Yahoo/yfinance ha restituito ZERO "
                f"scadenze per {ticker}. "
                f"Se il ticker dispone normalmente di "
                f"opzioni, il problema è probabilmente "
                f"nel collegamento a Yahoo Finance."
            )

            return [], diagnostic

        # ------------------------------------------
        # OK
        # ------------------------------------------

        diagnostic["status"] = "ok"

        diagnostic["message"] = (
            f"Trovate {len(expirations)} "
            f"scadenze per {ticker}."
        )

        return (
            expirations,
            diagnostic
        )

    except Exception as e:

        msg = str(e)

        diagnostic["status"] = "error"

        diagnostic["message"] = msg

        msg_lower = msg.lower()

        # ------------------------------------------
        # Classificazione errore
        # ------------------------------------------

        if (
            "429" in msg_lower
            or "rate limit" in msg_lower
            or "too many requests" in msg_lower
        ):

            diagnostic["error_type"] = (
                "RATE_LIMIT"
            )

        elif (
            "401" in msg_lower
            or "unauthorized" in msg_lower
        ):

            diagnostic["error_type"] = (
                "UNAUTHORIZED"
            )

        elif (
            "crumb" in msg_lower
        ):

            diagnostic["error_type"] = (
                "CRUMB"
            )

        elif (
            "timeout" in msg_lower
            or "timed out" in msg_lower
        ):

            diagnostic["error_type"] = (
                "TIMEOUT"
            )

        else:

            diagnostic["error_type"] = (
                "UNKNOWN"
            )

        return (
            [],
            diagnostic
        )


@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def fetch_history_1y(
    ticker: str
):

    ticker = (
        ticker
        .strip()
        .upper()
    )

    tk = yf.Ticker(ticker)

    hist = tk.history(
        period="1y",
        auto_adjust=False
    )

    if (
        hist is None
        or hist.empty
    ):

        raise ValueError(
            f"Storico prezzi non disponibile "
            f"per {ticker}."
        )

    return hist


@st.cache_data(
    ttl=300,
    show_spinner=False
)
def fetch_option_chain(
    ticker: str,
    expiration: str
):

    ticker = (
        ticker
        .strip()
        .upper()
    )

    expiration = (
        expiration
        .strip()
    )

    tk = yf.Ticker(ticker)

    try:

        oc = tk.option_chain(
            expiration
        )

    except Exception as e:

        msg = str(e)

        raise RuntimeError(

            f"Errore Yahoo/yfinance durante "
            f"il recupero option chain "
            f"{ticker} - {expiration}: {msg}"

        )

    if oc is None:

        raise RuntimeError(

            f"Yahoo ha restituito una option chain "
            f"vuota per {ticker} - {expiration}."

        )

    calls = oc.calls
    puts = oc.puts

    if calls is None:
        calls = pd.DataFrame()

    if puts is None:
        puts = pd.DataFrame()

    if (
        calls.empty
        and puts.empty
    ):

        raise RuntimeError(

            f"Option chain vuota per "
            f"{ticker} - {expiration}."

        )

    return calls, puts


# ======================================================
# STREAMLIT UI
# ======================================================

st.set_page_config(

    page_title="Options Analytics",

    layout="wide"
)


st.title(
    "Options Analytics – Market vs Black–Scholes"
)


st.caption(

    "Grafico 1: BARRE (Market(mid) − BS) + SOLO linea verticale S0. "
    "Grafico 2 invariato. "
    "Strategie multi-leg Call/Put (Buy/Sell)."

)


# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:

    st.header("Input")

    ticker = st.text_input(

        "Ticker (es. SPY, QQQ, IWM, AAPL)",

        value="SPY"

    ).strip().upper()


    r = st.number_input(

        "Risk-free rate (r)",

        min_value=0.0,

        max_value=1.0,

        value=0.05,

        step=0.005,

        format="%.3f"

    )


    # ==================================================
    # FILTRI
    # ==================================================

    st.divider()

    st.subheader(
        "Filtro tradabilità (senza spread)"
    )


    min_oi = st.number_input(

        "Min Open Interest",

        min_value=0,

        max_value=500000,

        value=10,

        step=1

    )


    min_vol = st.number_input(

        "Min Volume (opzionale)",

        min_value=0,

        max_value=500000,

        value=0,

        step=1

    )


    st.caption(

        "Nota: se bid/ask non sono disponibili, "
        "il mid-price usa il lastPrice (se > 0)."

    )


    # ==================================================
    # SCADENZA
    # ==================================================

    st.divider()

    st.subheader("Scadenza")


    if "expirations" not in st.session_state:

        st.session_state[
            "expirations"
        ] = []


    if (
        "expiration_diagnostic"
        not in st.session_state
    ):

        st.session_state[
            "expiration_diagnostic"
        ] = {}


    # ==================================================
    # CARICA SCADENZE
    # ==================================================

    if st.button(

        "Carica scadenze",

        use_container_width=True,

        disabled=(not ticker)

    ):

        # ----------------------------------------------
        # Elimina cache precedente:
        # vogliamo una nuova richiesta a Yahoo
        # ----------------------------------------------

        fetch_expirations.clear()


        with st.spinner(
            "Recupero scadenze da Yahoo..."
        ):

            (
                expirations_found,
                diagnostic

            ) = fetch_expirations(
                ticker
            )


        st.session_state[
            "expirations"
        ] = expirations_found


        st.session_state[
            "expiration_diagnostic"
        ] = diagnostic


        # ----------------------------------------------
        # SCADENZE TROVATE
        # ----------------------------------------------

        if expirations_found:

            st.success(

                f"Trovate "
                f"{len(expirations_found)} "
                f"scadenze per {ticker}."

            )

        # ----------------------------------------------
        # NESSUNA SCADENZA / ERRORE
        # ----------------------------------------------

        else:

            status = diagnostic.get(
                "status",
                ""
            )

            error_type = diagnostic.get(
                "error_type",
                ""
            )

            message = diagnostic.get(
                "message",
                ""
            )


            if (
                error_type
                == "RATE_LIMIT"
            ):

                st.error(

                    "Yahoo sta limitando "
                    "temporaneamente le richieste "
                    "(HTTP 429 / Rate Limit)."

                )


            elif error_type in (
                "UNAUTHORIZED",
                "CRUMB"
            ):

                st.error(

                    "Yahoo ha rifiutato "
                    "l'accesso all'endpoint "
                    "delle opzioni "
                    "(autenticazione / cookie / crumb)."

                )


            elif (
                error_type
                == "TIMEOUT"
            ):

                st.error(

                    "Timeout durante il "
                    "collegamento a Yahoo Finance."

                )


            elif (
                status
                == "empty"
            ):

                st.warning(

                    f"Yahoo/yfinance ha restituito "
                    f"ZERO scadenze per {ticker}."

                )

                st.caption(

                    "Se il ticker è ad esempio "
                    "SPY, QQQ, AAPL, MSFT o NVDA, "
                    "questo normalmente indica "
                    "un problema Yahoo/yfinance "
                    "e non l'assenza reale di opzioni."

                )


            else:

                st.error(

                    "Errore durante il recupero "
                    "delle scadenze."

                )


            if message:

                with st.expander(
                    "Dettaglio tecnico Yahoo/yfinance"
                ):

                    st.code(message)


    # ==================================================
    # SELECTBOX SCADENZE
    # ==================================================

    if st.session_state[
        "expirations"
    ]:

        expiration = st.selectbox(

            "Seleziona scadenza",

            st.session_state[
                "expirations"
            ],

            index=0

        )

    else:

        expiration = st.text_input(

            "Oppure inserisci scadenza (YYYY-MM-DD)",

            value=""

        )


    exp_ok = is_valid_date(
        expiration
    )


    # ==================================================
    # RANGE STRIKE
    # ==================================================

    st.divider()

    st.subheader(

        "Range strike (Grafico 2) – invariato"

    )


    range_low = st.slider(

        "Min strike (% di S0)",

        min_value=10,

        max_value=95,

        value=70,

        step=5

    )


    range_high = st.slider(

        "Max strike (% di S0)",

        min_value=105,

        max_value=300,

        value=130,

        step=5

    )


    n_points = st.slider(

        "Numero punti strike",

        min_value=10,

        max_value=200,

        value=60,

        step=10

    )


    st.divider()


    run = st.button(

        "Esegui analisi",

        type="primary",

        use_container_width=True,

        disabled=(
            not ticker
            or not exp_ok
        )

    )


# ======================================================
# ANALYSIS
# ======================================================

def run_analysis():

    # ==================================================
    # STORICO SOTTOSTANTE
    # ==================================================

    hist = fetch_history_1y(
        ticker
    )


    if hist.empty:

        raise ValueError(
            "Storico prezzi vuoto."
        )


    S0 = float(
        hist["Close"].iloc[-1]
    )


    sigma_hist = (
        compute_hist_vol(hist)
    )


    # ==================================================
    # TEMPO A SCADENZA
    # ==================================================

    expiration_date = (
        datetime.strptime(
            expiration,
            "%Y-%m-%d"
        )
    )


    T = (

        expiration_date
        - datetime.now()

    ).days / 365.0


    if T <= 0:

        raise ValueError(

            "Scadenza nel passato "
            "(o oggi)."

        )


    # ==================================================
    # OPTION CHAIN
    # ==================================================

    calls = None
    puts = None

    exp_effective = expiration

    err_first = None


    try:

        calls, puts = (
            fetch_option_chain(
                ticker,
                expiration
            )
        )

    except Exception as e:

        err_first = e

        calls = None
        puts = None


    # ==================================================
    # FALLBACK ALTRE SCADENZE
    # ==================================================

    if (
        calls is None
        or puts is None
    ):

        for exp_try in (

            st.session_state.get(
                "expirations",
                []
            )
            or []

        ):

            if (
                exp_try
                == expiration
            ):
                continue


            try:

                calls, puts = (
                    fetch_option_chain(
                        ticker,
                        exp_try
                    )
                )


                exp_effective = (
                    exp_try
                )


                expiration_date = (
                    datetime.strptime(
                        exp_effective,
                        "%Y-%m-%d"
                    )
                )


                T = (

                    expiration_date
                    - datetime.now()

                ).days / 365.0


                if T <= 0:

                    calls = None
                    puts = None

                    continue


                break


            except Exception:

                calls = None
                puts = None

                continue


    if (
        calls is None
        or puts is None
    ):

        raise ValueError(

            "Impossibile recuperare "
            "option chain da Yahoo. "
            f"Primo errore: {err_first}"

        )


    # ==================================================
    # GARANTISCE COLONNE
    # ==================================================

    calls = ensure_columns(
        calls
    )

    puts = ensure_columns(
        puts
    )


    # ==================================================
    # FILTRO TRADABILITÀ
    # ==================================================

    def apply_filters(
        df: pd.DataFrame
    ) -> pd.DataFrame:


        if df.empty:
            return df


        out = df.copy()


        out["mid"] = (
            out.apply(
                mid_price_row,
                axis=1
            )
        )


        # ----------------------------------------------
        # Bid / Ask validi
        # ----------------------------------------------

        has_mid = (

            (out["bid"] > 0)

            & (out["ask"] > 0)

            & (
                out["ask"]
                >= out["bid"]
            )

        )


        # ----------------------------------------------
        # Fallback last price
        # ----------------------------------------------

        has_last = (

            out[
                "lastPrice"
            ].notna()

            & (
                out[
                    "lastPrice"
                ] > 0
            )

        )


        cond = (

            (has_mid | has_last)

            & (

                out[
                    "openInterest"
                ].fillna(0)

                >= min_oi

            )

        )


        # ----------------------------------------------
        # Volume opzionale
        # ----------------------------------------------

        if (
            min_vol
            and min_vol > 0
        ):

            cond = (

                cond

                & (

                    out[
                        "volume"
                    ].fillna(0)

                    >= min_vol

                )

            )


        out = (
            out
            .loc[cond]
            .copy()
        )


        out = (
            out.dropna(
                subset=["mid"]
            )
        )


        return out


    calls_f = apply_filters(
        calls
    )

    puts_f = apply_filters(
        puts
    )


    if (
        calls_f.empty
        and puts_f.empty
    ):

        raise ValueError(

            "Dopo i filtri di tradabilità "
            "non rimane nessuna opzione. "
            "Riduci OI/volume "
            "(o mettili a 0)."

        )


    # ==================================================
    # STRIKE CALL E PUT
    # ==================================================

    calls_by_strike = (

        calls_f
        .drop_duplicates(
            subset=["strike"]
        )
        .set_index(
            "strike"
        )

    )


    puts_by_strike = (

        puts_f
        .drop_duplicates(
            subset=["strike"]
        )
        .set_index(
            "strike"
        )

    )


    strike_union = np.union1d(

        calls_by_strike.index.values,

        puts_by_strike.index.values

    )


    # ==================================================
    # TABELLA MARKET / BS
    # ==================================================

    rows = []


    for K in strike_union:

        K = float(K)


        # ==============================================
        # CALL
        # ==============================================

        if (
            K
            in calls_by_strike.index
        ):

            rc = (
                calls_by_strike.loc[K]
            )


            mkt_c = float(
                rc["mid"]
            )


            iv_c = (

                float(
                    rc[
                        "impliedVolatility"
                    ]
                )

                if pd.notna(
                    rc[
                        "impliedVolatility"
                    ]
                )

                else np.nan

            )


            bs_c = (

                bs_call(
                    S0,
                    K,
                    r,
                    T,
                    iv_c
                )

                if (
                    pd.notna(iv_c)
                    and iv_c > 0
                )

                else np.nan

            )


            diff_c = (

                mkt_c - bs_c

                if (
                    pd.notna(mkt_c)
                    and pd.notna(bs_c)
                )

                else np.nan

            )

        else:

            mkt_c = np.nan
            iv_c = np.nan
            bs_c = np.nan
            diff_c = np.nan


        # ==============================================
        # PUT
        # ==============================================

        if (
            K
            in puts_by_strike.index
        ):

            rp = (
                puts_by_strike.loc[K]
            )


            mkt_p = float(
                rp["mid"]
            )


            iv_p = (

                float(
                    rp[
                        "impliedVolatility"
                    ]
                )

                if pd.notna(
                    rp[
                        "impliedVolatility"
                    ]
                )

                else np.nan

            )


            bs_p = (

                bs_put(
                    S0,
                    K,
                    r,
                    T,
                    iv_p
                )

                if (
                    pd.notna(iv_p)
                    and iv_p > 0
                )

                else np.nan

            )


            diff_p = (

                mkt_p - bs_p

                if (
                    pd.notna(mkt_p)
                    and pd.notna(bs_p)
                )

                else np.nan

            )

        else:

            mkt_p = np.nan
            iv_p = np.nan
            bs_p = np.nan
            diff_p = np.nan


        rows.append({

            "Strike": K,

            "Market Call (mid)": mkt_c,

            "IV Call": iv_c,

            "BS Call (IV)": bs_c,

            "Diff (C) = M-BS": diff_c,

            "Market Put (mid)": mkt_p,

            "IV Put": iv_p,

            "BS Put (IV)": bs_p,

            "Diff (P) = M-BS": diff_p,

        })


    df = pd.DataFrame(
        rows
    )


    df = (

        df
        .sort_values("Strike")
        .reset_index(drop=True)

    )


    # ==================================================
    # DATI GRAFICO 2
    # ==================================================

    low = (

        S0
        * (
            range_low
            / 100.0
        )

    )


    high = (

        S0
        * (
            range_high
            / 100.0
        )

    )


    strike_range = np.linspace(

        low,

        high,

        n_points

    )


    df2_rows = []


    for K in strike_range:

        K = float(K)


        if (
            calls_f.empty
            or puts_f.empty
        ):

            break


        call_idx = (

            (
                calls_f["strike"]
                - K
            )
            .abs()
            .idxmin()

        )


        put_idx = (

            (
                puts_f["strike"]
                - K
            )
            .abs()
            .idxmin()

        )


        row_c = calls_f.loc[
            call_idx
        ]


        row_p = puts_f.loc[
            put_idx
        ]


        mkt_call = float(
            row_c["mid"]
        )


        mkt_put = float(
            row_p["mid"]
        )


        iv_call = (

            float(
                row_c[
                    "impliedVolatility"
                ]
            )

            if pd.notna(
                row_c[
                    "impliedVolatility"
                ]
            )

            else np.nan

        )


        iv_put = (

            float(
                row_p[
                    "impliedVolatility"
                ]
            )

            if pd.notna(
                row_p[
                    "impliedVolatility"
                ]
            )

            else np.nan

        )


        bs_call_iv = (

            bs_call(
                S0,
                K,
                r,
                T,
                iv_call
            )

            if (
                pd.notna(iv_call)
                and iv_call > 0
            )

            else np.nan

        )


        bs_put_iv = (

            bs_put(
                S0,
                K,
                r,
                T,
                iv_put
            )

            if (
                pd.notna(iv_put)
                and iv_put > 0
            )

            else np.nan

        )


        bs_call_hist = (
            bs_call(
                S0,
                K,
                r,
                T,
                sigma_hist
            )
        )


        bs_put_hist = (
            bs_put(
                S0,
                K,
                r,
                T,
                sigma_hist
            )
        )


        df2_rows.append({

            "Strike": K,

            "Market Call (mid)":
                mkt_call,

            "BS Call (IV strike)":
                bs_call_iv,

            "BS Call (Hist σ)":
                bs_call_hist,

            "Market Put (mid)":
                mkt_put,

            "BS Put (IV strike)":
                bs_put_iv,

            "BS Put (Hist σ)":
                bs_put_hist,

        })


    df2 = pd.DataFrame(
        df2_rows
    )


    # ==================================================
    # STRIKE DISPONIBILI STRATEGY BUILDER
    # ==================================================

    strikes_call = (

        df.loc[
            df[
                "Market Call (mid)"
            ].notna(),
            "Strike"
        ].values

    )


    strikes_put = (

        df.loc[
            df[
                "Market Put (mid)"
            ].notna(),
            "Strike"
        ].values

    )


    return {

        "ticker":
            ticker,

        "expiration":
            exp_effective,

        "S0":
            S0,

        "sigma_hist":
            sigma_hist,

        "T":
            T,

        "df":
            df,

        "df2":
            df2,

        "strikes_call":
            strikes_call,

        "strikes_put":
            strikes_put,

        "filters": {

            "min_oi":
                min_oi,

            "min_vol":
                min_vol,

        },

    }


# ======================================================
# ESECUZIONE ANALISI
# ======================================================

if run:

    try:

        st.session_state[
            "analysis_data"
        ] = run_analysis()


        st.session_state[
            "analysis_ready"
        ] = True


        st.session_state.setdefault(
            "strategy_legs",
            []
        )


    except Exception as e:

        st.session_state[
            "analysis_ready"
        ] = False


        st.error(
            str(e)
        )


# ======================================================
# BLOCCO FINCHÉ ANALISI NON ESEGUITA
# ======================================================

if not st.session_state.get(
    "analysis_ready",
    False
):

    st.info(

        "Esegui l'analisi per generare i grafici. "
        "Poi potrai costruire strategie multi-leg "
        "sotto senza rifare i grafici."

    )

    st.stop()


A = st.session_state[
    "analysis_data"
]


# ======================================================
# HEADER METRICS
# ======================================================

colA, colB, colC, colD = (
    st.columns(4)
)


colA.metric(

    "S0 (ultimo close)",

    f"{A['S0']:,.2f}"

)


colB.metric(

    "σ storica (1y)",

    f"{A['sigma_hist']:.4f}"

)


colC.metric(

    "T (anni)",

    f"{A['T']:.4f}"

)


colD.metric(

    "Scadenza effettiva",

    A["expiration"]

)


st.caption(

    f"Filtro tradabilità: "
    f"OI≥{A['filters']['min_oi']}"

    + (

        f", volume≥"
        f"{A['filters']['min_vol']}"

        if (
            A["filters"]["min_vol"]
            and
            A["filters"]["min_vol"] > 0
        )

        else ""

    )

    + ". "
      "(Se bid/ask mancano, "
      "mid=lastPrice se disponibile)"

)


# ======================================================
# TABELLA
# ======================================================

st.subheader(

    "Tabella: Market(mid) vs BS "
    "(IV strike-specifica) + "
    "Differenza (Market−BS)"

)


st.dataframe(

    A["df"],

    use_container_width=True

)


# ======================================================
# GRAFICO 1
# ======================================================

st.subheader(

    "Grafico 1: Differenza "
    "(Market − Black-Scholes) "
    "usando σ storica (mid-price)"

)


df_plot = A[
    "df"
].copy()


S0 = float(
    A["S0"]
)


T = float(
    A["T"]
)


sigma_hist = float(
    A["sigma_hist"]
)


diff_call_hist = []

diff_put_hist = []


for _, row in df_plot.iterrows():

    K = float(
        row["Strike"]
    )


    mkt_c = row[
        "Market Call (mid)"
    ]


    mkt_p = row[
        "Market Put (mid)"
    ]


    bs_c = bs_call(

        S0,
        K,
        r,
        T,
        sigma_hist

    )


    bs_p = bs_put(

        S0,
        K,
        r,
        T,
        sigma_hist

    )


    dc = (

        float(
            mkt_c - bs_c
        )

        if (
            pd.notna(mkt_c)
            and
            pd.notna(bs_c)
        )

        else np.nan

    )


    dp = (

        float(
            mkt_p - bs_p
        )

        if (
            pd.notna(mkt_p)
            and
            pd.notna(bs_p)
        )

        else np.nan

    )


    diff_call_hist.append(
        dc
    )


    diff_put_hist.append(
        dp
    )


x = (

    df_plot[
        "Strike"
    ]
    .astype(float)
    .values

)


call_y = np.array(

    diff_call_hist,

    dtype=float

)


put_y = np.array(

    diff_put_hist,

    dtype=float

)


fig1 = plt.figure(
    figsize=(14, 6)
)


plt.bar(

    x,

    call_y,

    label="Call (Market - BS)",

    alpha=0.9

)


plt.bar(

    x,

    put_y,

    label="Put (Market - BS)",

    alpha=0.9

)


plt.axhline(

    0,

    linestyle="--",

    linewidth=1

)


plt.axvline(

    S0,

    linestyle="--",

    linewidth=1,

    label="S0"

)


plt.title(

    "Market vs Black-Scholes — "
    "Price Difference"

)


plt.xlabel(
    "Strike"
)


plt.ylabel(

    "Difference "
    "(Market - BS)"

)


plt.grid(

    True,

    axis="y"

)


plt.legend()


plt.tight_layout()


st.pyplot(
    fig1
)


# ======================================================
# GRAFICO 2
# ======================================================

st.subheader(

    "Grafico 2: Market vs BS "
    "(σ storica vs IV strike-specifica) "
    "su range di strike"

)


df2 = A[
    "df2"
]


if (
    df2 is None
    or df2.empty
):

    st.warning(

        "Impossibile costruire il Grafico 2 "
        "(range strike o chain non validi "
        "dopo filtri)."

    )


else:

    fig2 = plt.figure(
        figsize=(14, 7)
    )


    plt.plot(

        df2["Strike"],

        df2[
            "Market Call (mid)"
        ],

        marker="o",

        label="Market Call (mid)"

    )


    plt.plot(

        df2["Strike"],

        df2[
            "BS Call (IV strike)"
        ],

        linestyle="--",

        label="BS Call (IV strike)"

    )


    plt.plot(

        df2["Strike"],

        df2[
            "BS Call (Hist σ)"
        ],

        linestyle="dotted",

        label="BS Call (Hist σ)"

    )


    plt.plot(

        df2["Strike"],

        df2[
            "Market Put (mid)"
        ],

        marker="o",

        label="Market Put (mid)"

    )


    plt.plot(

        df2["Strike"],

        df2[
            "BS Put (IV strike)"
        ],

        linestyle="--",

        label="BS Put (IV strike)"

    )


    plt.plot(

        df2["Strike"],

        df2[
            "BS Put (Hist σ)"
        ],

        linestyle="dotted",

        label="BS Put (Hist σ)"

    )


    plt.title(

        "Market(mid) vs "
        "Black–Scholes Option Prices"

    )


    plt.xlabel(
        "Strike"
    )


    plt.ylabel(
        "Option Price"
    )


    plt.grid(
        True
    )


    plt.legend()


    plt.tight_layout()


    st.pyplot(
        fig2
    )


    with st.expander(
        "Mostra tabella Grafico 2"
    ):

        st.dataframe(

            df2,

            use_container_width=True

        )


# ======================================================
# STRATEGY BUILDER
# ======================================================

st.divider()


st.subheader(

    "Strategie multi-leg "
    "(Call/Put, Buy/Sell) — "
    "senza ricalcolare i grafici"

)


if (
    "strategy_legs"
    not in st.session_state
):

    st.session_state[
        "strategy_legs"
    ] = []


c1, c2, c3, c4, c5 = (
    st.columns(
        [
            1.0,
            1.0,
            1.3,
            1.0,
            1.0
        ]
    )
)


side = c1.selectbox(

    "Side",

    [
        "Buy",
        "Sell"
    ],

    key="leg_side"

)


opt_type = c2.selectbox(

    "Tipo",

    [
        "Call",
        "Put"
    ],

    key="leg_type"

)


strikes_avail = (

    A["strikes_call"]

    if opt_type == "Call"

    else A["strikes_put"]

)


if (
    len(strikes_avail)
    == 0
):

    st.warning(

        "Nessuno strike disponibile "
        "(dopo filtri) "
        "per il tipo selezionato."

    )

    st.stop()


default_idx = int(

    np.argmin(

        np.abs(

            strikes_avail
            - A["S0"]

        )

    )

)


K_sel = float(

    c3.selectbox(

        "Strike",

        strikes_avail,

        index=default_idx,

        key="leg_strike"

    )

)


qty = int(

    c4.number_input(

        "Qty contratti",

        min_value=1,

        value=1,

        step=1,

        key="leg_qty"

    )

)


mult = int(

    c5.number_input(

        "Moltiplicatore",

        min_value=1,

        value=100,

        step=1,

        key="leg_mult"

    )

)


ST = st.number_input(

    "Prezzo sottostante a scadenza (ST) "
    "per valutazione strategia",

    min_value=0.0,

    value=float(
        A["S0"]
    ),

    step=float(

        max(
            1.0,
            A["S0"] * 0.01
        )

    ),

    key="strategy_ST"

)


# ======================================================
# PULSANTI STRATEGIA
# ======================================================

b1, b2, b3 = st.columns(

    [
        1.1,
        1.1,
        1.6
    ]

)


if b1.button(

    "Aggiungi gamba",

    type="primary"

):

    st.session_state[
        "strategy_legs"
    ].append({

        "Side":
            side,

        "Type":
            opt_type,

        "Strike":
            K_sel,

        "Qty":
            qty,

        "Mult":
            mult,

    })


if b2.button(

    "Rimuovi ultima gamba"

):

    if st.session_state[
        "strategy_legs"
    ]:

        st.session_state[
            "strategy_legs"
        ].pop()


if b3.button(

    "Svuota strategia"

):

    st.session_state[
        "strategy_legs"
    ] = []


legs = st.session_state[
    "strategy_legs"
]


if not legs:

    st.info(

        "Aggiungi una o più gambe. "
        "La strategia può includere "
        "sia Call che Put, "
        "Buy o Sell."

    )

    st.stop()


# ======================================================
# PREZZI STRATEGIA
# ======================================================

df_price = (

    A["df"]
    .set_index(
        "Strike"
    )

)


leg_rows = []


cashflow0_total = 0.0

payoff_total = 0.0

pnl_total = 0.0


for i, leg in enumerate(
    legs,
    start=1
):

    side_i = leg[
        "Side"
    ]

    typ_i = leg[
        "Type"
    ]

    K_i = float(
        leg["Strike"]
    )

    q_i = int(
        leg["Qty"]
    )

    m_i = int(
        leg["Mult"]
    )


    row = df_price.loc[
        K_i
    ]


    premium = float(

        row[
            "Market Call (mid)"
        ]

        if typ_i == "Call"

        else row[
            "Market Put (mid)"
        ]

    )


    pay_share = (
        payoff_at_expiry(
            typ_i,
            ST,
            K_i
        )
    )


    # ==================================================
    # BUY
    # ==================================================

    if (
        side_i
        == "Buy"
    ):

        cash0 = (

            -premium
            * q_i
            * m_i

        )


        payoff_pos = (

            +pay_share
            * q_i
            * m_i

        )


        pnl = (

            cash0
            + payoff_pos

        )


    # ==================================================
    # SELL
    # ==================================================

    else:

        cash0 = (

            +premium
            * q_i
            * m_i

        )


        payoff_pos = (

            -pay_share
            * q_i
            * m_i

        )


        pnl = (

            cash0
            + payoff_pos

        )


    cashflow0_total += cash0

    payoff_total += payoff_pos

    pnl_total += pnl


    leg_rows.append({

        "#":
            i,

        "Side":
            side_i,

        "Type":
            typ_i,

        "Strike":
            K_i,

        "Qty":
            q_i,

        "Mult":
            m_i,

        "Premium(mid) per share":
            premium,

        "Cashflow t0":
            cash0,

        "Payoff(pos) @ST":
            payoff_pos,

        "PnL @ST":
            pnl,

    })


legs_df = pd.DataFrame(
    leg_rows
)


st.dataframe(

    legs_df,

    use_container_width=True

)


# ======================================================
# METRICHE STRATEGIA
# ======================================================

notional_equiv = (

    A["S0"]

    * sum(

        int(
            l["Mult"]
        )

        * abs(
            int(
                l["Qty"]
            )
        )

        for l in legs

    )

)


m1, m2, m3, m4 = (
    st.columns(4)
)


m1.metric(

    "Cashflow totale t0",

    f"{cashflow0_total:,.2f}"

)


m2.metric(

    "Payoff posizione totale @ST",

    f"{payoff_total:,.2f}"

)


m3.metric(

    "PnL totale @ST",

    f"{pnl_total:,.2f}"

)


m4.metric(

    "Nozionale equivalente (oggi)",

    f"{notional_equiv:,.2f}"

)

    with st.expander("Mostra tabella df2"):
        st.dataframe(df2, use_container_width=True)

st.caption("Nota: i dati opzioni Yahoo possono avere lastPrice obsoleti o spread elevati; per confronti seri, spesso conviene usare mid-price (bid/ask).")
