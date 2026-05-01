"""
Planetary Regression — Streamlit App
OLS + Logistic Regression on Transit-to-Transit Aspects

Changes vs v1:
  1. Ephemeris loaded from GitHub (no upload widget)
  2. Chart 1 fitted lines rebased to actual price at window start (same scale)
  3. User can choose Inner Planets (multiselect, default kept)
  4. User can choose Sign Planets (multiselect, default kept)
"""

import warnings
import datetime
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import yfinance as yf
import statsmodels.api as sm
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, accuracy_score, balanced_accuracy_score,
)

warnings.filterwarnings("ignore")

# ── Ephemeris source — update this URL to match your GitHub repo ──────────────
EPHEMERIS_URL = (
    "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/"
    "main/planet_degrees.csv"
)

# ── Plot colours ──────────────────────────────────────────────────────────────
BG     = "#080818"
GOLD   = "#C8A84B"
TEAL   = "#00D4B4"
WHITE  = "#E8E8F4"
GREY   = "#2A2A4A"
ORANGE = "#FF8844"
GREEN  = "#44DD88"
RED    = "#FF4466"

# ── Planet universe ───────────────────────────────────────────────────────────
ALL_PLANETS_ORDERED = [
    "sun", "moon", "mercury", "venus", "mars",
    "jupiter", "saturn", "uranus", "neptune",
    "pluto", "true_node", "chiron",
]
DEFAULT_INNER = ["sun", "moon", "mercury", "venus", "mars"]
DEFAULT_SIGN  = ["mercury", "venus", "mars"]

ASPECTS   = [0, 60, 90, 120, 180]
ASP_NAMES = {0: "Conj", 60: "Sext", 90: "Sqr", 120: "Trin", 180: "Opp"}
SIGNS     = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Planetary Regression",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Rajdhani:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
        background-color: #080818;
        color: #E8E8F4;
    }
    .stApp { background-color: #080818; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0d0d2e 0%,#080818 100%);
        border-right: 1px solid #2A2A4A;
    }
    h1, h2, h3 { font-family: 'Cinzel', serif; color: #C8A84B; }
    .stButton > button {
        background: linear-gradient(135deg,#C8A84B,#a07830);
        color: #080818; font-family:'Cinzel',serif;
        font-weight:700; border:none; border-radius:4px;
        padding:10px 28px; font-size:14px; letter-spacing:1px;
        width:100%;
    }
    .stButton > button:hover { opacity:.88; }
    .metric-card {
        background:#0d0d28; border:1px solid #2A2A4A;
        border-radius:6px; padding:16px 20px; margin:4px 0;
    }
    .metric-label { color:#888; font-size:12px; text-transform:uppercase; letter-spacing:1px; }
    .metric-value { color:#C8A84B; font-size:24px; font-family:'Cinzel',serif; font-weight:600; }
    .metric-sub   { color:#aaa; font-size:11px; }
    .forecast-box {
        background:#0d0d28; border:1px solid #C8A84B40;
        border-radius:6px; padding:16px 20px; margin:8px 0;
    }
    .up-signal   { color:#44DD88; font-weight:700; }
    .down-signal { color:#FF4466; font-weight:700; }
    hr.gold { border:none; border-top:1px solid #C8A84B40; margin:20px 0; }
    div[data-testid="stDataFrame"] { background:#0d0d28; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def angular_diff(lon_a, lon_b):
    d = (lon_a - lon_b) % 360
    return np.where(d > 180, d - 360, d)


def valid_pairs(avail, inner_planets):
    """Planet pairs where at least one member is an inner planet."""
    planet_list = [p for p in ALL_PLANETS_ORDERED if p in avail]
    pairs, seen = [], set()
    for i, pa in enumerate(planet_list):
        for pb in planet_list[i + 1:]:
            if pa not in inner_planets and pb not in inner_planets:
                continue
            if (pa, pb) not in seen:
                seen.add((pa, pb))
                pairs.append((pa, pb))
    return pairs


def build_sign_dummies(eph_slice, feat_cols, sign_planets):
    for planet in sign_planets:
        if planet not in eph_slice.columns:
            continue
        lon      = eph_slice[planet].values.astype(float) % 360
        sign_idx = (lon // 30).astype(int)
        for s, sign in enumerate(SIGNS):
            feat_cols[f"{planet}_sign_{sign}"] = (sign_idx == s).astype(float)
    return feat_cols


def build_features(date_index, eph, orb_apply, orb_sep, inner_planets, sign_planets):
    eph_al   = eph.reindex(date_index, method="ffill")
    avail    = [p for p in ALL_PLANETS_ORDERED if p in eph_al.columns]
    pairs    = valid_pairs(avail, inner_planets)
    all_used = sorted(set(p for pair in pairs for p in pair))
    lons     = {p: eph_al[p].values.astype(float) % 360 for p in all_used}
    motion   = {p: np.gradient(np.unwrap(lons[p], period=360)) for p in all_used}

    feat_cols = {}
    for pa, pb in pairs:
        rel_motion = motion[pa] - motion[pb]
        for asp in ASPECTS:
            target  = (lons[pb] + asp) % 360
            gap     = angular_diff(lons[pa], target)
            abs_gap = np.abs(gap)
            applying   = (((rel_motion > 0) & (gap < 0)) |
                          ((rel_motion < 0) & (gap > 0)))
            separating = ~applying
            base = f"{pa}_T_{pb}__{asp}"
            feat_cols[f"{base}__apply"] = (applying   & (abs_gap <= orb_apply)).astype(float)
            feat_cols[f"{base}__sep"]   = (separating & (abs_gap <= orb_sep)).astype(float)
    feat_cols = build_sign_dummies(eph_al, feat_cols, sign_planets)

    feat_df = pd.DataFrame(feat_cols, index=date_index)
    feat_df = feat_df.loc[:, (feat_df > 0).any(axis=0)]
    return feat_df


def build_forecast_features(fut_dates, eph, feature_cols, orb_apply, orb_sep,
                             inner_planets, sign_planets):
    eph_fut  = eph.reindex(fut_dates, method="ffill")
    avail    = [p for p in ALL_PLANETS_ORDERED if p in eph_fut.columns]
    pairs    = valid_pairs(avail, inner_planets)
    all_used = sorted(set(p for pair in pairs for p in pair))
    lons     = {p: eph_fut[p].values.astype(float) % 360 for p in all_used}
    motion   = {p: np.gradient(np.unwrap(lons[p], period=360)) for p in all_used}

    feat_cols_d = {}
    for pa, pb in pairs:
        rel_motion = motion[pa] - motion[pb]
        for asp in ASPECTS:
            target     = (lons[pb] + asp) % 360
            gap        = angular_diff(lons[pa], target)
            abs_gap    = np.abs(gap)
            applying   = (((rel_motion > 0) & (gap < 0)) |
                          ((rel_motion < 0) & (gap > 0)))
            separating = ~applying
            base = f"{pa}_T_{pb}__{asp}"
            feat_cols_d[f"{base}__apply"] = (applying   & (abs_gap <= orb_apply)).astype(float)
            feat_cols_d[f"{base}__sep"]   = (separating & (abs_gap <= orb_sep)).astype(float)
    feat_cols_d = build_sign_dummies(eph_fut, feat_cols_d, sign_planets)

    fut_df = pd.DataFrame(feat_cols_d, index=fut_dates)
    return fut_df.reindex(columns=feature_cols, fill_value=0.0).values


def fig_to_st(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
    buf.seek(0)
    return buf


def metric_html(label, value, sub=""):
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"<div class='metric-sub'>{sub}</div></div>"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<h2 style='text-align:center;margin-bottom:4px'>🪐 Planetary</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='text-align:center;margin-top:0'>Regression</h2>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='gold'/>", unsafe_allow_html=True)

    # ── Stock settings ────────────────────────────────────────────────────────
    st.markdown("### 📈 Stock Settings")
    ticker = st.text_input(
        "Stock Ticker", value="GC=F",
        help="Yahoo Finance ticker, e.g. GC=F, SI=F, AAPL, BTC-USD",
    )
    stock_name = st.text_input(
        "Display Name", value="Gold Futures",
        help="Used in chart titles and labels",
    )

    today       = datetime.date.today()
    default_end = today + datetime.timedelta(days=365)

    start_date = st.date_input(
        "Data Start Date", value=datetime.date(2000, 1, 1),
        min_value=datetime.date(1950, 1, 1), max_value=today,
    )
    forecast_end = st.date_input(
        "Forecast End Date", value=default_end,
        min_value=today + datetime.timedelta(days=1),
    )

    st.markdown("<hr class='gold'/>", unsafe_allow_html=True)

    # ── Orb settings ──────────────────────────────────────────────────────────
    st.markdown("### 🔭 Orb Settings")
    orb_apply = st.slider("Applying Orb (°)", min_value=0.5, max_value=10.0,
                          value=4.0, step=0.5)
    orb_sep   = st.slider("Separating Orb (°)", min_value=0.5, max_value=5.0,
                          value=1.0, step=0.5)

    st.markdown("<hr class='gold'/>", unsafe_allow_html=True)

    # ── Planet selection ──────────────────────────────────────────────────────
    st.markdown("### 🪐 Planet Selection")

    inner_planets = st.multiselect(
        "Inner Planets",
        options=ALL_PLANETS_ORDERED,
        default=DEFAULT_INNER,
        help=(
            "Aspect pairs are built so at least one planet is an Inner Planet. "
            "Controls which planets are treated as 'inner'."
        ),
    )
    if not inner_planets:
        st.warning("⚠ Select at least one inner planet — using defaults.")
        inner_planets = DEFAULT_INNER

    sign_planets = st.multiselect(
        "Sign Planets (zodiac dummies)",
        options=ALL_PLANETS_ORDERED,
        default=DEFAULT_SIGN,
        help="Zodiac-sign dummy features are built for these planets.",
    )

    st.markdown("<hr class='gold'/>", unsafe_allow_html=True)
    run_btn = st.button("⚡ Run Analysis")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='text-align:center;letter-spacing:3px;margin-bottom:4px'>"
    "PLANETARY REGRESSION</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#888;font-size:14px;letter-spacing:2px'>"
    "OLS · LOGISTIC · TRANSIT-TO-TRANSIT ASPECTS</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='gold'/>", unsafe_allow_html=True)

if not run_btn:
    st.markdown(
        """
        <div style='text-align:center;padding:60px 20px;color:#555'>
            <div style='font-size:64px;margin-bottom:16px'>🪐</div>
            <div style='font-family:Cinzel,serif;font-size:20px;color:#C8A84B;margin-bottom:12px'>
                Configure Settings &amp; Run Analysis
            </div>
            <div style='font-size:14px;line-height:2.0'>
                1. Set your stock ticker and display name<br>
                2. Choose data start date and forecast end date<br>
                3. Adjust applying / separating orb angles<br>
                4. Optionally customise Inner Planets and Sign Planets<br>
                5. Click <strong style='color:#C8A84B'>⚡ Run Analysis</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD EPHEMERIS FROM GITHUB
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_ephemeris(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, index_col="date", parse_dates=True)
    return df

with st.spinner("Loading ephemeris from GitHub …"):
    try:
        eph = load_ephemeris(EPHEMERIS_URL)
        st.success(
            f"✓ Ephemeris loaded: {eph.shape[0]:,} rows, "
            f"{eph.shape[1]} planets  "
            f"({eph.index[0].date()} → {eph.index[-1].date()})"
        )
    except Exception as e:
        st.error(
            f"Could not load ephemeris from GitHub.\n\n"
            f"**Error:** {e}\n\n"
            f"**URL tried:** `{EPHEMERIS_URL}`\n\n"
            "Update `EPHEMERIS_URL` at the top of the script to point to your "
            "`planet_degrees.csv` raw file on GitHub."
        )
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD STOCK DATA
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner(f"Downloading {ticker} from Yahoo Finance …"):
    try:
        stock_end_str = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        raw = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=stock_end_str,
            progress=False,
            auto_adjust=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw["Close"]      = pd.to_numeric(raw["Close"], errors="coerce")
        raw               = raw[["Close"]].dropna()
        raw["log_price"]  = np.log(raw["Close"])
        raw["log_return"] = raw["log_price"].diff()
        raw               = raw.dropna(subset=["log_return"])
        raw["direction"]  = np.sign(raw["log_return"]).astype(int)
        raw["direction"]  = raw["direction"].replace(0, 1)
        st.success(
            f"✓ {ticker} downloaded: {len(raw):,} trading days "
            f"({raw.index[0].date()} → {raw.index[-1].date()})"
        )
    except Exception as e:
        st.error(f"Failed to download {ticker}: {e}")
        st.stop()

price_df  = raw
dates_all = price_df.index
y_ret     = price_df["log_return"].values
y_dir     = price_df["direction"].values
close_all = price_df["Close"].values


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Building feature matrix …"):
    feat_all     = build_features(dates_all, eph, orb_apply, orb_sep,
                                  inner_planets, sign_planets)
    feature_cols = feat_all.columns.tolist()
    n_avail      = len(feature_cols)
    pairs_used   = valid_pairs(
        [p for p in ALL_PLANETS_ORDERED if p in eph.columns], inner_planets
    )
    n_sign_cols = sum(1 for c in feature_cols if "_sign_" in c)
    n_asp_cols  = n_avail - n_sign_cols
    X_raw       = feat_all.values
    X_con       = np.hstack([np.ones((len(X_raw), 1)), X_raw])

st.markdown(
    f"<div class='metric-card'><span class='metric-label'>Features built</span><br>"
    f"<span class='metric-value'>{n_avail}</span> "
    f"<span class='metric-sub'>"
    f"{n_asp_cols} aspect + {n_sign_cols} sign dummies · "
    f"{len(pairs_used)} planet pairs · "
    f"Apply {orb_apply}° / Sep {orb_sep}° · "
    f"Inner: {', '.join(inner_planets)} · "
    f"Sign: {', '.join(sign_planets) if sign_planets else 'none'}"
    f"</span></div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  FIT FULL-SAMPLE MODELS
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Fitting OLS + Logistic on full sample …"):
    ols_full       = sm.OLS(y_ret, X_con).fit()
    y_fit_ols_full = np.asarray(ols_full.fittedvalues)

    sc_full   = StandardScaler()
    X_sc_full = sc_full.fit_transform(X_raw)
    log_full  = LogisticRegression(
        C=0.1, penalty="l2", solver="lbfgs",
        max_iter=1000, class_weight="balanced", random_state=42,
    )
    log_full.fit(X_sc_full, y_dir)
    y_fit_log_full = log_full.predict(X_sc_full)

# ── Full-history cumulative paths (rebased to first close) ────────────────────
# Used unchanged in Plots 2, 3, 4 — only Plot 1 uses a rebased sub-window.
first_close    = close_all[0]
cumret_ols_fit = first_close * np.exp(np.cumsum(y_fit_ols_full))
log_step       = y_fit_log_full * np.abs(y_ret)
cumret_log_fit = first_close * np.exp(np.cumsum(log_step))
last_actual    = close_all[-1]
last_ols_fit   = cumret_ols_fit[-1]   # used in Plots 2, 3 connectors
last_log_fit   = cumret_log_fit[-1]

ols_is_r2   = r2_score(y_ret, y_fit_ols_full)
log_is_acc  = accuracy_score(y_dir, y_fit_log_full)
log_is_bacc = balanced_accuracy_score(y_dir, y_fit_log_full)


# ══════════════════════════════════════════════════════════════════════════════
#  FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Generating forecast …"):
    forecast_start = (dates_all[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fut_dates      = pd.date_range(
        start=forecast_start,
        end=forecast_end.strftime("%Y-%m-%d"),
        freq="B",
    )
    n_fut = len(fut_dates)

    X_fut        = build_forecast_features(
        fut_dates, eph, feature_cols, orb_apply, orb_sep, inner_planets, sign_planets
    )
    ols_params   = np.asarray(ols_full.params)
    X_fut_con    = np.hstack([np.ones((n_fut, 1)), X_fut])
    y_fore_ols_r = X_fut_con @ ols_params
    fore_ols_cum = last_actual * np.exp(np.cumsum(y_fore_ols_r))

    X_fut_sc       = sc_full.transform(X_fut)
    y_fore_log     = log_full.predict(X_fut_sc)
    fore_log_cum   = np.cumsum(y_fore_log)
    mean_abs_ret   = np.abs(y_ret).mean()
    fore_log_price = last_actual * np.exp(np.cumsum(y_fore_log * mean_abs_ret))

n_up   = int((y_fore_log == 1).sum())
n_down = int((y_fore_log == -1).sum())


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS ROW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>📊 Model Summary</h2>", unsafe_allow_html=True)

chg_pct = (fore_ols_cum[-1] / last_actual - 1) * 100
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.markdown(metric_html("Last Price",        f"{last_actual:,.4f}",     f"{ticker} · {dates_all[-1].date()}"),    unsafe_allow_html=True)
c2.markdown(metric_html("OLS In-Sample R²",  f"{ols_is_r2:.4f}",        "Full sample fit"),                       unsafe_allow_html=True)
c3.markdown(metric_html("Logit Accuracy",    f"{log_is_acc:.4f}",       f"BalAcc {log_is_bacc:.4f}"),             unsafe_allow_html=True)
c4.markdown(metric_html("OLS Forecast End",  f"{fore_ols_cum[-1]:,.4f}", f"{chg_pct:+.2f}% vs last"),            unsafe_allow_html=True)
c5.markdown(metric_html("Up Days (Logit)",   f"{n_up}",                  f"{n_up/n_fut*100:.1f}% of {n_fut}d"),  unsafe_allow_html=True)
c6.markdown(metric_html("Down Days (Logit)", f"{n_down}",                f"{n_down/n_fut*100:.1f}% of {n_fut}d"),unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  NEXT 3 DAYS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>📅 Next 3 Trading Days Forecast</h2>", unsafe_allow_html=True)

cols = st.columns(3)
for k in range(min(3, n_fut)):
    o_ret  = y_fore_ols_r[k] * 100
    o_cum  = fore_ols_cum[k]
    l_sig  = y_fore_log[k]
    o_dir  = "▲ UP" if y_fore_ols_r[k] > 0 else "▼ DOWN"
    l_dir  = "▲ UP" if l_sig == 1 else "▼ DOWN"
    o_cls  = "up-signal" if y_fore_ols_r[k] > 0 else "down-signal"
    l_cls  = "up-signal" if l_sig == 1 else "down-signal"
    cons   = "CONSENSUS" if o_dir == l_dir else "SPLIT"
    cons_c = "#44DD88" if cons == "CONSENSUS" else "#FF8844"
    with cols[k]:
        st.markdown(
            f"""<div class='forecast-box'>
            <div style='color:#888;font-size:11px;text-transform:uppercase;letter-spacing:1px'>Day +{k+1}</div>
            <div style='font-family:Cinzel,serif;font-size:15px;color:#C8A84B;margin:4px 0'>{fut_dates[k].strftime('%B %d, %Y')}</div>
            <div style='font-size:20px;font-weight:700;margin:8px 0'>
                <span style='color:{cons_c}'>{cons}</span>
            </div>
            <div style='margin:6px 0'>
                <span style='color:#888;font-size:12px'>OLS → </span>
                <span class='{o_cls}' style='font-size:15px'>{o_dir}</span>
                <span style='color:#aaa;font-size:12px'> ({o_ret:+.4f}%)</span>
            </div>
            <div style='margin:6px 0'>
                <span style='color:#888;font-size:12px'>Logit → </span>
                <span class='{l_cls}' style='font-size:15px'>{l_dir}</span>
            </div>
            <div style='margin-top:10px;padding-top:10px;border-top:1px solid #2A2A4A'>
                <span style='color:#888;font-size:11px'>OLS cum. price: </span>
                <span style='color:#C8A84B;font-size:13px'>{o_cum:,.4f}</span>
            </div>
            </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — Last 12M fitted + 3-Month forecast
#
#  FIX: Both fitted series are rebased so they START at the actual price at
#  the beginning of the 12-month window, making all three lines directly
#  comparable on the same price scale.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>📈 Chart 1 — Last 12 Months + 3-Month Forecast</h2>", unsafe_allow_html=True)

TRADING_MONTH   = 63
last_year_start = dates_all[-1] - pd.DateOffset(months=12)
mask_ly         = dates_all >= last_year_start
dates_ly        = dates_all[mask_ly]
actual_ly       = close_all[mask_ly]

# Integer position of window start in the full arrays
window_start_pos = int(np.where(mask_ly)[0][0])

# OLS: cumulate log-returns within the window, rebase so first value = actual_ly[0]
ols_ret_win = y_fit_ols_full[window_start_pos:]
ols_fit_ly  = actual_ly[0] * np.exp(
    np.cumsum(ols_ret_win) - ols_ret_win[0]
)

# Logistic: cumulate signed-|return| steps within the window, same rebase
log_step_win = y_fit_log_full[window_start_pos:] * np.abs(y_ret[window_start_pos:])
log_fit_ly   = actual_ly[0] * np.exp(
    np.cumsum(log_step_win) - log_step_win[0]
)

# Endpoints of the rebased fitted series (used to connect to forecast lines)
last_ols_fit_ly = float(ols_fit_ly[-1])
last_log_fit_ly = float(log_fit_ly[-1])

fore_dates_3m = fut_dates[:TRADING_MONTH]
fore_ols_3m   = fore_ols_cum[:TRADING_MONTH]
fore_log_3m   = fore_log_price[:TRADING_MONTH]

all_vals  = np.concatenate([
    actual_ly, ols_fit_ly, log_fit_ly,
    fore_ols_3m if len(fore_ols_3m) else np.array([actual_ly[-1]]),
    fore_log_3m if len(fore_log_3m) else np.array([actual_ly[-1]]),
])
y_top_val = float(all_vals.max())

fig1, ax = plt.subplots(figsize=(18, 6), facecolor=BG)
ax.set_facecolor("#0D0D28")
for sp in ax.spines.values(): sp.set_color(GREY)
ax.tick_params(colors=WHITE, labelsize=8)

ax.plot(dates_ly, actual_ly,  color=TEAL,   lw=2.5, zorder=6,
        label=f"Actual {stock_name}")
ax.plot(dates_ly, ols_fit_ly, color=ORANGE, lw=1.5, alpha=0.85, zorder=4,
        label="OLS fitted (rebased to window start)")
ax.plot(dates_ly, log_fit_ly, color=GREEN,  lw=1.5, alpha=0.85, zorder=4,
        label="Logistic fitted (rebased to window start)")

if len(fore_dates_3m):
    ax.axvline(dates_ly[-1], color=GOLD, lw=1.5, ls=":", alpha=0.9, zorder=5)
    ax.text(dates_ly[-1], y_top_val * 1.002, "  Forecast→",
            color=GOLD, fontsize=8, va="bottom", fontweight="bold")
    ax.axvspan(fore_dates_3m[0], fore_dates_3m[-1], alpha=0.06, color=GOLD, zorder=1)

    # OLS connector: from rebased fitted end → first forecast point (anchored to last_actual)
    ax.plot([dates_ly[-1], fore_dates_3m[0]], [last_ols_fit_ly, fore_ols_3m[0]],
            color=ORANGE, lw=1.2, alpha=0.7, zorder=4)
    ax.plot(fore_dates_3m, fore_ols_3m, color=ORANGE, lw=2.0, ls="--", zorder=5,
            label=(f"OLS 3M  end={fore_ols_3m[-1]:,.4f}  "
                   f"({(fore_ols_3m[-1]/last_actual-1)*100:+.1f}%)"))
    ax.scatter([fore_dates_3m[-1]], [fore_ols_3m[-1]], color=ORANGE, s=60, zorder=8)
    ax.annotate(f"{fore_ols_3m[-1]:,.4f}",
                xy=(fore_dates_3m[-1], fore_ols_3m[-1]),
                xytext=(8, 4), textcoords="offset points",
                color=ORANGE, fontsize=8, fontweight="bold")

    # Logistic connector
    ax.plot([dates_ly[-1], fore_dates_3m[0]], [last_log_fit_ly, fore_log_3m[0]],
            color=GREEN, lw=1.2, alpha=0.7, zorder=4)
    ax.plot(fore_dates_3m, fore_log_3m, color=GREEN, lw=2.0, ls="--", zorder=5,
            label=(f"Logistic 3M  end={fore_log_3m[-1]:,.4f}  "
                   f"({(fore_log_3m[-1]/last_actual-1)*100:+.1f}%)"))
    ax.scatter([fore_dates_3m[-1]], [fore_log_3m[-1]], color=GREEN, s=60, zorder=8)
    ax.annotate(f"{fore_log_3m[-1]:,.4f}",
                xy=(fore_dates_3m[-1], fore_log_3m[-1]),
                xytext=(8, -14), textcoords="offset points",
                color=GREEN, fontsize=8, fontweight="bold")

ax.scatter([dates_ly[-1]], [last_actual], color=TEAL, s=70, zorder=8,
           label=f"Last actual: {last_actual:,.4f}")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Price Level", color=WHITE, fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.2f}"))
ax.set_title(
    f"{stock_name} ({ticker}) — Last 12M Fitted + 3-Month Forecast  (OLS & Logistic)\n"
    f"Fitted lines rebased to window-start price · "
    f"Pairs: {len(pairs_used)}  ·  Features: {n_avail}  ·  "
    f"Orb: apply={orb_apply}°/sep={orb_sep}°",
    color=GOLD, fontsize=11, fontweight="bold",
)
ax.legend(fontsize=8, facecolor="#1A1A38", labelcolor=WHITE, loc="upper left")
fig1.tight_layout()
st.image(fig_to_st(fig1), use_container_width=True)
plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — 90-day peaks & troughs
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>🔍 Chart 2 — 90-Day OLS Forecast (Peaks & Troughs)</h2>", unsafe_allow_html=True)

FORECAST_DAYS = 90
fore_dates_90 = fut_dates[:FORECAST_DAYS]
fore_ols_90   = fore_ols_cum[:FORECAST_DAYS]

last_6m_start = dates_all[-1] - pd.DateOffset(months=6)
mask_6m       = dates_all >= last_6m_start
dates_6m      = dates_all[mask_6m]
actual_6m     = close_all[mask_6m]
ols_fit_6m    = cumret_ols_fit[mask_6m]

peak_idx, trough_idx = [], []
if len(fore_ols_90) >= 7:
    W_pt = 3
    for i in range(W_pt, len(fore_ols_90) - W_pt):
        w = fore_ols_90[i - W_pt:i + W_pt + 1]
        if (fore_ols_90[i] == w.max()
                and fore_ols_90[i] > fore_ols_90[i - 1]
                and fore_ols_90[i] > fore_ols_90[i + 1]):
            peak_idx.append(i)
        if (fore_ols_90[i] == w.min()
                and fore_ols_90[i] < fore_ols_90[i - 1]
                and fore_ols_90[i] < fore_ols_90[i + 1]):
            trough_idx.append(i)
    ai = int(np.argmax(fore_ols_90)); ni = int(np.argmin(fore_ols_90))
    if ai not in peak_idx:   peak_idx.append(ai)
    if ni not in trough_idx: trough_idx.append(ni)
    peak_idx   = sorted(set(peak_idx))
    trough_idx = sorted(set(trough_idx))

fig2, ax = plt.subplots(figsize=(20, 7), facecolor=BG)
ax.set_facecolor("#0D0D28")
for sp in ax.spines.values(): sp.set_color(GREY)
ax.tick_params(colors=WHITE, labelsize=8.5)

ax.plot(dates_6m, actual_6m,  color=TEAL,   lw=2.5, zorder=6, label=f"Actual {stock_name} (6M)")
ax.plot(dates_6m, ols_fit_6m, color=ORANGE, lw=1.4, alpha=0.80, zorder=4, label="OLS fitted")

if len(fore_dates_90):
    ax.axvline(fore_dates_90[0], color=GOLD, lw=2.0, ls=":", alpha=0.95, zorder=5)
    ax.axvspan(fore_dates_90[0], fore_dates_90[-1], alpha=0.07, color=GOLD, zorder=1)
    ax.plot([dates_6m[-1], fore_dates_90[0]], [last_ols_fit, fore_ols_90[0]],
            color=ORANGE, lw=1.2, alpha=0.7, zorder=4)
    ax.plot(fore_dates_90, fore_ols_90, color=ORANGE, lw=2.4, ls="--", zorder=5,
            label=(f"OLS {FORECAST_DAYS}d  end={fore_ols_90[-1]:,.4f}  "
                   f"({(fore_ols_90[-1]/last_actual-1)*100:+.2f}%)"))

    all_y    = np.concatenate([actual_6m, ols_fit_6m, fore_ols_90])
    y_min_ax = all_y.min() * 0.993; y_max_ax = all_y.max() * 1.012
    lyt = y_max_ax * 0.9985; lyb = y_min_ax * 1.0015

    for i in peak_idx:
        d = fore_dates_90[i]; v = fore_ols_90[i]
        ax.axvline(d, color="#FF4466", lw=1.3, ls="--", alpha=0.85, zorder=6)
        ax.scatter([d], [v], color="#FF4466", s=55, zorder=8, marker="^")
        ax.text(d, lyt, f"  ▲ {d.strftime('%b %d')}\n  {v:,.2f}",
                color="#FF8899", fontsize=7.5, va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a0010",
                          edgecolor="#FF4466", alpha=0.85))

    for i in trough_idx:
        d = fore_dates_90[i]; v = fore_ols_90[i]
        ax.axvline(d, color="#44FF88", lw=1.3, ls="--", alpha=0.85, zorder=6)
        ax.scatter([d], [v], color="#44FF88", s=55, zorder=8, marker="v")
        ax.text(d, lyb, f"  ▼ {d.strftime('%b %d')}\n  {v:,.2f}",
                color="#88FFAA", fontsize=7.5, va="bottom", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#001a0a",
                          edgecolor="#44FF88", alpha=0.85))

    ax.set_ylim(y_min_ax, y_max_ax)

ax.scatter([dates_6m[-1]], [last_actual], color=TEAL, s=80, zorder=8,
           label=f"Last: {last_actual:,.4f}")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
ax.xaxis.grid(True, which="major", color=GREY, alpha=0.35, lw=0.6)
ax.yaxis.grid(True, color=GREY, alpha=0.25, lw=0.5)
ax.set_ylabel("Price Level", color=WHITE, fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.2f}"))
ax.set_title(
    f"{stock_name} — {FORECAST_DAYS}-Day OLS Forecast  (peaks & troughs)\n"
    f"Pairs: {len(pairs_used)}  ·  Features: {n_avail}  ·  "
    f"Orb: apply={orb_apply}°/sep={orb_sep}°",
    color=GOLD, fontsize=11, fontweight="bold",
)
ax.legend(fontsize=9, facecolor="#1A1A38", labelcolor=WHITE, loc="upper left")
fig2.tight_layout(pad=2.0)
st.image(fig_to_st(fig2), use_container_width=True)
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — OLS Full-Period Forecast
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>📉 Chart 3 — OLS Full-Period Forecast</h2>", unsafe_allow_html=True)

fig3, ax = plt.subplots(figsize=(20, 7), facecolor=BG)
ax.set_facecolor("#0D0D28")
for sp in ax.spines.values(): sp.set_color(GREY)
ax.tick_params(colors=WHITE, labelsize=8.5)

ax.fill_between(fut_dates, last_actual, fore_ols_cum,
                where=(fore_ols_cum >= last_actual),
                color="#00AA55", alpha=0.18, zorder=2, label="Cumulative gain")
ax.fill_between(fut_dates, last_actual, fore_ols_cum,
                where=(fore_ols_cum < last_actual),
                color="#FF3355", alpha=0.18, zorder=2, label="Cumulative loss")
ax.axhline(last_actual, color=GOLD, lw=1.2, ls=":", alpha=0.8, zorder=3,
           label=f"Last actual: {last_actual:,.4f}")
ax.plot(fut_dates, fore_ols_cum, color=ORANGE, lw=2.5, zorder=5, label="OLS forecast")

peak_yr   = int(np.argmax(fore_ols_cum))
trough_yr = int(np.argmin(fore_ols_cum))
for idx, col, marker, label_s in [
    (peak_yr,   "#FF4466", "^", "Peak"),
    (trough_yr, "#44FF88", "v", "Trough"),
]:
    d = fut_dates[idx]; v = fore_ols_cum[idx]
    ax.scatter([d], [v], color=col, s=100, zorder=9, marker=marker)
    ax.axvline(d, color=col, lw=1.0, ls="--", alpha=0.6, zorder=4)
    va = "bottom" if marker == "^" else "top"
    dy = v * 1.003 if marker == "^" else v * 0.997
    ax.annotate(
        f"  {label_s}\n  {d.strftime('%b %d, %Y')}\n  {v:,.4f}",
        xy=(d, v), xytext=(d, dy),
        color="#FF8899" if marker == "^" else "#88FFAA",
        fontsize=8.5, va=va, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="#1a0010" if marker == "^" else "#001a0a",
                  edgecolor=col, alpha=0.9),
    )

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8.5)
ax.xaxis.grid(True, which="major", color=GREY, alpha=0.40, lw=0.7)
ax.yaxis.grid(True, color=GREY, alpha=0.25, lw=0.5)
ax.set_ylabel("Price Level (OLS cumulative)", color=WHITE, fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.2f}"))
chg_pct3 = (fore_ols_cum[-1] / last_actual - 1) * 100
ax.set_title(
    f"{stock_name} ({ticker}) — OLS Forecast  "
    f"{fut_dates[0].strftime('%b %d, %Y')} → {fut_dates[-1].strftime('%b %d, %Y')}\n"
    f"End: {fore_ols_cum[-1]:,.4f}  ({chg_pct3:+.2f}% vs last)  ·  Features: {n_avail}",
    color=GOLD, fontsize=11, fontweight="bold",
)
ax.legend(fontsize=9, facecolor="#1A1A38", labelcolor=WHITE, loc="upper left", ncol=2)
fig3.tight_layout(pad=2.0)
st.image(fig_to_st(fig3), use_container_width=True)
plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 4 — Logistic Direction Forecast
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>🔮 Chart 4 — Logistic Direction Forecast</h2>", unsafe_allow_html=True)


def get_runs(dates, signals):
    runs = []
    if len(signals) == 0:
        return runs
    cur_sig = signals[0]; cur_start = dates[0]
    for i in range(1, len(signals)):
        if signals[i] != cur_sig:
            runs.append((cur_start, dates[i - 1], cur_sig))
            cur_sig = signals[i]; cur_start = dates[i]
    runs.append((cur_start, dates[-1], cur_sig))
    return runs


runs       = get_runs(fut_dates, y_fore_log)
cum_signal = np.cumsum(y_fore_log)

fig4, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(20, 9), facecolor=BG,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
)
for ax in (ax_top, ax_bot):
    ax.set_facecolor("#0D0D28")
    for sp in ax.spines.values(): sp.set_color(GREY)
    ax.tick_params(colors=WHITE, labelsize=8.5)

ax_top.axhline(last_actual, color=GOLD, lw=1.0, ls=":", alpha=0.7)
for start, end, sig in runs:
    col = "#00AA55" if sig == 1 else "#FF3355"
    ax_top.axvspan(start, end + pd.Timedelta(days=1), alpha=0.22, color=col, zorder=1)
ax_top.plot(fut_dates, fore_log_price, color=WHITE, lw=2.2, zorder=5,
            label="Logistic cum. direction (price-scaled)")
for i in range(1, n_fut):
    if y_fore_log[i] != y_fore_log[i - 1]:
        ax_top.axvline(fut_dates[i], color=GOLD, lw=0.8, ls="--", alpha=0.55, zorder=4)
ax_top.scatter([fut_dates[0]],  [fore_log_price[0]],  color=TEAL,  s=80, zorder=8)
ax_top.scatter([fut_dates[-1]], [fore_log_price[-1]], color=WHITE, s=80, zorder=8)
ax_top.annotate(
    f"{fore_log_price[-1]:,.4f}",
    xy=(fut_dates[-1], fore_log_price[-1]),
    xytext=(8, 4), textcoords="offset points",
    color=WHITE, fontsize=8.5, fontweight="bold",
)
up_patch   = mpatches.Patch(color="#00AA55", alpha=0.5, label="Up period (+1)")
down_patch = mpatches.Patch(color="#FF3355", alpha=0.5, label="Down period (-1)")
ax_top.legend(
    handles=[up_patch, down_patch] + ax_top.get_legend_handles_labels()[0],
    fontsize=8.5, facecolor="#1A1A38", labelcolor=WHITE, loc="upper left", ncol=2,
)
ax_top.set_ylabel("Price-scaled signal", color=WHITE, fontsize=10)
ax_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.2f}"))
ax_top.xaxis.set_visible(False)
ax_top.yaxis.grid(True, color=GREY, alpha=0.25, lw=0.5)

ax_bot.axhline(0, color=GOLD, lw=1.0, ls=":", alpha=0.7)
ax_bot.fill_between(fut_dates, 0, cum_signal,
                    where=(cum_signal >= 0), color="#00AA55", alpha=0.45, zorder=2)
ax_bot.fill_between(fut_dates, 0, cum_signal,
                    where=(cum_signal < 0), color="#FF3355", alpha=0.45, zorder=2)
ax_bot.plot(fut_dates, cum_signal, color=WHITE, lw=1.5, zorder=5)
ax_bot.set_ylabel("Cum. signal\n(+1/−1 sum)", color=WHITE, fontsize=9)
ax_bot.yaxis.grid(True, color=GREY, alpha=0.25, lw=0.5)
ax_bot.xaxis.set_major_locator(mdates.MonthLocator())
ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax_bot.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8.5)
ax_bot.xaxis.grid(True, which="major", color=GREY, alpha=0.40, lw=0.7)

fig4.suptitle(
    f"{stock_name} ({ticker}) — Logistic Direction Forecast\n"
    f"{fut_dates[0].strftime('%b %d, %Y')} → {fut_dates[-1].strftime('%b %d, %Y')}  ·  "
    f"Up: {n_up} ({n_up/n_fut*100:.1f}%)  ·  Down: {n_down} ({n_down/n_fut*100:.1f}%)  ·  "
    f"Features: {n_avail}  ·  C=0.1 L2",
    color=GOLD, fontsize=11, fontweight="bold", y=0.998,
)
fig4.tight_layout(pad=2.0)
st.image(fig_to_st(fig4), use_container_width=True)
plt.close(fig4)


# ══════════════════════════════════════════════════════════════════════════════
#  ACTIVE ASPECTS TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>🔭 Active Aspects — Last 7 Days + Next 30 Days</h2>", unsafe_allow_html=True)

col_names_ols   = ["const"] + feature_cols
ols_full_params = pd.Series(np.asarray(ols_full.params),  index=col_names_ols)
ols_full_pvals  = pd.Series(np.asarray(ols_full.pvalues), index=col_names_ols)

with st.spinner("Computing active aspects …"):
    last_date    = dates_all[-1]
    window_start = last_date - pd.Timedelta(days=10)
    window_end   = last_date + pd.Timedelta(days=30)
    past_dates_w = dates_all[dates_all >= window_start]
    fut_dates_w  = pd.date_range(
        start=last_date + pd.Timedelta(days=1), end=window_end, freq="B"
    )
    all_win_d = past_dates_w.append(fut_dates_w)

    eph_win  = eph.reindex(all_win_d, method="ffill")
    avail_w  = [p for p in ALL_PLANETS_ORDERED if p in eph_win.columns]
    pairs_w  = valid_pairs(avail_w, inner_planets)
    all_uw   = sorted(set(p for pr in pairs_w for p in pr))
    lons_w   = {p: eph_win[p].values.astype(float) % 360 for p in all_uw}
    motion_w = {p: np.gradient(np.unwrap(lons_w[p], period=360)) for p in all_uw}

    rows = []
    for pa, pb in pairs_w:
        rel_motion = motion_w[pa] - motion_w[pb]
        for asp in ASPECTS:
            gap     = angular_diff(lons_w[pa], (lons_w[pb] + asp) % 360)
            abs_gap = np.abs(gap)
            applying   = (((rel_motion > 0) & (gap < 0)) | ((rel_motion < 0) & (gap > 0)))
            separating = ~applying
            base = f"{pa}_T_{pb}__{asp}"
            for col, phase, mask, olim in [
                (f"{base}__apply", "Applying",   applying,   orb_apply),
                (f"{base}__sep",   "Separating", separating, orb_sep),
            ]:
                if col not in ols_full_params.index:
                    continue
                coef = ols_full_params[col]
                for i, date in enumerate(all_win_d):
                    if mask[i] and abs_gap[i] <= olim:
                        rows.append({
                            "Date":      date.date(),
                            "Period":    "PAST" if date <= last_date else "FUTURE",
                            "Planet A":  pa,
                            "Aspect":    ASP_NAMES[asp],
                            "Planet B":  pb,
                            "Phase":     phase,
                            "Orb °":     round(float(abs_gap[i]), 2),
                            "OLS Coef":  round(float(coef), 6),
                            "Effect %":  round(float(coef) * 100, 3),
                            "Direction": "▲ Bullish" if coef > 0 else "▼ Bearish",
                            "p-value":   round(float(ols_full_pvals[col]), 4),
                        })

    asp_table = pd.DataFrame(rows).sort_values(["Date"]).reset_index(drop=True)

if len(asp_table):
    tab1, tab2 = st.tabs(["📅 Last 7 Days", "🔮 Next 30 Days"])
    past_asp = asp_table[asp_table["Period"] == "PAST"].drop(columns=["Period"])
    fut_asp  = asp_table[asp_table["Period"] == "FUTURE"].drop(columns=["Period"])

    def style_aspect_df(df):
        def row_color(row):
            color = "#0a1a0f" if row["Direction"] == "▲ Bullish" else "#1a0a0f"
            dir_c = "#44DD88" if row["Direction"] == "▲ Bullish" else "#FF4466"
            return (["background-color:" + color] * (len(row) - 1)
                    + [f"color:{dir_c};background-color:{color}"])
        return df.style.apply(row_color, axis=1).format({
            "Orb °": "{:.2f}", "OLS Coef": "{:+.6f}",
            "Effect %": "{:+.3f}", "p-value": "{:.4f}",
        })

    with tab1:
        st.markdown(f"**{len(past_asp)} active aspect-days**")
        if len(past_asp):
            st.dataframe(style_aspect_df(past_asp), use_container_width=True, height=400)
        else:
            st.info("No active aspects in the last 7 trading days.")

    with tab2:
        st.markdown(f"**{len(fut_asp)} active aspect-days**")
        if len(fut_asp):
            st.dataframe(style_aspect_df(fut_asp), use_container_width=True, height=400)
        else:
            st.info("No active aspects in the next 30 calendar days.")
else:
    st.info("No active aspects found in the window.")


# ══════════════════════════════════════════════════════════════════════════════
#  CURRENT ZODIAC SIGNS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>♈ Current Zodiac Signs</h2>", unsafe_allow_html=True)

eph_today = eph.reindex([last_date], method="ffill")
sign_rows = []
for planet in sign_planets:
    if planet not in eph_today.columns:
        continue
    lon  = float(eph_today[planet].iloc[0]) % 360
    sign = SIGNS[int(lon // 30)]
    col  = f"{planet}_sign_{sign}"
    coef = float(ols_full_params[col]) if col in ols_full_params.index else float("nan")
    pval = float(ols_full_pvals[col])  if col in ols_full_params.index else float("nan")
    bias = "▲ Bullish" if coef > 0 else "▼ Bearish"
    sign_rows.append({
        "Planet":      planet.capitalize(),
        "Sign":        sign,
        "Longitude °": round(lon, 2),
        "OLS Coef":    round(coef, 6),
        "Effect %":    round(coef * 100, 3),
        "p-value":     round(pval, 4),
        "Bias":        bias,
    })

if sign_rows:
    sign_df = pd.DataFrame(sign_rows)
    def style_sign_df(df):
        def row_color(row):
            color = "#0a1a0f" if row["Bias"] == "▲ Bullish" else "#1a0a0f"
            dir_c = "#44DD88" if row["Bias"] == "▲ Bullish" else "#FF4466"
            return (["background-color:" + color] * (len(row) - 1)
                    + [f"color:{dir_c};background-color:{color}"])
        return df.style.apply(row_color, axis=1).format({
            "Longitude °": "{:.2f}", "OLS Coef": "{:+.6f}",
            "Effect %": "{:+.3f}", "p-value": "{:.4f}",
        })
    st.dataframe(style_sign_df(sign_df), use_container_width=True)
elif sign_planets:
    st.info("No sign-planet data found in ephemeris for the selected planets.")
else:
    st.info("No sign planets selected.")


# ══════════════════════════════════════════════════════════════════════════════
#  LAST 10 DAYS SIGNAL TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>🕐 Last 10 Trading Days — Directional Signal</h2>", unsafe_allow_html=True)

n_total   = len(dates_all)
last10    = list(range(max(0, n_total - 10), n_total))
hist_rows = []
for pos in last10:
    d         = dates_all[pos]
    close_val = float(price_df.loc[d, "Close"])
    prev      = float(price_df.iloc[pos - 1]["Close"]) if pos > 0 else float("nan")
    act_pct   = (close_val - prev) / prev * 100 if (not np.isnan(prev) and prev != 0) else float("nan")
    act_dir   = "▲ UP" if close_val > prev else "▼ DOWN"
    ols_ret   = y_fit_ols_full[pos]
    ols_dir   = "▲ UP" if ols_ret > 0 else "▼ DOWN"
    log_d     = y_fit_log_full[pos]
    log_dir   = "▲ UP" if log_d == 1 else "▼ DOWN"
    hist_rows.append({
        "Date":       str(d.date()),
        "Close":      round(close_val, 4),
        "Actual %":   round(act_pct, 2) if not np.isnan(act_pct) else None,
        "Actual Dir": act_dir,
        "OLS Ret %":  round(ols_ret * 100, 4),
        "OLS Dir":    ols_dir,
        "OLS ✓":      "✓" if ols_dir == act_dir else "✗",
        "Logit Dir":  log_dir,
        "Logit ✓":    "✓" if log_dir == act_dir else "✗",
    })

hist_df = pd.DataFrame(hist_rows)

def style_hist(df):
    def row_style(row):
        styles = []
        bg = "#0d0d28"
        for col in df.columns:
            val = row[col]
            if col in ("OLS ✓", "Logit ✓"):
                c = "#44DD88" if val == "✓" else "#FF4466"
                styles.append(f"color:{c};background-color:{bg};font-weight:700")
            elif col in ("OLS Dir", "Logit Dir", "Actual Dir"):
                c = "#44DD88" if "UP" in str(val) else "#FF4466"
                styles.append(f"color:{c};background-color:{bg}")
            else:
                styles.append(f"background-color:{bg};color:#E8E8F4")
        return styles
    return df.style.apply(row_style, axis=1)

st.dataframe(style_hist(hist_df), use_container_width=True)

ols_hits = sum(1 for r in hist_rows if r["OLS ✓"] == "✓")
log_hits = sum(1 for r in hist_rows if r["Logit ✓"] == "✓")
n_h      = len(hist_rows)
c1, c2   = st.columns(2)
c1.markdown(metric_html("OLS Hit Rate (last 10d)",   f"{ols_hits}/{n_h}", f"{ols_hits/n_h*100:.0f}%"), unsafe_allow_html=True)
c2.markdown(metric_html("Logit Hit Rate (last 10d)", f"{log_hits}/{n_h}", f"{log_hits/n_h*100:.0f}%"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD CSVs
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<h2>💾 Download Results</h2>", unsafe_allow_html=True)

forecast_df = pd.DataFrame({
    "date":             fut_dates,
    "ols_forecast_ret": y_fore_ols_r,
    "ols_forecast_cum": fore_ols_cum,
    "log_forecast_dir": y_fore_log,
    "log_cum_signal":   fore_log_cum,
    "log_cum_price":    fore_log_price,
})

fitted_df = pd.DataFrame({
    "date":              dates_all,
    "actual_close":      close_all,
    "actual_log_return": y_ret,
    "actual_direction":  y_dir,
    "ols_fitted_return": y_fit_ols_full,
    "ols_cumret_price":  cumret_ols_fit,
    "log_fitted_dir":    y_fit_log_full,
    "log_cumret_price":  cumret_log_fit,
})

col_a, col_b, col_c = st.columns(3)
col_a.download_button(
    "📥 Forecast CSV",
    data=forecast_df.to_csv(index=False).encode(),
    file_name=f"{ticker.replace('=', '_')}_forecast.csv",
    mime="text/csv",
)
col_b.download_button(
    "📥 Fitted Values CSV",
    data=fitted_df.to_csv(index=False).encode(),
    file_name=f"{ticker.replace('=', '_')}_fitted.csv",
    mime="text/csv",
)
if len(asp_table):
    col_c.download_button(
        "📥 Active Aspects CSV",
        data=asp_table.to_csv(index=False).encode(),
        file_name=f"{ticker.replace('=', '_')}_aspects.csv",
        mime="text/csv",
    )

st.markdown(
    f"<hr class='gold'/>"
    f"<p style='text-align:center;color:#555;font-size:11px;letter-spacing:1px'>"
    f"PLANETARY REGRESSION · {stock_name} ({ticker}) · "
    f"OLS R²={ols_is_r2:.4f} · Logit Acc={log_is_acc:.4f} · "
    f"Features={n_avail} · Apply={orb_apply}° Sep={orb_sep}°</p>",
    unsafe_allow_html=True,
)
