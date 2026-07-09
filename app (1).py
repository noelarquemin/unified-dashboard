"""
UNIFIED DASHBOARD v5 — version Streamlit
Analyse fondamentale + quantitative + technique d'une action, à la demande.
Adapté depuis le script Colab d'origine pour tourner sur Streamlit Community Cloud
(déclenché depuis un dépôt GitHub).
"""

import warnings
warnings.filterwarnings("ignore")

import io
import base64
import math
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from scipy import stats as scipy_stats
from sklearn.metrics import r2_score

import streamlit as st
import streamlit.components.v1 as components

import yfinance as yf
from yahooquery import Ticker as YQTicker

# ═══════════════════════════════════════════════════════════════
#  CONSTANTES FIXES (non exposées dans l'UI)
# ═══════════════════════════════════════════════════════════════
EXPORT_DPI = 110

# ── Palette ────────────────────────────────────────────────────
BG_DARK  = "#0B1120"; BG_SURF  = "#151E32"; BG_SURF2 = "#1D2A44"
COL_GRID = "#1E3A5F"; COL_TEXT = "#CBD5E1"; COL_FAINT= "#64748B"
C_BLUE="#60A5FA"; C_PINK="#EC4899"; C_GREEN="#34D399"
C_AMBER="#FBBF24"; C_RED="#F43F5E"; C_SKY="#38BDF8"
C_PURPLE="#A78BFA"; C_ORANGE="#FB923C"
C_HIST="#94A3B8"; C_TREND="#60A5FA"; C_PROJ="#EC4899"
C_BUY="#38BDF8"; C_SELL="#F472B6"; C_WEEKLY="#A78BFA"; C_MONTHLY="#FB923C"

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":10,
    "axes.facecolor":BG_SURF,"figure.facecolor":BG_DARK,
    "axes.edgecolor":"#2A3A57","axes.labelcolor":COL_FAINT,
    "xtick.color":COL_FAINT,"ytick.color":COL_FAINT,
    "text.color":COL_TEXT,"axes.spines.top":False,"axes.spines.right":False,
    "grid.color":COL_GRID,"grid.linewidth":0.7,
    "legend.framealpha":0.92,"legend.facecolor":BG_SURF2,
    "legend.edgecolor":"#2A3A57","legend.fontsize":8.5,
})

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=EXPORT_DPI,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    enc = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig); return enc

def safe(v, default=np.nan):
    try:
        if v is None: return default
        f = float(v)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except: return default

def pct(v, sign=False):
    if np.isnan(v): return "N/A"
    s = "+" if sign and v > 0 else ""
    return f"{s}{v*100:.1f}%"

def fmt_n(v, d=2):
    if np.isnan(v): return "N/A"
    return f"{v:,.{d}f}"

def fmt_big(v):
    if np.isnan(v): return "N/A"
    if abs(v)>=1e12: return f"${v/1e12:.2f}T"
    if abs(v)>=1e9:  return f"${v/1e9:.1f}B"
    if abs(v)>=1e6:  return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"

def fmt_big_mpl(v):
    if np.isnan(v): return "N/A"
    if abs(v)>=1e12: return f"{v/1e12:.2f}T"
    if abs(v)>=1e9:  return f"{v/1e9:.1f}B"
    if abs(v)>=1e6:  return f"{v/1e6:.0f}M"
    return f"{v:,.0f}"

def pfmt(x, _=None):
    if x is None or (isinstance(x, float) and (np.isnan(x) or x <= 0)): return ""
    return f"{x:,.0f}" if x >= 100 else f"{x:.2f}"

def score_color(s):
    if s >= 75: return C_GREEN
    if s >= 55: return C_BLUE
    if s >= 35: return C_AMBER
    return C_RED

def score_grade(s):
    if s >= 80: return "A+"
    if s >= 70: return "A"
    if s >= 60: return "B+"
    if s >= 50: return "B"
    if s >= 40: return "C"
    if s >= 30: return "D"
    return "F"

def add_price_label(ax, x, y, text, color, fontsize=8, ha="left"):
    ax.text(x, y, text, va="center", ha=ha, fontsize=fontsize,
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=BG_SURF2, ec=color, alpha=0.90, lw=0.9))


# ═══════════════════════════════════════════════════════════════
#  MULTIPLICATEURS SECTORIELS
# ═══════════════════════════════════════════════════════════════
SECTOR_MULTS = {
    "Technology":            (1.10, 1.10, 1.05, 1.10, 0.85, 0.90),
    "Financial Services":    (0.95, 0.90, 0.70, 1.00, 1.10, 1.05),
    "Industrials":           (1.00, 1.00, 1.00, 0.95, 1.05, 1.00),
    "Consumer Cyclical":     (1.00, 0.95, 1.00, 0.95, 1.05, 1.00),
    "Communication Services":(1.05, 1.00, 1.00, 1.05, 0.90, 0.95),
    "Healthcare":            (1.05, 1.05, 1.05, 1.05, 0.85, 0.95),
    "Consumer Defensive":    (0.90, 1.00, 1.00, 0.95, 0.95, 1.10),
    "Energy":                (0.85, 0.90, 1.10, 0.90, 1.20, 1.15),
    "Basic Materials":       (0.85, 0.90, 1.05, 0.90, 1.15, 1.10),
    "Real Estate":           (0.90, 0.85, 0.65, 1.00, 0.90, 1.20),
    "Utilities":             (0.80, 0.85, 0.80, 0.90, 1.00, 1.20),
    "N/A":                   (1.00, 1.00, 1.00, 1.00, 1.00, 1.00),
    "Unknown":               (1.00, 1.00, 1.00, 1.00, 1.00, 1.00),
}

def get_sector_mults(sector_str):
    if not sector_str: return SECTOR_MULTS["N/A"]
    for key in SECTOR_MULTS:
        if key.lower() in sector_str.lower():
            return SECTOR_MULTS[key]
    return SECTOR_MULTS["N/A"]

def m(v, mult):
    if np.isnan(v): return np.nan
    return v * mult

def weighted_score_v5(scores_dict, weights_dict):
    available = {k: v for k,v in scores_dict.items() if not np.isnan(v)}
    if not available: return 0.0
    total_w = sum(weights_dict[k] for k in available)
    if total_w == 0: return 0.0
    return float(np.clip(sum(available[k]*weights_dict[k]/total_w for k in available), 0, 100))

# ═══════════════════════════════════════════════════════════════
#  BARÈMES UNIVERSELS STRICTS (v3 calibrés)
# ═══════════════════════════════════════════════════════════════
def score_growth_v5(mults, rev_cagr3, eps_cagr3, fcf_cagr3, rev_growth_yoy):
    mg = mults[0]
    def sg(v):
        if np.isnan(v): return np.nan
        if v>=0.20: return 100
        elif v>=0.15: return 75
        elif v>=0.10: return 45
        elif v>=0.05: return 20
        elif v>=0.00: return 8
        else:         return 0
    def sr(v):
        if np.isnan(v): return np.nan
        if v>=0.15: return 100
        elif v>=0.10: return 75
        elif v>=0.05: return 45
        elif v>=0.00: return 20
        else:         return 0
    pts = {
        "Rev CAGR 3Y": sg(m(rev_cagr3,     mg)),
        "EPS CAGR 3Y": sg(m(eps_cagr3,     mg)),
        "FCF CAGR 3Y": sg(m(fcf_cagr3,     mg)),
        "Rev YoY":     sr(m(rev_growth_yoy, mg)),
    }
    w = {"Rev CAGR 3Y":0.30,"EPS CAGR 3Y":0.25,"FCF CAGR 3Y":0.25,"Rev YoY":0.20}
    details = {
        "Rev CAGR 3Y": (rev_cagr3,     f"réf >15% (sect×{mg:.2f})"),
        "EPS CAGR 3Y": (eps_cagr3,     f"réf >15% (sect×{mg:.2f})"),
        "FCF CAGR 3Y": (fcf_cagr3,     f"réf >12% (sect×{mg:.2f})"),
        "Rev YoY":     (rev_growth_yoy, f"réf >10% (sect×{mg:.2f})"),
    }
    return weighted_score_v5(pts, w), details

def score_profitability_v5(mults, roe, roic, roa, roce):
    mg = mults[1]
    def sp(v, t=(0.30,0.20,0.15,0.08)):
        if np.isnan(v): return np.nan
        if v>=t[0]: return 100
        elif v>=t[1]: return 75
        elif v>=t[2]: return 45
        elif v>=t[3]: return 20
        elif v>=0:    return 5
        else:         return 0
    pts = {
        "ROE":  sp(m(roe,  mg), (0.30,0.20,0.15,0.08)),
        "ROIC": sp(m(roic, mg), (0.20,0.15,0.10,0.05)),
        "ROA":  sp(m(roa,  mg), (0.12,0.08,0.05,0.02)),
        "ROCE": sp(m(roce, mg), (0.20,0.15,0.10,0.05)),
    }
    w = {"ROE":0.30,"ROIC":0.30,"ROA":0.20,"ROCE":0.20}
    details = {
        "ROE":  (roe,  f"réf >20% (sect×{mg:.2f})"),
        "ROIC": (roic, f"réf >15% (sect×{mg:.2f})"),
        "ROA":  (roa,  f"réf >8% (sect×{mg:.2f})"),
        "ROCE": (roce, f"réf >15% (sect×{mg:.2f})"),
    }
    return weighted_score_v5(pts, w), details

def score_balance_v5(mults, debt_ebitda, current_ratio, altman_z, quick_ratio):
    mg = mults[2]
    def sd(v):
        if np.isnan(v): return np.nan
        veff = v * mg
        if veff < 0: return 100
        elif veff < 1: return 100
        elif veff < 2: return 75
        elif veff < 3: return 45
        elif veff < 5: return 20
        else:          return 0
    def sc(v):
        if np.isnan(v): return np.nan
        veff = v * mg
        if veff>=2.0: return 100
        elif veff>=1.5: return 75
        elif veff>=1.0: return 45
        elif veff>=0.8: return 20
        else:           return 0
    def sa(v):
        if np.isnan(v): return np.nan
        veff = v * mg
        if veff>=3.0: return 100
        elif veff>=2.0: return 75
        elif veff>=1.5: return 45
        elif veff>=1.0: return 20
        else:           return 0
    def sq(v):
        if np.isnan(v): return np.nan
        veff = v * mg
        if veff>=1.5: return 100
        elif veff>=1.0: return 75
        elif veff>=0.7: return 45
        elif veff>=0.5: return 20
        else:           return 0
    pts = {
        "Dette/EBITDA":   sd(debt_ebitda),
        "Current Ratio":  sc(current_ratio),
        "Altman Z-Score": sa(altman_z),
        "Quick Ratio":    sq(quick_ratio),
    }
    w = {"Dette/EBITDA":0.35,"Current Ratio":0.25,"Altman Z-Score":0.25,"Quick Ratio":0.15}
    details = {
        "Dette/EBITDA":   (debt_ebitda,   f"réf <2× (sect×{mg:.2f})"),
        "Current Ratio":  (current_ratio, f"réf >1.5 (sect×{mg:.2f})"),
        "Altman Z-Score": (altman_z,      f"réf >3 (sect×{mg:.2f})"),
        "Quick Ratio":    (quick_ratio,   f"réf >1 (sect×{mg:.2f})"),
    }
    return weighted_score_v5(pts, w), details

def score_margins_v5(mults, gross_margin, op_margin, net_margin, fcf_margin):
    mg = mults[3]
    def smg(v, t=(0.60,0.40,0.30,0.20)):
        if np.isnan(v): return np.nan
        veff = v * mg
        if veff>=t[0]: return 100
        elif veff>=t[1]: return 75
        elif veff>=t[2]: return 45
        elif veff>=t[3]: return 20
        elif veff>=0:    return 5
        else:            return 0
    fcf_s = smg(fcf_margin * mg, (0.20,0.10,0.05,0.00)) \
            if not np.isnan(fcf_margin) and fcf_margin >= 0 else np.nan
    pts = {
        "Marge Brute":          smg(gross_margin, (0.60,0.40,0.30,0.20)),
        "Marge Opérationnelle": smg(op_margin,    (0.25,0.15,0.10,0.05)),
        "Marge Nette":          smg(net_margin,   (0.20,0.10,0.05,0.00)),
        "FCF Margin":           fcf_s,
    }
    w = {"Marge Brute":0.28,"Marge Opérationnelle":0.30,"Marge Nette":0.24,"FCF Margin":0.18}
    fcf_ref = f"réf >10% (sect×{mg:.2f})"
    if not np.isnan(fcf_margin) and fcf_margin < 0:
        fcf_ref += " ⚠ négatif — indicatif"
    details = {
        "Marge Brute":          (gross_margin, f"réf >40% (sect×{mg:.2f})"),
        "Marge Opérationnelle": (op_margin,    f"réf >15% (sect×{mg:.2f})"),
        "Marge Nette":          (net_margin,   f"réf >10% (sect×{mg:.2f})"),
        "FCF Margin":           (fcf_margin,   fcf_ref),
    }
    return weighted_score_v5(pts, w), details

def score_valuation_v5(mults, pe_ratio, pb_ratio, ps_ratio, peg_ratio, ev_ebitda, pfcf_ratio):
    mg = mults[4]
    def slo(v, t):
        if np.isnan(v) or v <= 0: return np.nan
        veff = v / mg
        if veff<=t[0]: return 100
        elif veff<=t[1]: return 75
        elif veff<=t[2]: return 45
        elif veff<=t[3]: return 20
        else:            return 0
    pe_v  = pe_ratio  if not np.isnan(pe_ratio)  and pe_ratio  > 0 else np.nan
    peg_v = peg_ratio if not np.isnan(peg_ratio) and peg_ratio > 0 else np.nan
    pb_v  = pb_ratio  if not np.isnan(pb_ratio)  and 0 < pb_ratio < 50 else np.nan
    ps_v  = ps_ratio  if not np.isnan(ps_ratio)  and ps_ratio  > 0 else np.nan
    pts = {
        "PE Ratio":  slo(pe_v,       (10,15,20,30)),
        "P/B Ratio": slo(pb_v,       (1.0,2.0,3.0,5.0)),
        "P/S Ratio": slo(ps_v,       (1.0,2.0,4.0,6.0)),
        "PEG Ratio": slo(peg_v,      (1.0,1.5,2.0,3.0)),
        "EV/EBITDA": slo(ev_ebitda,  (7,10,13,15)),
        "P/FCF":     slo(pfcf_ratio, (15,25,35,50)),
    }
    w = {"PE Ratio":0.22,"P/B Ratio":0.18,"P/S Ratio":0.18,"PEG Ratio":0.18,"EV/EBITDA":0.14,"P/FCF":0.10}
    details = {
        "PE Ratio":  (pe_v,       f"≤10 exc · ≤15 bon · ≤20 ok (sect×{mg:.2f})"),
        "P/B Ratio": (pb_v,       f"réf <2 (sect×{mg:.2f})"),
        "P/S Ratio": (ps_v,       f"réf <2 (sect×{mg:.2f})"),
        "PEG Ratio": (peg_v,      f"réf <1.5 (sect×{mg:.2f})"),
        "EV/EBITDA": (ev_ebitda,  f"réf <13 (sect×{mg:.2f})"),
        "P/FCF":     (pfcf_ratio, f"réf <25 (sect×{mg:.2f})"),
    }
    return weighted_score_v5(pts, w), details

def score_capital_v5(mults, fcf_yield, buyback_yield, div_yield, payout_ratio):
    mg = mults[5]
    div_y  = min(div_yield,    0.15) if not np.isnan(div_yield)    else np.nan
    payout = min(payout_ratio, 2.00) if not np.isnan(payout_ratio) and payout_ratio > 0 else np.nan
    fcf_y  = fcf_yield    if not np.isnan(fcf_yield)    and fcf_yield    > 0 else np.nan
    buyb_y = buyback_yield if not np.isnan(buyback_yield) and buyback_yield > 0 else np.nan
    def shi(v, t):
        if np.isnan(v) or v <= 0: return np.nan
        veff = v * mg
        if veff>=t[0]: return 100
        elif veff>=t[1]: return 75
        elif veff>=t[2]: return 45
        elif veff>=t[3]: return 20
        else:            return 0
    def spay(v):
        if np.isnan(v) or v <= 0: return np.nan
        veff = v / mg
        if veff<=0.30: return 100
        elif veff<=0.50: return 75
        elif veff<=0.70: return 45
        elif veff<=0.90: return 20
        else:            return 0
    pts = {
        "FCF Yield":       shi(fcf_y,  (0.08,0.05,0.03,0.01)),
        "Buyback Yield":   shi(buyb_y, (0.05,0.03,0.02,0.01)),
        "Dividende Yield": shi(div_y,  (0.04,0.02,0.01,0.00)),
        "Payout Ratio":    spay(payout),
    }
    w = {"FCF Yield":0.40,"Buyback Yield":0.25,"Dividende Yield":0.20,"Payout Ratio":0.15}
    details = {
        "FCF Yield":      (fcf_yield,     f"réf >4% (sect×{mg:.2f})"),
        "Buyback Yield":  (buyback_yield, f"réf >2% (sect×{mg:.2f})"),
        "Dividende Yield":(div_y,         f"réf >1% (sect×{mg:.2f})"),
        "Payout Ratio":   (payout_ratio,  f"réf <60% (sect×{mg:.2f})"),
    }
    return weighted_score_v5(pts, w), details


# ═══════════════════════════════════════════════════════════════
#  GRAPHIQUE TECHNIQUE — CHANDELIERS + MM200 + BOLLINGER + RSI
# ═══════════════════════════════════════════════════════════════
def build_tech_chart(ticker, last_date, last_price, years=5):
    start = last_date - pd.DateOffset(years=years+1)
    raw_w = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                        interval="1wk", progress=False)
    raw_w.index = pd.to_datetime(raw_w.index)
    raw_w = raw_w.dropna(how="all")
    if isinstance(raw_w.columns, pd.MultiIndex):
        raw_w.columns = raw_w.columns.droplevel(1)

    raw_m = yf.download(ticker, period="max", interval="1mo", progress=False)
    raw_m.index = pd.to_datetime(raw_m.index)
    raw_m = raw_m.dropna(how="all")
    if isinstance(raw_m.columns, pd.MultiIndex):
        raw_m.columns = raw_m.columns.droplevel(1)

    if raw_w.empty or len(raw_w) < 10:
        return None

    close_w = raw_w["Close"].squeeze()
    open_w  = raw_w["Open"].squeeze()
    high_w  = raw_w["High"].squeeze()
    low_w   = raw_w["Low"].squeeze()

    cutoff  = last_date - pd.DateOffset(years=years)
    mask    = raw_w.index >= cutoff
    dates_d = raw_w.index[mask]
    if len(dates_d) < 5:
        return None

    mm200w   = close_w.rolling(200).mean()

    close_m  = raw_m["Close"].squeeze()
    mm200m_s = close_m.rolling(200).mean().dropna()
    mm200m_r = mm200m_s.reindex(close_w.index, method="ffill") if len(mm200m_s) else pd.Series(np.nan, index=close_w.index)

    bb_mid   = close_w.rolling(20).mean()
    bb_std   = close_w.rolling(20).std()
    bb_upper = bb_mid + 2*bb_std
    bb_lower = bb_mid - 2*bb_std

    delta = close_w.diff()
    gain  = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss  = (-delta).clip(lower=0).ewm(com=13, min_periods=14).mean()
    rsi_w = 100 - (100/(1 + gain/loss.replace(0, np.nan)))

    def fw(s): return s.loc[dates_d] if hasattr(s,"loc") else s

    close_d  = fw(close_w);  open_d   = fw(open_w)
    high_d   = fw(high_w);   low_d    = fw(low_w)
    mm200w_d = fw(mm200w);   mm200m_d = fw(mm200m_r)
    bbu_d    = fw(bb_upper); bbm_d    = fw(bb_mid); bbl_d = fw(bb_lower)
    rsi_d    = fw(rsi_w)

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(18,11),
                                   gridspec_kw={"height_ratios":[3.5,1],"hspace":0.04})
    for ax in [ax1,ax2]:
        ax.set_facecolor(BG_SURF)
        ax.grid(True,ls="--",alpha=0.25,color=COL_GRID)
    fig.patch.set_facecolor(BG_DARK)

    x = np.arange(len(dates_d)); W=0.55
    o_arr = open_d.values; c_arr = close_d.values
    h_arr = high_d.values; l_arr = low_d.values

    for i in range(len(x)):
        o,c,h,l = o_arr[i],c_arr[i],h_arr[i],l_arr[i]
        if np.isnan(o) or np.isnan(c): continue
        col = C_GREEN if c >= o else C_RED
        ax1.bar(x[i], abs(c-o), bottom=min(o,c), width=W,
                color=col, alpha=0.85, zorder=3, linewidth=0)
        ax1.plot([x[i],x[i]], [l,h], color=col, lw=0.8, alpha=0.70, zorder=2)

    ax1.fill_between(x, bbl_d.values, bbu_d.values, alpha=0.07, color=C_BLUE, zorder=1)
    ax1.plot(x, bbu_d.values, color=C_BLUE, lw=0.9, ls="--", alpha=0.55, label="BB sup")
    ax1.plot(x, bbm_d.values, color=C_BLUE, lw=0.9, ls=":",  alpha=0.40, label="BB mid 20w")
    ax1.plot(x, bbl_d.values, color=C_BLUE, lw=0.9, ls="--", alpha=0.55, label="BB inf")

    mm200w_v = mm200w_d.ffill()
    if mm200w_v.dropna().size:
        v200w = float(mm200w_v.dropna().iloc[-1])
        ax1.plot(x, mm200w_v.values, color=C_AMBER, lw=2.2, alpha=0.92, zorder=4,
                 label=f"MM200w ({pfmt(v200w)})")

    mm200m_v = mm200m_d.ffill()
    if mm200m_v.dropna().size:
        v200m = float(mm200m_v.dropna().iloc[-1])
        ax1.plot(x, mm200m_v.values, color=C_RED, lw=2.5, alpha=0.85, ls="--", zorder=4,
                 label=f"MM200m ({pfmt(v200m)})")

    last_c = float(close_d.dropna().iloc[-1])
    ax1.axhline(last_c, color="#F8FAFC", lw=0.7, ls=":", alpha=0.35)
    ax1.text(len(x)+0.5, last_c, pfmt(last_c), va="center", ha="left",
             fontsize=9, color="#F8FAFC", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3",fc=BG_SURF2,ec="#F8FAFC",alpha=0.85,lw=0.8))

    step = max(1, len(dates_d)//12)
    ax1.set_xticks(x[::step])
    ax1.set_xticklabels([d.strftime("%b %Y") for d in dates_d[::step]],
                        rotation=0, fontsize=8.5, color=COL_FAINT)
    ax1.set_xlim(-1, len(x)+2)
    ax1.tick_params(labelbottom=False)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(pfmt))
    ax1.legend(loc="upper left", fontsize=8.5, ncol=5)
    ax1.set_ylabel("Prix (weekly)", color=COL_FAINT, fontsize=9)
    ax1.set_title(f"{ticker} — Chandeliers hebdomadaires · MM200w · MM200m · Bollinger 20w · {years} ans",
                  color="#F1F5F9", fontsize=11, fontweight="bold", pad=10)

    rsi_last = float(rsi_d.dropna().iloc[-1]) if rsi_d.dropna().size else np.nan
    rsi_col  = C_RED if not np.isnan(rsi_last) and rsi_last>=70 else \
               C_GREEN if not np.isnan(rsi_last) and rsi_last<=30 else C_BLUE
    ax2.plot(x, rsi_d.values, color=C_WEEKLY, lw=1.6, alpha=0.92,
             label=f"RSI 14 Weekly ({rsi_last:.1f})" if not np.isnan(rsi_last) else "RSI 14 Weekly")
    ax2.axhline(70,color=C_RED,   lw=0.9,ls="--",alpha=0.60)
    ax2.axhline(30,color=C_GREEN, lw=0.9,ls="--",alpha=0.60)
    ax2.axhline(50,color="#475569",lw=0.7,ls=":", alpha=0.50)
    rsi_v = rsi_d.values.astype(float)
    ax2.fill_between(x, rsi_v, 70, where=(rsi_v>=70), color=C_RED,   alpha=0.15, interpolate=True)
    ax2.fill_between(x, rsi_v, 30, where=(rsi_v<=30), color=C_GREEN, alpha=0.15, interpolate=True)
    if not np.isnan(rsi_last):
        ax2.scatter([x[-1]],[rsi_last],color=rsi_col,s=55,zorder=8,ec="#F8FAFC",lw=0.8)
    ax2.set_ylim(0,100); ax2.set_yticks([30,50,70])
    ax2.set_xlim(-1,len(x)+2)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels([d.strftime("%b %Y") for d in dates_d[::step]],
                        rotation=0,fontsize=8.5,color=COL_FAINT)
    ax2.legend(loc="upper left",fontsize=8.5)
    ax2.set_ylabel("RSI",color=COL_FAINT,fontsize=9)
    fig.subplots_adjust(left=0.05,right=0.94,top=0.95,bottom=0.05)
    return fig


# ═══════════════════════════════════════════════════════════════
#  WACC + DCF
# ═══════════════════════════════════════════════════════════════
def compute_wacc_dynamic(info, inc_df, total_debt, total_equity, market_cap):
    try:
        tnx = yf.download("^TNX", period="5d", progress=False)
        if isinstance(tnx.columns, pd.MultiIndex):
            tnx.columns = tnx.columns.droplevel(1)
        rf = float(tnx["Close"].dropna().iloc[-1]) / 100
    except:
        rf = 0.043
    rf = np.clip(rf, 0.02, 0.08)

    beta = safe(info.get("beta"), 1.0)
    beta = np.clip(beta, 0.1, 4.0)
    erp  = 0.055
    ke   = np.clip(rf + beta * erp, 0.05, 0.20)

    interest_exp = np.nan
    if not inc_df.empty:
        for k in ["Interest Expense","InterestExpense","Net Interest Income"]:
            if k in inc_df.index:
                interest_exp = safe(inc_df.loc[k].iloc[0])
                break
    if not np.isnan(interest_exp) and not np.isnan(total_debt) and total_debt > 0:
        kd_pretax = np.clip(abs(interest_exp)/total_debt, 0.02, 0.12)
    else:
        ratio = safe(total_debt,0)/max(safe(total_equity,1),1)
        kd_pretax = np.clip(0.035 + 0.015*min(ratio,5), 0.03, 0.10)
    kd = kd_pretax * 0.79

    total_cap = safe(market_cap,0) + safe(total_debt,0)
    if total_cap <= 0:
        return 0.09, rf, beta, ke, kd
    e_w = safe(market_cap,0)/total_cap
    d_w = safe(total_debt,0)/total_cap
    wacc = np.clip(ke*e_w + kd*d_w, 0.05, 0.18)
    return wacc, rf, beta, ke, kd

def dcf_model_v4(fcf_base, gr, tv_gr, wacc, years, sh, total_cash, total_debt, price):
    if np.isnan(fcf_base) or fcf_base <= 0:
        return np.nan, np.nan, np.nan, np.nan, []
    fcfs=[]; pv_sum=0.0
    for t_i in range(1, years+1):
        f  = fcf_base*(1+gr)**t_i
        pv = f/(1+wacc)**t_i
        fcfs.append((t_i,f,pv)); pv_sum+=pv
    tv    = fcf_base*(1+gr)**years*(1+tv_gr)/(wacc-tv_gr)
    pv_tv = tv/(1+wacc)**years
    net_c = safe(total_cash,0)-safe(total_debt,0)
    iv    = (pv_sum+pv_tv+net_c)/sh if sh and sh>0 else np.nan
    mg    = (iv-price)/price if not any(np.isnan([iv,price])) and price>0 else np.nan
    tv_pct= pv_tv/(pv_sum+pv_tv) if (pv_sum+pv_tv)>0 else np.nan
    return iv, pv_tv, mg, tv_pct, fcfs

def compute_quant_score_v4(pos_sigma, P_v, CAGR_v, MDD_v, t_v):
    if pos_sigma >= 2.0:
        return max(2.0, 10.0 - pos_sigma*3), 0,0,0,0
    if pos_sigma >= 1.0:
        return max(5.0, 25.0 - (pos_sigma-1.0)*18), 0,0,0,0
    if P_v > 0 and CAGR_v > 0:
        norm_P    = min(P_v/0.95,   1.0)
        norm_CAGR = min(CAGR_v/0.40,1.0)
        norm_MDD  = min(0.15/max(MDD_v,0.001),1.0)
        norm_t    = min(0.5/max(t_v,0.05),1.0)
        base = float(np.clip(0.35*norm_P+0.25*norm_CAGR+0.25*norm_MDD+0.15*norm_t,0,1))
        if pos_sigma >= 0:
            base *= (1 - pos_sigma*0.35)
        return float(np.clip(base*100,0,100)), norm_P, norm_CAGR, norm_MDD, norm_t
    return 0.0, 0,0,0,0


# ═══════════════════════════════════════════════════════════════
#  GÉNÉRATION COMPLÈTE DU RAPPORT POUR UN TICKER
# ═══════════════════════════════════════════════════════════════
def generate_dashboard(ticker, dcf_tv_growth=0.025, dcf_bear_gr=0.03, dcf_base_gr=0.08,
                       dcf_bull_gr=0.14, dcf_years=10, n_sim=1000, seed=42,
                       hist_years=100, tech_years=5):

    TICKER        = ticker.strip().upper()
    DCF_TV_GROWTH = dcf_tv_growth
    DCF_BEAR_GR   = dcf_bear_gr
    DCF_BASE_GR   = dcf_base_gr
    DCF_BULL_GR   = dcf_bull_gr
    DCF_YEARS     = dcf_years
    N_SIM         = n_sim
    SEED          = seed
    HIST_YEARS    = hist_years
    TECH_YEARS    = tech_years

    np.random.seed(SEED)

    # ─────────────────────────────────────────────────────────
    #  BLOC 1 — CHARGEMENT DONNÉES
    # ─────────────────────────────────────────────────────────
    print("\n"+"═"*65)
    print("  BLOC 1 — CHARGEMENT DONNÉES")
    print("═"*65)

    tk   = yf.Ticker(TICKER)
    yqtk = YQTicker(TICKER)
    info = tk.info or {}

    price      = safe(info.get("currentPrice") or info.get("regularMarketPrice"))
    market_cap = safe(info.get("marketCap"))
    shares_out = safe(info.get("sharesOutstanding"))
    sector     = info.get("sector","N/A")
    industry   = info.get("industry","N/A")
    company    = info.get("longName", TICKER)
    currency   = info.get("financialCurrency","USD")

    print(f"  {company} ({TICKER}) | {sector} → multiplicateurs sectoriels chargés")
    print(f"  Prix : ${price:.2f} | Market Cap : {fmt_big(market_cap)}")

    raw = yf.download(TICKER, period="max", interval="1d", progress=False)
    raw = raw[["Close"]].dropna()
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
    raw.columns = ["Close"]; raw.index = pd.to_datetime(raw.index)
    if raw.empty or len(raw) < 10:
        raise ValueError(f"Ticker '{TICKER}' introuvable ou historique insuffisant sur Yahoo Finance. "
                         f"Vérifiez le ticker (ex: AAPL, MSFT, ABEV) — les tickers européens peuvent nécessiter "
                         f"le suffixe de marché (ex: AIR.PA pour Airbus).")
    last_price = float(raw["Close"].iloc[-1]); last_date = raw.index[-1]
    S_full = raw["Close"].values.astype(float); dates_full = raw.index
    N_full = len(S_full); t_full = np.arange(N_full, dtype=float)
    first_date = dates_full[0]
    hist_years_actual = (last_date - first_date).days / 365.25

    def get_df(attr):
        try:
            df = getattr(tk, attr)
            if df is None or df.empty: return pd.DataFrame()
            if len(df.columns) and isinstance(df.columns[0], pd.Timestamp):
                df = df.sort_index(axis=1, ascending=False)
            return df
        except: return pd.DataFrame()

    inc = get_df("income_stmt")
    bal = get_df("balance_sheet")
    cf  = get_df("cash_flow")

    try:
        yq_fin = yqtk.financial_data.get(TICKER, {})
        yq_ks  = yqtk.key_stats.get(TICKER, {})
    except: yq_fin={};yq_ks={}

    def row(df, *keys):
        for k in keys:
            if k in df.index:
                return np.array([safe(v) for v in df.loc[k].values])
        return np.array([np.nan]*4)

    def row1(df, *keys):
        v = row(df, *keys); return safe(v[0]) if len(v) else np.nan

    def load_yq_stmt(kind):
        try:
            if   kind=="inc": df=yqtk.income_statement(frequency="a",trailing=False)
            elif kind=="cf":  df=yqtk.cash_flow(frequency="a",trailing=False)
            else:             df=yqtk.balance_sheet(frequency="a",trailing=False)
            if isinstance(df,pd.DataFrame) and not df.empty:
                df=df.reset_index()
                if "asOfDate" in df.columns:
                    df["asOfDate"]=pd.to_datetime(df["asOfDate"])
                    return df.dropna(subset=["asOfDate"]).sort_values("asOfDate")
        except: pass
        return pd.DataFrame()

    yq_inc_a=load_yq_stmt("inc"); yq_cf_a=load_yq_stmt("cf"); yq_bal_a=load_yq_stmt("bal")

    def yq_year_series(df,*cols):
        if df.empty: return pd.Series(dtype=float)
        for c in cols:
            if c in df.columns:
                s=pd.Series([safe(v) for v in df[c].values],index=df["asOfDate"].dt.year.values)
                s=s[~s.index.duplicated(keep="last")].dropna()
                if len(s): return s
        return pd.Series(dtype=float)

    def yf_year_series(df,*keys):
        if df.empty: return pd.Series(dtype=float)
        for k in keys:
            if k in df.index:
                years=[pd.to_datetime(c).year for c in df.columns]
                s=pd.Series([safe(v) for v in df.loc[k].values],index=years)
                s=s[~s.index.duplicated(keep="last")].dropna()
                if len(s): return s
        return pd.Series(dtype=float)

    def merged_metric(src_yf,yf_keys,src_yq,yq_cols):
        s_yf=yf_year_series(src_yf,*yf_keys)
        s_yq=yq_year_series(src_yq,*yq_cols)
        return s_yf.combine_first(s_yq).sort_index()

    rev_hist  = merged_metric(inc,["Total Revenue","Revenue"],       yq_inc_a,["TotalRevenue"])
    gp_hist   = merged_metric(inc,["Gross Profit","GrossProfit"],    yq_inc_a,["GrossProfit"])
    op_hist   = merged_metric(inc,["Operating Income","EBIT"],       yq_inc_a,["OperatingIncome","EBIT"])
    ni_hist   = merged_metric(inc,["Net Income","NetIncome"],        yq_inc_a,["NetIncome"])
    cfo_hist  = merged_metric(cf, ["Operating Cash Flow"],           yq_cf_a, ["OperatingCashFlow"])
    capex_hist= merged_metric(cf, ["Capital Expenditure","CapEx"],   yq_cf_a, ["CapitalExpenditure"])
    da_hist   = merged_metric(cf, ["Depreciation And Amortization"], yq_cf_a, ["DepreciationAndAmortization"])
    wc_hist   = merged_metric(cf, ["Change In Working Capital"],     yq_cf_a, ["ChangeInWorkingCapital"])
    sbc_hist  = merged_metric(cf, ["Stock Based Compensation"],      yq_cf_a, ["StockBasedCompensation"])
    divp_hist = merged_metric(cf, ["Payment Of Dividends"],          yq_cf_a, ["CashDividendsPaid"])
    acq_hist  = merged_metric(cf, ["Acquisitions Net"],              yq_cf_a, ["PurchaseOfBusiness"])

    _cf = cfo_hist.index.intersection(capex_hist.index)
    fcf_hist = pd.Series({y:cfo_hist[y]-abs(capex_hist[y]) for y in _cf}).sort_index()

    def margin_series(num,den):
        c=num.index.intersection(den.index)
        return pd.Series({y:num[y]/den[y] for y in c if den[y]>0}).sort_index()

    gm_hist=margin_series(gp_hist, rev_hist)
    om_hist=margin_series(op_hist, rev_hist)
    nm_hist=margin_series(ni_hist, rev_hist)
    fm_hist=margin_series(fcf_hist,rev_hist)

    fin_years=sorted(set(rev_hist.index)|set(fcf_hist.index)|set(ni_hist.index))
    n_fin_years=len(fin_years)
    print(f"  Historique financier : {n_fin_years} exercices ({fin_years[0] if fin_years else '—'} → {fin_years[-1] if fin_years else '—'})")

    rev_arr  = row(inc,"Total Revenue","Revenue","TotalRevenue")
    rev_ttm  = safe(info.get("totalRevenue")) or safe(rev_arr[0])
    op_arr   = row(inc,"Operating Income","EBIT","OperatingIncome")
    ni_arr   = row(inc,"Net Income","NetIncome")
    ebitda_v = safe(info.get("ebitda"))
    if np.isnan(ebitda_v):
        da0=row1(cf,"Depreciation And Amortization","Depreciation")
        ebitda_v=safe(op_arr[0])+safe(da0)

    cfo_arr  = row(cf,"Operating Cash Flow","Cash From Operations")
    capex_arr= row(cf,"Capital Expenditure","CapEx","Purchase Of Property Plant And Equipment")
    fcf_arr  = np.array([safe(c)-abs(safe(x)) for c,x in zip(cfo_arr,capex_arr)])
    fcf_ttm  = safe(info.get("freeCashflow"))
    if np.isnan(fcf_ttm) and not np.isnan(fcf_arr[0]): fcf_ttm=fcf_arr[0]

    total_debt   = safe(info.get("totalDebt"))     or row1(bal,"Total Debt","Long Term Debt")
    total_cash   = safe(info.get("totalCash"))     or row1(bal,"Cash And Cash Equivalents")
    total_assets = row1(bal,"Total Assets")
    total_equity = (safe(info.get("bookValue",np.nan))*safe(shares_out)
                    if not np.isnan(safe(info.get("bookValue",np.nan)))
                    else row1(bal,"Stockholders Equity"))
    current_assets= row1(bal,"Current Assets","Total Current Assets")
    current_liab  = row1(bal,"Current Liabilities","Total Current Liabilities")
    retained_earn = row1(bal,"Retained Earnings")
    net_debt      = total_debt-total_cash if not any(np.isnan([total_debt,total_cash])) else np.nan
    ev            = safe(info.get("enterpriseValue"))
    if np.isnan(ev) and not any(np.isnan([market_cap,net_debt])): ev=market_cap+net_debt

    pe_ratio   = safe(info.get("trailingPE") or info.get("forwardPE"))
    peg_ratio  = safe(info.get("pegRatio"))
    pb_ratio   = safe(info.get("priceToBook"))
    ps_ratio   = safe(info.get("priceToSalesTrailing12Months"))
    ev_ebitda  = safe(info.get("enterpriseToEbitda"))
    ev_sales   = safe(info.get("enterpriseToRevenue"))
    pfcf_ratio = (market_cap/fcf_ttm) if not any(np.isnan([market_cap,fcf_ttm])) and fcf_ttm>0 else np.nan
    book_value_ps= safe(info.get("bookValue"))
    cash_ps      = safe(info.get("totalCashPerShare"))
    cash_ebitda  = (total_cash/ebitda_v)    if not any(np.isnan([total_cash,ebitda_v]))    and ebitda_v>0    else np.nan
    cash_equity  = (total_cash/total_equity) if not any(np.isnan([total_cash,total_equity])) and total_equity>0 else np.nan
    cash_mktcap  = (total_cash/market_cap)  if not any(np.isnan([total_cash,market_cap]))  and market_cap>0  else np.nan
    netcash_mktcap=((total_cash-total_debt)/market_cap) if not any(np.isnan([total_cash,total_debt,market_cap])) and market_cap>0 else np.nan

    roe        = safe(info.get("returnOnEquity"))
    roa        = safe(info.get("returnOnAssets"))
    gross_margin= safe(info.get("grossMargins"))
    op_margin  = safe(info.get("operatingMargins"))
    net_margin = safe(info.get("profitMargins"))
    fcf_margin = (fcf_ttm/rev_ttm) if not any(np.isnan([fcf_ttm,rev_ttm])) and rev_ttm>0 else np.nan
    try:
        nopat=safe(op_arr[0])*(1-0.21); ic=total_equity+total_debt-total_cash
        roic=nopat/ic if not np.isnan(nopat) and ic>0 else np.nan
    except: roic=np.nan
    try:
        ce=total_assets-current_liab; roce=safe(op_arr[0])/ce if not np.isnan(safe(op_arr[0])) and ce>0 else np.nan
    except: roce=np.nan

    def cagr_hist(s,n=3):
        s=s.dropna()
        if len(s)<2: return np.nan
        n=min(n,len(s)-1)
        v0,vn=s.iloc[-1],s.iloc[-1-n]
        if v0<=0 or vn<=0: return np.nan
        return float((v0/vn)**(1/n)-1)

    def cagr_calc(arr,n=3):
        arr=np.array([safe(v) for v in arr]); valid=arr[~np.isnan(arr)]
        if len(valid)<2: return np.nan
        n=min(n,len(valid)-1)
        if valid[0]<=0 or valid[n]<=0: return np.nan
        return float((valid[0]/valid[n])**(1/n)-1)

    eps_arr   = row(inc,"Diluted EPS","Basic EPS","Earnings Per Share")
    rev_cagr3 = cagr_hist(rev_hist,3) if len(rev_hist)>=2 else cagr_calc(rev_arr,3)
    eps_cagr3 = cagr_calc(eps_arr,3)
    fcf_cagr3 = cagr_hist(fcf_hist,3) if len(fcf_hist)>=2 else cagr_calc(fcf_arr,3)
    rev_growth_yoy = safe(info.get("revenueGrowth"))

    debt_ebitda   = (net_debt/ebitda_v) if not any(np.isnan([net_debt,ebitda_v])) and ebitda_v>0 else np.nan
    current_ratio = safe(info.get("currentRatio")) or (current_assets/current_liab if not any(np.isnan([current_assets,current_liab])) and current_liab>0 else np.nan)
    quick_ratio   = safe(info.get("quickRatio"))
    try:
        wc=current_assets-current_liab
        altman_z=(1.2*(wc/total_assets)+1.4*(safe(retained_earn)/total_assets)+
                  3.3*(safe(op_arr[0])/total_assets)+0.6*(market_cap/(total_debt if total_debt>0 else 1))+
                  1.0*(rev_ttm/total_assets)) if not np.isnan(total_assets) and total_assets>0 else np.nan
    except: altman_z=np.nan

    div_yield    = safe(info.get("dividendYield"))
    payout_ratio = safe(info.get("payoutRatio"))
    buyback_arr  = row(cf,"Repurchase Of Stock","Common Stock Repurchased","Repurchase of Capital Stock")
    buyback_ttm  = abs(safe(buyback_arr[0]))
    buyback_yield= (buyback_ttm/market_cap) if not any(np.isnan([buyback_ttm,market_cap])) and market_cap>0 else np.nan
    fcf_yield    = (fcf_ttm/market_cap) if not any(np.isnan([fcf_ttm,market_cap])) and market_cap>0 else np.nan

    DCF_WACC, wacc_rf, wacc_beta, wacc_ke, wacc_kd = compute_wacc_dynamic(
        info, inc, total_debt, total_equity, market_cap)
    print(f"  WACC dynamique : {DCF_WACC*100:.2f}% (Rf={wacc_rf*100:.2f}%, β={wacc_beta:.2f}, Ke={wacc_ke*100:.2f}%, Kd={wacc_kd*100:.2f}%)")
    print(f"  Rev CAGR 3Y: {pct(rev_cagr3)} | EPS CAGR 3Y: {pct(eps_cagr3)} | FCF CAGR 3Y: {pct(fcf_cagr3)}")
    print(f"  PE: {fmt_n(pe_ratio)} | PB: {fmt_n(pb_ratio)} | PS: {fmt_n(ps_ratio)}")
    print(f"  ROE: {pct(roe)} | ROIC: {pct(roic)} | ROA: {pct(roa)}")

    # ─────────────────────────────────────────────────────────
    #  BLOC 2 — ANALYSE QUANTITATIVE
    # ─────────────────────────────────────────────────────────
    print("\n"+"═"*65); print("  BLOC 2 — ANALYSE QUANTITATIVE"); print("═"*65)

    P0=S_full[0]; logP_full=np.log(S_full/P0)
    slope,intercept,r_value,p_value,std_err=scipy_stats.linregress(t_full,logP_full)
    def trend_logP(t): return intercept+slope*t
    trend_logP_full=trend_logP(t_full); r2_full=r2_score(logP_full,trend_logP_full)
    resid_log=logP_full-trend_logP_full; sigma_log=float(np.std(resid_log))
    last_logP=float(np.log(last_price/P0))
    pos_sigma=(last_logP-trend_logP_full[-1])/sigma_log
    drift_annual=float(np.exp(slope*252)-1)

    cutoff=last_date-pd.DateOffset(years=HIST_YEARS)
    mask10=dates_full>=cutoff; dates10=dates_full[mask10]
    S10=S_full[mask10]; t10_abs=t_full[mask10]
    trend_logP_10=trend_logP(t10_abs); trend_px_10=P0*np.exp(trend_logP_10)
    bands={}
    for k in [1,2]:
        bands[f"+{k}s"]=P0*np.exp(trend_logP_10+k*sigma_log)
        bands[f"-{k}s"]=P0*np.exp(trend_logP_10-k*sigma_log)
    bp2_now=float(bands["+2s"][-1]); bp1_now=float(bands["+1s"][-1])
    bm1_now=float(bands["-1s"][-1]); bm2_now=float(bands["-2s"][-1])
    tr_now=float(trend_px_10[-1])

    N_PROJ_DAYS=252
    proj_dates=pd.bdate_range(last_date+pd.Timedelta(days=1),periods=N_PROJ_DAYS)
    t_proj_abs=np.arange(N_full,N_full+N_PROJ_DAYS,dtype=float)
    trend_px_proj=P0*np.exp(trend_logP(t_proj_abs))
    bands_proj={f"+{k}s":P0*np.exp(trend_logP(t_proj_abs)+k*sigma_log) for k in [1,2]}
    bands_proj.update({f"-{k}s":P0*np.exp(trend_logP(t_proj_abs)-k*sigma_log) for k in [1,2]})
    dates_ext=np.concatenate([dates10,proj_dates])
    trend_ext=np.concatenate([trend_px_10,trend_px_proj])
    bands_ext={k:np.concatenate([bands[k],bands_proj[k]]) for k in bands}

    ZONES={"Bulle >+2σ":(2.0,np.inf),"Surévalué +1→+2σ":(1.0,2.0),"Fair 0→+1σ":(0.0,1.0),
           "Sous-val -1→0σ":(-1.0,0.0),"Bon marché -2→-1σ":(-2.0,-1.0),"Excès <-2σ":(-np.inf,-2.0)}
    ZONE_KEYS=list(ZONES.keys())
    ZONE_COLORS=["#F43F5E","#F97316","#34D399","#34D399","#38BDF8","#3B82F6"]
    ZONE_SHORT=[">+2σ","+1→+2σ","0→+1σ","-1→0σ","-2→-1σ","<-2σ"]

    def get_zone(e):
        s=e/sigma_log
        for name,(lo,hi) in ZONES.items():
            if lo<=s<hi: return name
        return ZONE_KEYS[-1]

    zone_seq=[get_zone(e) for e in resid_log]
    weights_z={z:zone_seq.count(z)/len(zone_seq) for z in ZONE_KEYS}
    pos_zone=get_zone(resid_log[-1]); pos_idx=ZONE_KEYS.index(pos_zone)

    if pos_zone in ("Bon marché -2→-1σ","Excès <-2σ"):
        sig_txt="✅ ZONE ACHAT"; sig_color=C_BUY
        sig_bg="rgba(56,189,248,0.12)"; sig_border="rgba(56,189,248,0.35)"
    elif pos_zone in ("Surévalué +1→+2σ","Bulle >+2σ"):
        sig_txt="⚠ ZONE VENTE"; sig_color=C_SELL
        sig_bg="rgba(244,114,182,0.12)"; sig_border="rgba(244,114,182,0.35)"
    else:
        sig_txt="⏳ ZONE NEUTRE"; sig_color=C_BLUE
        sig_bg="rgba(96,165,250,0.10)"; sig_border="rgba(96,165,250,0.30)"

    rho1=float(np.clip(np.corrcoef(resid_log[:-1],resid_log[1:])[0,1],0.01,0.9999))
    kappa_ar1=float(-np.log(rho1))
    drift_sig=abs(drift_annual)/sigma_log
    kappa_adj=kappa_ar1*r2_full*(1.0/(1.0+drift_sig))
    HL_MIN=max(sigma_log*252*0.25,60.0); HL_MAX=min(hist_years_actual*252*0.40,1260.0)
    kappa_f=float(np.clip(kappa_adj,np.log(2)/HL_MAX,np.log(2)/HL_MIN))
    hl_final=np.log(2)/kappa_f; hl_yr=hl_final/252
    sigma_innov=float(np.std(resid_log[1:]-rho1*resid_log[:-1])); e0=float(resid_log[-1])

    ct=int(np.clip(round(hl_final),21,504))
    mt=int(np.clip(round(hl_final*3),126,1260))
    lt=int(np.clip(round(hl_final*7),252,5040))
    def hl_lbl(n): return f"{n/252:.1f}an" if n<504 else f"{n/252:.0f}ans"
    HORIZONS={f"CT {hl_lbl(ct)}":ct,f"MT {hl_lbl(mt)}":mt,f"LT {hl_lbl(lt)}":lt}
    N_DAYS_MAX=max(max(HORIZONS.values()),N_PROJ_DAYS)

    fut_dates_mc=pd.bdate_range(last_date+pd.Timedelta(days=1),periods=N_DAYS_MAX)
    t0=float(t_full[-1]); t_fut=t0+np.arange(1,N_DAYS_MAX+1,dtype=float)
    tr_fut=trend_logP(t_fut); decay=1.0-kappa_f
    np.random.seed(SEED)
    sims=np.zeros((N_SIM,N_DAYS_MAX))
    for s in range(N_SIM):
        e=e0; noise=np.random.normal(0,sigma_innov,N_DAYS_MAX)
        for i in range(N_DAYS_MAX):
            e=decay*e+noise[i]; sims[s,i]=P0*np.exp(tr_fut[i]+e)
    sims=np.clip(sims,last_price/500,last_price*500)

    mc_p025=np.percentile(sims,2.5,axis=0); mc_p25=np.percentile(sims,25,axis=0)
    mc_med=np.percentile(sims,50,axis=0);   mc_p75=np.percentile(sims,75,axis=0)
    mc_p975=np.percentile(sims,97.5,axis=0)

    def _mdd_sim(path): pk=np.maximum.accumulate(path); return float(np.min((path-pk)/pk))
    risk={}
    for lbl,h in HORIZONS.items():
        fp=sims[:,h-1]; fr=(fp-last_price)/last_price
        v95=float(np.percentile(fr,5))
        cv=float(np.mean(fr[fr<=v95])) if np.any(fr<=v95) else v95
        mdd_arr=np.array([_mdd_sim(sims[s,:h]) for s in range(N_SIM)])
        risk[lbl]=dict(v95=v95,cv=cv,esp=float(np.mean(fr)),med=float(np.median(fr)),
                       pp=float(np.mean(fr>0)),mdd_m=float(np.mean(mdd_arr)),h=h,fr=fr)

    def first_hit_zone(lo,hi,sim_days=None):
        if sim_days is None: sim_days=N_DAYS_MAX
        times,returns,mdds=[],[],[]
        for s in range(N_SIM):
            found=False
            for i in range(sim_days):
                t_i=t0+i+1; e_sim=np.log(np.clip(sims[s,i],1e-9,None)/P0)-trend_logP(t_i)
                s_sim=e_sim/sigma_log
                if lo<=s_sim<hi:
                    times.append(i+1); returns.append((sims[s,i]-last_price)/last_price)
                    pk=np.maximum.accumulate(sims[s,:i+1])
                    mdds.append(float(np.min((sims[s,:i+1]-pk)/pk))); found=True; break
            if not found: times.append(np.nan); returns.append(np.nan)
        t_arr=np.array(times); r_arr=np.array(returns); mdd_arr=np.array(mdds)
        mask=~np.isnan(t_arr); p=float(mask.mean())
        if p==0 or mask.sum()<5:
            return dict(p=0,t_med=np.nan,r_med=np.nan,mdd_med=np.nan,cagr=np.nan,
                        fr_final=np.array([]),t_p25=np.nan,t_p75=np.nan)
        tm=t_arr[mask]; rm=r_arr[mask]; md=mdd_arr
        t_med=float(np.median(tm)); r_med=float(np.median(rm)); mdd_m=float(np.median(md))
        t_yrs=t_med/252; cagr_v=float((1+r_med)**(1/max(t_yrs,0.1))-1) if t_yrs>0 else r_med
        t_med_idx=min(int(t_med)-1,N_DAYS_MAX-1)
        return dict(p=p,t_med=t_med,r_med=r_med,mdd_med=mdd_m,cagr=cagr_v,
                    fr_final=(sims[:,t_med_idx]-last_price)/last_price,
                    t_p25=float(np.percentile(tm,25)),t_p75=float(np.percentile(tm,75)))

    mc_target=first_hit_zone(1.0,2.0)
    P_v=mc_target["p"]; CAGR_v=mc_target.get("cagr",0.0) or 0.0
    MDD_v=abs(mc_target.get("mdd_med",-0.20) or 0.20)
    t_v=(mc_target.get("t_med",252) or 252)/252.0

    quant_score_100,norm_P,norm_CAGR,norm_MDD,norm_t = compute_quant_score_v4(
        pos_sigma,P_v,CAGR_v,MDD_v,t_v)
    quant_score_01=quant_score_100/100

    print(f"  R²={r2_full:.4f} | σ={sigma_log*100:.1f}% | Drift={drift_annual:+.2%}/an | Pos={pos_sigma:+.2f}σ")
    print(f"  Bandes à t : +2σ={pfmt(bp2_now)} +1σ={pfmt(bp1_now)} Tend={pfmt(tr_now)} -1σ={pfmt(bm1_now)} -2σ={pfmt(bm2_now)}")
    print(f"  Zone: {pos_zone} | Signal: {sig_txt}")

    cyc_ok = False
    cyc_periods, cyc_r2 = [], 0.0
    cyc_fit_hist = np.zeros(N_full); e_cyc_fut = np.zeros(N_DAYS_MAX)
    cyc_path = mc_med.copy(); cyc_path_hist = None
    cyc_T_dom = np.nan; cyc_amp_sig = np.nan; cyc_amp_pct = np.nan
    cyc_phase_txt = "—"; cyc_next_peak = None; cyc_next_trough = None
    try:
        _res_c = resid_log - resid_log.mean()
        _n_res = len(_res_c)
        _fftv  = np.fft.rfft(_res_c * np.hanning(_n_res))
        _freqs = np.fft.rfftfreq(_n_res, d=1.0)
        _power = np.abs(_fftv)**2
        with np.errstate(divide="ignore"):
            _periods = np.where(_freqs > 0, 1.0/np.maximum(_freqs, 1e-12), np.inf)
        _valid = (_periods >= 126) & (_periods <= 0.75*_n_res)
        if _valid.sum() >= 2:
            _order = np.argsort(np.where(_valid, _power, -1.0))[::-1]
            for _i in _order:
                if not _valid[_i]: break
                _T = float(_periods[_i])
                if all(abs(_T-_p)/_p > 0.30 for _p in cyc_periods):
                    cyc_periods.append(_T)
                if len(cyc_periods) >= 3: break
        if cyc_periods:
            _cols = [np.ones(_n_res)]
            for _T in cyc_periods:
                _w = 2*np.pi/_T
                _cols += [np.cos(_w*t_full), np.sin(_w*t_full)]
            _X = np.column_stack(_cols)
            _coef, *_ = np.linalg.lstsq(_X, resid_log, rcond=None)
            cyc_fit_hist = _X @ _coef
            _ss_r = float(np.sum((resid_log-cyc_fit_hist)**2))
            _ss_t = float(np.sum((resid_log-resid_log.mean())**2))
            cyc_r2 = max(0.0, 1.0-_ss_r/_ss_t) if _ss_t > 0 else 0.0
            _cols_f = [np.ones(N_DAYS_MAX)]
            for _T in cyc_periods:
                _w = 2*np.pi/_T
                _cols_f += [np.cos(_w*t_fut), np.sin(_w*t_fut)]
            e_cyc_fut = np.column_stack(_cols_f) @ _coef
            _gap0 = e0 - float(cyc_fit_hist[-1])
            e_cyc_fut = e_cyc_fut + _gap0*np.exp(-kappa_f*np.arange(1, N_DAYS_MAX+1))
            e_cyc_fut = np.clip(e_cyc_fut, -2.3*sigma_log, 2.3*sigma_log)
            cyc_path      = P0*np.exp(tr_fut + e_cyc_fut)
            cyc_path_hist = P0*np.exp(trend_logP_full + cyc_fit_hist)
            cyc_T_dom   = cyc_periods[0]
            cyc_amp_sig = float(np.std(cyc_fit_hist)/sigma_log*np.sqrt(2))
            cyc_amp_pct = float(np.exp(np.std(cyc_fit_hist)*np.sqrt(2))-1)
            _slope0 = float(e_cyc_fut[min(63, N_DAYS_MAX-1)] - e_cyc_fut[0])
            cyc_phase_txt = "ascendante ↗" if _slope0 > 0 else "descendante ↘"
            _pk = [i for i in range(1, N_DAYS_MAX-1)
                   if e_cyc_fut[i] >= e_cyc_fut[i-1] and e_cyc_fut[i] > e_cyc_fut[i+1]]
            _tr = [i for i in range(1, N_DAYS_MAX-1)
                   if e_cyc_fut[i] <= e_cyc_fut[i-1] and e_cyc_fut[i] < e_cyc_fut[i+1]]
            if _pk: cyc_next_peak   = (fut_dates_mc[_pk[0]], float(cyc_path[_pk[0]]))
            if _tr: cyc_next_trough = (fut_dates_mc[_tr[0]], float(cyc_path[_tr[0]]))
            cyc_ok = True
            print(f"  ✓ Cycles dominants: {', '.join(f'{_T/252:.1f} ans' for _T in cyc_periods)}"
                  f" | R² cyclique={cyc_r2:.0%} | phase {cyc_phase_txt}")
    except Exception as _e:
        print(f"  ⚠ Extraction des cycles impossible ({_e}) — médiane MC conservée")

    # ─────────────────────────────────────────────────────────
    #  BLOC 3 — SCORECARD FONDAMENTALE v5
    # ─────────────────────────────────────────────────────────
    print("\n"+"═"*65); print(f"  BLOC 3 — SCORECARD FONDAMENTALE v5 ({sector})"); print("═"*65)

    sector_mults = get_sector_mults(sector)

    s_growth,  d_growth  = score_growth_v5(sector_mults, rev_cagr3, eps_cagr3, fcf_cagr3, rev_growth_yoy)
    s_profit,  d_profit  = score_profitability_v5(sector_mults, roe, roic, roa, roce)
    s_balance, d_balance = score_balance_v5(sector_mults, debt_ebitda, current_ratio, altman_z, quick_ratio)
    s_margins, d_margins = score_margins_v5(sector_mults, gross_margin, op_margin, net_margin, fcf_margin)
    s_val,     d_val     = score_valuation_v5(sector_mults, pe_ratio, pb_ratio, ps_ratio, peg_ratio, ev_ebitda, pfcf_ratio)
    s_capital, d_capital = score_capital_v5(sector_mults, fcf_yield, buyback_yield, div_yield, payout_ratio)

    FUND_PILLARS=[
        ("Croissance",    0.20, s_growth,  d_growth,  C_GREEN),
        ("Rentabilité",   0.20, s_profit,  d_profit,  C_BLUE),
        ("Bilan/Risque",  0.20, s_balance, d_balance, C_SKY),
        ("Marges",        0.15, s_margins, d_margins, C_PURPLE),
        ("Valorisation",  0.15, s_val,     d_val,     C_AMBER),
        ("Alloc.Capital", 0.10, s_capital, d_capital, C_PINK),
    ]
    pillar_scores  = {n:s for n,w,s,d,c in FUND_PILLARS}
    pillar_details = {n:d for n,w,s,d,c in FUND_PILLARS}
    fund_score     = sum(w*s for n,w,s,d,c in FUND_PILLARS)
    global_score   = (fund_score + quant_score_100) / 2.0

    for n,w,s,d,c in FUND_PILLARS:
        print(f"  {n:<22} {s:5.1f}/100  [{score_grade(s)}]  (poids {w*100:.0f}%)")
    print(f"\n  SCORE FONDAMENTAL : {fund_score:.1f}/100 [{score_grade(fund_score)}]")
    print(f"  SCORE GLOBAL      : {global_score:.1f}/100 [{score_grade(global_score)}]")

    sh=shares_out if not np.isnan(shares_out) else (market_cap/price if not any(np.isnan([market_cap,price])) and price>0 else np.nan)
    dcf_bear=dcf_model_v4(fcf_ttm,DCF_BEAR_GR,DCF_TV_GROWTH,DCF_WACC,DCF_YEARS,sh,total_cash,total_debt,price)
    dcf_base=dcf_model_v4(fcf_ttm,DCF_BASE_GR,DCF_TV_GROWTH,DCF_WACC,DCF_YEARS,sh,total_cash,total_debt,price)
    dcf_bull=dcf_model_v4(fcf_ttm,DCF_BULL_GR,DCF_TV_GROWTH,DCF_WACC,DCF_YEARS,sh,total_cash,total_debt,price)
    def dcf_fmt(r): iv,_,mg,tv_pct,_2=r; return (f"${iv:.2f}" if not np.isnan(iv) else "N/A", pct(mg) if not np.isnan(mg) else "N/A")

    # ─────────────────────────────────────────────────────────
    #  BLOC 3-bis — SÉRIES EXPLICATIVES
    # ─────────────────────────────────────────────────────────
    print("\n  ── 3-bis : séries explicatives ──")

    def abs_series(s):
        return s.apply(lambda v: abs(v)) if len(s) else s

    debt_repay_hist = abs_series(merged_metric(
        cf, ["Repayment Of Debt","Long Term Debt Payments","Repayments Of Long Term Debt"],
        yq_cf_a, ["RepaymentOfDebt","LongTermDebtPayments"]))
    debt_issue_hist = abs_series(merged_metric(
        cf, ["Issuance Of Debt","Long Term Debt Issuance","Issuance Of Long Term Debt"],
        yq_cf_a, ["IssuanceOfDebt","LongTermDebtIssuance"]))
    _yrs_df = sorted(set(debt_repay_hist.index) | set(debt_issue_hist.index))
    debt_netflow_hist = pd.Series(
        {y: safe(debt_issue_hist.get(y, 0.0), 0.0) - safe(debt_repay_hist.get(y, 0.0), 0.0)
         for y in _yrs_df}).sort_index()

    tdebt_hist = merged_metric(bal, ["Total Debt","TotalDebt"], yq_bal_a, ["TotalDebt"])
    cashbs_hist = merged_metric(
        bal, ["Cash And Cash Equivalents","Cash Cash Equivalents And Short Term Investments"],
        yq_bal_a, ["CashAndCashEquivalents","CashCashEquivalentsAndShortTermInvestments"])
    _yrs_nd = tdebt_hist.index.intersection(cashbs_hist.index)
    netdebt_hist = pd.Series({y: tdebt_hist[y] - cashbs_hist[y] for y in _yrs_nd}).sort_index()

    divpaid_hist = abs_series(divp_hist)
    buyback_hist_s = abs_series(merged_metric(
        cf, ["Repurchase Of Capital Stock","Common Stock Payments","Repurchase Of Stock"],
        yq_cf_a, ["RepurchaseOfCapitalStock","CommonStockPayments"]))

    intexp_hist = abs_series(merged_metric(
        inc, ["Interest Expense","InterestExpense"], yq_inc_a, ["InterestExpense"]))
    acq_hist_abs = abs_series(acq_hist)
    sbc_hist_s   = sbc_hist.dropna()

    print(f"    ΔBFR: {len(wc_hist.dropna())} ex. | Dette repay/issue: {len(debt_repay_hist)}/{len(debt_issue_hist)} ex. | "
          f"Dette nette: {len(netdebt_hist)} ex.")
    print(f"    Dividendes: {len(divpaid_hist)} ex. | Rachats: {len(buyback_hist_s)} ex. | "
          f"Intérêts: {len(intexp_hist)} ex. | SBC: {len(sbc_hist_s)} ex. | Acquisitions: {len(acq_hist_abs)} ex.")

    cover_years = sorted(set(fcf_hist.dropna().index) &
                         (set(divpaid_hist.index) | set(buyback_hist_s.index)))
    coverage_data = []
    for _y in cover_years:
        _fcf = safe(fcf_hist.get(_y, np.nan))
        _dv  = safe(divpaid_hist.get(_y, 0.0), 0.0)
        _bb  = safe(buyback_hist_s.get(_y, 0.0), 0.0)
        _tot = _dv + _bb
        _cov = _fcf/_tot if _tot > 0 and not np.isnan(_fcf) else np.nan
        coverage_data.append((_y, _fcf, _dv, _bb, _tot, _cov))

    def generate_explain_comment():
        L = []
        wc = wc_hist.dropna()
        if len(wc):
            v, y = float(wc.iloc[-1]), wc.index[-1]
            if v < 0:
                L.append(f"<b>Δ BFR ({y})</b> : {fmt_big(v)} — le besoin en fonds de roulement a <b>consommé du cash</b> "
                         f"(hausse des stocks/créances ou baisse des dettes fournisseurs), ce qui pèse mécaniquement sur le FCF "
                         f"sans refléter la rentabilité opérationnelle.")
            else:
                L.append(f"<b>Δ BFR ({y})</b> : +{fmt_big(v)} — libération de cash favorable au FCF "
                         f"(effet potentiellement non récurrent).")
            if len(wc) >= 3 and (wc.iloc[-2:] < 0).all():
                L.append("⚠ Le BFR consomme du cash sur <b>plusieurs exercices consécutifs</b> — à surveiller "
                         "(croissance gourmande en BFR ou dégradation du cycle client/fournisseur).")
        if len(debt_netflow_hist):
            nf, y = float(debt_netflow_hist.iloc[-1]), debt_netflow_hist.index[-1]
            rp = safe(debt_repay_hist.get(y, np.nan))
            if nf < 0:
                L.append(f"<b>Dette ({y})</b> : désendettement net de {fmt_big(abs(nf))} "
                         f"(remboursements {fmt_big(rp)}) — cela <b>ampute la trésorerie disponible</b> mais renforce le bilan "
                         f"et réduira la charge d'intérêts future.")
            elif nf > 0:
                L.append(f"<b>Dette ({y})</b> : endettement net de +{fmt_big(nf)} — le cash levé peut financer capex, "
                         f"dividendes ou rachats ; vérifier que le FCF suit.")
        if len(netdebt_hist) >= 2:
            nd0, nd1 = float(netdebt_hist.iloc[0]), float(netdebt_hist.iloc[-1])
            trend_nd = "en baisse ✅" if nd1 < nd0 else "en hausse ⚠"
            L.append(f"<b>Dette nette</b> : {fmt_big(nd0)} ({netdebt_hist.index[0]}) → {fmt_big(nd1)} "
                     f"({netdebt_hist.index[-1]}) — {trend_nd}.")
        ie = intexp_hist.dropna()
        if len(ie) >= 2:
            i0, i1 = float(ie.iloc[0]), float(ie.iloc[-1])
            if i1 > i0 * 1.25:
                L.append(f"<b>Charge d'intérêts</b> : {fmt_big(i0)} → {fmt_big(i1)} — la hausse du coût de la dette "
                         f"<b>comprime directement le bénéfice net et la marge nette</b>.")
            elif i1 < i0 * 0.80:
                L.append(f"<b>Charge d'intérêts</b> : {fmt_big(i0)} → {fmt_big(i1)} — l'allègement soutient le bénéfice net.")
        if coverage_data:
            _y, _fcf, _dv, _bb, _tot, _cov = coverage_data[-1]
            if not np.isnan(_cov):
                if _cov < 1.0:
                    L.append(f"<b>Retours actionnaires ({_y})</b> : dividendes + rachats = {fmt_big(_tot)} pour un FCF de "
                             f"{fmt_big(_fcf)} (couverture {_cov:.2f}×) — les retours <b>excèdent le FCF</b> et sont donc "
                             f"financés par la trésorerie ou la dette : non soutenable durablement.")
                elif _cov < 1.5:
                    L.append(f"<b>Retours actionnaires ({_y})</b> : {fmt_big(_tot)} vs FCF {fmt_big(_fcf)} "
                             f"(couverture {_cov:.2f}×) — couvert, mais marge de manœuvre limitée.")
                else:
                    L.append(f"<b>Retours actionnaires ({_y})</b> : {fmt_big(_tot)} vs FCF {fmt_big(_fcf)} "
                             f"(couverture {_cov:.2f}×) — confortablement couverts par le FCF.")
        _cx = capex_hist.dropna(); _rv = rev_hist.dropna()
        _cyrs = sorted(set(_cx.index) & set(_rv.index))
        if len(_cyrs) >= 2:
            c0 = abs(_cx[_cyrs[0]])/_rv[_cyrs[0]] if _rv[_cyrs[0]] > 0 else np.nan
            c1 = abs(_cx[_cyrs[-1]])/_rv[_cyrs[-1]] if _rv[_cyrs[-1]] > 0 else np.nan
            if not any(np.isnan([c0, c1])):
                if c1 > c0 * 1.3:
                    L.append(f"<b>Intensité capex</b> : {pct(c0)} → {pct(c1)} du CA — l'investissement s'intensifie et "
                             f"<b>explique une partie de la pression sur le FCF</b> (à relier à la croissance future attendue).")
                elif c1 < c0 * 0.7:
                    L.append(f"<b>Intensité capex</b> : {pct(c0)} → {pct(c1)} du CA — allègement favorable au FCF.")
        aq = acq_hist_abs.dropna()
        if len(aq) and float(aq.iloc[-1]) > 0:
            _y = aq.index[-1]
            L.append(f"<b>Acquisitions ({_y})</b> : {fmt_big(float(aq.iloc[-1]))} — sortie de cash ponctuelle "
                     f"(hors FCF classique mais impacte la trésorerie et la dette nette).")
        return "<br><br>".join(L) if L else "Données de cash-flow insuffisantes pour générer une analyse explicative."

    explain_comment_html = generate_explain_comment()

    # ─────────────────────────────────────────────────────────
    #  BLOC 4 — GRAPHIQUES
    # ─────────────────────────────────────────────────────────
    print("\n"+"═"*65); print("  BLOC 4 — GRAPHIQUES"); print("═"*65)

    fig1,ax1=plt.subplots(figsize=(16,7))
    fig1.patch.set_facecolor(BG_DARK); ax1.set_facecolor(BG_SURF)
    ax1.fill_between(dates_ext,bands_ext["-2s"],bands_ext["+2s"],color=C_TREND,alpha=0.05)
    ax1.fill_between(dates_ext,bands_ext["-1s"],bands_ext["+1s"],color=C_TREND,alpha=0.10)
    label_x=last_date+pd.Timedelta(days=18)
    for key,y_now,col,ls,al,name in [("+2s",bp2_now,C_SELL,":",0.70,"+2σ"),
                                      ("+1s",bp1_now,C_SELL,"--",0.85,"+1σ"),
                                      ("-1s",bm1_now,C_BUY,"--",0.85,"−1σ"),
                                      ("-2s",bm2_now,C_BUY,":",0.70,"−2σ")]:
        ax1.plot(dates_ext,bands_ext[key],color=col,lw=1.0,ls=ls,alpha=al)
        add_price_label(ax1,label_x,y_now,f"{name} {pfmt(y_now)}",col,8)
    ax1.plot(dates_ext,trend_ext,color=C_TREND,lw=2.0,label=f"Tendance R²={r2_full:.3f}")
    add_price_label(ax1,label_x,tr_now,f"Tend {pfmt(tr_now)}",C_TREND,8)
    ax1.plot(dates10,S10,color=C_HIST,lw=1.2,alpha=0.90,label="Cours")
    ax1.scatter([last_date],[last_price],color="#F8FAFC",s=60,zorder=10,ec=C_TREND,lw=1.5)
    add_price_label(ax1,proj_dates[min(60,len(proj_dates)-1)],last_price,
                    f"{pfmt(last_price)} ({pos_sigma:+.2f}σ)","#F8FAFC",9)
    ax1.axvline(last_date,color="#475569",lw=1.2,ls=":",zorder=4)
    ax1.set_yscale("log"); ax1.yaxis.set_major_formatter(mticker.FuncFormatter(pfmt))
    ax1.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax1.set_xlim(dates10[0],proj_dates[-1]+pd.Timedelta(days=int(hist_years_actual*18)))
    ax1.grid(True,ls="--",alpha=0.30); ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2 if hist_years_actual>10 else 1))
    ax1.legend(loc="upper left",fontsize=8.5)
    ax1.set_title(f"{TICKER} — Régression log-prix | R²={r2_full:.4f} | σ={sigma_log*100:.1f}% | {pos_sigma:+.2f}σ «{pos_zone}»",
                  color="#F1F5F9",fontsize=10,fontweight="bold",pad=10)
    fig1.subplots_adjust(left=0.06,right=0.90,top=0.93,bottom=0.08)
    b64_reg=fig_to_b64(fig1); print("  ✓ régression")

    fig2,ax2=plt.subplots(figsize=(16,7))
    fig2.patch.set_facecolor(BG_DARK); ax2.set_facecolor(BG_SURF)
    ax2.plot(dates_full,S_full,color=C_HIST,lw=1.1,alpha=0.88,label="Historique",zorder=6)
    ax2.fill_between(fut_dates_mc,mc_p025,mc_p975,color=C_PROJ,alpha=0.07,label="IC 95%")
    ax2.fill_between(fut_dates_mc,mc_p25,mc_p75,color=C_PROJ,alpha=0.15,label="IQR 25-75%")
    ax2.plot(fut_dates_mc,mc_med,color=C_PROJ,lw=2.0,zorder=7,label=f"Médiane ({N_SIM} sim.)")
    t_fut_full=np.arange(N_full+N_DAYS_MAX,dtype=float)
    all_dates_ext=pd.bdate_range(first_date,periods=len(t_fut_full),freq="B")
    ax2.plot(all_dates_ext,P0*np.exp(trend_logP(t_fut_full)),color=C_TREND,lw=1.3,ls="--",alpha=0.55,label="Tendance")
    if cyc_ok and cyc_path_hist is not None:
        ax2.plot(dates_full, cyc_path_hist, color=C_ORANGE, lw=1.2, alpha=0.60, ls="-", zorder=5, label="Fit cyclique (hist.)")
        cyc_lbl = f"Cyclique ({', '.join(f'{_T/252:.1f}a' for _T in cyc_periods[:2])})"
        ax2.plot(fut_dates_mc, cyc_path, color=C_ORANGE, lw=2.2, alpha=0.90, ls="-", zorder=8, label=cyc_lbl)
        add_price_label(ax2, fut_dates_mc[-1]+pd.Timedelta(days=5), float(cyc_path[-1]),
                        f"Cyc {pfmt(float(cyc_path[-1]))}", C_ORANGE, 8.0)
        if cyc_next_peak:
            ax2.scatter([cyc_next_peak[0]], [cyc_next_peak[1]], color=C_GREEN, s=80, zorder=10, marker="^",
                        label=f"Prochain pic ~{cyc_next_peak[0].strftime('%b %Y')} ({pfmt(cyc_next_peak[1])})")
        if cyc_next_trough:
            ax2.scatter([cyc_next_trough[0]], [cyc_next_trough[1]], color=C_RED, s=80, zorder=10, marker="v",
                        label=f"Prochain creux ~{cyc_next_trough[0].strftime('%b %Y')} ({pfmt(cyc_next_trough[1])})")
    add_price_label(ax2,fut_dates_mc[-1]+pd.Timedelta(days=5),mc_med[-1], f"Med {pfmt(mc_med[-1])}",C_PROJ,8.0)
    add_price_label(ax2,fut_dates_mc[-1]+pd.Timedelta(days=5),mc_p975[-1],f"P97 {pfmt(mc_p975[-1])}",C_PROJ,7.5)
    add_price_label(ax2,fut_dates_mc[-1]+pd.Timedelta(days=5),mc_p025[-1],f"P3  {pfmt(mc_p025[-1])}",C_PROJ,7.5)
    ax2.axvline(last_date,color="#475569",lw=1.2,ls=":",zorder=4)
    ax2.scatter([last_date],[last_price],color="#F8FAFC",s=60,zorder=10,ec=C_TREND,lw=1.5)
    ax2.set_yscale("log"); ax2.yaxis.set_major_formatter(mticker.FuncFormatter(pfmt))
    ax2.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax2.set_xlim(first_date,fut_dates_mc[-1]+pd.Timedelta(days=int(hist_years_actual*18)))
    ax2.grid(True,ls="--",alpha=0.28); ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2 if hist_years_actual>10 else 1))
    cyc_title = f" | Cycles {', '.join(f'{_T/252:.1f}a' for _T in cyc_periods[:2])} R²={cyc_r2:.0%}" if cyc_ok else ""
    ax2.legend(loc="upper left",ncol=3,fontsize=8)
    ax2.set_title(f"{TICKER} — Monte Carlo OU | {N_SIM} sim. | HL={hl_final:.0f}j | {sig_txt}{cyc_title}",
                  color="#F1F5F9",fontsize=10,fontweight="bold",pad=10)
    fig2.subplots_adjust(left=0.05,right=0.84,top=0.93,bottom=0.07)
    b64_mc=fig_to_b64(fig2); print("  ✓ Monte Carlo" + (" + scénario cyclique" if cyc_ok else ""))

    fig3,axes3=plt.subplots(1,3,figsize=(15,5))
    fig3.patch.set_facecolor(BG_DARK)
    for i,(lbl,rd) in enumerate(risk.items()):
        ax=axes3[i]; ax.set_facecolor(BG_SURF)
        fr_pct=rd["fr"]*100; vl=rd["v95"]*100
        ax.hist(fr_pct,bins=55,color=C_TREND,alpha=0.40,edgecolor="none")
        ax.hist(fr_pct[fr_pct<=vl],bins=25,color=C_PROJ,alpha=0.65,edgecolor="none")
        ax.axvline(vl,color=C_PROJ,lw=1.8,ls="--",label=f"VaR95 {rd['v95']:+.0%}")
        ax.axvline(rd["cv"]*100,color="#FDA4AF",lw=1.2,ls=":",label=f"CVaR  {rd['cv']:+.0%}")
        ax.axvline(rd["esp"]*100,color=C_TREND,lw=1.8,ls="--",label=f"Esp.  {rd['esp']:+.0%}")
        ax.axvline(rd["med"]*100,color=C_BUY,lw=1.2,ls="-.",label=f"Méd.  {rd['med']:+.0%}")
        ax.axvline(0,color="#475569",lw=0.8,alpha=0.6)
        ax.set_title(f"{lbl} · P(>0)={rd['pp']:.0%}\nMDD moy {rd['mdd_m']:+.0%}",
                     color="#F1F5F9",fontsize=9,fontweight="600",pad=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:.0f}%"))
        ax.legend(fontsize=8); ax.grid(True,ls="--",alpha=0.28)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig3.subplots_adjust(left=0.06,right=0.97,top=0.88,bottom=0.12,wspace=0.28)
    b64_dist=fig_to_b64(fig3); print("  ✓ distributions risque")

    fig4,ax4=plt.subplots(figsize=(7,7),subplot_kw=dict(polar=True))
    fig4.patch.set_facecolor(BG_DARK); ax4.set_facecolor(BG_DARK)
    lbls_r=[f"{n}\n({pillar_scores[n]:.0f})" for n,w,s,d,c in FUND_PILLARS]
    vals_r=[pillar_scores[n] for n,w,s,d,c in FUND_PILLARS]; N_ax=len(vals_r)
    angles=[i/N_ax*2*math.pi for i in range(N_ax)]+[0]; vals_r2=vals_r+[vals_r[0]]
    ax4.plot(angles,vals_r2,color=C_BLUE,lw=2); ax4.fill(angles,vals_r2,alpha=0.22,color=C_BLUE)
    for lv,col,al in [(80,C_GREEN,0.04),(60,C_BLUE,0.04),(40,C_AMBER,0.04)]:
        lv_=[lv]*N_ax+[lv]; ax4.fill(angles,lv_,alpha=al,color=col)
        ax4.plot(angles,lv_,color=col,lw=0.6,alpha=0.3,ls="--")
    ax4.set_xticks(angles[:-1]); ax4.set_xticklabels(lbls_r,size=9,color=COL_TEXT)
    ax4.set_ylim(0,100); ax4.set_yticks([20,40,60,80])
    ax4.set_yticklabels(["20","40","60","80"],size=7.5,color=COL_FAINT)
    ax4.grid(color=COL_GRID,linewidth=0.6)
    ax4.set_title(f"{company}\n{sector}\nFondamental {fund_score:.0f}/100 [{score_grade(fund_score)}]",
                  color="#F1F5F9",fontsize=10,fontweight="bold",pad=22)
    b64_radar=fig_to_b64(fig4); print("  ✓ radar fondamental")

    fig5,ax5=plt.subplots(figsize=(10,6))
    fig5.patch.set_facecolor(BG_DARK); ax5.set_facecolor(BG_SURF)
    names_b=[n for n,w,s,d,c in FUND_PILLARS]; scrs_b=[pillar_scores[n] for n in names_b]
    cols_b=[c for n,w,s,d,c in FUND_PILLARS]; y_pos=np.arange(len(names_b))
    ax5.barh(y_pos,[100]*len(names_b),height=0.55,color=BG_SURF2,alpha=0.5,zorder=0)
    ax5.barh(y_pos,scrs_b,height=0.55,color=cols_b,alpha=0.85)
    weights_b=[w for n,w,s,d,c in FUND_PILLARS]
    for i,(s,n) in enumerate(zip(scrs_b,names_b)):
        ax5.text(min(s+1.5,97),i,f"{s:.0f}/100  {score_grade(s)}",va="center",ha="left",fontsize=9.5,color="#F1F5F9",fontweight="600")
        ax5.text(-2,i,f"{weights_b[i]*100:.0f}%",va="center",ha="right",fontsize=8.5,color=COL_FAINT)
    ax5.set_yticks(y_pos); ax5.set_yticklabels(names_b,fontsize=10.5)
    ax5.set_xlim(-8,110); ax5.axvline(50,color="#475569",lw=0.8,ls="--",alpha=0.6)
    ax5.axvline(70,color=C_GREEN,lw=0.8,ls=":",alpha=0.5)
    ax5.set_title(f"{TICKER} ({sector}) — Scorecard | {fund_score:.0f}/100 [{score_grade(fund_score)}]",
                  color="#F1F5F9",fontsize=11,fontweight="bold",pad=10)
    ax5.grid(True,axis="x",ls="--",alpha=0.3)
    b64_bars=fig_to_b64(fig5); print("  ✓ barres piliers")

    def bar_hist_series(ax,s,col,title):
        ax.set_facecolor(BG_SURF); ax.set_title(title,color="#F1F5F9",fontsize=10,fontweight="bold")
        s=s.dropna()
        if s.empty: ax.text(0.5,0.5,"N/A",ha="center",va="center",transform=ax.transAxes,color=COL_FAINT); return
        ys=[str(y) for y in s.index]; vs=list(s.values)
        bars=ax.bar(ys,vs,color=col,alpha=0.80,width=0.6)
        vmax=max(abs(v) for v in vs) or 1
        for bar,v in zip(bars,vs):
            ax.text(bar.get_x()+bar.get_width()/2,v+(vmax*0.02 if v>=0 else -vmax*0.02),
                    fmt_big_mpl(v),ha="center",va="bottom" if v>=0 else "top",
                    fontsize=7.5,color="#F1F5F9",fontweight="600")
        ax.axhline(0,color="#475569",lw=0.8); ax.grid(True,axis="y",ls="--",alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(colors=COL_FAINT,labelsize=8)

    fig6,axes6=plt.subplots(2,3,figsize=(15,9)); fig6.patch.set_facecolor(BG_DARK)
    bar_hist_series(axes6[0,0],rev_hist, C_BLUE,  "Revenus")
    bar_hist_series(axes6[0,1],fcf_hist, C_GREEN, "Free Cash Flow")
    bar_hist_series(axes6[0,2],ni_hist,  C_PURPLE,"Bénéfice Net")
    bar_hist_series(axes6[1,0],cfo_hist, C_SKY,   "Cash Flow Opérationnel")
    bar_hist_series(axes6[1,1],capex_hist.apply(lambda v:-abs(v)),C_RED,"Capex")
    bar_hist_series(axes6[1,2],om_hist.apply(lambda v:v*100),C_AMBER,"Marge Opé. (%)")
    fig6.subplots_adjust(wspace=0.32,hspace=0.42,left=0.06,right=0.97,top=0.94,bottom=0.07)
    b64_hist=fig_to_b64(fig6); print(f"  ✓ historique financier ({n_fin_years} exercices)")

    fig6b,ax6b=plt.subplots(figsize=(14,6))
    fig6b.patch.set_facecolor(BG_DARK); ax6b.set_facecolor(BG_SURF)
    any_margin=False
    for s_m,col,lbl,mk in [(gm_hist,C_SKY,"Marge brute","o"),(om_hist,C_BLUE,"Marge opé.","s"),
                            (nm_hist,C_PURPLE,"Marge nette","^"),(fm_hist,C_GREEN,"Marge FCF","D")]:
        s_m=s_m.dropna()
        if len(s_m)>=2:
            any_margin=True
            ax6b.plot([str(y) for y in s_m.index],s_m.values*100,color=col,lw=2.0,
                      marker=mk,ms=6,label=lbl,alpha=0.92)
            ax6b.annotate(f"{s_m.values[-1]*100:.1f}%",(len(s_m)-1,s_m.values[-1]*100),
                          textcoords="offset points",xytext=(8,0),fontsize=8.5,color=col,fontweight="bold")
    if any_margin:
        ax6b.axhline(0,color="#475569",lw=0.8,ls="--",alpha=0.6)
        ax6b.legend(loc="best",fontsize=9,ncol=4)
        ax6b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:.0f}%"))
        ax6b.grid(True,ls="--",alpha=0.3)
    ax6b.set_title(f"{TICKER} — Évolution des marges ({n_fin_years} exercices)",
                   color="#F1F5F9",fontsize=11,fontweight="bold",pad=10)
    b64_margins=fig_to_b64(fig6b); print("  ✓ évolution des marges")

    def bar_signed_series(ax, s, title, pos_lbl="libération de cash", neg_lbl="consommation de cash"):
        ax.set_facecolor(BG_SURF)
        ax.set_title(title, color="#F1F5F9", fontsize=10, fontweight="bold")
        s = s.dropna()
        if s.empty:
            ax.text(0.5,0.5,"N/A",ha="center",va="center",transform=ax.transAxes,color=COL_FAINT); return
        ys=[str(y) for y in s.index]; vs=list(s.values)
        cols=[C_GREEN if v>=0 else C_RED for v in vs]
        bars=ax.bar(ys,vs,color=cols,alpha=0.82,width=0.6)
        vmax=max(abs(v) for v in vs) or 1
        for bar,v in zip(bars,vs):
            ax.text(bar.get_x()+bar.get_width()/2, v+(vmax*0.03 if v>=0 else -vmax*0.03),
                    fmt_big_mpl(v), ha="center", va="bottom" if v>=0 else "top",
                    fontsize=7.5, color="#F1F5F9", fontweight="600")
        ax.axhline(0,color="#475569",lw=0.9)
        ax.grid(True,axis="y",ls="--",alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(colors=COL_FAINT,labelsize=8)
        ax.text(0.01,0.97,f"▲ {pos_lbl}   ▼ {neg_lbl}",transform=ax.transAxes,
                fontsize=7,color=COL_FAINT,va="top")

    fig9, axes9 = plt.subplots(2,3, figsize=(16,9.5))
    fig9.patch.set_facecolor(BG_DARK)

    bar_signed_series(axes9[0,0], wc_hist, "Δ BFR — impact cash (CF)",
                      "BFR libère du cash", "BFR consomme du cash")

    ax_d = axes9[0,1]; ax_d.set_facecolor(BG_SURF)
    ax_d.set_title("Dette — émission vs remboursement", color="#F1F5F9", fontsize=10, fontweight="bold")
    if _yrs_df:
        _ysd=[str(y) for y in _yrs_df]; _xd=np.arange(len(_yrs_df))
        _iss=[safe(debt_issue_hist.get(y,0.0),0.0) for y in _yrs_df]
        _rep=[-safe(debt_repay_hist.get(y,0.0),0.0) for y in _yrs_df]
        ax_d.bar(_xd-0.18,_iss,width=0.34,color=C_SKY, alpha=0.80,label="Émission")
        ax_d.bar(_xd+0.18,_rep,width=0.34,color=C_RED, alpha=0.80,label="Remboursement")
        _nf=[debt_netflow_hist.get(y,np.nan) for y in _yrs_df]
        ax_d.plot(_xd,_nf,color=C_AMBER,lw=2.0,marker="o",ms=5,label="Flux net",zorder=6)
        ax_d.axhline(0,color="#475569",lw=0.9)
        ax_d.set_xticks(_xd); ax_d.set_xticklabels(_ysd,fontsize=8,color=COL_FAINT)
        ax_d.legend(fontsize=8,loc="best")
        ax_d.grid(True,axis="y",ls="--",alpha=0.3)
        ax_d.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:fmt_big_mpl(v)))
    else:
        ax_d.text(0.5,0.5,"N/A",ha="center",va="center",transform=ax_d.transAxes,color=COL_FAINT)
    ax_d.spines["top"].set_visible(False); ax_d.spines["right"].set_visible(False)
    ax_d.tick_params(colors=COL_FAINT,labelsize=8)

    ax_nd=axes9[0,2]; ax_nd.set_facecolor(BG_SURF)
    ax_nd.set_title("Dette nette (bilan)", color="#F1F5F9", fontsize=10, fontweight="bold")
    _nds=netdebt_hist.dropna()
    if len(_nds):
        _ysn=[str(y) for y in _nds.index]
        _colsn=[C_AMBER if v>0 else C_GREEN for v in _nds.values]
        bars_nd=ax_nd.bar(_ysn,_nds.values,color=_colsn,alpha=0.82,width=0.6)
        _vmaxn=max(abs(v) for v in _nds.values) or 1
        for bar,v in zip(bars_nd,_nds.values):
            ax_nd.text(bar.get_x()+bar.get_width()/2, v+(_vmaxn*0.03 if v>=0 else -_vmaxn*0.03),
                       fmt_big_mpl(v), ha="center", va="bottom" if v>=0 else "top",
                       fontsize=7.5, color="#F1F5F9", fontweight="600")
        ax_nd.axhline(0,color="#475569",lw=0.9)
        ax_nd.grid(True,axis="y",ls="--",alpha=0.3)
        ax_nd.text(0.01,0.97,"▼ dette nette négative = net cash",transform=ax_nd.transAxes,
                   fontsize=7,color=COL_FAINT,va="top")
    else:
        ax_nd.text(0.5,0.5,"N/A",ha="center",va="center",transform=ax_nd.transAxes,color=COL_FAINT)
    ax_nd.spines["top"].set_visible(False); ax_nd.spines["right"].set_visible(False)
    ax_nd.tick_params(colors=COL_FAINT,labelsize=8)

    ax_r=axes9[1,0]; ax_r.set_facecolor(BG_SURF)
    ax_r.set_title("Retours actionnaires vs FCF", color="#F1F5F9", fontsize=10, fontweight="bold")
    if cover_years:
        _ysr=[str(y) for y in cover_years]; _xr=np.arange(len(cover_years))
        _dvs=[safe(divpaid_hist.get(y,0.0),0.0) for y in cover_years]
        _bbs=[safe(buyback_hist_s.get(y,0.0),0.0) for y in cover_years]
        _fcs=[safe(fcf_hist.get(y,np.nan)) for y in cover_years]
        ax_r.bar(_xr,_dvs,width=0.55,color=C_BLUE, alpha=0.80,label="Dividendes")
        ax_r.bar(_xr,_bbs,width=0.55,bottom=_dvs,color=C_PINK,alpha=0.80,label="Rachats")
        ax_r.plot(_xr,_fcs,color=C_GREEN,lw=2.2,marker="D",ms=5,label="FCF",zorder=6)
        ax_r.axhline(0,color="#475569",lw=0.9)
        ax_r.set_xticks(_xr); ax_r.set_xticklabels(_ysr,fontsize=8,color=COL_FAINT)
        ax_r.legend(fontsize=8,loc="best")
        ax_r.grid(True,axis="y",ls="--",alpha=0.3)
        ax_r.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:fmt_big_mpl(v)))
        ax_r.text(0.01,0.97,"Barres > ligne FCF = retours financés par dette/cash",
                  transform=ax_r.transAxes,fontsize=7,color=COL_FAINT,va="top")
    else:
        ax_r.text(0.5,0.5,"N/A",ha="center",va="center",transform=ax_r.transAxes,color=COL_FAINT)
    ax_r.spines["top"].set_visible(False); ax_r.spines["right"].set_visible(False)
    ax_r.tick_params(colors=COL_FAINT,labelsize=8)

    bar_hist_series(axes9[1,1], intexp_hist.apply(lambda v:-abs(v)), C_RED,
                    "Charge d'intérêts (pèse sur le bénéfice net)")

    ax_sa=axes9[1,2]; ax_sa.set_facecolor(BG_SURF)
    ax_sa.set_title("SBC & Acquisitions", color="#F1F5F9", fontsize=10, fontweight="bold")
    _ysa_all=sorted(set(sbc_hist_s.index)|set(acq_hist_abs.dropna().index))
    if _ysa_all:
        _xsa=np.arange(len(_ysa_all))
        _sbcv=[safe(sbc_hist_s.get(y,np.nan)) for y in _ysa_all]
        _acqv=[-safe(acq_hist_abs.get(y,0.0),0.0) for y in _ysa_all]
        ax_sa.bar(_xsa-0.18,[0 if np.isnan(v) else v for v in _sbcv],width=0.34,
                  color=C_PURPLE,alpha=0.80,label="SBC (dilution)")
        ax_sa.bar(_xsa+0.18,_acqv,width=0.34,color=C_ORANGE,alpha=0.80,label="Acquisitions (sortie)")
        ax_sa.axhline(0,color="#475569",lw=0.9)
        ax_sa.set_xticks(_xsa); ax_sa.set_xticklabels([str(y) for y in _ysa_all],fontsize=8,color=COL_FAINT)
        ax_sa.legend(fontsize=8,loc="best")
        ax_sa.grid(True,axis="y",ls="--",alpha=0.3)
        ax_sa.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:fmt_big_mpl(v)))
    else:
        ax_sa.text(0.5,0.5,"N/A",ha="center",va="center",transform=ax_sa.transAxes,color=COL_FAINT)
    ax_sa.spines["top"].set_visible(False); ax_sa.spines["right"].set_visible(False)
    ax_sa.tick_params(colors=COL_FAINT,labelsize=8)

    fig9.suptitle(f"{TICKER} — Facteurs explicatifs de la tendance financière (BFR · dette · retours · intérêts)",
                  color="#F1F5F9",fontsize=12,fontweight="bold",y=0.985)
    fig9.subplots_adjust(wspace=0.30,hspace=0.40,left=0.06,right=0.97,top=0.91,bottom=0.06)
    b64_explain=fig_to_b64(fig9); print("  ✓ facteurs explicatifs (BFR · dette · retours · intérêts · SBC)")

    def build_fcf_bridge_chart():
        common=[y for y in fcf_hist.dropna().index if y in ni_hist.dropna().index]
        if not common: return None, None
        y=common[-1]
        fcf_v=safe(fcf_hist.get(y))
        steps=[("Bénéfice net", safe(ni_hist.get(y)))]
        _da=safe(da_hist.get(y,np.nan))
        if not np.isnan(_da): steps.append(("+ D&A", abs(_da)))
        _wc=safe(wc_hist.get(y,np.nan))
        if not np.isnan(_wc): steps.append(("Δ BFR", _wc))
        _cx=safe(capex_hist.get(y,np.nan))
        if not np.isnan(_cx): steps.append(("− Capex", -abs(_cx)))
        if np.isnan(fcf_v) or np.isnan(steps[0][1]) or len(steps)<2: return None, None
        autres=fcf_v - sum(v for _,v in steps)
        steps.append(("Autres (non-cash…)", autres))

        fig,ax=plt.subplots(figsize=(13,5.8))
        fig.patch.set_facecolor(BG_DARK); ax.set_facecolor(BG_SURF)
        labels=[s[0] for s in steps]+["FCF"]
        cum=0.0; xs=np.arange(len(labels))
        for i,(lbl,v) in enumerate(steps):
            if i==0:
                col=C_BLUE; ax.bar(i,v,width=0.62,color=col,alpha=0.88)
                ax.text(i,v/2,fmt_big_mpl(v),ha="center",va="center",fontsize=9,
                        color="#F1F5F9",fontweight="700")
                cum=v
            else:
                col=C_GREEN if v>=0 else C_RED
                ax.bar(i,v,bottom=cum,width=0.62,color=col,alpha=0.88)
                ax.text(i,cum+v/2,f"{'+' if v>=0 else ''}{fmt_big_mpl(v)}",ha="center",va="center",
                        fontsize=8.5,color="#F1F5F9",fontweight="700")
                ax.plot([i-1+0.31,i-0.31],[cum,cum],color=COL_FAINT,lw=0.9,ls=":",alpha=0.7)
                cum+=v
        ax.bar(len(steps),fcf_v,width=0.62,color=C_SKY,alpha=0.92)
        ax.text(len(steps),fcf_v/2,fmt_big_mpl(fcf_v),ha="center",va="center",fontsize=9.5,
                color="#0B1120",fontweight="800")
        ax.plot([len(steps)-1+0.31,len(steps)-0.31],[cum,cum],color=COL_FAINT,lw=0.9,ls=":",alpha=0.7)
        ax.axhline(0,color="#475569",lw=0.9)
        ax.set_xticks(xs); ax.set_xticklabels(labels,fontsize=9,color=COL_TEXT)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:fmt_big_mpl(v)))
        ax.grid(True,axis="y",ls="--",alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_title(f"{TICKER} — Pont FCF · exercice {y} | Bénéfice net → FCF",
                     color="#F1F5F9",fontsize=11,fontweight="bold",pad=10)
        return fig, y

    fig10, bridge_year = build_fcf_bridge_chart()
    if fig10:
        b64_bridge=fig_to_b64(fig10); print(f"  ✓ pont FCF (exercice {bridge_year})")
    else:
        b64_bridge=None; bridge_year="—"
        print("  ⚠ pont FCF — données insuffisantes (NI/D&A/BFR/Capex manquants)")

    fig7,ax7=plt.subplots(figsize=(12,5))
    fig7.patch.set_facecolor(BG_DARK); ax7.set_facecolor(BG_SURF)
    _,_,_,_,fcfs_b=dcf_base
    if fcfs_b:
        ts=[f[0] for f in fcfs_b]; pvs=[f[2]/1e9 for f in fcfs_b]; fvs=[f[1]/1e9 for f in fcfs_b]
        x=np.arange(len(ts))
        ax7.bar(x,fvs,width=0.35,color=C_BLUE,alpha=0.70,label="FCF projeté (B)",zorder=3)
        ax7.bar(x+0.38,pvs,width=0.35,color=C_GREEN,alpha=0.70,label="PV FCF (B)",zorder=3)
        for sc,gr,res,col,ls_ in [("Bear",DCF_BEAR_GR,dcf_bear,C_RED,"--"),
                                   ("Base",DCF_BASE_GR,dcf_base,C_AMBER,"-"),
                                   ("Bull",DCF_BULL_GR,dcf_bull,C_GREEN,":")]:
            iv,_,mg,tv_pct,_2=res
            if not np.isnan(iv):
                lbl_dcf=f"{sc} VI={iv:.2f} ({'+' if mg>0 else ''}{mg*100:.0f}%) TV={tv_pct*100:.0f}%"
                ax7.axhline(iv/1e9 if iv>1e6 else iv,color=col,lw=1.5,ls=ls_,alpha=0.85,label=lbl_dcf)
        ax7.set_xticks(x+0.19); ax7.set_xticklabels([f"Année {t}" for t in ts],fontsize=8.5,rotation=30)
        ax7.legend(loc="upper left",fontsize=8.5); ax7.grid(True,axis="y",ls="--",alpha=0.3)
        ax7.set_title(f"DCF {TICKER} | FCF={fmt_big_mpl(fcf_ttm)} | WACC={DCF_WACC*100:.2f}% (β={wacc_beta:.2f}) | VI base={dcf_fmt(dcf_base)[0]}",
                      color="#F1F5F9",fontsize=10,fontweight="bold")
    else:
        ax7.text(0.5,0.5,"FCF insuffisant",ha="center",va="center",transform=ax7.transAxes,color=COL_FAINT,fontsize=13)
    b64_dcf=fig_to_b64(fig7); print("  ✓ DCF")

    fig_tech=build_tech_chart(TICKER,last_date,last_price,years=TECH_YEARS)
    if fig_tech:
        b64_tech=fig_to_b64(fig_tech); print("  ✓ graphique technique (chandeliers weekly)")
    else:
        b64_tech=None; print("  ⚠ graphique technique — données insuffisantes")

    print("\n  ✅ Tous les graphiques générés")

    # ─────────────────────────────────────────────────────────
    #  BLOC 5 — EXPORT HTML
    # ─────────────────────────────────────────────────────────
    print("\n"+"═"*65); print("  BLOC 5 — EXPORT HTML"); print("═"*65)

    now_str=datetime.now().strftime("%d/%m/%Y %H:%M")

    def score_bar_html(s):
        p=min(s,100); c=score_color(s)
        return f'<div class="bar"><div class="bar-fill" style="width:{p:.0f}%;background:{c}"></div></div>'

    def pill_row(label,val_str,color=None,ref=""):
        col=color or COL_TEXT
        ref_html=f'<span class="ref">{ref}</span>' if ref else ""
        return f'<div class="pill-row"><span class="pill-lbl">{label}</span><span class="pill-val" style="color:{col}">{val_str}{ref_html}</span></div>'

    def pillar_card_html(name,weight,color):
        s=pillar_scores[name]; d=pillar_details[name]; rows=""
        for metric,(val,ref) in d.items():
            if np.isnan(val): val_s="N/A"; col=COL_FAINT
            elif "%" in ref and val<1: val_s=pct(val); col=C_GREEN if val>0 else C_RED
            else: val_s=fmt_n(val); col=COL_TEXT
            rows+=pill_row(metric,val_s,col,ref)
        return f"""<div class="pillar-card" style="border-left-color:{color}">
          <div class="pillar-head"><span style="color:{color}">{name}</span>
            <span class="pillar-score" style="color:{score_color(s)}">{s:.0f}<small>/100 · {score_grade(s)}</small></span></div>
          {score_bar_html(s)}<div style="margin-top:11px">{rows}</div>
          <div class="pillar-weight">poids {weight*100:.0f}%</div></div>"""

    pillar_cards_html='<div class="pillar-grid">'
    for n,w,s,d,c in FUND_PILLARS: pillar_cards_html+=pillar_card_html(n,w,c)
    pillar_cards_html+="</div>"

    dcf_rows=""
    for sc,gr,res in [("🐻 Bear",DCF_BEAR_GR,dcf_bear),("📊 Base",DCF_BASE_GR,dcf_base),("🚀 Bull",DCF_BULL_GR,dcf_bull)]:
        iv,_,mg,tv_pct,_2=res
        iv_s=f"${iv:.2f}" if not np.isnan(iv) else "N/A"
        mg_s=pct(mg) if not np.isnan(mg) else "N/A"
        tv_s=f"{tv_pct*100:.0f}%" if not np.isnan(tv_pct) else "N/A"
        tv_warn=' ⚠' if not np.isnan(tv_pct) and tv_pct>0.75 else ''
        cls="pos" if not np.isnan(mg) and mg>0 else "neg"
        tv_cls="neg" if not np.isnan(tv_pct) and tv_pct>0.75 else "warnc" if not np.isnan(tv_pct) and tv_pct>0.60 else ""
        dcf_rows+=f"<tr><td>{sc}</td><td class='pos'>{gr*100:.0f}%/an</td><td style='font-weight:700'>{iv_s}</td><td class='{cls}'>{mg_s}</td><td class='{tv_cls}'>{tv_s}{tv_warn}</td></tr>"

    risk_rows=""
    for lbl,rd in risk.items():
        pp_cls="pos" if rd["pp"]>0.55 else "neg" if rd["pp"]<0.45 else "warnc"
        risk_rows+=f"""<tr><td>{lbl}</td><td class="{pp_cls}">{rd['pp']:.0%}</td>
          <td class="{'pos' if rd['esp']>0 else 'neg'}">{rd['esp']:+.0%}</td>
          <td class="{'pos' if rd['med']>0 else 'neg'}">{rd['med']:+.0%}</td>
          <td class="neg">{rd['v95']:+.0%}</td><td class="neg">{rd['mdd_m']:+.0%}</td></tr>"""

    cover_rows=""
    for _y,_fcf,_dv,_bb,_tot,_cov in coverage_data[-10:]:
        if np.isnan(_cov): cov_s="N/A"; cov_cls=""
        else:
            cov_s=f"{_cov:.2f}×"
            cov_cls="pos" if _cov>=1.5 else "warnc" if _cov>=1.0 else "neg"
        cover_rows+=(f"<tr><td>{_y}</td><td class='pos'>{fmt_big(_fcf)}</td>"
                     f"<td>{fmt_big(_dv)}</td><td>{fmt_big(_bb)}</td>"
                     f"<td style='font-weight:700'>{fmt_big(_tot)}</td>"
                     f"<td class='{cov_cls}' style='font-weight:700'>{cov_s}</td></tr>")
    if not cover_rows:
        cover_rows="<tr><td colspan='6' style='text-align:center;color:var(--faint)'>Données insuffisantes</td></tr>"

    ZONE_BG_CSS=["rgba(244,63,94,.60)","rgba(251,146,60,.55)","rgba(52,211,153,.42)",
                 "rgba(52,211,153,.25)","rgba(56,189,248,.48)","rgba(59,130,246,.65)"]
    zone_bar_html=""
    for zi,zk in enumerate(ZONE_KEYS):
        pct_z=weights_z[zk]*100; short=ZONE_SHORT[zi]
        outline="2px solid #F8FAFC" if zk==pos_zone else "none"
        zone_bar_html+=(f'<div style="flex:{max(pct_z,2):.1f};background:{ZONE_BG_CSS[zi]};'
                        f'outline:{outline};outline-offset:-1px;display:flex;align-items:center;'
                        f'justify-content:center;font-size:10px;font-family:monospace;font-weight:700">'
                        f'{short if pct_z>5 else ""}</div>')

    def band_row(name,v,col):
        gap=(v/last_price-1) if not np.isnan(v) and last_price>0 else np.nan
        cls="pos" if not np.isnan(gap) and gap>0 else "neg"
        return f"<tr><td style='color:{col};font-weight:700'>{name}</td><td>{pfmt(v)}</td><td class='{cls}'>{pct(gap,sign=True)}</td></tr>"

    bands_now_rows=(band_row("+2σ",bp2_now,C_SELL)+band_row("+1σ",bp1_now,C_SELL)
                   +band_row("Tendance",tr_now,C_TREND)
                   +f"<tr style='background:rgba(96,165,250,.06)'><td style='color:#F8FAFC;font-weight:700'>Cours actuel</td><td style='font-weight:700'>{pfmt(last_price)}</td><td>{pos_sigma:+.2f}σ</td></tr>"
                   +band_row("−1σ",bm1_now,C_BUY)+band_row("−2σ",bm2_now,C_BUY))

    pos_color_z=ZONE_COLORS[pos_idx]
    col_g=score_color(global_score); col_f=score_color(fund_score)
    col_q=C_GREEN if quant_score_100>=70 else C_BLUE if quant_score_100>=50 else C_AMBER if quant_score_100>=30 else C_RED
    grade_g=score_grade(global_score); grade_f=score_grade(fund_score); grade_q=score_grade(quant_score_100)
    t_mo_v=round(t_v*12)

    def generate_comment():
        lines=[]
        if global_score>=75: lines.append(f"<b>{company}</b> ressort avec un profil exceptionnel.")
        elif global_score>=60: lines.append(f"<b>{company}</b> présente un profil globalement positif.")
        elif global_score>=45: lines.append(f"<b>{company}</b> affiche un profil mitigé, des forces réelles coexistent avec des points de vigilance.")
        else: lines.append(f"<b>{company}</b> présente un profil sous les seuils cibles. Prudence recommandée.")
        if quant_score_100>=65: lines.append(f"Sur le plan quantitatif ({quant_score_100:.0f}/100), la position à {pos_sigma:+.2f}σ est favorable, avec une probabilité de {P_v:.0%} d'atteindre +1σ et un CAGR médian estimé à {CAGR_v:+.1%}/an.")
        elif quant_score_100>=45: lines.append(f"Sur le plan quantitatif ({quant_score_100:.0f}/100), la position à {pos_sigma:+.2f}σ est neutre.")
        else: lines.append(f"Sur le plan quantitatif ({quant_score_100:.0f}/100), la position à {pos_sigma:+.2f}σ («{pos_zone}») est défavorable.")
        best=max(pillar_scores,key=pillar_scores.get); worst=min(pillar_scores,key=pillar_scores.get)
        lines.append(f"Fondamental ({fund_score:.0f}/100, barème <b>{sector}</b>) : pilier fort = <b>{best}</b> ({pillar_scores[best]:.0f}/100), vigilance sur <b>{worst}</b> ({pillar_scores[worst]:.0f}/100).")
        if len(fm_hist.dropna())>=3:
            fm_v=fm_hist.dropna(); trend_txt="en amélioration" if fm_v.iloc[-1]>fm_v.iloc[0] else "en dégradation"
            lines.append(f"Sur {len(fm_v)} exercices, la marge FCF est {trend_txt} ({pct(fm_v.iloc[0])} → {pct(fm_v.iloc[-1])}).")
        iv_b,_,mg_b,tv_pct_b,_2=dcf_base
        if not np.isnan(mg_b):
            tv_warn=f" (attention : valeur terminale = {tv_pct_b*100:.0f}% de la valeur totale)" if not np.isnan(tv_pct_b) and tv_pct_b>0.75 else ""
            if mg_b>0.20: lines.append(f"Le DCF base (WACC {DCF_WACC*100:.2f}%, β={wacc_beta:.2f}) suggère {dcf_fmt(dcf_base)[0]}, marge de sécurité {pct(mg_b)}.{tv_warn}")
            elif mg_b>-0.10: lines.append(f"DCF base ({dcf_fmt(dcf_base)[0]}, WACC {DCF_WACC*100:.2f}%) proche du cours — valorisation juste.{tv_warn}")
            else: lines.append(f"DCF base ({dcf_fmt(dcf_base)[0]}) signale une possible surévaluation de {pct(abs(mg_b))}.{tv_warn}")
        return "<br><br>".join(lines)

    comment_html=generate_comment()

    bridge_img_html=(f'<img class="chart" src="data:image/png;base64,{b64_bridge}">' if b64_bridge
                     else '<div style="padding:40px;text-align:center;color:var(--faint)">Données insuffisantes pour le pont FCF</div>')

    tech_img_html=(f'<img class="chart" src="data:image/png;base64,{b64_tech}">' if b64_tech
                   else '<div style="padding:40px;text-align:center;color:var(--faint)">Historique insuffisant pour les chandeliers</div>')

    html=f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UnifiedDash v5 — {TICKER}</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#0B1120;--surf:#151E32;--surf2:#1D2A44;--border:rgba(148,163,184,.10);--border2:rgba(148,163,184,.18);--faint:#64748B;--muted:#94A3B8;--bright:#F1F5F9;--blue:#60A5FA;--green:#34D399;--amber:#FBBF24;--red:#F43F5E;--sky:#38BDF8;--pink:#EC4899;--purple:#A78BFA;--orange:#FB923C;--mono:"JetBrains Mono","Fira Mono",ui-monospace,monospace;--radius:12px}}
html{{scroll-behavior:smooth}}
body{{background:radial-gradient(1100px 500px at 85% -10%,rgba(96,165,250,.07),transparent 60%),var(--bg);color:#CBD5E1;font-family:"Segoe UI",system-ui,sans-serif;line-height:1.6;font-size:15px}}
.tape{{position:sticky;top:0;z-index:50;background:rgba(11,17,32,.85);backdrop-filter:blur(12px);border-bottom:1px solid rgba(96,165,250,.18);padding:11px 28px;display:flex;align-items:center;gap:18px;flex-wrap:wrap}}
.brand{{font-family:var(--mono);font-size:17px;font-weight:700;color:var(--blue)}}
.dot{{width:7px;height:7px;border-radius:50%;background:var(--green);display:inline-block;box-shadow:0 0 8px var(--green);animation:pulse 2.4s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
.nav{{display:flex;gap:6px;margin-left:auto;flex-wrap:wrap}}
.nav a{{font-size:11.5px;color:var(--muted);text-decoration:none;padding:5px 12px;border-radius:999px;border:1px solid transparent;transition:all .15s;font-weight:600}}
.nav a:hover{{color:var(--bright);border-color:var(--border2);background:rgba(96,165,250,.08)}}
.stamp{{display:flex;align-items:center;gap:7px;background:rgba(96,165,250,.08);border:1px solid rgba(96,165,250,.25);border-radius:999px;padding:4px 13px;font-size:11.5px;color:var(--blue);font-family:var(--mono)}}
.wrap{{max-width:1360px;margin:0 auto;padding:28px 28px 70px}}
.hero{{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:20px}}
.hero h1{{font-size:26px;font-weight:800;color:var(--bright);letter-spacing:-.02em}}
.hero .tick{{font-family:var(--mono);font-size:15px;color:var(--blue);background:rgba(96,165,250,.10);border:1px solid rgba(96,165,250,.25);border-radius:7px;padding:2px 10px}}
.sector-badge{{font-size:12px;background:rgba(167,139,250,.12);color:var(--purple);border:1px solid rgba(167,139,250,.25);border-radius:6px;padding:2px 10px;font-weight:600}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:11px;margin-bottom:22px}}
.kpi{{background:linear-gradient(160deg,var(--surf),#121A2C);border:1px solid var(--border2);border-radius:var(--radius);padding:15px 18px 14px;position:relative;overflow:hidden;transition:transform .15s,border-color .15s}}
.kpi::before{{content:"";position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,rgba(96,165,250,.55),transparent 70%)}}
.kpi:hover{{transform:translateY(-2px);border-color:rgba(96,165,250,.35)}}
.kpi .lbl{{font-size:9.5px;color:var(--faint);letter-spacing:.11em;text-transform:uppercase;margin-bottom:7px;font-weight:700}}
.kpi .val{{font-family:var(--mono);font-size:22px;font-weight:800;line-height:1.05;letter-spacing:-.02em;color:var(--bright)}}
.kpi .sub{{font-size:11px;color:var(--muted);margin-top:6px}}
.section-title{{font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin:34px 0 14px;padding-bottom:8px;border-bottom:1px solid rgba(96,165,250,.18);display:flex;align-items:center;gap:10px;scroll-margin-top:70px}}
.section-title .line{{flex:1;height:1px;background:rgba(96,165,250,.10)}}
.card{{background:linear-gradient(170deg,var(--surf),#111A2D);border:1px solid var(--border);border-radius:var(--radius);padding:20px 24px;margin-bottom:14px;box-shadow:0 8px 24px rgba(0,0,0,.22)}}
.card h2{{font-size:12.5px;font-weight:700;color:var(--blue);margin-bottom:10px;letter-spacing:.02em}}
.bar{{height:7px;background:rgba(255,255,255,.06);border-radius:4px;overflow:hidden;margin-top:6px}}
.bar-fill{{height:100%;border-radius:4px}}
.pill-row{{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.04)}}
.pill-lbl{{font-size:12px;color:var(--muted)}}
.pill-val{{font-family:var(--mono);font-size:13px;font-weight:700}}
.ref{{font-size:10px;color:var(--faint);margin-left:6px;font-family:"Segoe UI",sans-serif;font-weight:400}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;padding:8px 11px;font-size:10px;color:var(--blue);letter-spacing:.08em;text-transform:uppercase;border-bottom:1px solid var(--border2);background:var(--surf2)}}
td{{padding:9px 11px;border-bottom:1px solid rgba(255,255,255,.03);font-family:var(--mono);font-size:13px}}
tr:last-child td{{border-bottom:none}}
td:first-child{{font-family:inherit;color:var(--bright);font-weight:500}}
.pos{{color:var(--green)}} .neg{{color:var(--red)}} .warnc{{color:var(--amber)}}
img.chart{{width:100%;border-radius:9px;display:block;border:1px solid rgba(255,255,255,.05);margin-top:5px}}
.zone-bar{{display:flex;height:22px;border-radius:7px;overflow:hidden;margin:12px 0 6px;border:1px solid var(--border2)}}
.pillar-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-bottom:6px}}
.pillar-card{{background:var(--surf);border:1px solid var(--border);border-left:3px solid var(--blue);border-radius:10px;padding:16px 18px}}
.pillar-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;font-size:13px;font-weight:700}}
.pillar-score{{font-family:var(--mono);font-size:19px;font-weight:800}}
.pillar-score small{{font-size:11px;color:var(--faint);font-weight:400}}
.pillar-weight{{font-size:10px;color:var(--faint);margin-top:7px}}
.score-trio{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-bottom:20px}}
.score-box{{border-radius:14px;padding:24px;text-align:center;border:1px solid var(--border2);background:linear-gradient(165deg,var(--surf),#101a2e)}}
.score-num-lg{{font-family:var(--mono);font-size:62px;font-weight:900;line-height:1;letter-spacing:-.03em}}
.comment-box{{background:rgba(96,165,250,.05);border:1px solid rgba(96,165,250,.18);border-left:3px solid var(--blue);border-radius:var(--radius);padding:22px 26px;font-size:13.5px;color:#CBD5E1;line-height:1.85}}
.explain-box{{background:rgba(251,146,60,.05);border:1px solid rgba(251,146,60,.20);border-left:3px solid var(--orange);border-radius:var(--radius);padding:20px 24px;font-size:13px;color:#CBD5E1;line-height:1.85;margin-bottom:14px}}
.grid-2{{display:grid;grid-template-columns:1fr 1.4fr;gap:14px;margin-bottom:14px}}
.wacc-strip{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;padding:12px 16px;border-radius:9px;background:rgba(96,165,250,.05);border:1px solid rgba(96,165,250,.18);margin-bottom:12px;font-size:11px;color:var(--muted)}}
.wacc-strip .wk{{font-family:var(--mono);font-size:15px;font-weight:700;color:var(--bright)}}
@media(max-width:900px){{.grid-2{{grid-template-columns:1fr}}.nav{{display:none}}}}
footer{{margin-top:42px;padding-top:18px;border-top:1px solid var(--border);font-size:11px;color:#475569;line-height:1.9}}
</style>
</head>
<body>
<header class="tape">
  <div class="brand">▲ Unified<b style="color:#F1F5F9">Dash</b> <span style="color:var(--purple)">v5</span></div>
  <nav class="nav">
    <a href="#quant">① Quantitatif</a>
    <a href="#tech">② Technique</a>
    <a href="#fond">③ Fondamental</a>
    <a href="#explain">🔍 Facteurs</a>
    <a href="#hist">④ Historique</a>
    <a href="#synthese">⑤ Synthèse</a>
  </nav>
  <div class="stamp"><span class="dot"></span><span>{now_str}</span></div>
</header>
<div class="wrap">
  <div class="hero">
    <h1>{company}</h1><span class="tick">{TICKER}</span>
    <span class="sector-badge">{sector}</span>
    <span style="font-size:12px;color:var(--faint)">{industry} · {currency}</span>
  </div>
  <div class="kpis">
    <div class="kpi"><div class="lbl">Cours</div><div class="val">${price:.2f}</div><div class="sub">{last_date.strftime("%d/%m/%Y")}</div></div>
    <div class="kpi"><div class="lbl">Market Cap</div><div class="val" style="color:var(--blue)">{fmt_big(market_cap)}</div><div class="sub">EV : {fmt_big(ev)}</div></div>
    <div class="kpi"><div class="lbl">Signal Quant</div><div class="val" style="color:{sig_color};font-size:15px">{sig_txt}</div><div class="sub">{pos_sigma:+.2f}σ · «{pos_zone}»</div></div>
    <div class="kpi"><div class="lbl">Revenus TTM</div><div class="val" style="color:var(--sky)">{fmt_big(rev_ttm)}</div><div class="sub">{pct(rev_growth_yoy,sign=True)} YoY</div></div>
    <div class="kpi"><div class="lbl">FCF TTM</div><div class="val" style="color:var(--green)">{fmt_big(fcf_ttm)}</div><div class="sub">Marge {pct(fcf_margin)}</div></div>
    <div class="kpi"><div class="lbl">P/E · P/B · P/S</div><div class="val" style="color:var(--amber);font-size:16px">{fmt_n(pe_ratio,1)}× · {fmt_n(pb_ratio,1)}× · {fmt_n(ps_ratio,1)}×</div><div class="sub">EV/EBITDA {fmt_n(ev_ebitda,1)}</div></div>
    <div class="kpi"><div class="lbl">ROE / ROIC</div><div class="val" style="color:var(--purple)">{pct(roe)} / {pct(roic)}</div><div class="sub">ROA {pct(roa)}</div></div>
    <div class="kpi"><div class="lbl">WACC dynamique</div><div class="val" style="color:var(--sky)">{DCF_WACC*100:.2f}%</div><div class="sub">β={wacc_beta:.2f} · Ke={wacc_ke*100:.1f}% · Kd={wacc_kd*100:.1f}%</div></div>
  </div>

  <div class="section-title" id="quant">
    <span style="font-family:var(--mono);color:var(--blue)">①</span><span>Analyse Quantitative</span><span class="line"></span>
    <span style="color:{col_q};font-family:var(--mono)">{quant_score_100:.0f}/100 [{grade_q}]</span>
  </div>
  <div class="card">
    <div style="display:flex;align-items:flex-start;gap:32px;flex-wrap:wrap">
      <div style="min-width:200px">
        <div style="font-size:10px;color:var(--sky);text-transform:uppercase;letter-spacing:.09em;font-weight:700;margin-bottom:8px">Score Quantitatif</div>
        <div style="font-family:var(--mono);font-size:62px;font-weight:900;line-height:1;color:{col_q}">{quant_score_100:.0f}</div>
        <div style="font-size:12px;color:var(--faint);margin-top:3px">/ 100  [{grade_q}]</div>
        {score_bar_html(quant_score_100)}
        <div style="margin-top:14px;padding:10px 14px;border-radius:9px;background:{sig_bg};border:1px solid {sig_border}">
          <div style="color:{sig_color};font-weight:700;font-size:14px">{sig_txt}</div>
          <div style="color:var(--faint);font-size:11px;margin-top:3px">{pos_sigma:+.2f}σ · «{pos_zone}»</div>
        </div>
        {'<div style="margin-top:8px;font-size:11px;color:var(--amber)">⚠ Décote appliquée : position >' + f'{pos_sigma:.1f}σ</div>' if pos_sigma > 0 else ''}
      </div>
      <div style="flex:1;min-width:280px">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
          <div style="background:rgba(56,189,248,.07);border:1px solid rgba(56,189,248,.20);border-radius:9px;padding:12px">
            <div style="font-size:10px;color:var(--faint);text-transform:uppercase">P(+1σ)</div>
            <div style="font-family:var(--mono);font-size:22px;font-weight:700;color:{'#34D399' if P_v>=0.70 else '#60A5FA'};margin-top:4px">{P_v:.0%}</div>
            <div style="font-size:10px;color:var(--faint)">norm={norm_P:.3f} · poids 35%</div>
          </div>
          <div style="background:rgba(52,211,153,.07);border:1px solid rgba(52,211,153,.20);border-radius:9px;padding:12px">
            <div style="font-size:10px;color:var(--faint);text-transform:uppercase">CAGR médian</div>
            <div style="font-family:var(--mono);font-size:22px;font-weight:700;color:{'#34D399' if CAGR_v>=0.25 else '#60A5FA'};margin-top:4px">{CAGR_v:+.1%}</div>
            <div style="font-size:10px;color:var(--faint)">norm={norm_CAGR:.3f} · poids 25%</div>
          </div>
          <div style="background:rgba(244,114,182,.07);border:1px solid rgba(244,114,182,.20);border-radius:9px;padding:12px">
            <div style="font-size:10px;color:var(--faint);text-transform:uppercase">MDD médian</div>
            <div style="font-family:var(--mono);font-size:22px;font-weight:700;color:{'#34D399' if MDD_v<=0.15 else '#FBBF24' if MDD_v<=0.30 else '#F43F5E'};margin-top:4px">{MDD_v:.0%}</div>
            <div style="font-size:10px;color:var(--faint)">norm={norm_MDD:.3f} · poids 25%</div>
          </div>
          <div style="background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.20);border-radius:9px;padding:12px">
            <div style="font-size:10px;color:var(--faint);text-transform:uppercase">Durée médiane</div>
            <div style="font-family:var(--mono);font-size:22px;font-weight:700;color:{'#34D399' if t_v<=0.5 else '#60A5FA'};margin-top:4px">{t_mo_v} mois</div>
            <div style="font-size:10px;color:var(--faint)">norm={norm_t:.3f} · poids 15%</div>
          </div>
        </div>
      </div>
      <div style="min-width:220px">
        <div style="font-size:10px;color:var(--sky);text-transform:uppercase;letter-spacing:.09em;font-weight:700;margin-bottom:8px">Niveaux ±σ à l'instant t</div>
        <table><tbody>{bands_now_rows}</tbody></table>
        <div style="font-size:11px;color:var(--faint);margin-top:8px;line-height:1.8">
          R² <span style="font-family:var(--mono);color:var(--bright)">{r2_full:.4f}</span> ·
          σ <span style="font-family:var(--mono);color:var(--bright)">{sigma_log*100:.1f}%</span> ·
          Drift <span style="font-family:var(--mono);color:{'#34D399' if drift_annual>0 else '#F43F5E'}">{drift_annual:+.2%}/an</span><br>
          HL <span style="font-family:var(--mono);color:var(--bright)">{hl_final:.0f}j ({hl_yr:.1f}ans)</span>
        </div>
      </div>
    </div>
  </div>
  <div class="zone-bar">{zone_bar_html}</div>
  <div style="font-size:11px;color:var(--muted);margin-bottom:16px;font-family:var(--mono)">
    <span style="color:{pos_color_z};font-weight:700">▲ «{pos_zone}» · {pos_sigma:+.2f}σ</span>&nbsp;&nbsp;|&nbsp;&nbsp;
    {' · '.join([f'<span>{ZONE_SHORT[i]} {weights_z[ZONE_KEYS[i]]:.0%}</span>' for i in range(6)])}
  </div>
  <div class="card"><h2>Régression log-prix · bandes ±σ — niveaux à l'instant t</h2><img class="chart" src="data:image/png;base64,{b64_reg}"></div>
  <div class="card"><h2>Monte Carlo Ornstein-Uhlenbeck · {N_SIM} simulations</h2><img class="chart" src="data:image/png;base64,{b64_mc}"></div>
  <div class="grid-2">
    <div class="card"><h2>Risque par horizon</h2>
      <table><thead><tr><th>Horizon</th><th>P(&gt;0)</th><th>Esp.</th><th>Méd.</th><th>VaR95</th><th>MDD</th></tr></thead>
      <tbody>{risk_rows}</tbody></table></div>
    <div class="card"><h2>Distributions de rendement</h2><img class="chart" src="data:image/png;base64,{b64_dist}"></div>
  </div>

  <div class="section-title" id="tech">
    <span style="font-family:var(--mono);color:var(--blue)">②</span><span>Analyse Technique</span><span class="line"></span>
    <span style="font-size:11px;color:var(--muted)">Chandeliers Weekly · MM200w · MM200m · Bollinger 20w · RSI 14</span>
  </div>
  <div class="card"><h2>Chandeliers hebdomadaires · MM200w · MM200m · Bollinger 20w · RSI 14 — {TECH_YEARS} ans</h2>
    {tech_img_html}</div>

  <div class="section-title" id="fond">
    <span style="font-family:var(--mono);color:var(--blue)">③</span><span>Analyse Fondamentale</span><span class="line"></span>
    <span style="color:var(--purple);font-size:11px">Barème : {sector}</span><span class="line"></span>
    <span style="color:{col_f};font-family:var(--mono)">{fund_score:.0f}/100 [{grade_f}]</span>
  </div>
  <div class="card">
    <div style="display:flex;align-items:flex-start;gap:32px;flex-wrap:wrap">
      <div style="min-width:200px">
        <div style="font-size:10px;color:var(--sky);text-transform:uppercase;letter-spacing:.09em;font-weight:700;margin-bottom:8px">Score Fondamental</div>
        <div style="font-family:var(--mono);font-size:62px;font-weight:900;line-height:1;color:{col_f}">{fund_score:.0f}</div>
        <div style="font-size:12px;color:var(--faint);margin-top:3px">/ 100  [{grade_f}]</div>
        {score_bar_html(fund_score)}
        <div style="margin-top:10px;font-size:11px;color:var(--purple);background:rgba(167,139,250,.08);border:1px solid rgba(167,139,250,.20);border-radius:7px;padding:7px 10px">
          Barème sectoriel : <b>{sector}</b><br>
          <span style="color:var(--faint)">Pondération dynamique N/A active</span>
        </div>
      </div>
      <div style="flex:1;min-width:280px">
        {"".join([f'''<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
          <div style="width:130px;font-size:12px;color:var(--muted)">{n}</div>
          <div style="flex:1;height:6px;background:rgba(255,255,255,.05);border-radius:3px;overflow:hidden">
            <div style="height:100%;width:{pillar_scores[n]:.0f}%;background:{c};border-radius:3px"></div></div>
          <div style="font-family:var(--mono);font-size:12px;color:{score_color(pillar_scores[n])};min-width:75px;text-align:right">
            {pillar_scores[n]:.0f}/100 {score_grade(pillar_scores[n])}</div></div>''' for n,w,s,d,c in FUND_PILLARS])}
      </div>
    </div>
  </div>
  <div class="card"><h2>Valorisation — multiples & structure de cash</h2>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px">
      <div>
        <div style="font-size:11px;color:var(--amber);font-weight:700;margin-bottom:8px;text-transform:uppercase">Multiples (barème {sector})</div>
        {pill_row("PE Ratio",   fmt_n(pe_ratio),  C_AMBER,f"≤10 exc · ≤15 bon · ≤20 ok (×{sector_mults[4]:.2f})")}
        {pill_row("P/B Ratio",  fmt_n(pb_ratio),  C_AMBER,f"réf <2 (×{sector_mults[4]:.2f})")}
        {pill_row("P/S Ratio",  fmt_n(ps_ratio),  C_AMBER,f"réf <2 (×{sector_mults[4]:.2f})")}
        {pill_row("PEG Ratio",  fmt_n(peg_ratio), C_AMBER,f"réf <1.5 (×{sector_mults[4]:.2f})")}
        {pill_row("EV/EBITDA",  fmt_n(ev_ebitda), C_AMBER,f"réf <13 (×{sector_mults[4]:.2f})")}
        {pill_row("P/FCF",      fmt_n(pfcf_ratio),C_AMBER,f"réf <25 (×{sector_mults[4]:.2f})")}
      </div>
      <div>
        <div style="font-size:11px;color:var(--green);font-weight:700;margin-bottom:8px;text-transform:uppercase">Structure de cash</div>
        {pill_row("Cash / Market Cap",     pct(cash_mktcap),    C_GREEN if not np.isnan(cash_mktcap) and cash_mktcap>0.1 else COL_FAINT,">20% = value play")}
        {pill_row("Net Cash / Market Cap", pct(netcash_mktcap), C_GREEN if not np.isnan(netcash_mktcap) and netcash_mktcap>0 else C_RED,">0 = net cash positif")}
        {pill_row("Cash / EBITDA",         fmt_n(cash_ebitda),  C_GREEN if not np.isnan(cash_ebitda) and cash_ebitda>0.5 else COL_FAINT,"ratio liquidité")}
        {pill_row("Cash / Equity",         pct(cash_equity),    C_GREEN if not np.isnan(cash_equity) and cash_equity>0.2 else COL_FAINT,">20% = fort")}
        {pill_row("Total Cash",            fmt_big(total_cash), C_GREEN,"")}
        {pill_row("Total Debt",            fmt_big(total_debt), C_RED,"")}
      </div>
    </div>
  </div>
  <div class="grid-2">
    <div class="card"><h2>Radar — 6 piliers</h2><img class="chart" src="data:image/png;base64,{b64_radar}"></div>
    <div class="card"><h2>Scorecard détaillée</h2><img class="chart" src="data:image/png;base64,{b64_bars}"></div>
  </div>
  <div class="card"><h2>Détail des 6 piliers fondamentaux</h2>{pillar_cards_html}</div>

  <div class="section-title" id="explain">
    <span style="font-family:var(--mono);color:var(--orange)">🔍</span><span>Facteurs explicatifs — bénéfices, marges &amp; FCF</span><span class="line"></span>
    <span style="font-size:11px;color:var(--muted)">Δ BFR · dette · retours actionnaires · intérêts · SBC · acquisitions</span>
  </div>
  <div class="explain-box">
    <div style="font-size:11px;color:var(--orange);text-transform:uppercase;letter-spacing:.08em;font-weight:700;margin-bottom:12px">💡 Lecture explicative — {company} ({TICKER})</div>
    <div>{explain_comment_html}</div>
    <div style="margin-top:14px;padding-top:10px;border-top:1px solid rgba(251,146,60,.15);font-size:11px;color:var(--faint);line-height:1.7">
      Rappel : une baisse de FCF n'implique pas forcément une dégradation opérationnelle — elle peut venir d'un Δ BFR défavorable (souvent réversible),
      d'un pic de capex ou d'acquisitions. Inversement, un remboursement massif de dette réduit la trésorerie mais <b>améliore</b> le profil de risque.
    </div>
  </div>
  <div class="card"><h2>Décomposition annuelle — Δ BFR · dette · retours actionnaires · intérêts · SBC/acquisitions</h2>
    <img class="chart" src="data:image/png;base64,{b64_explain}"></div>
  <div class="grid-2">
    <div class="card"><h2>Couverture des retours actionnaires par le FCF</h2>
      <table><thead><tr><th>Exercice</th><th>FCF</th><th>Dividendes</th><th>Rachats</th><th>Total retours</th><th>Couverture</th></tr></thead>
      <tbody>{cover_rows}</tbody></table>
      <div style="font-size:11px;color:var(--faint);margin-top:10px;line-height:1.7">
        Couverture = FCF / (dividendes + rachats).<br>
        <span class="pos">≥1.5× confortable</span> · <span class="warnc">1.0–1.5× juste</span> · <span class="neg">&lt;1× financé par dette/cash</span>
      </div>
    </div>
    <div class="card"><h2>Pont FCF — exercice {bridge_year} · Bénéfice net → FCF</h2>{bridge_img_html}</div>
  </div>

  <div class="card"><h2>Modèle DCF — 3 scénarios</h2>
    <div class="wacc-strip">
      <div><div style="color:var(--faint);font-size:9px;text-transform:uppercase;letter-spacing:.08em">WACC</div><div class="wk">{DCF_WACC*100:.2f}%</div></div>
      <div><div style="color:var(--faint);font-size:9px;text-transform:uppercase;letter-spacing:.08em">Beta</div><div class="wk">{wacc_beta:.2f}</div></div>
      <div><div style="color:var(--faint);font-size:9px;text-transform:uppercase;letter-spacing:.08em">Rf (10Y US)</div><div class="wk">{wacc_rf*100:.2f}%</div></div>
      <div><div style="color:var(--faint);font-size:9px;text-transform:uppercase;letter-spacing:.08em">Ke (CAPM)</div><div class="wk">{wacc_ke*100:.2f}%</div></div>
      <div><div style="color:var(--faint);font-size:9px;text-transform:uppercase;letter-spacing:.08em">Kd (après IS)</div><div class="wk">{wacc_kd*100:.2f}%</div></div>
      <div><div style="color:var(--faint);font-size:9px;text-transform:uppercase;letter-spacing:.08em">TV Growth</div><div class="wk">{DCF_TV_GROWTH*100:.1f}%</div></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1.8fr;gap:20px;align-items:start">
      <div>
        <table><thead><tr><th>Scénario</th><th>Croiss.</th><th>VI/action</th><th>Marge</th><th>TV%</th></tr></thead>
        <tbody>{dcf_rows}</tbody></table>
        <div style="margin-top:10px;font-size:11px;color:#475569;line-height:1.9">
          Cours actuel : <b style="color:#F1F5F9">${price:.2f}</b><br>
          FCF TTM : <b style="color:var(--green)">{fmt_big(fcf_ttm)}</b><br>
          <span style="color:var(--amber)">⚠ TV% > 75% = fiabilité réduite</span><br>
          <i>⚠ DCF sensible aux hypothèses</i>
        </div>
      </div>
      <div><img class="chart" src="data:image/png;base64,{b64_dcf}"></div>
    </div>
  </div>

  <div class="section-title" id="hist">
    <span style="font-family:var(--mono);color:var(--blue)">④</span><span>Historique financier</span><span class="line"></span>
    <span style="color:#2DD4BF;font-family:var(--mono)">{n_fin_years} exercices ({fin_years[0] if fin_years else '—'} → {fin_years[-1] if fin_years else '—'})</span>
  </div>
  <div style="font-size:12px;color:var(--muted);margin-bottom:12px">Sources : yfinance ∪ yahooquery (fusion par année fiscale — historique le plus long disponible)</div>
  <div class="card"><h2>Évolution des marges</h2><img class="chart" src="data:image/png;base64,{b64_margins}"></div>
  <div class="card"><h2>Séries annuelles — Revenus · FCF · Bénéfice · CFO · Capex · Marge opé.</h2><img class="chart" src="data:image/png;base64,{b64_hist}"></div>

  <div class="section-title" id="synthese">
    <span style="font-family:var(--mono);color:var(--blue)">⑤</span><span>Score Global &amp; Synthèse</span><span class="line"></span>
    <span style="color:{col_g};font-family:var(--mono)">{global_score:.0f}/100 [{grade_g}]</span>
  </div>
  <div class="score-trio">
    <div class="score-box" style="border-color:rgba(96,165,250,.22)">
      <div style="font-size:10px;color:var(--sky);text-transform:uppercase;letter-spacing:.09em;font-weight:700;margin-bottom:10px">① Quantitatif</div>
      <div class="score-num-lg" style="color:{col_q}">{quant_score_100:.0f}</div>
      <div style="font-size:13px;color:var(--faint);margin-top:4px">/ 100 · [{grade_q}]</div>
      <div class="bar" style="margin-top:12px"><div class="bar-fill" style="width:{quant_score_100:.0f}%;background:{col_q}"></div></div>
      <div style="font-size:11px;color:var(--faint);margin-top:8px">poids 50%</div>
    </div>
    <div class="score-box" style="border-color:rgba(167,139,250,.22)">
      <div style="font-size:10px;color:var(--purple);text-transform:uppercase;letter-spacing:.09em;font-weight:700;margin-bottom:10px">③ Fondamental</div>
      <div class="score-num-lg" style="color:{col_f}">{fund_score:.0f}</div>
      <div style="font-size:13px;color:var(--faint);margin-top:4px">/ 100 · [{grade_f}]</div>
      <div class="bar" style="margin-top:12px"><div class="bar-fill" style="width:{fund_score:.0f}%;background:{col_f}"></div></div>
      <div style="font-size:11px;color:var(--faint);margin-top:8px">poids 50% · barème {sector}</div>
    </div>
    <div class="score-box" style="border:2px solid {col_g}">
      <div style="font-size:10px;color:var(--sky);text-transform:uppercase;letter-spacing:.09em;font-weight:700;margin-bottom:10px">⑤ Score Global</div>
      <div class="score-num-lg" style="color:{col_g}">{global_score:.0f}</div>
      <div style="font-size:13px;color:var(--faint);margin-top:4px">/ 100 · [{grade_g}]</div>
      <div class="bar" style="margin-top:12px"><div class="bar-fill" style="width:{global_score:.0f}%;background:{col_g}"></div></div>
      <div style="font-size:11px;color:var(--faint);margin-top:8px">50% Quant + 50% Fond.</div>
    </div>
  </div>
  <div class="comment-box">
    <div style="font-size:11px;color:var(--blue);text-transform:uppercase;letter-spacing:.08em;font-weight:700;margin-bottom:12px">📋 Synthèse — {company} ({TICKER}) · {sector}</div>
    <div>{comment_html}</div>
    <div style="margin-top:16px;padding-top:12px;border-top:1px solid rgba(96,165,250,.15);font-size:11px;color:var(--faint);line-height:1.7">⚠ Ce commentaire est généré automatiquement. Il ne constitue pas un conseil en investissement.</div>
  </div>
  <footer>
    <b style="color:var(--blue)">UnifiedDash v5</b> · {TICKER} · {company} · {sector}<br>
    Quant {quant_score_100:.0f}/100 [{grade_q}] · Fond. {fund_score:.0f}/100 [{grade_f}] · Global {global_score:.0f}/100 [{grade_g}]<br>
    Signal : {sig_txt} · {pos_sigma:+.2f}σ · ±σ à t : +2σ {pfmt(bp2_now)} / Tend {pfmt(tr_now)} / −2σ {pfmt(bm2_now)}<br>
    WACC {DCF_WACC*100:.2f}% (β={wacc_beta:.2f}, Rf={wacc_rf*100:.2f}%, Ke={wacc_ke*100:.2f}%, Kd={wacc_kd*100:.2f}%)<br>
    Historique : {n_fin_years} exercices · Généré le {now_str} · ⚠ pas un conseil en investissement.
  </footer>
</div>
</body>
</html>"""

    print(f"  ✓ HTML généré ({len(html)//1024} Ko)")
    print("═"*65)
    print(f"  ✅ TERMINÉ — Score Global : {global_score:.1f}/100 [{grade_g}]")
    print("═"*65)

    summary = dict(
        ticker=TICKER, company=company, sector=sector,
        global_score=global_score, fund_score=fund_score, quant_score=quant_score_100,
        grade_g=grade_g, grade_f=grade_f, grade_q=grade_q,
        sig_txt=sig_txt, pos_sigma=pos_sigma, price=price,
    )
    return html, summary


# ═══════════════════════════════════════════════════════════════
#  INTERFACE STREAMLIT
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="UnifiedDash v5", page_icon="📈", layout="wide")

st.title("📈 UnifiedDash v5 — Analyse Actions")
st.caption("Analyse fondamentale + quantitative + technique, générée à la demande à partir de Yahoo Finance. "
           "Ceci n'est pas un conseil en investissement.")

with st.sidebar:
    st.header("⚙️ Paramètres")
    ticker_input = st.text_input("Ticker", value="AAPL",
                                  help="Ex : AAPL, MSFT, ABEV — pour les tickers européens, ajoutez le suffixe "
                                       "de marché (ex : AIR.PA pour Airbus).").strip().upper()

    with st.expander("Hypothèses DCF"):
        dcf_bear = st.slider("Croissance FCF — Bear (%)", 0.0, 20.0, 3.0, 0.5) / 100
        dcf_base = st.slider("Croissance FCF — Base (%)", 0.0, 30.0, 8.0, 0.5) / 100
        dcf_bull = st.slider("Croissance FCF — Bull (%)", 0.0, 40.0, 14.0, 0.5) / 100
        dcf_tv   = st.slider("Croissance terminale (%)", 0.0, 5.0, 2.5, 0.25) / 100
        dcf_yrs  = st.slider("Horizon DCF (années)", 5, 15, 10, 1)

    with st.expander("Monte Carlo & Technique"):
        n_sim    = st.select_slider("Simulations Monte Carlo", options=[200, 500, 1000, 2000], value=1000)
        hist_yrs = st.slider("Historique pour les bandes σ (années)", 10, 100, 100, 5)
        tech_yrs = st.slider("Fenêtre du graphique technique (années)", 2, 15, 5, 1)

    run_btn = st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("⚠️ Les données proviennent de Yahoo Finance (yfinance / yahooquery) et peuvent être "
               "retardées, incomplètes ou temporairement indisponibles selon le ticker.")

if "dashboard_html" not in st.session_state:
    st.session_state.dashboard_html = None
    st.session_state.dashboard_summary = None

if run_btn:
    if not ticker_input:
        st.warning("Merci de renseigner un ticker.")
    else:
        with st.spinner(f"Analyse de {ticker_input} en cours… (30 s à 2 min selon le ticker)"):
            try:
                html_out, summary = generate_dashboard(
                    ticker_input,
                    dcf_tv_growth=dcf_tv, dcf_bear_gr=dcf_bear, dcf_base_gr=dcf_base,
                    dcf_bull_gr=dcf_bull, dcf_years=dcf_yrs,
                    n_sim=n_sim, hist_years=hist_yrs, tech_years=tech_yrs,
                )
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse de **{ticker_input}** : {e}")
                st.session_state.dashboard_html = None
                st.session_state.dashboard_summary = None
            else:
                st.session_state.dashboard_html = html_out
                st.session_state.dashboard_summary = summary

if st.session_state.dashboard_html:
    summary = st.session_state.dashboard_summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score Global", f"{summary['global_score']:.0f}/100", summary['grade_g'])
    c2.metric("Score Fondamental", f"{summary['fund_score']:.0f}/100", summary['grade_f'])
    c3.metric("Score Quantitatif", f"{summary['quant_score']:.0f}/100", summary['grade_q'])
    c4.metric("Signal", summary['sig_txt'], f"{summary['pos_sigma']:+.2f}σ")

    st.download_button(
        "⬇️ Télécharger le rapport HTML complet",
        data=st.session_state.dashboard_html,
        file_name=f"{summary['ticker']}_unified_dashboard.html",
        mime="text/html",
        use_container_width=True,
    )

    components.html(st.session_state.dashboard_html, height=7200, scrolling=True)
else:
    st.info("👈 Renseignez un ticker dans la barre latérale puis cliquez sur **Lancer l'analyse**.")
