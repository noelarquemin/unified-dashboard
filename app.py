"""
app.py — Interface Streamlit pour UnifiedDash v4
Déploiement : streamlit.io (gratuit)
"""
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="UnifiedDash v4",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS minimal ───────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .stTextInput > div > div > input { font-size: 18px; font-weight: 700; }
    .metric-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }
    .metric-box {
        background: #151E32; border: 1px solid rgba(96,165,250,.25);
        border-radius: 10px; padding: 12px 18px; min-width: 150px;
        font-family: monospace;
    }
    .metric-box .lbl { font-size: 10px; color: #64748B; text-transform: uppercase;
                       letter-spacing: .1em; font-weight: 700; }
    .metric-box .val { font-size: 20px; font-weight: 800; color: #F1F5F9; margin-top: 4px; }
    .metric-box .sub { font-size: 11px; color: #94A3B8; margin-top: 3px; }
    .score-badge {
        display: inline-block; padding: 4px 14px; border-radius: 999px;
        font-family: monospace; font-size: 14px; font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:20px">
  <div style="font-family:monospace;font-size:26px;font-weight:900;color:#60A5FA">
    ▲ Unified<span style="color:#F1F5F9">Dash</span>
    <span style="color:#A78BFA;font-size:16px">v4</span>
  </div>
  <div style="font-size:12px;color:#475569;padding-top:6px">
    Analyse quantitative · Fondamentale · Technique
  </div>
</div>
""", unsafe_allow_html=True)

# ── Input ticker ──────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 5])
with col1:
    ticker_input = st.text_input(
        "Ticker",
        value="",
        placeholder="ex: AAPL, EXE, ICE…",
        label_visibility="collapsed",
    ).strip().upper()
with col2:
    run_btn = st.button("🔍 Analyser", type="primary", use_container_width=True)

if not ticker_input:
    st.markdown("""
    <div style="margin-top:60px;text-align:center;color:#475569">
      <div style="font-size:48px;margin-bottom:16px">▲</div>
      <div style="font-size:16px;font-weight:600;color:#64748B">Entrez un ticker pour lancer l'analyse</div>
      <div style="font-size:13px;margin-top:8px">Exemples : AAPL · MSFT · EXE · ICE · FERG · GRBK</div>
      <div style="font-size:11px;margin-top:24px;color:#334155">
        Données : yfinance · yahooquery — Durée estimée : 30 à 90 secondes selon le ticker
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if run_btn or ticker_input:
    if not run_btn and "last_ticker" in st.session_state and st.session_state.last_ticker == ticker_input:
        # Déjà calculé — afficher le résultat en cache
        result = st.session_state.last_result
    else:
        # Lancer l'analyse
        st.session_state.last_ticker = ticker_input
        progress_bar = st.progress(0, text="Initialisation...")
        status_text  = st.empty()

        def update_progress(p, msg=""):
            progress_bar.progress(min(p, 1.0), text=msg or "Analyse en cours…")
            if msg:
                status_text.markdown(f"<div style='font-size:12px;color:#64748B'>{msg}</div>",
                                     unsafe_allow_html=True)

        try:
            with st.spinner(""):
                from dashboard_engine import run_analysis
                result = run_analysis(ticker_input, progress_cb=update_progress)
            st.session_state.last_result = result
            progress_bar.empty()
            status_text.empty()
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Erreur lors de l'analyse de **{ticker_input}** : `{e}`")
            st.info("Vérifiez que le ticker est valide sur Yahoo Finance (ex: AAPL, MSFT, EXE).")
            st.stop()

    # ── Résumé scores ────────────────────────────────────────
    SCORE_COLORS = {
        "green": "#34D399", "blue": "#60A5FA",
        "amber": "#FBBF24", "red":  "#F43F5E",
    }
    def score_col(s):
        if s >= 75: return SCORE_COLORS["green"]
        if s >= 55: return SCORE_COLORS["blue"]
        if s >= 35: return SCORE_COLORS["amber"]
        return SCORE_COLORS["red"]

    g = result["global_score"]; f = result["fund_score"]; q = result["quant_score_100"]
    sig = result["sig_txt"]; sigma = result["pos_sigma"]

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;
                background:#151E32;border:1px solid rgba(96,165,250,.2);
                border-radius:12px;padding:16px 20px;margin-bottom:16px">
      <div style="font-size:15px;font-weight:800;color:#F1F5F9">
        {result['company']}
        <span style="font-family:monospace;font-size:12px;color:#60A5FA;
               background:rgba(96,165,250,.1);border:1px solid rgba(96,165,250,.25);
               border-radius:6px;padding:2px 8px;margin-left:8px">{result['ticker']}</span>
        <span style="font-size:11px;color:#A78BFA;margin-left:8px">{result['sector']}</span>
      </div>
      <div style="margin-left:auto;display:flex;gap:10px;flex-wrap:wrap">
        <div class="score-badge" style="color:{score_col(q)};border:1px solid {score_col(q)}33;background:{score_col(q)}11">
          Quant {q:.0f} [{result['grade_q']}]
        </div>
        <div class="score-badge" style="color:{score_col(f)};border:1px solid {score_col(f)}33;background:{score_col(f)}11">
          Fond. {f:.0f} [{result['grade_f']}]
        </div>
        <div class="score-badge" style="color:{score_col(g)};border:2px solid {score_col(g)};background:{score_col(g)}15;font-size:16px">
          Score {g:.0f} [{result['grade_g']}]
        </div>
        <div class="score-badge" style="color:#60A5FA;border:1px solid rgba(96,165,250,.3);background:rgba(96,165,250,.08)">
          {sig} · {sigma:+.2f}σ
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dashboard HTML complet ────────────────────────────────
    st.markdown("### 📊 Dashboard complet")
    components.html(result["html"], height=12000, scrolling=True)

    # ── Bouton téléchargement HTML ────────────────────────────
    st.download_button(
        label="⬇️ Télécharger le dashboard HTML",
        data=result["html"].encode("utf-8"),
        file_name=f"unified_dashboard_{ticker_input}.html",
        mime="text/html",
        use_container_width=False,
    )
