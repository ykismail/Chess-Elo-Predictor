import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chess Analytics · Executive Dashboard",
    page_icon="♟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0d0f14; color: #e8e4dc; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1400px !important; }

[data-testid="stSidebar"] {
    background: #131620 !important;
    border-right: 1px solid #1e2230 !important;
}

[data-testid="stMetric"] {
    background: #131620;
    border: 1px solid #1e2230;
    border-radius: 10px;
    padding: 1.1rem 1.25rem 1rem !important;
    overflow: hidden;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: #2e3450; }
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important;
    font-weight: 600 !important;
    color: #e8e4dc !important;
    line-height: 1.15 !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
}

hr { border-color: #1e2230 !important; margin: 1.5rem 0 !important; }

h2 {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: #c9c3b8 !important;
}

.insight-box {
    background: #131620;
    border: 1px solid #1e2230;
    border-left: 3px solid #c9a227;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    font-size: 13px;
    color: #9ca3af;
    line-height: 1.6;
}
.insight-box strong { color: #e8e4dc; font-weight: 500; }

.tag {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    padding: 2px 9px;
    border-radius: 20px;
    letter-spacing: 0.04em;
}
.tag-green  { background: #1a3328; color: #5a9e6f; border: 1px solid #244a37; }
.tag-red    { background: #2d1a1a; color: #c0392b; border: 1px solid #4a2222; }
.tag-amber  { background: #2d2410; color: #c9a227; border: 1px solid #4a3a18; }
.tag-slate  { background: #1a1e2a; color: #7c8a9e; border: 1px solid #252c3a; }

.dash-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #1e2230;
}
.dash-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #c9a227;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.dash-main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #e8e4dc;
    line-height: 1.1;
    margin: 0;
}
.dash-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #4b5563;
    margin-top: 6px;
}
.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #5a9e6f;
    background: #0f1f16;
    border: 1px solid #1e3a28;
    border-radius: 20px;
    padding: 5px 14px;
}
.live-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #5a9e6f;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

.panel-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 500;
    color: #c9c3b8;
    margin-bottom: 2px;
}
.panel-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    color: #4b5563;
    margin-bottom: 0.75rem;
    line-height: 1.4;
}

.sb-wrap { display:flex; flex-direction:column; gap:7px; margin-top:4px; }
.sb-row  { display:flex; align-items:center; gap:10px; }
.sb-lbl  {
    font-family:'DM Mono',monospace; font-size:10px; color:#6b7280;
    width:100px; text-align:right; flex-shrink:0;
}
.sb-track {
    flex:1; height:22px; border-radius:3px; overflow:hidden;
    display:flex; background:#0d0f14; border:1px solid #1e2230;
}
.sb-seg {
    height:100%; display:flex; align-items:center; justify-content:center;
    font-family:'DM Mono',monospace; font-size:10px;
    color:rgba(255,255,255,0.85); transition:opacity 0.15s;
}
.sb-seg:hover { opacity:0.8; }

.acl-wrap { display:flex; flex-direction:column; gap:10px; padding-top:6px; }
.acl-row  { display:flex; align-items:center; gap:10px; }
.acl-lbl  {
    font-family:'DM Mono',monospace; font-size:10px; color:#6b7280;
    width:78px; text-align:right; flex-shrink:0;
}
.acl-track {
    flex:1; height:32px; background:#0d0f14;
    border:1px solid #1e2230; border-radius:3px; position:relative;
}
.acl-zero  { position:absolute; left:50%; top:0; bottom:0; width:1px; background:#2e3450; }
.acl-whisker { position:absolute; top:50%; height:1px; background:#4b5563; }
.acl-box   { position:absolute; top:5px; bottom:5px; border-radius:2px; border:1.5px solid; }
.acl-median{ position:absolute; top:3px; bottom:3px; width:2.5px; border-radius:1px; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Load & normalise ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(
        r"C:\Users\abdelrahman\Desktop\college\data_science"
        r"\Chess-Elo-Predictor\data\merged_games.csv"
    )

    # ── winner_multiclass: 0=Black, 1=Draw, 2=White ──
    df["outcome"] = df["winner_multiclass"].map({0: "black", 1: "draw", 2: "white"})

    # ── derived columns (guard against pre-computed ones) ──
    if "elo_gap" not in df.columns:
        df["elo_gap"] = df["white_elo"] - df["black_elo"]
    if "avg_elo" not in df.columns:
        df["avg_elo"] = (df["white_elo"] + df["black_elo"]) / 2
    if "acl_gap" not in df.columns:
        df["acl_gap"] = df["white_acl"] - df["black_acl"]

    # short opening label
    df["eco_family"] = df["eco_family"].fillna("?").astype(str)

    return df


df = load_data()
df_sf = df[df["has_stockfish"] == True]

# ── Colour palette ────────────────────────────────────────────────────────────
WHT_COL = "#5a9e6f"
BLK_COL = "#c0392b"
DRW_COL = "#7c8a9e"
GOLD = "#c9a227"
BORDER = "#1e2230"
MUTED = "#6b7280"

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color=MUTED, size=11),
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(
        gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(family="DM Mono", size=10)
    ),
    yaxis=dict(
        gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(family="DM Mono", size=10)
    ),
)

# ── Baseline stats ────────────────────────────────────────────────────────────
total_games = len(df)
base_white = (df["outcome"] == "white").mean()
base_black = (df["outcome"] == "black").mean()
base_draw = (df["outcome"] == "draw").mean()
base_elo = df["avg_elo"].mean()
elo_min = int(df[["white_elo", "black_elo"]].min().min())
elo_max = int(df[["white_elo", "black_elo"]].max().max())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ♟ Filters")
    st.markdown("---")
    elo_range = st.slider("Elo range", elo_min, elo_max, (elo_min, elo_max), step=50)
    outcomes = st.multiselect(
        "Outcomes",
        ["white", "black", "draw"],
        default=["white", "black", "draw"],
    )
    use_sf = st.checkbox("Stockfish games only", value=False)

    sources = sorted(df["source"].dropna().unique().tolist())
    sel_src = st.multiselect("Data source", sources, default=sources)

    terms = sorted(df["termination"].dropna().unique().tolist())
    sel_term = st.multiselect("Termination", terms, default=terms)

    st.markdown("---")
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:10px;"
        "color:#3a4050;text-align:center'>Chess Elo Predictor · v2.1</div>",
        unsafe_allow_html=True,
    )

# ── Apply filters ─────────────────────────────────────────────────────────────
mask = (
    (df["white_elo"] >= elo_range[0])
    & (df["white_elo"] <= elo_range[1])
    & (df["black_elo"] >= elo_range[0])
    & (df["black_elo"] <= elo_range[1])
    & (df["outcome"].isin(outcomes))
)
if use_sf:
    mask &= df["has_stockfish"] == True
if sel_src:
    mask &= df["source"].isin(sel_src)
if sel_term:
    mask &= df["termination"].isin(sel_term)

dff = df[mask].copy()
dff_sf = dff[dff["has_stockfish"] == True]

n_games = len(dff)
avg_elo_val = dff["avg_elo"].mean() if n_games else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""
<div class="dash-header">
  <div>
    <div class="dash-eyebrow">Executive Dashboard</div>
    <div class="dash-main-title">Chess Match Analytics</div>
    <div class="dash-subtitle">
      {n_games:,} games &nbsp;·&nbsp; Lichess + Kaggle PGN
      &nbsp;·&nbsp; Elo {elo_range[0]}–{elo_range[1]}
    </div>
  </div>
  <div class="live-pill"><span class="live-dot"></span>Live filters active</div>
</div>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ═══════════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric(
        "Total Games",
        f"{n_games:,}",
        delta=f"{n_games / max(total_games,1) * 100:.0f}% of dataset",
    )
with k2:
    wr = (dff["outcome"] == "white").mean() if n_games else 0
    st.metric(
        "White Win Rate",
        f"{wr*100:.1f}%",
        delta=f"{(wr - base_white)*100:+.1f}pp vs baseline",
    )
with k3:
    br = (dff["outcome"] == "black").mean() if n_games else 0
    st.metric(
        "Black Win Rate",
        f"{br*100:.1f}%",
        delta=f"{(br - base_black)*100:+.1f}pp vs baseline",
    )
with k4:
    dr = (dff["outcome"] == "draw").mean() if n_games else 0
    st.metric(
        "Draw Rate",
        f"{dr*100:.1f}%",
        delta=f"{(dr - base_draw)*100:+.1f}pp vs baseline",
    )
with k5:
    st.metric(
        "Avg Elo",
        f"{avg_elo_val:.0f}",
        delta=f"{avg_elo_val - base_elo:+.0f} vs all games",
    )

st.markdown("<hr>", unsafe_allow_html=True)

# Guard: nothing to show
if n_games == 0:
    st.warning("No games match the current filters. Please widen your selection.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Elo gap stacked bars  |  ACL box-and-whisker
# ═══════════════════════════════════════════════════════════════════════════════
col_l, col_r = st.columns(2, gap="medium")

# ── Elo gap ───────────────────────────────────────────────────────────────────
with col_l:
    st.markdown(
        """
    <div class="panel-title">Elo Gap → Match Outcome</div>
    <div class="panel-desc">Win probability by rating difference bin &nbsp;
    <span class="tag tag-amber">pre-game predictor</span></div>
    """,
        unsafe_allow_html=True,
    )

    dff2 = dff.copy()
    dff2["elo_bin"] = pd.cut(
        dff2["elo_gap"],
        bins=[-9999, -200, -50, 50, 200, 9999],
        labels=[
            "Black ≫ White",
            "Black › White",
            "Even",
            "White › Black",
            "White ≫ Black",
        ],
    )
    grp = (
        dff2.groupby(["elo_bin", "outcome"], observed=True).size().reset_index(name="n")
    )
    totals = grp.groupby("elo_bin", observed=True)["n"].transform("sum")
    grp["pct"] = (grp["n"] / totals * 100).round(1)

    bin_labels = [
        "Black ≫ White",
        "Black › White",
        "Even",
        "White › Black",
        "White ≫ Black",
    ]
    html_rows = ""
    for lbl in bin_labels:
        sub = grp[grp["elo_bin"] == lbl]

        def pct(o, _s=sub):
            r = _s[_s["outcome"] == o]["pct"]
            return float(r.values[0]) if len(r) else 0.0

        wp, dp, bp = pct("white"), pct("draw"), pct("black")
        html_rows += (
            f'<div class="sb-row">'
            f'<span class="sb-lbl">{lbl}</span>'
            f'<div class="sb-track">'
            f'<div class="sb-seg" style="width:{wp}%;background:{WHT_COL}cc" '
            f'title="White {wp:.0f}%">{f"{wp:.0f}%" if wp > 12 else ""}</div>'
            f'<div class="sb-seg" style="width:{dp}%;background:{DRW_COL}cc" '
            f'title="Draw {dp:.0f}%">{f"{dp:.0f}%" if dp > 10 else ""}</div>'
            f'<div class="sb-seg" style="width:{bp}%;background:{BLK_COL}cc" '
            f'title="Black {bp:.0f}%">{f"{bp:.0f}%" if bp > 12 else ""}</div>'
            f"</div></div>"
        )

    st.markdown(
        '<div style="margin-bottom:10px">'
        '<span class="tag tag-green">■ White</span>&nbsp;&nbsp;'
        '<span class="tag tag-slate">■ Draw</span>&nbsp;&nbsp;'
        '<span class="tag tag-red">■ Black</span></div>'
        f'<div class="sb-wrap">{html_rows}</div>',
        unsafe_allow_html=True,
    )

# ── ACL box-and-whisker ───────────────────────────────────────────────────────
with col_r:
    st.markdown(
        """
    <div class="panel-title">Engine Accuracy Gap (ACL)</div>
    <div class="panel-desc">white_acl − black_acl by outcome &nbsp;
    <span class="tag tag-amber">Stockfish only</span></div>
    """,
        unsafe_allow_html=True,
    )

    if len(dff_sf) > 30:
        ACL_RANGE = 200  # maps -100…+100 → 0…100%
        label_map = {"white": "White wins", "draw": "Draw", "black": "Black wins"}
        color_map = {"white": WHT_COL, "draw": DRW_COL, "black": BLK_COL}

        html_acl = ""
        for oc, label in label_map.items():
            vals = dff_sf[dff_sf["outcome"] == oc]["acl_gap"].dropna()
            if len(vals) < 5:
                continue
            s = dict(
                q1=vals.quantile(0.25),
                q3=vals.quantile(0.75),
                med=vals.median(),
                lo=vals.quantile(0.05),
                hi=vals.quantile(0.95),
            )
            color = color_map[oc]
            to_x = lambda v: f"{((v + 100) / ACL_RANGE * 100):.2f}%"
            html_acl += (
                f'<div class="acl-row">'
                f'<span class="acl-lbl">{label}</span>'
                f'<div class="acl-track">'
                f'<div class="acl-zero"></div>'
                f'<div class="acl-whisker" style="left:{to_x(s["lo"])};'
                f'width:{abs(s["hi"]-s["lo"])/ACL_RANGE*100:.2f}%"></div>'
                f'<div class="acl-box" style="left:{to_x(min(s["q1"],s["q3"]))};'
                f'width:{abs(s["q3"]-s["q1"])/ACL_RANGE*100:.2f}%;'
                f'background:{color}22;border-color:{color}"></div>'
                f'<div class="acl-median" style="left:{to_x(s["med"])};'
                f'background:{color}"></div>'
                f"</div></div>"
            )
        st.markdown(
            '<div style="margin-bottom:10px;font-family:DM Mono,monospace;'
            'font-size:10px;color:#4b5563">'
            "Box = IQR &nbsp;·&nbsp; Line = median &nbsp;·&nbsp; Whiskers = 5–95th pct"
            "</div>"
            f'<div class="acl-wrap">{html_acl}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "Not enough Stockfish games for this filter. "
            "Uncheck 'Stockfish games only' or widen the Elo range."
        )

st.markdown("<hr>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Game length  |  Rating density
# ═══════════════════════════════════════════════════════════════════════════════
col_a, col_b = st.columns(2, gap="medium")

with col_a:
    st.markdown(
        """
    <div class="panel-title">Game Length Distribution</div>
    <div class="panel-desc">num_moves per game by outcome &nbsp;
    <span class="tag tag-slate">histogram</span></div>
    """,
        unsafe_allow_html=True,
    )

    dff3 = dff.copy()
    dff3["move_bin"] = pd.cut(
        dff3["num_moves"],
        bins=[0, 20, 40, 60, 80, 100, 9999],
        labels=["1–20", "21–40", "41–60", "61–80", "81–100", "100+"],
    )
    fig_mv = go.Figure()
    for oc, color, name in [
        ("white", WHT_COL, "White wins"),
        ("draw", DRW_COL, "Draw"),
        ("black", BLK_COL, "Black wins"),
    ]:
        sub = dff3[dff3["outcome"] == oc]
        cnts = (
            sub["move_bin"]
            .value_counts()
            .reindex(
                ["1–20", "21–40", "41–60", "61–80", "81–100", "100+"], fill_value=0
            )
        )
        fig_mv.add_trace(
            go.Bar(
                x=cnts.index.tolist(),
                y=cnts.values,
                name=name,
                marker_color=color,
                marker_line_width=0,
                hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:,}}<extra></extra>",
            )
        )
    fig_mv.update_layout(
        **PLOTLY_BASE,
        barmode="group",
        bargap=0.15,
        bargroupgap=0.05,
        legend=dict(
            orientation="h",
            y=-0.28,
            x=0,
            font=dict(family="DM Mono", size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=280,
    )
    st.plotly_chart(fig_mv, use_container_width=True, config={"displayModeBar": False})

with col_b:
    st.markdown(
        """
    <div class="panel-title">Rating Distribution</div>
    <div class="panel-desc">white_elo vs black_elo across filtered games &nbsp;
    <span class="tag tag-slate">density</span></div>
    """,
        unsafe_allow_html=True,
    )

    fig_elo = go.Figure()
    for col_name, color, name in [
        ("white_elo", WHT_COL, "White Elo"),
        ("black_elo", BLK_COL, "Black Elo"),
    ]:
        vals = dff[col_name].dropna().values
        hist, edges = np.histogram(vals, bins=40)
        centers = (edges[:-1] + edges[1:]) / 2
        fig_elo.add_trace(
            go.Scatter(
                x=centers,
                y=hist,
                name=name,
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=f"{color}",
                hovertemplate=f"<b>{name}</b><br>Elo %{{x:.0f}}: %{{y:,}}<extra></extra>",
            )
        )
    fig_elo.update_layout(
        **PLOTLY_BASE,
        legend=dict(
            orientation="h",
            y=-0.28,
            x=0,
            font=dict(family="DM Mono", size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=280,
    )
    st.plotly_chart(fig_elo, use_container_width=True, config={"displayModeBar": False})

st.markdown("<hr>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3 — ECO family  |  Blunders & mistakes
# ═══════════════════════════════════════════════════════════════════════════════
col_c, col_d = st.columns(2, gap="medium")

with col_c:
    st.markdown(
        """
    <div class="panel-title">ECO Family Win Rates</div>
    <div class="panel-desc">Outcome share by opening family (eco_family A–E) &nbsp;
    <span class="tag tag-amber">strategic</span></div>
    """,
        unsafe_allow_html=True,
    )

    eco_rows = []
    for fam in sorted(dff["eco_family"].dropna().unique()):
        sub = dff[dff["eco_family"] == fam]
        eco_rows.append(
            {
                "family": fam,
                "games": len(sub),
                "white": (sub["outcome"] == "white").mean() * 100,
                "draw": (sub["outcome"] == "draw").mean() * 100,
                "black": (sub["outcome"] == "black").mean() * 100,
            }
        )
    eco_df = pd.DataFrame(eco_rows).sort_values("games", ascending=True)

    fig_eco = go.Figure()
    for oc, color, name in [
        ("white", WHT_COL, "White"),
        ("draw", DRW_COL, "Draw"),
        ("black", BLK_COL, "Black"),
    ]:
        fig_eco.add_trace(
            go.Bar(
                y=eco_df["family"],
                x=eco_df[oc],
                name=name,
                orientation="h",
                marker_color=color,
                marker_line_width=0,
                hovertemplate=f"<b>{name}</b><br>Family %{{y}}: %{{x:.1f}}%<extra></extra>",
            )
        )
    fig_eco.update_layout(**PLOTLY_BASE)
    fig_eco.update_layout(
        barmode="stack",
        bargap=0.3,
        xaxis=dict(
            gridcolor=BORDER,
            ticksuffix="%",
            zerolinecolor=BORDER,
            tickfont=dict(family="DM Mono", size=10),
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            zerolinecolor="rgba(0,0,0,0)",
            tickfont=dict(family="DM Mono", size=10),
        ),
        legend=dict(
            orientation="h",
            y=-0.2,
            x=0,
            font=dict(family="DM Mono", size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=300,
    )
    st.plotly_chart(fig_eco, use_container_width=True, config={"displayModeBar": False})

with col_d:
    st.markdown(
        """
    <div class="panel-title">Blunders & Mistakes by Outcome</div>
    <div class="panel-desc">Avg errors per game — white/black blunders & mistakes &nbsp;
    <span class="tag tag-slate">accuracy</span></div>
    """,
        unsafe_allow_html=True,
    )

    err_cols = {"white_blunders", "black_blunders", "white_mistakes", "black_mistakes"}
    if err_cols.issubset(dff.columns):
        err_rows = []
        for oc, label in [
            ("white", "White wins"),
            ("draw", "Draw"),
            ("black", "Black wins"),
        ]:
            sub = dff[dff["outcome"] == oc]
            err_rows.append(
                {
                    "outcome": label,
                    "white_blunders": sub["white_blunders"].mean(),
                    "black_blunders": sub["black_blunders"].mean(),
                    "white_mistakes": sub["white_mistakes"].mean(),
                    "black_mistakes": sub["black_mistakes"].mean(),
                }
            )
        err_df = pd.DataFrame(err_rows)

        fig_err = go.Figure()
        for col_name, color, name in [
            ("white_blunders", WHT_COL, "White blunders"),
            ("black_blunders", BLK_COL, "Black blunders"),
            ("white_mistakes", WHT_COL, "White mistakes"),
            ("black_mistakes", BLK_COL, "Black mistakes"),
        ]:
            fig_err.add_trace(
                go.Bar(
                    x=err_df["outcome"],
                    y=err_df[col_name].round(2),
                    name=name,
                    marker_color=color,
                    marker_line_width=0,
                    hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.2f}} avg<extra></extra>",
                )
            )
        fig_err.update_layout(
            **PLOTLY_BASE,
            barmode="group",
            bargap=0.2,
            bargroupgap=0.05,
            legend=dict(
                orientation="h",
                y=-0.28,
                x=0,
                font=dict(family="DM Mono", size=10),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=300,
        )
        st.plotly_chart(
            fig_err, use_container_width=True, config={"displayModeBar": False}
        )
    else:
        st.info("Blunder / mistake columns not found in dataset.")

st.markdown("<hr>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ROW 4 — Scatter (Elo gap vs ACL gap)  |  Termination pie
# ═══════════════════════════════════════════════════════════════════════════════
col_e, col_f = st.columns([1.4, 1], gap="medium")

with col_e:
    st.markdown(
        """
    <div class="panel-title">Elo Gap vs ACL Gap</div>
    <div class="panel-desc">Pre-game skill vs post-game accuracy, coloured by outcome &nbsp;
    <span class="tag tag-amber">correlation</span></div>
    """,
        unsafe_allow_html=True,
    )

    if len(dff_sf) > 50:
        sample = dff_sf.sample(min(3000, len(dff_sf)), random_state=42)
        fig_sc = go.Figure()
        for oc, color in [("white", WHT_COL), ("draw", DRW_COL), ("black", BLK_COL)]:
            sub = sample[sample["outcome"] == oc]
            fig_sc.add_trace(
                go.Scatter(
                    x=sub["elo_gap"],
                    y=sub["acl_gap"],
                    mode="markers",
                    name=oc.capitalize(),
                    marker=dict(color=color, size=3, opacity=0.45),
                    hovertemplate=(
                        f"<b>{oc.capitalize()}</b><br>"
                        "elo_gap: %{x:.0f}<br>acl_gap: %{y:.1f}<extra></extra>"
                    ),
                )
            )
        fig_sc.add_hline(y=0, line_width=1, line_color=BORDER)
        fig_sc.add_vline(x=0, line_width=1, line_color=BORDER)
        fig_sc.update_layout(
            **PLOTLY_BASE,
            xaxis_title="elo_gap (white_elo − black_elo)",
            yaxis_title="acl_gap (white_acl − black_acl)",
            legend=dict(
                orientation="h",
                y=-0.2,
                x=0,
                font=dict(family="DM Mono", size=10),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=310,
        )
        st.plotly_chart(
            fig_sc, use_container_width=True, config={"displayModeBar": False}
        )
    else:
        st.info("Broaden filters or enable Stockfish games to see the scatter plot.")

with col_f:
    st.markdown(
        """
    <div class="panel-title">Game Termination Type</div>
    <div class="panel-desc">How games end — checkmate, resignation, draw &nbsp;
    <span class="tag tag-slate">termination</span></div>
    """,
        unsafe_allow_html=True,
    )

    term_counts = dff["termination"].value_counts()
    fig_pie = go.Figure(
        go.Pie(
            labels=term_counts.index.tolist(),
            values=term_counts.values,
            hole=0.55,
            marker=dict(
                colors=[WHT_COL, DRW_COL, BLK_COL, GOLD][: len(term_counts)],
                line=dict(color="#0d0f14", width=2),
            ),
            textfont=dict(family="DM Mono", size=10),
            hovertemplate="<b>%{label}</b><br>%{value:,} games (%{percent})<extra></extra>",
        )
    )
    fig_pie.update_layout(
        **PLOTLY_BASE,
        legend=dict(
            orientation="h",
            y=-0.12,
            x=0,
            font=dict(family="DM Mono", size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=310,
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

st.markdown("<hr>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# OUTCOME TREND BY ELO BAND  (uses avg_elo column)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="panel-title" style="margin-bottom:4px">Outcome Share Across Elo Bands</div>
<div class="panel-desc">How win rates shift as average game Elo rises — drawn from avg_elo</div>
""",
    unsafe_allow_html=True,
)

dff4 = dff.copy()
dff4["elo_band"] = pd.cut(
    dff4["avg_elo"],
    bins=[0, 1000, 1200, 1400, 1600, 1800, 2000, 9999],
    labels=[
        "<1000",
        "1000–1200",
        "1200–1400",
        "1400–1600",
        "1600–1800",
        "1800–2000",
        "2000+",
    ],
)
band_grp = (
    dff4.groupby(["elo_band", "outcome"], observed=True).size().unstack(fill_value=0)
)
for c in ["white", "black", "draw"]:
    if c not in band_grp.columns:
        band_grp[c] = 0
band_grp = band_grp.div(band_grp.sum(axis=1), axis=0) * 100

fig_trend = go.Figure()
for oc, color, name in [
    ("white", WHT_COL, "White wins"),
    ("draw", DRW_COL, "Draw"),
    ("black", BLK_COL, "Black wins"),
]:
    fig_trend.add_trace(
        go.Scatter(
            x=band_grp.index.astype(str).tolist(),
            y=band_grp[oc].round(1).tolist(),
            name=name,
            line=dict(color=color, width=2.5),
            mode="lines+markers",
            marker=dict(size=6, color=color),
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        )
    )

fig_trend.update_layout(**PLOTLY_BASE)
fig_trend.update_layout(
    yaxis=dict(
        gridcolor=BORDER,
        ticksuffix="%",
        zerolinecolor=BORDER,
        tickfont=dict(family="DM Mono", size=10),
        range=[0, 70],
    ),
    xaxis=dict(
        gridcolor="rgba(0,0,0,0)",
        zerolinecolor="rgba(0,0,0,0)",
        tickfont=dict(family="DM Mono", size=10),
    ),
    legend=dict(
        orientation="h",
        y=-0.2,
        x=0,
        font=dict(family="DM Mono", size=10),
        bgcolor="rgba(0,0,0,0)",
    ),
    height=240,
)
st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

st.markdown("<hr>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGIC INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## Strategic Modeling Insights")
st.markdown("<br>", unsafe_allow_html=True)

ic1, ic2, ic3 = st.columns(3, gap="medium")
with ic1:
    st.markdown(
        """
    <div class="insight-box">
      <strong>♟ Class Imbalance</strong><br>
      <code>winner_multiclass</code> (0=Black, 1=Draw, 2=White) is skewed — draws (~20%) will
      be suppressed. Apply <strong>class_weight</strong> or <strong>SMOTE</strong> before training.
    </div>
    """,
        unsafe_allow_html=True,
    )
with ic2:
    st.markdown(
        """
    <div class="insight-box">
      <strong>◈ Feature Hierarchy</strong><br>
      <strong>Pre-game:</strong> <code>elo_gap</code> dominates. <strong>Post-game:</strong>
      <code>acl_gap</code>, <code>white_blunders</code>, and <code>black_blunders</code>
      explain &gt;60% of outcome variance.
    </div>
    """,
        unsafe_allow_html=True,
    )
with ic3:
    st.markdown(
        """
    <div class="insight-box">
      <strong>⊞ Stockfish Coverage</strong><br>
      Only <code>kaggle_pgn</code> rows carry Stockfish features (<code>has_stockfish=True</code>).
      Train separate models or impute NaN rows from <code>lichess_csv</code>.
    </div>
    """,
        unsafe_allow_html=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
sf_count = len(dff_sf)
st.markdown(
    f"""
<div style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid #1e2230;
     display:flex;justify-content:space-between;align-items:center;
     font-family:DM Mono,monospace;font-size:10px;color:#3a4050;">
  <span>Chess Elo Predictor · Executive Dashboard v2.1</span>
  <span>{n_games:,} games · {sf_count:,} with Stockfish · avg Elo {avg_elo_val:.0f}</span>
</div>
""",
    unsafe_allow_html=True,
)
