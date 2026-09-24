import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ------------------------------------------------------------------ sozlamalar
TEAM_ID = "C9B71210"
DATA_PATH = "train_signals.csv"
TARGET_HINTS = ["label", "target", "is_alert", "alert", "suspicious", "is_fraud",
                "fraud", "sar", "flag", "class", "y", "eskalatsiya"]

# Ranglar: tilla FAQAT raqamlar uchun, qolgan interfeys sovuq (ko'k/kulrang) tonlarda
GOLD = "#F5C451"
GOLD_SCALE = [[0, "#5A4416"], [0.5, "#C9902B"], [1, "#FFE29A"]]
BLUE = "#5B7CFA"
CORAL = "#FF6B8B"
MUTED = "#94A3B8"

st.set_page_config(page_title="WIUT Hackathon - AML AI Dashboard", page_icon="🛡️", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
.stApp {
    background:
        radial-gradient(1100px 600px at 8% -10%, rgba(91,124,250,.18), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(168,85,247,.12), transparent 55%),
        #0a0e1a;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1400px; }
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,.03);
    border-right: 1px solid rgba(255,255,255,.06);
}
h1, h2, h3, h4 { color: #e5e7eb; letter-spacing: -0.01em; }

/* ---------- Hero ---------- */
.hero {
    padding: 28px 32px; margin-bottom: 22px; border-radius: 22px;
    background: linear-gradient(135deg, rgba(91,124,250,.16), rgba(255,255,255,.03) 55%);
    border: 1px solid rgba(255,255,255,.08);
    box-shadow: 0 20px 60px rgba(0,0,0,.35);
}
.hero .pill {
    display: inline-block; padding: 5px 12px; border-radius: 999px; font-size: .78rem;
    color: #c7d2fe; background: rgba(91,124,250,.16); border: 1px solid rgba(91,124,250,.35);
}
.hero h1 {
    margin: 12px 0 6px; font-size: 2.5rem; font-weight: 800;
    background: linear-gradient(90deg, #ffffff, #9db4ff);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { margin: 0; color: #94a3b8; font-size: 1.02rem; }
.hero .team {
    display: inline-block; margin-top: 14px; padding: 6px 12px; border-radius: 10px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85rem;
    color: #cbd5e1; background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.08);
}

/* ---------- Metrika kartalari (raqamlar = tilla) ---------- */
[data-testid="stMetric"] {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
    border-radius: 18px; padding: 16px 18px; backdrop-filter: blur(10px);
    transition: border-color .2s ease, transform .2s ease;
}
[data-testid="stMetric"]:hover { border-color: rgba(245,196,81,.45); transform: translateY(-2px); }
[data-testid="stMetricLabel"] p {
    color: #94a3b8; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em;
}
[data-testid="stMetricValue"] { color: #F5C451; }
@supports (-webkit-background-clip: text) {
    [data-testid="stMetricValue"] div {
        font-weight: 800; font-variant-numeric: tabular-nums;
        background: linear-gradient(135deg, #FFE29A 0%, #F5C451 45%, #C9902B 100%);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }
}

/* ---------- Tablar ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(255,255,255,.08); }
.stTabs [data-baseweb="tab"] { padding: 10px 18px; border-radius: 12px 12px 0 0; color: #94a3b8; }
.stTabs [aria-selected="true"] { color: #ffffff; background: rgba(255,255,255,.05); }

/* ---------- Kartalar, jadval, expander ---------- */
.card {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
    border-radius: 18px; padding: 18px 20px; height: 100%;
}
.card h4 { margin: 0 0 6px; font-size: 1.02rem; }
.card p { margin: 0; color: #94a3b8; font-size: .92rem; line-height: 1.5; }
[data-testid="stDataFrame"] {
    border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,.08);
}
[data-testid="stExpander"] {
    border-radius: 14px; border: 1px solid rgba(255,255,255,.08); background: rgba(255,255,255,.02);
}
.stButton > button, .stDownloadButton > button {
    border-radius: 12px; border: 1px solid rgba(255,255,255,.12);
    background: rgba(255,255,255,.05); color: #e5e7eb;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #5B7CFA; color: #ffffff; background: rgba(91,124,250,.18);
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ yordamchilar
@st.cache_data(show_spinner="Ma'lumotlar yuklanmoqda...")
def load_csv(source):
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    df = pd.read_csv(source)
    for c in df.columns:
        if df[c].dtype == "object" and any(k in c.lower() for k in ("date", "time", "sana", "vaqt")):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.8:
                df[c] = parsed
    return df


def guess_target(df):
    lower = {c.lower(): c for c in df.columns}
    for h in TARGET_HINTS:
        if h in lower:
            return lower[h]
    for c in df.columns:
        if df[c].nunique(dropna=True) == 2:
            return c
    return None


def to_binary(s):
    vals = s.dropna().unique()
    if len(vals) != 2:
        return None
    if set(vals) <= {0, 1, True, False}:
        return s.astype(float)
    minority = s.value_counts().idxmin()  # kam uchraydigan sinf = "alert"
    return (s == minority).astype(float).where(s.notna())


def make_X(frame, feats, columns=None):
    """Modelga tayyor matritsa: raqamlar + kategoriyalar (one-hot)."""
    X = pd.get_dummies(frame[feats]).astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X.fillna(0)


@st.cache_data(show_spinner="AI model o'qitilmoqda va halol baholanmoqda (cross-validation)...")
def train_and_score(X, y, n_estimators, max_depth, n_splits):
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=3,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # out-of-fold: har bir qator o'zi ishtirok etmagan modelda baholanadi
    oof = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
    model.fit(X, y)
    return oof, model, model.feature_importances_, roc_auc_score(y, oof), average_precision_score(y, oof)


def topk_stats(y, score, pct):
    k = max(1, int(len(y) * pct / 100))
    top = np.argsort(-score)[:k]
    caught = y[top].sum()
    recall = caught / max(y.sum(), 1)
    precision = caught / k
    lift = precision / max(y.mean(), 1e-9)
    return k, recall, precision, lift


def style_fig(fig, height=420, gold_x=False, gold_y=False):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
        margin=dict(l=10, r=10, t=56, b=10), height=height,
        title=dict(font=dict(size=16, color="#e2e8f0")),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#111827", font=dict(color="#f8fafc")),
    )
    grid = "rgba(255,255,255,0.06)"
    fig.update_xaxes(gridcolor=grid, zeroline=False, tickfont=dict(color=GOLD if gold_x else MUTED))
    fig.update_yaxes(gridcolor=grid, zeroline=False, tickfont=dict(color=GOLD if gold_y else MUTED))
    return fig


def show(fig, **kw):
    st.plotly_chart(style_fig(fig, **kw), use_container_width=True, config={"displayModeBar": False})


def gold_table(frame, extra=None):
    """Jadvalda raqamli ustunlar tilla rangda."""
    nums = frame.select_dtypes(include=[np.number]).columns.tolist()
    sty = frame.style
    if nums:
        sty = sty.set_properties(subset=nums, **{"color": GOLD, "font-weight": "600"})
    if extra:
        sty = extra(sty)
    return sty


# ------------------------------------------------------------------ sarlavha
st.markdown(
    f"""
    <div class="hero">
        <span class="pill">WIUT Fintech Hackathon 2026 · AI in Finance Track</span>
        <h1>AML Alert Prioritization</h1>
        <p>Shubhali signallarni sun'iy intellekt yordamida xavf darajasi bo'yicha ustuvorlashtirish paneli</p>
        <span class="team">Jamoa ID · {TEAM_ID}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ ma'lumot
with st.sidebar:
    st.header("⚙️ Sozlamalar")
    uploaded = st.file_uploader("Boshqa CSV yuklash (ixtiyoriy)", type="csv")

try:
    df_full = load_csv(uploaded.getvalue() if uploaded else DATA_PATH)
except FileNotFoundError:
    st.error(f"`{DATA_PATH}` topilmadi. Faylni app.py yoniga qo'ying yoki chapdan yuklang.")
    st.stop()
except Exception as e:
    st.error(f"Faylni o'qishda xatolik: {e}")
    st.stop()

num_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
dt_cols = df_full.select_dtypes(include=["datetime64[ns]", "datetime"]).columns.tolist()
cat_cols = [c for c in df_full.columns if c not in num_cols and c not in dt_cols]

with st.sidebar:
    guessed = guess_target(df_full)
    options = ["(yo'q)"] + df_full.columns.tolist()
    target_col = st.selectbox(
        "Target (alert/risk) ustuni", options,
        index=options.index(guessed) if guessed else 0,
        help="AI shu ustunni bashorat qilishni o'rganadi.",
    )

y_full = None
if target_col != "(yo'q)":
    y_full = to_binary(df_full[target_col])
    if y_full is None:
        st.sidebar.warning("Target ustuni aynan 2 ta qiymatdan iborat bo'lishi kerak.")
        target_col = "(yo'q)"
    elif not set(df_full[target_col].dropna().unique()) <= {0, 1, True, False}:
        st.sidebar.caption("Kam uchraydigan sinf «alert» deb qabul qilindi.")

# filtrlar faqat ko'rinishga ta'sir qiladi — model to'liq ma'lumotda o'qitiladi
view = df_full
with st.sidebar:
    st.subheader("🔎 Filtrlar")
    low_card = [c for c in cat_cols if df_full[c].nunique() <= 30][:5]
    for c in low_card:
        chosen = st.multiselect(c, sorted(df_full[c].dropna().astype(str).unique()))
        if chosen:
            view = view[view[c].astype(str).isin(chosen)]

if view.empty:
    st.warning("Filtrlardan keyin ma'lumot qolmadi.")
    st.stop()

y_view = y_full.reindex(view.index) if y_full is not None else None

# ------------------------------------------------------------------ metrikalar
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Jami signallar", f"{len(view):,}")
m2.metric("Ustunlar", len(view.columns))
m3.metric("Bo'sh qiymatlar", f"{view.isnull().mean().mean():.1%}")
m4.metric("Haqiqiy alertlar", f"{int(y_view.sum()):,}" if y_view is not None else "—")
m5.metric("Alert ulushi", f"{y_view.mean():.2%}" if y_view is not None else "—")
st.write("")

tabs = st.tabs(["📌 Xulosa", "🤖 AI Risk Prioritization", "📈 Taqsimot", "🔗 Bog'liqlik", "📋 Ma'lumotlar"])

# ------------------------------------------------------------------ 1. Xulosa
with tabs[0]:
    a, b, c = st.columns(3)
    a.markdown('<div class="card"><h4>🎯 Maqsad</h4><p>Compliance xodimlarining vaqtini tejash: '
               'eng xavfli signallar navbatning boshiga chiqariladi.</p></div>', unsafe_allow_html=True)
    b.markdown('<div class="card"><h4>🧠 Yondashuv</h4><p>Random Forest modeli har bir signalga xavf balli beradi. '
               'Ballar cross-validation orqali olinadi, shuning uchun natija ishonchli.</p></div>',
               unsafe_allow_html=True)
    c.markdown('<div class="card"><h4>📊 Natija</h4><p>Ustuvor ro\'yxat, eng muhim omillar va '
               'test fayl uchun tayyor bashorat.</p></div>', unsafe_allow_html=True)
    st.write("")

    if y_view is not None:
        vc = view[target_col].astype(str).value_counts().reset_index()
        vc.columns = [target_col, "soni"]
        fig = px.bar(vc, x=target_col, y="soni", text="soni", color=target_col,
                     color_discrete_sequence=[BLUE, CORAL], title="Target taqsimoti")
        fig.update_traces(textfont=dict(color=GOLD, size=14), textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        show(fig, height=360, gold_y=True)
        if y_view.mean() < 0.1 or y_view.mean() > 0.9:
            st.info("⚠️ Sinflar nomutanosib (imbalanced), shuning uchun modelni baholashda "
                    "ROC-AUC bilan birga PR-AUC ham ko'rsatiladi.")

    if num_cols:
        st.subheader("Statistik ko'rsatkichlar")
        st.dataframe(gold_table(view[num_cols].describe().T).format("{:,.2f}"), use_container_width=True)

# ------------------------------------------------------------------ 2. AI Risk
with tabs[1]:
    st.subheader("🤖 AI yordamida signallarni ustuvorlashtirish")
    st.caption("Har bir signalga 0–1 oralig'ida xavf balli beriladi. Ballar **out-of-fold** usulida hisoblanadi: "
               "model o'zi o'qigan qatorni baholamaydi.")

    if y_full is None:
        st.info("ℹ️ Chap menyudan **Target** ustunini tanlang.")
    else:
        id_like = [c for c in df_full.columns if df_full[c].nunique() == len(df_full) and len(df_full) > 1]
        cand = [c for c in num_cols + [x for x in cat_cols if df_full[x].nunique() <= 20]
                if c != target_col]
        default = [c for c in cand if c not in id_like]

        with st.expander("⚙️ Model sozlamalari"):
            feats = st.multiselect("Model omillari", cand, default=default,
                                   help="ID ustunlari avtomatik olib tashlandi.")
            s1, s2 = st.columns(2)
            n_est = s1.slider("Daraxtlar soni", 50, 500, 200, step=50)
            depth = s2.slider("Maksimal chuqurlik", 3, 20, 8)

        mask = y_full.notna().values
        y_m = y_full[mask].astype(int).values
        n_splits = min(5, int(y_m.sum()), int(len(y_m) - y_m.sum()))

        if not feats:
            st.warning("Kamida bitta omil tanlang.")
        elif n_splits < 2:
            st.warning("Har bir sinfda kamida 2 ta kuzatuv bo'lishi kerak.")
        else:
            X_all = make_X(df_full, feats)
            X_m = X_all[mask]
            oof, model, imps, roc, pr = train_and_score(X_m, y_m, n_est, depth, n_splits)

            scores = pd.Series(np.nan, index=df_full.index)
            scores.loc[mask] = oof

            pct = st.slider("Ko'rib chiqiladigan signallar ulushi (%)", 1, 50, 10,
                            help="Xodimlar navbatning yuqori qismidan shuncha signalni tekshiradi.")
            k, recall, precision, lift = topk_stats(y_m, oof, pct)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("ROC-AUC", f"{roc:.3f}")
            k2.metric("PR-AUC", f"{pr:.3f}")
            k3.metric(f"Top {pct}% qamrovi", f"{recall:.1%}")
            k4.metric("Lift", f"{lift:.1f}×")
            st.caption(f"Top {pct}% = {k:,} ta signal. Ular ichida haqiqiy alertlarning {recall:.1%} qismi ushlanadi; "
                       f"tasodifiy tanlashda bu ~{pct}% bo'lar edi.")
            if roc > 0.99:
                st.warning("⚠️ ROC-AUC juda yuqori. Bu target'ni bilvosita oshkor qiluvchi ustun "
                           "(data leakage) borligini bildirishi mumkin — omillarni tekshiring.")
            st.write("")

            g1, g2 = st.columns(2)
            with g1:
                order = np.argsort(-oof)
                cum = np.cumsum(y_m[order]) / max(y_m.sum(), 1)
                n = len(order)
                idx = np.unique(np.linspace(0, n - 1, min(n, 400)).astype(int))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=(idx + 1) / n * 100, y=cum[idx] * 100, mode="lines", name="AI model",
                                         line=dict(color=GOLD, width=3), fill="tozeroy",
                                         fillcolor="rgba(245,196,81,0.10)"))
                fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Tasodifiy",
                                         line=dict(color="#64748B", dash="dash")))
                fig.add_trace(go.Scatter(x=[pct], y=[recall * 100], mode="markers", name=f"Top {pct}%",
                                         marker=dict(color="#FFE29A", size=11, line=dict(color=GOLD, width=2))))
                fig.update_layout(title="Qamrov egri chizig'i (cumulative gain)",
                                  xaxis_title="Ko'rilgan signallar, %", yaxis_title="Ushlangan alertlar, %")
                show(fig, height=380, gold_x=True, gold_y=True)
            with g2:
                sd = pd.DataFrame({"AI_Risk_Score": oof, "Sinf": np.where(y_m == 1, "Alert", "Normal")})
                fig = px.histogram(sd, x="AI_Risk_Score", color="Sinf", nbins=40, barmode="overlay", opacity=0.75,
                                   color_discrete_map={"Normal": BLUE, "Alert": CORAL},
                                   title="Xavf ballari taqsimoti")
                show(fig, height=380, gold_x=True, gold_y=True)

            imp = (pd.DataFrame({"Omil": X_m.columns, "Muhimlik": imps})
                   .sort_values("Muhimlik").tail(10))
            fig = px.bar(imp, x="Muhimlik", y="Omil", orientation="h", color="Muhimlik",
                         color_continuous_scale=GOLD_SCALE, text_auto=".3f",
                         title="Xavfni aniqlovchi asosiy omillar (Feature Importance)")
            fig.update_coloraxes(showscale=False)
            fig.update_traces(textfont=dict(color=GOLD), textposition="outside", cliponaxis=False)
            show(fig, height=400, gold_x=True)

            st.subheader("🚨 Eng yuqori ustuvorlikdagi signallar")
            ranked = view.assign(AI_Risk_Score=scores.reindex(view.index)).sort_values(
                "AI_Risk_Score", ascending=False)
            ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
            ranked = ranked[["Rank", "AI_Risk_Score"] + [c for c in ranked.columns
                                                          if c not in ("Rank", "AI_Risk_Score")]]
            top_n = st.slider("Ko'rsatiladigan qatorlar", 10, 200, 25, step=5)
            st.dataframe(
                gold_table(
                    ranked.head(top_n),
                    lambda s: s.format({"AI_Risk_Score": "{:.3f}"}).bar(
                        subset=["AI_Risk_Score"], color="rgba(245,196,81,0.35)", vmin=0, vmax=1),
                ),
                use_container_width=True, hide_index=True,
            )
            st.download_button("⬇️ To'liq ustuvor ro'yxat (CSV)", ranked.to_csv(index=False).encode("utf-8"),
                               file_name="ai_scored_signals.csv", mime="text/csv")

            with st.expander("📤 Test faylni baholash (submission)"):
                test_file = st.file_uploader("Test CSV faylini yuklang", type="csv", key="test_up")
                if test_file:
                    tdf = pd.read_csv(test_file)
                    missing = [f for f in feats if f not in tdf.columns]
                    if missing:
                        st.error(f"Test faylda yetishmayotgan ustunlar: {', '.join(missing)}")
                    else:
                        Xt = make_X(tdf, feats, columns=X_m.columns)
                        tdf["AI_Risk_Score"] = model.predict_proba(Xt)[:, 1]
                        tdf["AI_Risk_Rank"] = tdf["AI_Risk_Score"].rank(ascending=False, method="first").astype(int)
                        tdf = tdf.sort_values("AI_Risk_Rank")
                        st.dataframe(gold_table(tdf.head(25), lambda s: s.format({"AI_Risk_Score": "{:.3f}"})),
                                     use_container_width=True, hide_index=True)
                        st.download_button("⬇️ Bashoratlarni yuklab olish", tdf.to_csv(index=False).encode("utf-8"),
                                           file_name="test_predictions.csv", mime="text/csv")

# ------------------------------------------------------------------ 3. Taqsimot
with tabs[2]:
    if not num_cols:
        st.info("Raqamli ustunlar topilmadi.")
    else:
        t1, t2, t3 = st.columns([2, 1, 1])
        col = t1.selectbox("Tahlil uchun ustun", num_cols, key="dist_col")
        bins = t2.slider("Bins", 10, 100, 40)
        logy = t3.checkbox("Log shkala (Y)")
        color = view[target_col].astype(str) if y_view is not None else None

        fig = px.histogram(view, x=col, nbins=bins, color=color, barmode="overlay", opacity=0.75,
                           color_discrete_sequence=[BLUE, CORAL], title=f"{col} taqsimoti")
        if logy:
            fig.update_yaxes(type="log")
        show(fig, gold_x=True, gold_y=True)

        fig = px.box(view, x=color, y=col, color=color, color_discrete_sequence=[BLUE, CORAL],
                     title=f"{col} — box plot") if y_view is not None else \
            px.box(view, y=col, title=f"{col} — box plot", color_discrete_sequence=[BLUE])
        fig.update_layout(showlegend=False)
        show(fig, height=380, gold_y=True)

# ------------------------------------------------------------------ 4. Bog'liqlik
with tabs[3]:
    if len(num_cols) < 2:
        st.info("Kamida 2 ta raqamli ustun kerak.")
    else:
        cols = num_cols[:25]
        if len(num_cols) > 25:
            st.caption("Ko'rinish aniq bo'lishi uchun dastlabki 25 ta raqamli ustun ko'rsatilmoqda.")
        corr = view[cols].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                        color_continuous_scale=[[0, BLUE], [0.5, "#111827"], [1, GOLD]],
                        title="Korrelyatsiya matritsasi")
        show(fig, height=560)

        if y_view is not None:
            tc = view[[c for c in cols if c != target_col]].corrwith(y_view).dropna()
            tc = tc.reindex(tc.abs().sort_values(ascending=False).index).head(15)[::-1]
            fig = px.bar(tc.rename("corr"), orientation="h", color="corr", text_auto=".2f",
                         color_continuous_scale=[[0, BLUE], [0.5, "#334155"], [1, GOLD]],
                         range_color=[-1, 1], title="Targetga eng bog'liq ustunlar")
            fig.update_coloraxes(showscale=False)
            fig.update_traces(textfont=dict(color=GOLD), textposition="outside", cliponaxis=False)
            fig.update_layout(xaxis_title="Korrelyatsiya", yaxis_title=None)
            show(fig, height=460, gold_x=True)

# ------------------------------------------------------------------ 5. Ma'lumotlar
with tabs[4]:
    n = st.slider("Qatorlar soni", 5, 200, 20)
    st.dataframe(gold_table(view.head(n)), use_container_width=True)

    st.subheader("Ustunlar haqida")
    info = pd.DataFrame({
        "Tur": view.dtypes.astype(str),
        "Bo'sh soni": view.isnull().sum(),
        "Bo'sh %": (view.isnull().mean() * 100).round(2),
        "Noyob qiymatlar": view.nunique(),
    })
    st.dataframe(gold_table(info), use_container_width=True)
    st.download_button("⬇️ Filtrlangan CSV", view.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_signals.csv", mime="text/csv")
