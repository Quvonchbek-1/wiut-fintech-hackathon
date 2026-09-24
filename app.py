"""
WIUT Fintech Hackathon 2026 - AML Alert Prioritization Dashboard
------------------------------------------------------------------
Streamlit + scikit-learn asosidagi AML (Anti-Money Laundering) signallarni
xavf darajasi bo'yicha ustuvorlashtiruvchi interaktiv panel.
"""

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ============================================================
# SOZLAMALAR
# ============================================================
TEAM_ID = "C9B71210"
DATA_PATH = "train_signals.csv"
TARGET_HINTS = [
    "label", "target", "is_alert", "alert", "suspicious", "is_fraud",
    "fraud", "sar", "flag", "class", "y", "eskalatsiya",
]

GOLD = "#F5C451"
BLUE = "#5B7CFA"
CORAL = "#FF6B8B"
GREEN = "#34D399"
MUTED = "#94A3B8"
RISK_COLORS = {"Past": GREEN, "O'rta": GOLD, "Yuqori": CORAL}

st.set_page_config(page_title="WIUT Hackathon - AML AI Dashboard", page_icon="🛡️", layout="wide")

# ============================================================
# STIL
# ============================================================
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
[data-testid="stMetric"] {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
    border-radius: 18px; padding: 16px 18px; backdrop-filter: blur(10px);
}
[data-testid="stMetricLabel"] p { color: #94a3b8; font-size: .78rem; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #F5C451; }
.card {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
    border-radius: 18px; padding: 18px 20px; height: 100%;
}
.card h4 { margin: 0 0 6px; font-size: 1.02rem; }
.card p { margin: 0; color: #94a3b8; font-size: .92rem; line-height: 1.5; }
.risk-badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: .78rem; font-weight: 700;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# YORDAMCHI FUNKSIYALAR
# ============================================================
@st.cache_data(show_spinner="Ma'lumotlar yuklanmoqda...")
def load_csv(source) -> pd.DataFrame:
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    df = pd.read_csv(source)
    if df.empty:
        raise ValueError("Fayl bo'sh yoki noto'g'ri formatda.")
    for c in df.columns:
        if df[c].dtype == "object" and any(k in c.lower() for k in ("date", "time", "sana", "vaqt")):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.8:
                df[c] = parsed
    return df


def guess_target(df: pd.DataFrame) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for h in TARGET_HINTS:
        if h in lower:
            return lower[h]
    binary_cols = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    return binary_cols[0] if binary_cols else None


def to_binary(s: pd.Series) -> pd.Series | None:
    vals = s.dropna().unique()
    if len(vals) != 2:
        return None
    if set(vals) <= {0, 1, True, False}:
        return s.astype(float)
    minority = s.value_counts().idxmin()
    return (s == minority).astype(float).where(s.notna())


MAX_ONE_HOT_CARDINALITY = 30


def make_X(frame: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, list[str]]:
    sub = frame[feats].copy()
    notes: list[str] = []

    for c in sub.columns:
        if pd.api.types.is_datetime64_any_dtype(sub[c]):
            sub[c] = sub[c].astype("int64") // 10**9
            sub[c] = sub[c].where(frame[c].notna())
        elif sub[c].dtype == "object" or str(sub[c].dtype) == "category":
            n_unique = sub[c].nunique(dropna=True)
            if n_unique > MAX_ONE_HOT_CARDINALITY:
                freq = sub[c].value_counts()
                sub[c] = sub[c].map(freq)
                notes.append(
                    f"'{c}' ustuni {n_unique} noyob qiymatga ega — RAM tejash uchun "
                    f"one-hot o'rniga chastota kodlash qo'llanildi."
                )

    X = pd.get_dummies(sub).astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(0), notes


@dataclass(frozen=True)
class ModelResult:
    oof: np.ndarray
    model: RandomForestClassifier
    importances: pd.Series
    roc_auc: float
    pr_auc: float
    fpr: np.ndarray
    tpr: np.ndarray
    precision: np.ndarray
    recall: np.ndarray


@st.cache_resource(show_spinner="AI model o'qitilmoqda...")
def train_and_score(X: pd.DataFrame, y: np.ndarray, n_estimators: int, max_depth: int, n_splits: int) -> ModelResult:
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=1,
        class_weight="balanced_subsample", n_jobs=2, random_state=42,
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
    model.fit(X, y)
    fpr, tpr, _ = roc_curve(y, oof)
    precision, recall, _ = precision_recall_curve(y, oof)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return ModelResult(
        oof=oof, model=model, importances=importances,
        roc_auc=roc_auc_score(y, oof), pr_auc=average_precision_score(y, oof),
        fpr=fpr, tpr=tpr, precision=precision, recall=recall,
    )


def risk_band(score: float, low_cut: float, high_cut: float) -> str:
    if pd.isna(score):
        return "—"
    if score >= high_cut:
        return "Yuqori"
    if score >= low_cut:
        return "O'rta"
    return "Past"


def style_fig(fig, height=420):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
        margin=dict(l=10, r=10, t=56, b=10), height=height,
    )
    return fig


def show(fig, **kw):
    st.plotly_chart(style_fig(fig, **kw), use_container_width=True, config={"displayModeBar": False})


def gold_table(frame: pd.DataFrame):
    nums = frame.select_dtypes(include=[np.number]).columns.tolist()
    sty = frame.style
    if nums:
        sty = sty.set_properties(subset=nums, **{"color": GOLD, "font-weight": "600"})
    return sty


# ============================================================
# SARLAVHA
# ============================================================
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

# ============================================================
# MA'LUMOT YUKLASH VA CHAP MENYU SOZLAMALARI
# ============================================================
with st.sidebar:
    st.header("⚙️ Boshqaruv paneli")
    uploaded = st.file_uploader("Boshqa CSV yuklash", type="csv")
    
    guessed = guess_target(load_csv(uploaded.getvalue() if uploaded else DATA_PATH) if True else None) # type: ignore
    
try:
    df_full = load_csv(uploaded.getvalue() if uploaded else DATA_PATH)
except FileNotFoundError:
    st.error(f"❌ `{DATA_PATH}` topilmadi. Chap menyudan CSV faylni yuklang.")
    st.stop()
except Exception as e:
    st.error(f"❌ Faylni o'qishda xatolik: {e}")
    st.stop()

num_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
dt_cols = df_full.select_dtypes(include=["datetime64[ns]", "datetime"]).columns.tolist()
cat_cols = [c for c in df_full.columns if c not in num_cols and c not in dt_cols]

with st.sidebar:
    st.divider()
    guessed = guess_target(df_full)
    options = ["(yo'q)"] + df_full.columns.tolist()
    default_idx = options.index(guessed) if guessed else 0
    target_col = st.selectbox("🎯 Target (alert/risk) ustuni", options, index=default_idx)
    st.caption(f"📄 {len(df_full):,} qator · {len(df_full.columns)} ustun")

    st.divider()
    with st.expander("🎛️ Kengaytirilgan model sozlamalari", expanded=False):
        st.markdown("Hakamlar uchun texnik giperparametrlar:")
        n_est = st.slider("Daraxtlar soni (n_estimators)", 50, 300, 100, step=10)
        depth = st.slider("Maksimal chuqurlik (max_depth)", 3, 15, 6)
        n_splits_ui = st.slider("CV bo'laklari (folds)", 2, 5, 5)

    if st.button("🔄 Keshni tozalash", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

y_full = to_binary(df_full[target_col]) if target_col != "(yo'q)" else None
if target_col != "(yo'q)" and y_full is None:
    st.sidebar.warning("⚠️ Tanlangan ustun ikkilik (binary) emas — target sifatida ishlatib bo'lmaydi.")

view = df_full
y_view = y_full.reindex(view.index) if y_full is not None else None

# ============================================================
# YUQORI METRIKALAR
# ============================================================
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Jami signallar", f"{len(view):,}")
m2.metric("Ustunlar", len(view.columns))
m3.metric("Bo'sh qiymatlar", f"{view.isnull().mean().mean():.1%}")
m4.metric("Haqiqiy alertlar", f"{int(y_view.sum()):,}" if y_view is not None else "—")
m5.metric("Alert ulushi", f"{y_view.mean():.2%}" if y_view is not None else "—")
st.write("")

tabs = st.tabs([
    "📌 Xulosa", "🤖 AI Risk Prioritization", "📈 Taqsimot",
    "🔗 Bog'liqlik", "🕒 Vaqt tendensiyasi", "📋 Ma'lumotlar",
])

# ---------- 1. Xulosa ----------
with tabs[0]:
    st.subheader("💡 Loyiha haqida")
    st.markdown(
        "Ushbu panel AML (pul yuvishga qarshi kurash) signallarini sun'iy intellekt "
        "yordamida xavf darajasi bo'yicha ustuvorlashtirish uchun mo'ljallangan. "
        "Random Forest modeli tarixiy belgilangan (label) ma'lumotlar asosida o'qitiladi "
        "va har bir yangi signalga 0 dan 1 gacha xavf balli beradi."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h4>1️⃣ Ma\'lumot</h4><p>CSV formatidagi tranzaksiya/signal ma\'lumotlari yuklanadi va avtomatik tozalanadi.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h4>2️⃣ Model</h4><p>Random Forest cross-validation orqali o\'qitiladi, ROC-AUC va PR-AUC bilan baholanadi.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><h4>3️⃣ Natija</h4><p>Signallar xavf balli bo\'yicha saralanadi va tekshiruvchilar uchun ustuvorlashtiriladi.</p></div>', unsafe_allow_html=True)

    st.write("")
    st.subheader("🧪 Ma'lumot sifati")
    miss = df_full.isnull().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        st.success("Bo'sh qiymatlar topilmadi ✅")
    else:
        show(px.bar(miss.head(15) * 100, template="plotly_dark", title="Ustunlar bo'yicha bo'sh qiymatlar (%)",
                    labels={"value": "%", "index": "Ustun"}, color_discrete_sequence=[CORAL]), height=360)

# ---------- 2. AI Risk Prioritization ----------
with tabs[1]:
    st.subheader("🤖 AI yordamida signallarni ustuvorlashtirish")
    if y_full is None:
        st.info("ℹ️ Chap menyudan ikkilik (binary) **Target** ustunini tanlang.")
    else:
        cand = [c for c in df_full.columns if c != target_col]

        with st.expander("⚙️ Model omillari (Features)", expanded=True):
            feats = st.multiselect("Modelga beriladigan ustunlar", cand, default=cand)

        mask = y_full.notna().values
        y_m = y_full[mask].astype(int).values
        n_pos, n_neg = int(y_m.sum()), int(len(y_m) - y_m.sum())

        if not feats:
            st.warning("Kamida bitta omil tanlang.")
        elif n_pos < 2 or n_neg < 2:
            st.error("Modelni o'qitish uchun har ikki sinfdan (0 va 1) yetarli namuna yo'q.")
        else:
            n_splits = max(2, min(n_splits_ui, n_pos, n_neg))
            X_all, encoding_notes = make_X(df_full, feats)

            if encoding_notes:
                with st.expander(f"ℹ️ Kodlash haqida eslatma ({len(encoding_notes)})"):
                    for note in encoding_notes:
                        st.caption(f"• {note}")

            if X_all.shape[1] == 0:
                st.error("❌ Tanlangan omillardan modelga yaroqli ustun hosil bo'lmadi.")
                st.stop()

            X_m = X_all[mask]
            result = train_and_score(X_m, y_m, n_est, depth, n_splits)

            scores = pd.Series(np.nan, index=df_full.index)
            scores.loc[mask] = result.oof

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("ROC-AUC", f"{result.roc_auc:.3f}")
            k2.metric("PR-AUC", f"{result.pr_auc:.3f}")
            k3.metric("CV bo'laklari", n_splits)
            k4.metric("Model holati", "Ishlayapti 🟢")

            st.write("")
            g1, g2 = st.columns(2)
            with g1:
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=result.fpr, y=result.tpr, mode="lines",
                                             line=dict(color=BLUE, width=3), name="Model"))
                roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                             line=dict(color=MUTED, dash="dash"), name="Tasodifiy"))
                roc_fig.update_layout(title="ROC egri chizig'i", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                show(roc_fig, height=360)
            with g2:
                pr_fig = go.Figure()
                pr_fig.add_trace(go.Scatter(x=result.recall, y=result.precision, mode="lines",
                                            line=dict(color=GOLD, width=3), name="Model"))
                pr_fig.update_layout(title="Precision-Recall egri chizig'i", xaxis_title="Recall", yaxis_title="Precision")
                show(pr_fig, height=360)

            st.write("")
            st.subheader("🔑 Eng muhim omillar (Feature importance)")
            top_imp = result.importances.head(15).sort_values()
            show(px.bar(top_imp, orientation="h", template="plotly_dark",
                        labels={"value": "Ahamiyat darajasi", "index": "Omil"},
                        color=top_imp.values, color_continuous_scale=[BLUE, GOLD]), height=420)

            st.write("")
            st.subheader("🎚️ Xavf darajalari chegarasi")
            t1, t2 = st.columns(2)
            low_cut = t1.slider("Past ↔ O'rta chegara", 0.0, 1.0, 0.3, 0.01)
            high_cut = t2.slider("O'rta ↔ Yuqori chegara", 0.0, 1.0, 0.7, 0.01)
            if low_cut > high_cut:
                high_cut = low_cut

            bands = scores.apply(lambda s: risk_band(s, low_cut, high_cut))
            band_counts = bands.value_counts().reindex(["Yuqori", "O'rta", "Past"]).fillna(0)
            b1, b2, b3 = st.columns(3)
            for col, label, color in zip((b1, b2, b3), ["Yuqori", "O'rta", "Past"], [CORAL, GOLD, GREEN]):
                col.markdown(
                    f'<div class="card"><span class="risk-badge" style="background:{color}22;color:{color};">{label}</span>'
                    f'<h4 style="margin-top:10px;">{int(band_counts[label]):,} ta</h4></div>',
                    unsafe_allow_html=True,
                )

            st.write("")
            st.subheader("🚨 Eng yuqori ustuvorlikdagi signallar")
            ranked = view.assign(AI_Risk_Score=scores.reindex(view.index).round(4),
                                 Xavf_Darajasi=bands.reindex(view.index))
            ranked = ranked.sort_values("AI_Risk_Score", ascending=False)
            ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
            top_n = st.slider("Ko'rsatiladigan signallar soni", 10, min(200, len(ranked)), min(25, len(ranked)))
            st.dataframe(gold_table(ranked.head(top_n)), use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Barcha natijalarni CSV sifatida yuklab olish",
                ranked.to_csv(index=False).encode("utf-8"),
                file_name="ai_scored_signals.csv", mime="text/csv",
            )

# ---------- 3. Taqsimot ----------
with tabs[2]:
    st.subheader("📈 Ustun taqsimoti")
    if num_cols:
        col1, col2 = st.columns([2, 1])
        col = col1.selectbox("Ustun", num_cols)
        bins = col2.slider("Bo'laklar soni", 10, 100, 30)
        color_by = False
        if y_full is not None:
            color_by = st.checkbox("Target bo'yicha rangla", value=True)
        fig = px.histogram(
            view, x=col, nbins=bins, template="plotly_dark", title=f"{col} taqsimoti",
            color=y_full.reindex(view.index).map({0: "Normal", 1: "Alert"}) if (color_by and y_full is not None) else None,
            color_discrete_sequence=[BLUE, CORAL],
        )
        show(fig)
    else:
        st.info("Raqamli ustunlar topilmadi.")

# ---------- 4. Bog'liqlik (Zaxira bilan to'ldirilgan) ----------
with tabs[3]:
    st.subheader("🔗 Ustunlar orasidagi korrelyatsiya")
    if len(num_cols) >= 2:
        show(px.imshow(view[num_cols].corr(), text_auto=".2f", template="plotly_dark",
                       color_continuous_scale=[[0, CORAL], [0.5, "#111827"], [1, BLUE]],
                       title="Korrelyatsiya matritsasi"), height=500)
    else:
        st.warning("⚠️ Korrelyatsiya matritsasini chizish uchun datasetda kamida 2 ta raqamli ustun bo'lishi kerak.")
        st.info("Ma'lumotlar jadvalidagi ustun turlarini tekshiring yoki boshqa CSV yuklang.")

# ---------- 5. Vaqt tendensiyasi (Zaxira bilan to'ldirilgan) ----------
with tabs[4]:
    st.subheader("🕒 Vaqt bo'yicha tendensiya")
    if dt_cols:
        dcol = st.selectbox("Sana/vaqt ustuni", dt_cols)
        freq = st.radio("Davriylik", ["Kun", "Hafta", "Oy"], horizontal=True)
        rule = {"Kun": "D", "Hafta": "W", "Oy": "M"}[freq]
        ts = view.dropna(subset=[dcol]).set_index(dcol)
        if y_full is not None:
            grouped = pd.DataFrame({
                "Jami signal": ts.resample(rule).size(),
                "Alertlar": y_full.reindex(ts.index).resample(rule).sum(),
            }).fillna(0)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=grouped.index, y=grouped["Jami signal"], name="Jami signal",
                                     line=dict(color=BLUE, width=2)))
            fig.add_trace(go.Scatter(x=grouped.index, y=grouped["Alertlar"], name="Alertlar",
                                     line=dict(color=CORAL, width=2)))
            fig.update_layout(title=f"{freq} kesimida signallar soni")
            show(fig)
        else:
            counts = ts.resample(rule).size()
            show(px.line(counts, template="plotly_dark", title=f"{freq} kesimida signallar soni",
                         color_discrete_sequence=[BLUE]))
    else:
        st.warning("⚠️ Datasetda sana/vaqt (datetime) formatidagi ustun aniqlanmadi.")
        st.info("Agar vaqt bo'yicha tahlil kerak bo'lsa, CSV faylingizga sana ustunini qo'shing yoki formatini to'g'rilang.")

# ---------- 6. Ma'lumotlar ----------
with tabs[5]:
    st.subheader("📋 Xom va filtrlangan ma'lumotlar")
    n_rows = st.slider("Ko'rsatiladigan qatorlar", 10, min(500, len(view)), min(20, len(view)))
    search_col = st.selectbox("Filtr uchun ustun (ixtiyoriy)", ["(yo'q)"] + cat_cols)
    filtered = view
    if search_col != "(yo'q)":
        q = st.text_input(f"'{search_col}' bo'yicha qidirish")
        if q:
            filtered = filtered[filtered[search_col].astype(str).str.contains(q, case=False, na=False)]
    st.dataframe(gold_table(filtered.head(n_rows)), use_container_width=True)
    st.download_button("⬇️ Ko'rinishni CSV sifatida yuklab olish",
                       filtered.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_signals.csv", mime="text/csv")
