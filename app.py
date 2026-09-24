import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

TEAM_ID = "C9B71210"
DATA_PATH = "train_signals.csv"
TARGET_HINTS = ["label", "target", "is_alert", "alert", "suspicious", "is_fraud",
                "fraud", "sar", "flag", "class", "y", "eskalatsiya"]

st.set_page_config(page_title="WIUT Hackathon - AML AI Dashboard", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0a0e1a; color: #e5e7eb; font-family: 'Inter', sans-serif; }
.hero {
    padding: 24px; border-radius: 16px; margin-bottom: 20px;
    background: linear-gradient(135deg, rgba(91,124,250,.2), rgba(255,255,255,.02));
    border: 1px solid rgba(255,255,255,.08);
}
[data-testid="stMetricValue"] { color: #F5C451 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_csv(source):
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    return pd.read_csv(source)

# Sarlavha
st.markdown(f"""
<div class="hero">
    <span style="color: #93c5fd; font-size: 0.85rem;">WIUT Fintech Hackathon 2026 · AI in Finance Track</span>
    <h1 style="margin: 5px 0; color: #ffffff;">AML Alert Prioritization & AI Risk Dashboard</h1>
    <p style="margin: 0; color: #94a3b8;">Shubhali tranzaksiyalarni sun'iy intellekt yordamida ustuvorlashtirish tizimi</p>
    <div style="margin-top: 10px; display: inline-block; padding: 4px 10px; background: rgba(255,255,255,.05); border-radius: 6px; font-family: monospace;">Jamoa ID: {TEAM_ID}</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Sozlamalar")
    uploaded = st.file_uploader("Boshqa CSV yuklash", type="csv")

try:
    df = load_csv(uploaded.getvalue() if uploaded else DATA_PATH)
except Exception as e:
    st.error(f"Faylni yuklashda xatolik: {e}")
    st.stop()

# Targetni avtomatik topish
guessed = None
for h in TARGET_HINTS:
    for c in df.columns:
        if h in c.lower():
            guessed = c
            break
    if guessed: break

if not guessed:
    for c in df.columns:
        if df[c].nunique(dropna=True) == 2:
            guessed = c
            break

with st.sidebar:
    options = ["(yo'q)"] + df.columns.tolist()
    target_col = st.selectbox("Target (alert/risk) ustuni", options, index=options.index(guessed) if guessed and guessed in options else 0)

y = None
if target_col != "(yo'q)":
    vals = df[target_col].dropna().unique()
    if len(vals) == 2:
        minority = df[target_col].value_counts().idxmin()
        y = (df[target_col] == minority).astype(float)

# Metrikalar
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jami Signallar", f"{len(df):,}")
c2.metric("Ustunlar Soni", len(df.columns))
c3.metric("Haqiqiy Alertlar", f"{int(y.sum()):,}" if y is not None else "Aniqlanmadi")
c4.metric("Alert Ulushi", f"{y.mean():.2%}" if y is not None else "Aniqlanmadi")
st.markdown("---")

tabs = st.tabs(["📌 Xulosa", "🤖 AI Risk Prioritization", "📈 Taqsimot", "📋 Ma'lumotlar"])

with tabs[0]:
    st.subheader("💡 Loyiha haqida qisqacha")
    st.markdown("Ushbu tizim bank va moliya institutlari uchun AML (Anti-Money Laundering) qoidabuzarliklarini tezkor aniqlash va xavf darajasiga ko'ra birinchi o'ringa chiqarishga mo'ljallangan.")
    if y is not None:
        vc = df[target_col].astype(str).value_counts().reset_index()
        vc.columns = ["Sinf", "Soni"]
        fig = px.bar(vc, x="Sinf", y="Soni", color="Sinf", template="plotly_dark", title="Target taqsimoti")
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("🤖 AI Model va Risk Prioritization")
    if y is None:
        st.warning("⚠️ AI ishlashi uchun chap menyudan to'g'ri **Target** ustunini tanlang.")
    else:
        try:
            # Target'dan boshqa barcha ustunlarni feature sifatida olamiz
            features = [c for c in df.columns if c != target_col]
            
            if not features:
                st.error("Model uchun ustunlar topilmadi.")
            else:
                with st.spinner("AI model o'qitilmoqda va risklar baholanmoqda..."):
                    # Matnli/kategorik ustunlarni ham avtomatik raqamga o'tkazamiz (One-Hot Encoding)
                    X = pd.get_dummies(df[features]).astype(float).fillna(0)
                    
                    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight="balanced")
                    
                    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    oof = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
                    model.fit(X, y)
                    
                    roc = roc_auc_score(y, oof)
                    pr = average_precision_score(y, oof)
                
                k1, k2, k3 = st.columns(3)
                k1.metric("ROC-AUC", f"{roc:.3f}")
                k2.metric("PR-AUC", f"{pr:.3f}")
                k3.metric("Model Holati", "Faol 🟢")
                
                res_df = df.copy()
                res_df["AI_Risk_Score"] = oof
                res_df = res_df.sort_values(by="AI_Risk_Score", ascending=False)
                res_df.insert(0, "Rank", range(1, len(res_df) + 1))
                
                st.subheader("🚨 Eng yuqori xavfli signallar (Top Prioritized)")
                st.dataframe(res_df.head(25), use_container_width=True, hide_index=True)
                
                st.download_button("⬇️ Ustuvor ro'yxatni yuklab olish (CSV)", res_df.to_csv(index=False).encode("utf-8"), file_name="ai_scored_signals.csv", mime="text/csv")
        except Exception as ex:
            st.error(f"AI modelini ishga tushirishda xatolik: {ex}")

with tabs[2]:
    if len(df.columns) > 0:
        col = st.selectbox("Grafik uchun ustun", df.columns.tolist())
        fig = px.histogram(df, x=col, template="plotly_dark", title=f"{col} taqsimoti")
        st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("📋 Ma'lumotlar jadvali")
    n = st.slider("Qatorlar soni", 5, 100, 20)
    st.dataframe(df.head(n), use_container_width=True)
