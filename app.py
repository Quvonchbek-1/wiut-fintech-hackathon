import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

TEAM_ID = "C9B71210"
DATA_PATH = "train_signals.csv"
TARGET_HINTS = ["label", "target", "is_alert", "alert", "suspicious",
                "is_fraud", "fraud", "sar", "flag", "class", "y"]
MAX_POINTS = 20_000  # scatter uchun maksimal nuqtalar soni

st.set_page_config(page_title="WIUT Hackathon - AML EDA", page_icon="🚀", layout="wide")


# ---------------------------------------------------------------- yordamchilar
@st.cache_data(show_spinner="Ma'lumotlar yuklanmoqda...")
def load_csv(source):
    """source: fayl yo'li (str) yoki yuklangan fayl baytlari."""
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    df = pd.read_csv(source)
    # sana/vaqt ustunlarini avtomatik aniqlash
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
    """Target'ni 0/1 ga o'tkazadi. Nomaqbul bo'lsa None qaytaradi."""
    vals = s.dropna().unique()
    if len(vals) != 2:
        return None
    if set(vals) <= {0, 1, True, False}:
        return s.astype(float)
    minority = s.value_counts().idxmin()  # kam uchraydigan sinf = "alert" deb olinadi
    return (s == minority).astype(float)


def show(fig):
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------- sarlavha
st.title("📊 AML Alert Prioritization & Fintech EDA Dashboard")
st.markdown(f"**Jamoa ID:** `{TEAM_ID}` | WIUT Fintech Hackathon 2026")

# ---------------------------------------------------------------- ma'lumot
with st.sidebar:
    st.header("⚙️ Sozlamalar")
    uploaded = st.file_uploader("Boshqa CSV yuklash (ixtiyoriy)", type="csv")

try:
    df = load_csv(uploaded.getvalue() if uploaded else DATA_PATH)
except FileNotFoundError:
    st.error(f"`{DATA_PATH}` topilmadi. Faylni app.py yoniga qo'ying yoki chapdan yuklang.")
    st.stop()
except Exception as e:
    st.error(f"Faylni o'qishda xatolik: {e}")
    st.stop()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime"]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols and c not in dt_cols]

# target tanlash
with st.sidebar:
    guessed = guess_target(df)
    options = ["(yo'q)"] + df.columns.tolist()
    target_col = st.selectbox(
        "Target (alert/label) ustuni",
        options,
        index=options.index(guessed) if guessed else 0,
        help="Avtomatik topildi. Kerak bo'lsa o'zgartiring.",
    )

    # filtrlar
    st.subheader("🔎 Filtrlar")
    low_card = [c for c in cat_cols if df[c].nunique() <= 30][:5]
    for c in low_card:
        chosen = st.multiselect(c, sorted(df[c].dropna().astype(str).unique()))
        if chosen:
            df = df[df[c].astype(str).isin(chosen)]
    if num_cols:
        fcol = st.selectbox("Son bo'yicha filtr", ["(yo'q)"] + num_cols)
        if fcol != "(yo'q)":
            lo, hi = float(df[fcol].min()), float(df[fcol].max())
            if lo < hi:
                r = st.slider(fcol, lo, hi, (lo, hi))
                df = df[df[fcol].between(*r)]

if df.empty:
    st.warning("Filtrlardan keyin ma'lumot qolmadi.")
    st.stop()

y = None
if target_col != "(yo'q)":
    y = to_binary(df[target_col])
    if y is None:
        st.sidebar.warning("Target ustuni ikki qiymatli emas — target tahlili o'chirildi.")
        target_col = "(yo'q)"

# ---------------------------------------------------------------- metrikalar
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Jami signallar", f"{len(df):,}")
c2.metric("Ustunlar", len(df.columns))
c3.metric("Bo'sh qiymatlar", f"{df.isnull().mean().mean():.1%}")
c4.metric("Takroriy qatorlar", f"{df.duplicated().sum():,}")
c5.metric("Alert ulushi", f"{y.mean():.2%}" if y is not None else "—")
st.markdown("---")

tabs = st.tabs(["📌 Xulosa", "🧹 Sifat", "📈 Taqsimot", "🔗 Bog'liqlik",
                "🏷️ Kategoriyalar", "🕒 Vaqt", "📋 Jadval"])

# ---------------------------------------------------------------- 1. Xulosa
with tabs[0]:
    st.subheader("Loyiha haqida")
    st.markdown(
        "Dashboard AML (Anti-Money Laundering) signallarini ustuvorlashtirish uchun "
        "ma'lumotlarni chuqur o'rganish (EDA) imkonini beradi: sifat, taqsimot, "
        "bog'liqlik va target bilan aloqasi."
    )
    if y is not None:
        st.subheader("Target taqsimoti")
        vc = df[target_col].value_counts().reset_index()
        vc.columns = [target_col, "soni"]
        vc[target_col] = vc[target_col].astype(str)
        show(px.bar(vc, x=target_col, y="soni", text="soni", color=target_col,
                    template="plotly_dark"))
        if y.mean() < 0.1 or y.mean() > 0.9:
            st.info("⚠️ Sinflar nomutanosib (imbalanced). Modelda class weight / PR-AUC "
                    "ishlatish tavsiya etiladi.")
    st.subheader("Statistik ko'rsatkichlar")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

# ---------------------------------------------------------------- 2. Sifat
with tabs[1]:
    info = pd.DataFrame({
        "Ma'lumot turi": df.dtypes.astype(str),
        "Bo'sh soni": df.isnull().sum(),
        "Bo'sh %": (df.isnull().mean() * 100).round(2),
        "Noyob qiymatlar": df.nunique(),
    })
    st.dataframe(info, use_container_width=True)
    miss = info[info["Bo'sh soni"] > 0].reset_index().rename(columns={"index": "Ustun"})
    if miss.empty:
        st.success("Bo'sh qiymatlar yo'q ✅")
    else:
        show(px.bar(miss.sort_values("Bo'sh %"), x="Bo'sh %", y="Ustun",
                    orientation="h", template="plotly_dark", title="Bo'sh qiymatlar ulushi"))
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if const:
        st.warning(f"Doimiy (foydasiz) ustunlar: {', '.join(const)}")

# ---------------------------------------------------------------- 3. Taqsimot
with tabs[2]:
    if not num_cols:
        st.info("Raqamli ustunlar yo'q.")
    else:
        colA, colB, colC = st.columns([2, 1, 1])
        col = colA.selectbox("Ustun", num_cols, key="dist_col")
        bins = colB.slider("Bins", 10, 100, 40)
        logy = colC.checkbox("Log shkala (Y)", value=False)
        color = df[target_col].astype(str) if y is not None else None
        fig = px.histogram(df, x=col, nbins=bins, color=color, barmode="overlay",
                           opacity=0.7, template="plotly_dark", title=f"{col} taqsimoti")
        if logy:
            fig.update_yaxes(type="log")
        show(fig)
        show(px.box(df, x=color, y=col, color=color, template="plotly_dark",
                    title=f"{col} — box plot") if y is not None
             else px.box(df, y=col, template="plotly_dark", title=f"{col} — box plot"))

# ---------------------------------------------------------------- 4. Bog'liqlik
with tabs[3]:
    if len(num_cols) < 2:
        st.info("Kamida 2 ta raqamli ustun kerak.")
    else:
        corr = df[num_cols].corr()
        show(px.imshow(corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                       color_continuous_scale="RdBu_r", template="plotly_dark",
                       title="Korrelyatsiya matritsasi"))

        if y is not None:
            st.subheader("Targetga eng bog'liq ustunlar")
            tc = df[[c for c in num_cols if c != target_col]].corrwith(y).dropna()
            tc = tc.reindex(tc.abs().sort_values(ascending=False).index).head(15)
            show(px.bar(tc[::-1], orientation="h", template="plotly_dark",
                        labels={"value": "Korrelyatsiya", "index": "Ustun"}))

        st.subheader("Scatter")
        x_col = st.selectbox("X", num_cols, index=0, key="sx")
        y_col = st.selectbox("Y", num_cols, index=min(1, len(num_cols) - 1), key="sy")
        sample = df.sample(min(len(df), MAX_POINTS), random_state=42)
        if len(df) > MAX_POINTS:
            st.caption(f"Tezlik uchun {MAX_POINTS:,} ta tasodifiy nuqta ko'rsatilmoqda.")
        show(px.scatter(sample, x=x_col, y=y_col,
                        color=sample[target_col].astype(str) if y is not None else None,
                        opacity=0.6, template="plotly_dark"))

# ---------------------------------------------------------------- 5. Kategoriyalar
with tabs[4]:
    if not cat_cols:
        st.info("Kategorik ustunlar yo'q.")
    else:
        col = st.selectbox("Ustun", cat_cols, key="cat_col")
        top_n = st.slider("Top N", 3, 30, 10)
        counts = df[col].astype(str).value_counts().head(top_n).reset_index()
        counts.columns = [col, "soni"]
        show(px.bar(counts, x=col, y="soni", template="plotly_dark", title=f"{col} — Top {top_n}"))
        if y is not None:
            tmp = pd.DataFrame({col: df[col].astype(str), "y": y})
            rate = (tmp.groupby(col)["y"].agg(["mean", "count"])
                    .query("count >= 5").sort_values("mean", ascending=False).head(top_n)
                    .reset_index())
            if not rate.empty:
                show(px.bar(rate, x=col, y="mean", hover_data=["count"], template="plotly_dark",
                            labels={"mean": "Alert ulushi"},
                            title=f"{col} bo'yicha alert ulushi (kamida 5 ta kuzatuv)"))

# ---------------------------------------------------------------- 6. Vaqt
with tabs[5]:
    if not dt_cols:
        st.info("Sana/vaqt ustuni topilmadi.")
    else:
        col = st.selectbox("Sana ustuni", dt_cols)
        freq = st.radio("Davr", ["D", "W", "M"], horizontal=True,
                        format_func=lambda f: {"D": "Kun", "W": "Hafta", "M": "Oy"}[f])
        ts = df.set_index(col)
        cnt = ts.resample(freq).size().rename("signallar")
        show(px.line(cnt, template="plotly_dark", title="Signallar soni"))
        if y is not None:
            rt = y.set_axis(df[col]).resample(freq).mean().rename("alert ulushi")
            show(px.line(rt, template="plotly_dark", title="Alert ulushi dinamikasi"))

# ---------------------------------------------------------------- 7. Jadval
with tabs[6]:
    n = st.slider("Ko'rsatiladigan qatorlar", 5, 200, 20)
    st.dataframe(df.head(n), use_container_width=True)
    st.download_button("⬇️ Filtrlangan CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_signals.csv", mime="text/csv")
