import streamlit as st
import pandas as pd
import numpy as np

# Sahifa sozlamalari
st.set_page_config(
    page_title="WIUT Hackathon - Fintech EDA",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AML Alert Prioritization & Fintech EDA Dashboard")
st.markdown("**Jamoa ID:** `C9B71210` | WIUT Hackathon 2026")

# Ma'lumotlarni yuklash funksiyasi
@st.cache_data
def load_data():
    try:
        train_signals = pd.read_csv('train_signals.csv')
        return train_signals
    except Exception as e:
        return None

df = load_data()

if df is not None:
    st.success("Ma'lumotlar muvaffaqiyatli yuklandi!")
    
    # Asosiy metrikalar
    col1, col2, col3 = st.columns(3)
    col1.metric("Jami signallar soni", len(df))
    col2.metric("Ustunlar soni", len(df.columns))
    col3.metric("Jamoa ID", "C9B71210")
    
    st.subheader("🔍 Ma'lumotlar jadvali (Dataset Head)")
    st.dataframe(df.head(10))
    
    st.subheader("📈 Statistik tahlil (Describe)")
    st.write(df.describe())
else:
    st.warning("`train_signals.csv` fayli topilmadi. Iltimos, fayl papkada borligini tekshiring.")