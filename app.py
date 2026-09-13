import os, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="FYP Predictor", page_icon="🚀", layout="wide")

@st.cache_data
def load():
    f = pd.read_csv(os.path.join(BASE, "tiktok_funny_hashtag_videos.csv"))
    t = pd.read_csv(os.path.join(BASE, "trending_videos.csv"))
    return f, t

funny, trending = load()

def hashtags(s):
    return len(re.findall(r"#\w+", str(s)))

def features(df):
    x = df.copy()
    x["duration"] = pd.to_numeric(x["video_duration"], errors="coerce").fillna(0)
    x["followers"] = pd.to_numeric(x["author_followerCount"], errors="coerce").fillna(0)
    x["hearts"] = pd.to_numeric(x["author_heartCount"], errors="coerce").fillna(0)
    x["shares"] = pd.to_numeric(x["video_shareCount"], errors="coerce").fillna(0)
    x["comments"] = pd.to_numeric(x["video_commentCount"], errors="coerce").fillna(0)
    x["desc_len"] = x["video_desc"].fillna("").astype(str).str.len()
    x["hashtag_count"] = x["video_desc"].fillna("").astype(str).apply(hashtags)
    # Proxy engagement signals that can be known/estimated for a candidate post
    x["share_comment"] = np.log1p(x["shares"] + x["comments"])
    x["follower_log"] = np.log1p(x["followers"])
    return x

Xdf = features(funny)

# Define a practical FYP-like target from the observed distribution:
# top 20% by plays = "high discovery potential".
threshold = Xdf["video_playCount"].quantile(.80)
Xdf["fyp_target"] = (Xdf["video_playCount"] >= threshold).astype(int)

cols = ["duration","followers","hearts","shares","comments","desc_len","hashtag_count","share_comment","follower_log"]
X = Xdf[cols].replace([np.inf,-np.inf],np.nan).fillna(0)
y = Xdf["fyp_target"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
model = RandomForestClassifier(
    n_estimators=400, max_depth=8, min_samples_leaf=3,
    class_weight="balanced", random_state=42
)
model.fit(Xtr,ytr)
proba = model.predict_proba(Xte)[:,1]
auc = roc_auc_score(yte, proba)
acc = accuracy_score(yte, (proba >= .5).astype(int))

def fmt(n):
    n=float(n)
    if n>=1e9: return f"{n/1e9:.1f}B"
    if n>=1e6: return f"{n/1e6:.1f}M"
    if n>=1e3: return f"{n/1e3:.1f}K"
    return f"{n:,.0f}"

st.markdown(
    """
    <style>
        :root {
            --bg: #f5f7fb;
            --panel: #ffffff;
            --soft: #f1f5f9;
            --line: #e2e8f0;
            --text: #0f172a;
            --muted: #475569;
            --primary: #2563eb;
            --primary-soft: #dbeafe;
            --pink: #ec4899;
            --green: #16a34a;
            --amber: #f59e0b;
            --red: #ef4444;
        }
        .stApp {
            background: var(--bg);
            color: var(--text);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        div[data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid var(--line);
        }
        .stSidebar > div {
            padding-top: 1rem;
        }
        h1, h2, h3, h4, p, div, span, label {
            color: var(--text) !important;
        }
        .stTabs [role="tablist"] {
            gap: 0.6rem;
            margin-bottom: 1.4rem;
        }
        .stTabs [role="tab"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 0.8rem;
            padding: 0.6rem 1rem;
            color: var(--muted);
            font-weight: 600;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #eff6ff, #fdf2f8);
            border-color: rgba(37,99,235,0.18);
            color: var(--primary);
        }
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: var(--text) !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.8rem;
            color: var(--muted) !important;
        }
        .stButton > button {
            border-radius: 0.75rem;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            border: none;
            font-weight: 700;
            padding: 0.75rem 1.1rem;
            box-shadow: 0 8px 18px rgba(37,99,235,0.18);
        }
        .stButton > button:hover {
            filter: brightness(1.03);
        }
        .stProgress > div > div {
            background: linear-gradient(90deg, #22c55e, #3b82f6, #8b5cf6);
        }
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea > div > textarea,
        .stSelectbox > div > div > div,
        .stSlider > div > div {
            background: white;
            color: var(--text);
            border: 1px solid var(--line);
            border-radius: 0.75rem;
        }
        .css-1d391kg, .css-2trqyj {
            color: var(--text) !important;
        }
        .block-container .stAlert {
            background: #f8fafc;
            border: 1px solid var(--line);
            border-radius: 0.9rem;
            color: var(--text);
        }
        .stDataFrame {
            background: white;
            border-radius: 0.9rem;
            border: 1px solid var(--line);
        }
        .stDataFrame * {
            color: var(--text) !important;
        }
        .element-container > div {
            border-radius: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("🚀 FYP Predictor")
st.sidebar.caption("Probabilitas konten masuk FYP berdasarkan pola dataset historis.")
st.sidebar.metric("Dataset", f"{len(Xdf):,} video")
st.sidebar.metric("Threshold FYP", fmt(threshold), "plays")
st.sidebar.caption(f"Model validation • AUC {auc:.2f} • Accuracy {acc:.2f}")
st.sidebar.divider()

st.title("🎯 Prediksi Potensi FYP")
st.caption("Penilaian didasarkan pada kemiripan pola konten Anda dengan video yang masuk kategori top 20% performa dataset.")

pred_tab, pattern_tab, ranking_tab = st.tabs(["Prediksi", "Pola Dataset", "Ranking"])

with pred_tab:
    with st.form("fyp_form"):
        lcol, rcol = st.columns([1.5, 1.2])

        with lcol:
            st.markdown("### Data konten")
            duration = st.slider("Durasi video (detik)", 1, 180, 15)
            followers = st.number_input("Followers creator", 0, 100_000_000, 100_000, 10_000)
            hearts = st.number_input("Total hearts creator", 0, 1_000_000_000, 1_000_000, 50_000)
            shares = st.number_input("Estimasi shares", 0, 10_000_000, 500, 50)
            comments = st.number_input("Estimasi comments", 0, 10_000_000, 300, 50)
            caption = st.text_area("Caption + hashtag", "#fyp #viral #funny", height=120)
            submitted = st.form_submit_button("Prediksi sekarang", use_container_width=True)

        with rcol:
            st.markdown("### Hasil evaluasi")
            if submitted:
                desc_len = len(caption)
                tag_count = hashtags(caption)

                row = pd.DataFrame([{
                    "duration": duration,
                    "followers": followers,
                    "hearts": hearts,
                    "shares": shares,
                    "comments": comments,
                    "desc_len": desc_len,
                    "hashtag_count": tag_count,
                    "share_comment": np.log1p(shares + comments),
                    "follower_log": np.log1p(followers)
                }])[cols]

                score = float(model.predict_proba(row)[0, 1] * 100)
                prob = float(model.predict_proba(row)[0, 1])

                if score >= 75:
                    label, msg, color = "🔥 Sangat Berpeluang", "Pola konten sangat mirip dengan video yang tampil di kelompok performa tinggi.", "#16a34a"
                elif score >= 55:
                    label, msg, color = "🟢 Berpotensi", "Ada sinyal kuat, tetapi masih perlu optimasi agar lebih dekat dengan pola FYP.", "#2563eb"
                elif score >= 35:
                    label, msg, color = "🟡 Sedang", "Belum cukup kuat untuk dipastikan masuk FYP berdasarkan dataset saat ini.", "#f59e0b"
                else:
                    label, msg, color = "🔴 Rendah", "Sinyal performa tinggi masih lemah pada input saat ini.", "#ef4444"

                st.markdown(
                    f"""
                    <div style="padding: 1.2rem; border-radius: 1rem; background: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 10px 30px rgba(15,23,42,0.06); margin-bottom: 1rem;">
                        <div style="font-size: 0.75rem; color: #475569; letter-spacing: 0.08em; text-transform: uppercase;">Probabilitas FYP</div>
                        <div style="font-size: 3rem; font-weight: 800; color: {color}; margin: 0.5rem 0; line-height: 1.1;">{score:.0f}%</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a;">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(min(score / 100, 1.0))
                st.info(msg)

                c1, c2, c3 = st.columns(3)
                c1.metric("Probabilitas", f"{prob:.2%}")
                c2.metric("Caption", f"{desc_len} karakter")
                c3.metric("Hashtag", f"{tag_count} tag")

                st.subheader("Insight optimasi")
                tips = []
                if duration > 45:
                    tips.append("Durasi video Anda cukup panjang. Coba uji versi 15–30 detik agar hook lebih cepat masuk.")
                elif duration < 5:
                    tips.append("Video terlalu pendek bisa kehilangan konteks. Pastikan ada hook yang jelas di awal.")
                if tag_count == 0:
                    tips.append("Tambahkan hashtag yang relevan dengan niche agar pola pencarian lebih kuat.")
                if shares + comments < 10:
                    tips.append("Perkuat CTA agar audiens terdorong membagikan atau meninggalkan komentar.")
                if desc_len < 15:
                    tips.append("Caption terlalu pendek. Tambahkan konteks atau rasa penasaran agar lebih menarik.")
                if not tips:
                    tips.append("Input Anda sudah masuk pola yang kuat. Coba variasikan hook, thumbnail, dan intro.")
                for t in tips:
                    st.write("•", t)

                st.caption("Catatan: label 'FYP' di sini adalah proxy analitik dari dataset top 20% plays, bukan data internal TikTok.")
            else:
                st.markdown(
                    """
                    <div style="padding: 1.1rem; border-radius: 1rem; background: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 8px 20px rgba(15,23,42,0.04);">
                        <div style="font-size: 0.75rem; color: #475569; letter-spacing: 0.08em; text-transform: uppercase;">Status</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-top: 0.5rem; line-height: 1.1;">Siap diprediksi</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.info("Isi form di sebelah kiri lalu klik tombol prediksi untuk melihat hasilnya.")

with pattern_tab:
    st.subheader("Pola video dengan performa tinggi")
    x = Xdf.copy()
    x["kategori"] = np.where(x["fyp_target"] == 1, "Top 20% plays", "80% lainnya")

    c1, c2, c3 = st.columns(3)
    c1.metric("Batas top 20%", fmt(threshold), "plays")
    c2.metric("Total video", f"{len(x):,}")
    c3.metric("Persentase kandidat", "20%")

    fig = px.box(x, x="kategori", y="duration", title="Durasi vs kelompok performa")
    st.plotly_chart(fig, use_container_width=True)

    importance = pd.DataFrame({
        "Feature": cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig_importance = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Fitur yang paling memengaruhi prediksi",
        color="Importance",
        color_continuous_scale="Blues"
    )
    fig_importance.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white")
    st.plotly_chart(fig_importance, use_container_width=True)

    summary = x.groupby("kategori")[cols[:7]].mean().T
    summary.columns = ["80% lainnya", "Top 20% plays"] if len(summary.columns) == 2 else summary.columns
    st.dataframe(summary.round(2), use_container_width=True)

with ranking_tab:
    st.subheader("Konten dengan sinyal performa tertinggi")
    x = Xdf.copy()
    x["fyp_score"] = model.predict_proba(x[cols].replace([np.inf, -np.inf], np.nan).fillna(0))[:, 1] * 100
    rank = x.sort_values("fyp_score", ascending=False).head(25)
    cols_show = ["author_nickname", "video_desc", "video_duration", "video_playCount", "video_shareCount", "video_commentCount", "fyp_score"]
    st.dataframe(rank[cols_show].rename(columns={
        "author_nickname": "Creator",
        "video_desc": "Caption",
        "video_duration": "Durasi",
        "video_playCount": "Plays",
        "video_shareCount": "Shares",
        "video_commentCount": "Comments",
        "fyp_score": "FYP Score",
    }), use_container_width=True, hide_index=True)
