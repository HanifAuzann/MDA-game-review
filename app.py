import streamlit as st
import pickle
import numpy as np
import requests
import pandas as pd
import altair as alt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =========================
# MODEL CNN
# =========================
model = Sequential([
    Embedding(input_dim=10000, output_dim=100),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.build(input_shape=(None, 100))
model.load_weights("cnn_weights.weights.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

analyzer = SentimentIntensityAnalyzer()

# =========================
# CONFIG
# =========================
MAX_LEN = 100

label_map = {
    0: "Aesthetics",
    1: "Dynamics",
    2: "Mechanics"
}

game_options = {
    "Elden Ring": 1245620,
    "Dark Souls Remastered": 570940,
    "Sekiro: Shadows Die Twice": 814380,
    "Devil May Cry 5": 601150
}

# =========================
# STEAM API
# =========================
def get_steam_reviews(app_id, target=100):
    cursor = "*"
    reviews = []

    while len(reviews) < target:

        url = f"https://store.steampowered.com/appreviews/{app_id}"

        params = {
            "json": 1,
            "language": "english",
            "filter": "recent",
            "num_per_page": 100,
            "cursor": cursor
        }

        response = requests.get(url, params=params)
        data = response.json()

        batch = data.get("reviews", [])
        if not batch:
            break

        for r in batch:
            reviews.append(r["review"])

        cursor = data["cursor"]

    return reviews[:target]

# =========================
# HEADER
# =========================
st.title("🎮 Game Review AI Dashboard (MDA + Sentiment)")
st.write("CNN + VADER based Steam review analytics system")

st.divider()

# =========================
# GAME INPUT
# =========================
st.subheader("🎮 Game Selection")

input_mode = st.radio(
    "Mode Input Game",
    ["Pilih dari daftar game", "Input App ID manual"]
)

if input_mode == "Pilih dari daftar game":

    selected_game = st.selectbox(
        "Pilih Game",
        list(game_options.keys())
    )

    app_id = game_options[selected_game]
    st.success(f"{selected_game} (ID: {app_id})")

else:
    app_id = st.text_input("Steam App ID", value="1245620")

# =========================
# MODE REVIEW
# =========================
st.subheader("⚙️ Mode Analisis")

mode = st.radio(
    "Jumlah Review",
    ["Test (100 review)", "Full (5000 review)"]
)

target = 100 if mode == "Test (100 review)" else 5000

st.info(f"Target review: {target}")

st.divider()

# =========================
# RUN ANALYSIS
# =========================
if st.button("🚀 Analisis Game"):

    with st.spinner("Mengambil review Steam..."):
        reviews = get_steam_reviews(app_id, target)

    if len(reviews) == 0:
        st.error("Review tidak ditemukan")

    else:

        results = []

        for r in reviews:

            seq = tokenizer.texts_to_sequences([r])
            padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

            pred = model.predict(padded, verbose=0)
            mda = label_map[np.argmax(pred)]

            sentiment_score = analyzer.polarity_scores(r)
            compound = sentiment_score["compound"]

            sentiment = (
                "Positive" if compound >= 0.05 else
                "Negative" if compound <= -0.05 else
                "Neutral"
            )

            results.append([mda, sentiment])

        df = pd.DataFrame(results, columns=["MDA", "Sentiment"])

        # =========================
        # PIE MDA
        # =========================
        st.subheader("📊 Distribusi MDA")

        mda_counts = df["MDA"].value_counts().reset_index()
        mda_counts.columns = ["MDA", "Count"]

        chart_mda = alt.Chart(mda_counts).mark_arc(innerRadius=50).encode(
            theta="Count",
            color="MDA",
            tooltip=["MDA", "Count"]
        )

        st.altair_chart(chart_mda, use_container_width=True)

        # =========================
        # PIE SENTIMENT
        # =========================
        st.subheader("❤️ Distribusi Sentiment")

        sent_counts = df["Sentiment"].value_counts().reset_index()
        sent_counts.columns = ["Sentiment", "Count"]

        chart_sent = alt.Chart(sent_counts).mark_arc(innerRadius=50).encode(
            theta="Count",
            color="Sentiment",
            tooltip=["Sentiment", "Count"]
        )

        st.altair_chart(chart_sent, use_container_width=True)

        # =========================
        # BAR CHART
        # =========================
        st.subheader("📊 Sentiment per MDA")

        pivot = df.groupby(["MDA", "Sentiment"]).size().reset_index(name="Count")

        chart_bar = alt.Chart(pivot).mark_bar().encode(
            x="MDA",
            y="Count",
            color="Sentiment",
            tooltip=["MDA", "Sentiment", "Count"]
        )

        st.altair_chart(chart_bar, use_container_width=True)

        # =========================
        # AUTO INSIGHT GENERATOR
        # =========================
        st.subheader("🧠 Auto Insight Generator")

        total_reviews = len(df)
        top_mda = df["MDA"].value_counts().idxmax()
        top_sent = df["Sentiment"].value_counts().idxmax()

        mda_dist = df["MDA"].value_counts(normalize=True) * 100
        sent_dist = df["Sentiment"].value_counts(normalize=True) * 100

        worst_sent = df[df["Sentiment"] == "Negative"]["MDA"].value_counts().idxmax() if "Negative" in df["Sentiment"].values else None
        best_sent = df[df["Sentiment"] == "Positive"]["MDA"].value_counts().idxmax() if "Positive" in df["Sentiment"].values else None

        insight_text = f"""
🎮 Total review dianalisis: {total_reviews}

📌 Aspek paling dominan: **{top_mda}** ({mda_dist[top_mda]:.2f}%)

❤️ Sentimen dominan: **{top_sent}** ({sent_dist[top_sent]:.2f}%)

"""

        if worst_sent:
            insight_text += f"⚠️ Aspek paling banyak dikritik (negative): **{worst_sent}**\n"

        if best_sent:
            insight_text += f"✨ Aspek paling disukai (positive): **{best_sent}**\n"

        if top_mda == "Dynamics" and top_sent == "Negative":
            insight_text += "\n⚠️ Kesimpulan: Gameplay banyak dibahas tetapi banyak kritik → balancing issue kemungkinan tinggi"

        elif top_mda == "Aesthetics" and top_sent == "Negative":
            insight_text += "\n⚠️ Kesimpulan: Visual/graphical experience menjadi titik lemah game"

        elif top_mda == "Mechanics" and top_sent == "Negative":
            insight_text += "\n⚠️ Kesimpulan: Sistem gameplay/controls perlu improvement"

        st.info(insight_text)