import streamlit as st
import requests
import pandas as pd
import altair as alt
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Game Review MDA & Sentiment",
    page_icon="🎮",
    layout="wide"
)


# =========================================================
# MDA KEYWORDS
# =========================================================

mechanics_keywords = {
    "combat", "weapon", "weapons", "attack", "defense",
    "skill", "skills", "level", "leveling", "upgrade",
    "build", "stats", "inventory", "equipment", "movement",
    "input", "camera", "hit", "dodge", "roll", "parry",
    "damage", "sword", "magic", "spell", "armor", "shield",
    "stamina", "health", "mana"
}

dynamics_keywords = {
    "challenge", "challenging", "difficulty", "difficult",
    "hard", "easy", "tactics", "tactical", "strategy",
    "strategic", "playstyle", "exploration", "explore",
    "progression", "multiplayer", "coop", "co-op",
    "cooperation", "interaction", "interactions",
    "encounter", "encounters", "replayability", "replay"
}

aesthetics_keywords = {
    "graphics", "graphic", "visual", "visuals", "design",
    "art", "artstyle", "beautiful", "stunning", "gorgeous",
    "environment", "scenery", "landscape", "music", "sound",
    "audio", "soundtrack", "atmosphere", "immersive",
    "immersion", "story", "narrative", "lore", "cinematic",
    "cinematics", "fantasy", "aesthetic", "presentation"
}


# =========================================================
# VADER
# =========================================================

analyzer = SentimentIntensityAnalyzer()


# =========================================================
# GAME OPTIONS
# =========================================================

game_options = {
    "Elden Ring": 1245620,
    "Dark Souls Remastered": 570940,
    "Sekiro: Shadows Die Twice": 814380,
    "Devil May Cry 5": 601150
}


# =========================================================
# STEAM API
# =========================================================

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

        try:
            response = requests.get(
                url,
                params=params,
                timeout=30
            )

            response.raise_for_status()
            data = response.json()

        except requests.RequestException:
            break

        batch = data.get("reviews", [])

        if not batch:
            break

        for review in batch:

            text = review.get("review", "").strip()

            if text:
                reviews.append(text)

            if len(reviews) >= target:
                break

        new_cursor = data.get("cursor")

        if not new_cursor or new_cursor == cursor:
            break

        cursor = new_cursor

    return reviews[:target]


# =========================================================
# PREPROCESSING UNTUK MDA
# =========================================================

def preprocess_for_mda(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    tokens = text.split()

    return tokens


# =========================================================
# MDA SCORING
# =========================================================

def score_mda(tokens):

    mechanics_score = sum(
        1 for word in tokens
        if word in mechanics_keywords
    )

    dynamics_score = sum(
        1 for word in tokens
        if word in dynamics_keywords
    )

    aesthetics_score = sum(
        1 for word in tokens
        if word in aesthetics_keywords
    )

    return (
        mechanics_score,
        dynamics_score,
        aesthetics_score
    )


# =========================================================
# MDA LABEL
# =========================================================

def get_mda_label(
    mechanics_score,
    dynamics_score,
    aesthetics_score
):

    scores = {
        "Mechanics": mechanics_score,
        "Dynamics": dynamics_score,
        "Aesthetics": aesthetics_score
    }

    max_score = max(scores.values())

    # Tidak memiliki keyword MDA
    if max_score == 0:
        return None

    # Cek apakah terdapat skor tertinggi yang sama
    max_count = list(scores.values()).count(max_score)

    if max_count > 1:
        return None

    return max(
        scores,
        key=scores.get
    )


# =========================================================
# SENTIMENT
# =========================================================

def analyze_sentiment(text):

    scores = analyzer.polarity_scores(
        str(text)
    )

    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "Positive"

    elif compound <= -0.05:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return compound, sentiment


# =========================================================
# HEADER
# =========================================================

st.title("🎮 Game Review MDA & Sentiment Dashboard")

st.write(
    "Analisis ulasan game berdasarkan framework "
    "Mechanics, Dynamics, and Aesthetics (MDA) "
    "serta analisis sentimen menggunakan VADER."
)

st.divider()


# =========================================================
# GAME SELECTION
# =========================================================

st.subheader("🎮 Game Selection")

input_mode = st.radio(
    "Mode Input Game",
    [
        "Pilih dari daftar game",
        "Input App ID manual"
    ]
)

if input_mode == "Pilih dari daftar game":

    selected_game = st.selectbox(
        "Pilih Game",
        list(game_options.keys())
    )

    app_id = game_options[selected_game]

    st.success(
        f"{selected_game} (App ID: {app_id})"
    )

else:

    app_id = st.text_input(
        "Steam App ID",
        value="1245620"
    )

    try:
        app_id = int(app_id)

    except ValueError:
        st.error("Steam App ID harus berupa angka.")
        st.stop()


# =========================================================
# REVIEW MODE
# =========================================================

st.subheader("⚙️ Mode Analisis")

mode = st.radio(
    "Jumlah Review",
    [
        "Test (100 review)",
        "Full (5000 review)"
    ]
)

target = (
    100
    if mode == "Test (100 review)"
    else 5000
)

st.info(
    f"Target review: {target}"
)

st.divider()


# =========================================================
# RUN ANALYSIS
# =========================================================

if st.button(
    "🚀 Analisis Game",
    type="primary"
):

    # -----------------------------------------------------
    # GET REVIEWS
    # -----------------------------------------------------

    with st.spinner(
        "Mengambil review Steam..."
    ):

        reviews = get_steam_reviews(
            app_id,
            target
        )

    if len(reviews) == 0:

        st.error(
            "Review tidak ditemukan atau "
            "terjadi masalah saat mengambil data Steam."
        )

        st.stop()


    # -----------------------------------------------------
    # PROCESS DATA
    # -----------------------------------------------------

    results = []

    progress = st.progress(0)

    for i, review in enumerate(reviews):

        # Preprocessing untuk MDA
        tokens = preprocess_for_mda(
            review
        )

        # MDA scoring
        mechanics_score, dynamics_score, aesthetics_score = score_mda(
            tokens
        )

        # MDA label
        mda_label = get_mda_label(
            mechanics_score,
            dynamics_score,
            aesthetics_score
        )

        # Review tanpa label MDA tidak digunakan
        if mda_label is None:

            progress.progress(
                (i + 1) / len(reviews)
            )

            continue


        # Sentiment menggunakan teks asli
        compound, sentiment = analyze_sentiment(
            review
        )

        results.append({
            "Review": review,
            "MDA": mda_label,
            "Mechanics Score": mechanics_score,
            "Dynamics Score": dynamics_score,
            "Aesthetics Score": aesthetics_score,
            "Compound": compound,
            "Sentiment": sentiment
        })

        progress.progress(
            (i + 1) / len(reviews)
        )

    progress.empty()


    # -----------------------------------------------------
    # DATAFRAME
    # -----------------------------------------------------

    df = pd.DataFrame(results)


    if df.empty:

        st.error(
            "Tidak ada review yang berhasil "
            "memperoleh label MDA."
        )

        st.stop()


    # -----------------------------------------------------
    # DATA SUMMARY
    # -----------------------------------------------------

    total_collected = len(reviews)
    total_mda = len(df)

    neutral_count = (
        df["Sentiment"] == "Neutral"
    ).sum()

    df_final = df[
        df["Sentiment"].isin(
            ["Positive", "Negative"]
        )
    ].copy()

    total_final = len(df_final)


    # =====================================================
    # SUMMARY METRICS
    # =====================================================

    st.subheader("📌 Ringkasan Data")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Review Diambil",
        total_collected
    )

    col2.metric(
        "Berlabel MDA",
        total_mda
    )

    col3.metric(
        "Neutral",
        neutral_count
    )

    col4.metric(
        "Data Akhir",
        total_final
    )


    # =====================================================
    # MDA DISTRIBUTION
    # =====================================================

    st.subheader(
        "📊 Distribusi Aspek MDA"
    )

    mda_counts = (
        df["MDA"]
        .value_counts()
        .reindex(
            ["Mechanics", "Dynamics", "Aesthetics"],
            fill_value=0
        )
        .reset_index()
    )

    mda_counts.columns = [
        "MDA",
        "Count"
    ]

    chart_mda = (
        alt.Chart(mda_counts)
        .mark_arc(innerRadius=50)
        .encode(
            theta="Count:Q",
            color="MDA:N",
            tooltip=[
                "MDA:N",
                "Count:Q"
            ]
        )
    )

    st.altair_chart(
        chart_mda,
        use_container_width=True
    )


    # =====================================================
    # SENTIMENT DISTRIBUTION
    # =====================================================

    st.subheader(
        "❤️ Distribusi Sentimen"
    )

    sent_counts = (
        df["Sentiment"]
        .value_counts()
        .reindex(
            ["Positive", "Negative", "Neutral"],
            fill_value=0
        )
        .reset_index()
    )

    sent_counts.columns = [
        "Sentiment",
        "Count"
    ]

    chart_sent = (
        alt.Chart(sent_counts)
        .mark_arc(innerRadius=50)
        .encode(
            theta="Count:Q",
            color="Sentiment:N",
            tooltip=[
                "Sentiment:N",
                "Count:Q"
            ]
        )
    )

    st.altair_chart(
        chart_sent,
        use_container_width=True
    )


    # =====================================================
    # SENTIMENT PER MDA
    # =====================================================

    st.subheader(
        "📊 Sentimen Berdasarkan Aspek MDA"
    )

    pivot = (
        df_final
        .groupby(
            ["MDA", "Sentiment"]
        )
        .size()
        .reset_index(
            name="Count"
        )
    )

    chart_bar = (
        alt.Chart(pivot)
        .mark_bar()
        .encode(
            x=alt.X(
                "MDA:N",
                sort=[
                    "Mechanics",
                    "Dynamics",
                    "Aesthetics"
                ]
            ),
            y="Count:Q",
            color="Sentiment:N",
            tooltip=[
                "MDA:N",
                "Sentiment:N",
                "Count:Q"
            ]
        )
    )

    st.altair_chart(
        chart_bar,
        use_container_width=True
    )


    # =====================================================
    # SENTIMENT PERCENTAGE
    # =====================================================

    st.subheader(
        "📈 Persentase Sentimen per Aspek"
    )

    percentage = (
        pd.crosstab(
            df_final["MDA"],
            df_final["Sentiment"],
            normalize="index"
        )
        * 100
    )

    percentage = (
        percentage
        .reindex(
            ["Mechanics", "Dynamics", "Aesthetics"]
        )
        .reindex(
            columns=["Positive", "Negative"],
            fill_value=0
        )
        .round(2)
    )

    percentage_display = (
        percentage
        .reset_index()
        .melt(
            id_vars="MDA",
            var_name="Sentiment",
            value_name="Percentage"
        )
    )

    chart_percentage = (
        alt.Chart(percentage_display)
        .mark_bar()
        .encode(
            x=alt.X(
                "MDA:N",
                sort=[
                    "Mechanics",
                    "Dynamics",
                    "Aesthetics"
                ]
            ),
            y=alt.Y(
                "Percentage:Q",
                scale=alt.Scale(domain=[0, 100])
            ),
            color="Sentiment:N",
            tooltip=[
                "MDA:N",
                "Sentiment:N",
                alt.Tooltip(
                    "Percentage:Q",
                    format=".2f"
                )
            ]
        )
    )

    st.altair_chart(
        chart_percentage,
        use_container_width=True
    )


    # =====================================================
    # AUTO INSIGHT
    # =====================================================

    st.subheader(
        "🧠 Ringkasan Hasil Analisis"
    )

    if not df_final.empty:

        # Persentase positif
        positive_percentage = (
            len(
                df_final[
                    df_final["Sentiment"] == "Positive"
                ]
            )
            / len(df_final)
            * 100
        )

        # Persentase negatif
        negative_percentage = (
            len(
                df_final[
                    df_final["Sentiment"] == "Negative"
                ]
            )
            / len(df_final)
            * 100
        )


        # Persentase per MDA
        mda_percentage = (
            pd.crosstab(
                df_final["MDA"],
                df_final["Sentiment"],
                normalize="index"
            )
            * 100
        )


        positive_mda = (
            mda_percentage["Positive"]
            .idxmax()
        )

        negative_mda = (
            mda_percentage["Negative"]
            .idxmax()
        )


        insight_text = f"""
**Total data akhir:** {total_final} review

**Sentimen positif:** {positive_percentage:.2f}%

**Sentimen negatif:** {negative_percentage:.2f}%

**Aspek dengan persentase positif tertinggi:** {positive_mda} ({mda_percentage.loc[positive_mda, "Positive"]:.2f}%)

**Aspek dengan persentase negatif tertinggi:** {negative_mda} ({mda_percentage.loc[negative_mda, "Negative"]:.2f}%)
"""

        st.info(
            insight_text
        )


    # =====================================================
    # DATA TABLE
    # =====================================================

    with st.expander(
        "🔎 Lihat Data Hasil Analisis"
    ):

        st.dataframe(
            df_final[
                [
                    "Review",
                    "MDA",
                    "Compound",
                    "Sentiment"
                ]
            ],
            use_container_width=True
        )