# 🎮 CNN-MDA Game Review

A Streamlit-based web application for **game review analysis** using a **Convolutional Neural Network (CNN)** to classify reviews into **Mechanics, Dynamics, and Aesthetics (MDA)** aspects, followed by **sentiment analysis** using the VADER lexicon.

This project was developed as part of an undergraduate thesis in Information Systems.

---

## 📌 Features

- Classify Steam game reviews into MDA aspects:
  - ⚙️ Mechanics
  - 🎮 Dynamics
  - 🎨 Aesthetics
- Perform sentiment analysis using VADER.
- Retrieve game reviews directly from the Steam API.
- Visualize:
  - MDA distribution
  - Sentiment distribution
  - Sentiment by MDA aspect
- Generate automatic insights based on the analysis results.

---

## 🛠️ Technologies Used

- Python
- Streamlit
- TensorFlow / Keras
- Pandas
- NumPy
- Altair
- Steam Web API
- VADER Sentiment

---

## 📂 Project Structure

```
CNN-MDA-game-review/
│
├── app.py
├── cnn_weights.weights.h5
├── tokenizer.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone this repository:

```bash
git clone https://github.com/YOUR_USERNAME/CNN-MDA-game-review.git
```

Move into the project directory:

```bash
cd CNN-MDA-game-review
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🎮 Supported Games

Currently, the application provides predefined support for:

- Elden Ring
- Dark Souls Remastered
- Sekiro: Shadows Die Twice
- Devil May Cry 5

Users can also analyze any Steam game by entering its App ID manually.

---

## 🧠 Model Architecture

The classification model uses a Convolutional Neural Network (CNN) consisting of:

- Embedding Layer
- Conv1D Layer
- Global Max Pooling Layer
- Dense Layer (ReLU)
- Softmax Output Layer

The model classifies each review into one of three MDA categories:

- Mechanics
- Dynamics
- Aesthetics

---

## 📊 Analysis Workflow

1. Retrieve reviews from the Steam API.
2. Convert reviews into sequences using the trained tokenizer.
3. Pad sequences to a fixed length.
4. Classify each review using the CNN model.
5. Perform sentiment analysis using VADER.
6. Display interactive charts and automatically generated insights.

---

## 📷 Application Preview

![CNN-MDA Dashboard](dashboard.png)

---

## 👨‍💻 Author

**Hanif Auzan Abdillah**

Undergraduate Student  
Information Systems  
Universitas Ahmad Dahlan

---

## 📄 License

This project is intended for academic and research purposes.
