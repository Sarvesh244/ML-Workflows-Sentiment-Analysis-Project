# 🏸 ML-Workflows-Sentiment-Analysis-Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Project Overview
This project focuses on classifying customer reviews from Flipkart as **Positive** or **Negative** and identifying the key pain points driving customer dissatisfaction. The specific focus is on the "YONEX MAVIS 350 Nylon Shuttle".

Unlike a standard notebook approach, this project implements a **structured MLOps workflow** using **MLflow**. This ensures that every experiment is tracked, reproducible, and comparable, allowing for a scientific approach to selecting the best model.

## 📊 Dataset
The dataset consists of **8,518 real-time reviews** scraped from Flipkart.
- **Product:** YONEX MAVIS 350 Nylon Shuttle
- **Key Features:** Review text, Ratings, Reviewer Name, Date, Up/Down Votes.
- **Target Variable:** Derived from `Ratings` (4-5 stars = Positive, 1-2 stars = Negative). Neutral ratings (3 stars) were excluded to focus on binary classification.

## 🛠 Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK
- **Experiment Tracking:** MLflow
- **Visualization:** Matplotlib / MLflow UI

## ⚙️ Workflow & Methodology

### 1. Data Preprocessing
- **Cleaning:** Removed HTML tags, special characters, and punctuation.
- **Normalization:** Applied lowercasing.
- **Stopword Logic:** Carefully tuned to **preserve negation words** (e.g., "not", "don't") to ensure phrases like "not good" are correctly classified as negative.
- **Handling Nulls:** Removed empty review rows to ensure data integrity.

### 2. Experimentation with MLflow
I conducted a series of controlled experiments to optimize model performance. Each run was logged with a specific naming convention to allow for easy comparison:

**Run Name Format:** `LR_C-1.0_Feat-5000`
- **`LR`**: Algorithm (Logistic Regression).
- **`C-1.0`**: Regularization strength (Inverse of regularization).
- **`Feat-5000`**: Max features (vocabulary size) used in the Bag-of-Words model.

 

### 3. Model Training & Results
I benchmarked multiple configurations:
- **Models:** Logistic Regression, Naive Bayes, Random Forest
- **Features:** Bag-of-Words (BoW) vs. TF-IDF
- **Hyperparameters:** Varied `C` values (0.1, 1.0, 10.0) and `max_features` (3000, 5000).

**🏆 Best Model:** Logistic Regression (BoW)
- **Accuracy:** ~92%
- **F1-Score:** ~0.96

### 4. Insights & Pain Points
Analysis of the model coefficients revealed the major drivers of negative sentiment:
- **Product Condition:** Keywords like **"damaged"**, **"old"**, **"dried"**.
- **Authenticity:** Keywords like **"fake"**, **"duplicate"**, **"cheated"**.
- **Price/Value:** Keywords like **"expensive"**, **"waste"**.



## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/Sarvesh244/ML-Workflows-Sentiment-Analysis-Project.git](https://github.com/Sarvesh244/ML-Workflows-Sentiment-Analysis-Project.git)
cd ML-Workflows-Sentiment-Analysis-Project

```

### 2. Set up Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the Training Script

This script will train the models and log all experiments to MLflow.

```bash
python train_model_mlflow.py

```

### 5. View MLflow Dashboard

To visualize the results and compare runs:

```bash
mlflow ui

```

Then open your browser to `http://127.0.0.1:5000`.

## 📂 Directory Structure

```
├── data.csv                 # Raw dataset
├── train_model_mlflow.py    # Main script for training and MLflow logging
├── requirements.txt         # Project dependencies
├── mlruns/                  # MLflow tracking data (generated automatically)
└── README.md                # Project documentation

```

## 🤝 Future Improvements

* **Deep Learning:** Integrate LSTM or BERT models to capture more complex context.
* **Dashboard:** Create a Streamlit dashboard to visualize sentiment trends over time.
* **Deployment:** Deploy the best model as a REST API using FastAPI or Flask.

---

*Created by Sarvesh*

```

```
