# 🎭 Unveiling Emotions with Social Media Sentiment Analysis 

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![Flask](https://img.shields.io/badge/Flask-Backend-green)

An advanced, production-grade platform that leverages **Deep Learning Neural Networks (Transformers)** and **Data Mining** to uncover complex emotions hidden within public and social media text streams. 

This system breaks past basic polarity algorithms, identifying multidimensional emotional arrays (Joy, Anger, Sadness, Fear, Surprise, Love) while processing live dynamic data points. Its stunning glassmorphism dashboard makes it highly applicable for **marketing agencies, public opinion monitoring, and behavioral research**.

---

## ⚡ Core Upgraded Architecture

1. **Neural AI Engine:** Utilizes Hugging Face Transformers (`distilbert/distilbert-base-uncased-finetuned-sst-2-english`) via PyTorch to parse heavy sentence structures contextually for ultra-high accuracy emotion detection.
2. **Apriori Data Mining:** Implements `mlxtend`'s Apriori Algorithms to automatically establish Confidence and Lift mathematical connections between keyword associations across large internet clusters.
3. **Live Internet Scraper:** Connects autonomously to live public RSS feeds (e.g., Reddit) to fetch, process, and map real-time sociological chatter.
4. **Premium UI & Reporting:** An immersive frontend engineered with modern glassmorphism, dynamic animations, dynamically rendered Chart.js, and instant `html2pdf.js` PDF reporting capabilities.

---

## ⚙️ System Requirements  
- Python 3.10+
- Internet Connectivity
- At least 3GB of free disk space (to store the local PyTorch Deep Learning Models!)

---

## 🛠️ Tool Requirements  
- **Backend Frame**: Python, Flask
- **Deep Learning**: PyTorch, Hugging Face `transformers`
- **Analytics Models**: `mlxtend`, `scikit-learn`, Pandas
- **Live Scraper**: Python `requests`, XML ElementTree
- **Frontend**: Tailwind CSS, Chart.js, Vanilla JS, HTML2PDF

---

## 🚀 Installation & Setup

1. **Clone the repository.**
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```
3. **Activate the Environment**:
   - On Windows: `.\venv\Scripts\activate`
   - On Mac/Linux: `source venv/bin/activate`
4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Launch the Engine**:
   ```bash
   python app.py
   ```
   *Note: During the very first launch, it may take 1-3 minutes for the `transformers` pipeline to download the DistilBERT model onto your machine.*
6. **Access the Application**:
   Navigate your browser to `http://localhost:5000/`.

---

## 📊 Available Usage Flows

- **Live Internet Scan:** Type a trending topic (ex: "OpenAI") into the UI to scrape dynamic chatter from Reddit and score the sentiment instantly.
- **Neural Text Processor:** Test single complex statements against the neural network.
- **Batch CSV Processing:** Upload a dataset containing a `text` column (like the included `tweets.csv` file) and extract both sentiment distributions and Data Mining Apriori Rules. 
- **Export Report:** Instantly grab an executive PDF screenshot of your findings mapping the mathematical computations!

---

## 📱 Future Enhancements
- Fine-Tuning the local Transformer model explicitly on customized brand dataset databases.
- Integration of a database (MongoDB / PostgreSQL) for historical query tracking.
