# IMDB Sentiment Analysis using NLP

A Deep Learning web application built with **Simple RNN (Recurrent Neural Network)** and **Streamlit** to analyze and classify IMDB movie reviews as Positive or Negative.

## 🚀 Features
- **Interactive UI**: Clean Streamlit interface for entering and classifying movie reviews in real time.
- **NLP Preprocessing**: Tokenization and sequence padding using the IMDB dataset vocabulary.
- **Deep Learning Model**: Simple RNN architecture trained on the IMDB sentiment dataset.

## 📁 Repository Structure
- `main.py`: Streamlit application script.
- `simple_rnn_imdb.h5`: Trained Keras RNN model file.
- `simpleRNN.ipynb`: Training and model architecture notebook.
- `embedding.ipynb`: Embedding exploration notebook.
- `prediction.ipynb`: Testing and inference notebook.
- `requirements.txt`: Python package requirements.

## 🛠️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Vibhavan7/IMDB-sentiment-analysis-using-NLP.git
cd IMDB-sentiment-analysis-using-NLP
```

### 2. Set up virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run main.py
```
