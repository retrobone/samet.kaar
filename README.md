# samet.kaar — Multilingual Text Summarizer

A web-based NLP tool for summarizing text and PDF documents in English, Hindi, Punjabi, Malayalam, Telugu, and Urdu.

## How It Works
- **Preprocessing (`preprocessor.py`):** Detects language using `langdetect` and splits long text into 350-word chunks at sentence boundaries using `indic-nlp-library`.
- **Summarization (`summarizer.py`):** Uses `csebuetnlp/mT5_multilingual_XLSum` for abstractive summaries and `scikit-learn` TF-IDF vectorization for extractive sentence ranking and keyword extraction.
- **Backend & UI (`main.py`, `index.html`):** Built with FastAPI, `pypdf` for PDF text extraction, and Tailwind CSS.

## Setup & Run
1. `cd backend`
2. `pip install -r requirements.txt`
3. `uvicorn app.main:app --host 127.0.0.1 --port 8000`
4. Open `http://localhost:8000`