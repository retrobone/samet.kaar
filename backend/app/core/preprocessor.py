import re
from langdetect import detect, DetectorFactory
from indicnlp.tokenize import sentence_tokenize

DetectorFactory.seed = 0

SUPPORTED_LANGS = {
    "en": "English",
    "hi": "Hindi",
    "pa": "Punjabi",
    "ml": "Malayalam",
    "te": "Telugu",
    "ur": "Urdu"
}

def clean_text(text: str) -> str:
    """Remove excessive whitespace, newlines, and noisy artifacts."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def detect_language(text: str) -> str:
    """Detect ISO 639-1 language code; fallback to 'en' if uncertain."""
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED_LANGS else "en"
    except Exception:
        return "en"

def chunk_text(text: str, lang: str = "en", max_words: int = 350) -> list[str]:
    """Split long text into chunks at proper sentence boundaries."""
    cleaned = clean_text(text)
    
    if lang in ["hi", "pa", "ml", "te"]:
        sentences = sentence_tokenize.sentence_split(cleaned, lang=lang)
    else:
        sentences = re.split(r'(?<=[.!?।])\s+', cleaned)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if current_word_count + len(words) > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = words
            current_word_count = len(words)
        else:
            current_chunk.extend(words)
            current_word_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks