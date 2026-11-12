import io
import os
from pypdf import PdfReader
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.core.summarizer import MultilingualSummarizer

app = FastAPI(title="Summate AI - Multilingual NLP Summarizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

summarizer = MultilingualSummarizer()

class SummarizeRequest(BaseModel):
    text: str
    target_lang: str = "auto"
    length_mode: str = "standard"
    style: str = "paragraph"

@app.get("/")
def serve_ui():
    frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend/index.html"))
    return FileResponse(frontend_path)

@app.get("/health")
def health_check():
    return {"status": "online", "model": "mT5_multilingual_XLSum"}

@app.post("/api/summarize")
def summarize_text(req: SummarizeRequest):
    if not req.text or len(req.text.strip()) < 30:
        raise HTTPException(status_code=400, detail="Please enter at least 30 characters to summarize.")
    try:
        return summarizer.summarize(
            text=req.text,
            target_lang=req.target_lang,
            length_mode=req.length_mode,
            style=req.style
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize-pdf")
async def summarize_pdf(
    file: UploadFile = File(...),
    target_lang: str = Form("auto"),
    length_mode: str = Form("standard"),
    style: str = Form("paragraph")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        pdf_bytes = await file.read()
        reader = PdfReader(io.BytesIO(pdf_bytes))
        extracted_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        
        if len(extracted_text.strip()) < 30:
            raise HTTPException(status_code=400, detail="Could not extract readable text from this PDF.")
            
        result = summarizer.summarize(
            text=extracted_text,
            target_lang=target_lang,
            length_mode=length_mode,
            style=style
        )
        result["extracted_preview"] = extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else "")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")