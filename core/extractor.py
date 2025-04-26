# extractor.py

import io
import os
import logging
from typing import List, Dict, Optional

import pdfplumber        # PDF Plumber: teks & tabel :contentReference[oaicite:3]{index=3}
import fitz              # PyMuPDF: Page.get_text(...) :contentReference[oaicite:4]{index=4}
from pdf2image import convert_from_path  # Konversi halaman jadi gambar :contentReference[oaicite:5]{index=5}
import pytesseract       # OCR wrapper Tesseract :contentReference[oaicite:6]{index=6}

# ----------------------------------------
# Logger Setup
# ----------------------------------------
def setup_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logger("extractor")

# ----------------------------------------
# PDF Metadata Extraction
# ----------------------------------------
def extract_metadata(pdf_path: str) -> Dict[str, str]:
    """Ambil metadata PDF seperti author, title, creator."""
    try:
        with fitz.open(pdf_path) as doc:
            meta = doc.metadata
            logger.info("Metadata extracted: %s", {k: meta[k] for k in meta if meta[k]})
            return meta
    except Exception as e:
        logger.warning("Metadata extraction failed: %s", e)
        return {}

# ----------------------------------------
# Text Extraction: pdfplumber
# ----------------------------------------
def extract_text_pdfplumber(pdf_path: str) -> str:
    """Ekstrak semua teks dengan pdfplumber (mesin-generated PDF)."""
    text_chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                txt = page.extract_text() or ""
                text_chunks.append(txt)
                logger.debug("pdfplumber page %d length: %d chars", i, len(txt))
    except Exception as e:
        logger.error("pdfplumber failed: %s", e)
    return "\n".join(text_chunks)

# ----------------------------------------
# Text Extraction: PyMuPDF
# ----------------------------------------
def extract_text_pymupdf(pdf_path: str) -> str:
    """Ekstrak semua teks dengan PyMuPDF; faster, block‐aware."""
    text_chunks = []
    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                # "text" sort=True untuk reading order :contentReference[oaicite:7]{index=7}
                txt = page.get_text("text", sort=True)
                text_chunks.append(txt)
                logger.debug("PyMuPDF page %d length: %d chars", i, len(txt))
    except Exception as e:
        logger.error("PyMuPDF failed: %s", e)
    return "\n".join(text_chunks)

# ----------------------------------------
# OCR Fallback for Scanned PDFs
# ----------------------------------------
def ocr_text(pdf_path: str,
             dpi: int = 300,
             ocr_lang: str = "eng") -> str:
    """
    OCR tiap halaman:
    1) convert_from_path → list PIL.Images :contentReference[oaicite:8]{index=8}
    2) pytesseract.image_to_string → teks :contentReference[oaicite:9]{index=9}
    """
    text_chunks = []
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        for idx, img in enumerate(images):
            txt = pytesseract.image_to_string(img, lang=ocr_lang)
            text_chunks.append(txt)
            logger.debug("OCR page %d length: %d chars", idx, len(txt))
    except Exception as e:
        logger.error("OCR fallback failed: %s", e)
    return "\n".join(text_chunks)

# ----------------------------------------
# Table Extraction (optional)
# ----------------------------------------
def extract_tables(pdf_path: str) -> List[List]:
    """Ekstrak semua tabel dengan pdfplumber."""
    all_tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tbl in tables:
                    all_tables.append(tbl)
        logger.info("Tables extracted: %d", len(all_tables))
    except Exception as e:
        logger.warning("Table extraction failed: %s", e)
    return all_tables

# ----------------------------------------
# Unified Extractor
# ----------------------------------------
def extract_pdf(pdf_path: str,
                prefer_ocr: bool = False) -> Dict[str, Optional[str]]:
    """
    Orkestrasi:
    1) Metadata
    2) pdfplumber → jika kosong atau prefer_ocr → fallback ke PyMuPDF → OCR
    3) Tables
    """
    result = {"metadata": None, "text": None, "tables": None}
    result["metadata"] = extract_metadata(pdf_path)

    # Utama: pdfplumber
    text = extract_text_pdfplumber(pdf_path)
    if not text.strip() or prefer_ocr:
        logger.info("pdfplumber empty or OCR preferred; trying PyMuPDF")
        text = extract_text_pymupdf(pdf_path)

    # Jika masih kosong, OCR
    if not text.strip():
        logger.info("No text found; invoking OCR fallback")
        text = ocr_text(pdf_path)

    result["text"] = text
    result["tables"] = extract_tables(pdf_path)
    return result

# ----------------------------------------
# CLI / Contoh Penggunaan
# ----------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract text/tables/metadata from PDF")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--ocr", action="store_true", help="Force OCR fallback")
    args = parser.parse_args()

    output = extract_pdf(args.pdf, prefer_ocr=args.ocr)
    # Simpan hasil ke file .txt & .json
    base, _ = os.path.splitext(args.pdf)
    txt_path = f"{base}_extracted.txt"
    json_path = f"{base}_tables.json"

    with io.open(txt_path, "w", encoding="utf8") as f:
        f.write(output["text"])
    logger.info("Extracted text saved to %s", txt_path)

    # Tabel disimpan dalam JSON sederhana
    try:
        import json
        with io.open(json_path, "w", encoding="utf8") as f:
            json.dump(output["tables"], f, ensure_ascii=False, indent=2)
        logger.info("Extracted tables saved to %s", json_path)
    except Exception as e:
        logger.warning("Failed saving tables JSON: %s", e)