#!/usr/bin/env python3
"""
summarizer.py: Advanced financial text summarizer using Google Gemini API
with a local HuggingFace BART fallback. Outputs summary in TXT, JSON, and PDF.
"""

import requests
import time
import logging
import json
from fpdf import FPDF
import torch
from transformers import pipeline

# Configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s %(levelname)s:%(name)s: %(message)s")

class GeminiSummarizer:
    """
    Summarizer using Google Gemini API via HTTP requests.
    Retries on failure with exponential backoff.
    """
    def __init__(self, api_key, model="gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
    
    def summarize(self, text):
        """Call Gemini generateContent and return the summary text."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        params = {"key": self.api_key}
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": text}]}]}
        
        logger.info("Calling Gemini API for summarization...")
        for attempt in range(1, 4):  # up to 3 attempts
            try:
                response = requests.post(url, params=params, headers=headers, json=payload, timeout=10)
                response.raise_for_status()
                data = response.json()
                # Parse the first candidate's content parts
                parts = data["candidates"][0]["content"]["parts"]
                summary = "".join([p["text"] for p in parts])
                logger.info("Gemini summarization succeeded.")
                return summary
            except Exception as e:
                logger.error(f"Gemini API attempt {attempt} failed: {e}")
                if attempt < 3:
                    backoff = 2 ** (attempt - 1)
                    logger.info(f"Retrying after {backoff}s...")
                    time.sleep(backoff)
                else:
                    # After final attempt, propagate exception to trigger fallback
                    logger.error("Gemini API failed after 3 attempts.")
                    raise

class LocalSummarizer:
    """
    Summarizer using a local HuggingFace model (facebook/bart-large-cnn).
    Detects GPU/CPU and runs on appropriate device.
    """
    def __init__(self):
        # Detect device: 0 for first GPU, -1 for CPU
        self.device = 0 if torch.cuda.is_available() else -1
        model_name = "facebook/bart-large-cnn"
        logger.info(f"Loading local summarization model on device {self.device}...")
        # Initialize the pipeline
        self.summarizer = pipeline("summarization", model=model_name, device=self.device)
        logger.info("Local summarizer loaded.")
    
    def summarize(self, text):
        """Generate summary using the local BART model."""
        logger.info("Running local summarizer...")
        # You may set max_length/min_length as needed for financial text style
        result = self.summarizer(text, max_length=200, min_length=50, do_sample=False)
        # Extract the 'summary_text' from the pipeline output
        summary = result[0]['summary_text']
        logger.info("Local summarization completed.")
        return summary

class SummarizationEngine:
    """
    Orchestrates summarization: tries Gemini API first, falls back to local model.
    Saves output to TXT, JSON, and PDF.
    """
    def __init__(self, api_key):
        self.api_summarizer = GeminiSummarizer(api_key)
        self.local_summarizer = LocalSummarizer()
    
    def summarize(self, text):
        """
        Perform summarization with fallback. Returns the summary text.
        """
        try:
            # Try Google Gemini first
            summary = self.api_summarizer.summarize(text)
        except Exception:
            # On failure, use local summarizer
            logger.info("Falling back to local summarizer...")
            summary = self.local_summarizer.summarize(text)
        
        # Save outputs
        self._save_txt(summary)
        self._save_json(summary)
        self._save_pdf(summary)
        
        return summary
    
    def _save_txt(self, summary):
        """Save the summary to a TXT file."""
        try:
            with open("summary.txt", "w", encoding="utf-8") as f:
                f.write(summary)
            logger.info("Saved summary to summary.txt")
        except Exception as e:
            logger.error(f"Failed to save TXT: {e}")
    
    def _save_json(self, summary):
        """Save the summary to a JSON file."""
        try:
            with open("summary.json", "w", encoding="utf-8") as f:
                json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)
            logger.info("Saved summary to summary.json")
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
    
    def _save_pdf(self, summary):
        """Save the summary to a PDF using FPDF (Latin-1 encoding)."""
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            # Write text; handle encoding errors by replacement
            try:
                pdf.multi_cell(0, 10, summary)
            except Exception:
                safe_text = summary.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 10, safe_text)
            pdf.output("summary.pdf")
            logger.info("Saved summary to summary.pdf")
        except Exception as e:
            logger.error(f"Failed to save PDF: {e}")

# Example usage (for demonstration; in practice integrate with your input pipeline)
if __name__ == "__main__":
    # Load input text from a file or define inline
    try:
        text = open("prospectus.txt", "r", encoding="utf-8").read()
    except FileNotFoundError:
        text = input("Enter text to summarize: ")
    api_key = "AIzaSyBOkgDdN-NkjOegEwO_UcuTlgU2NloaJgs"
    
    engine = SummarizationEngine(api_key)
    summary = engine.summarize(text)
    logger.info("Summary process completed.")
