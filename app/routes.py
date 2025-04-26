import os
import tempfile
import logging
from typing import Any, Dict

from flask import Flask, request, jsonify

# Core modules
from core.extractor import extract_pdf
from core.parser import ProspectusParser
from core.nlp import ProspectusNLP
from core.summarizer import ProspectusSummarizer

# Configure logger
logger = logging.getLogger(__name__)

def register_routes(app: Flask) -> None:
    """
    Register all API endpoints on the Flask app.
    """
    @app.route("/health", methods=["GET"])
    def health_check() -> Any:
        """Simple health check endpoint."""
        return jsonify({"status": "ok"}), 200

    @app.route("/summarize", methods=["POST"])
    def summarize() -> Any:
        """
        Accepts multipart/form-data with 'file' (PDF prospectus),
        returns JSON with extraction, parsing, NLP, and summary.
        """
        # 1. Validate upload
        if "file" not in request.files:
            return jsonify({"error": "Missing 'file' field"}), 400
        pdf_file = request.files["file"]
        if pdf_file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # 2. Save to temp file
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            pdf_path = tmp.name
            pdf_file.save(pdf_path)
            tmp.close()
            logger.info("Saved uploaded PDF to %s", pdf_path)
        except Exception as e:
            logger.exception("Failed saving uploaded file")
            return jsonify({"error": "Failed to save file", "details": str(e)}), 500

        # 3. Extraction
        try:
            extraction: Dict[str, Any] = extract_pdf(pdf_path)
            text = extraction.get("text", "")
            tables = extraction.get("tables", [])
            metadata = extraction.get("metadata", {})
        except Exception as e:
            logger.exception("Extraction error")
            os.unlink(pdf_path)
            return jsonify({"error": "Extraction failed", "details": str(e)}), 500

        # 4. Parsing
        try:
            parser = ProspectusParser()
            parsed = parser.parse_all(text)
        except Exception as e:
            logger.exception("Parsing error")
            os.unlink(pdf_path)
            return jsonify({"error": "Parsing failed", "details": str(e)}), 500

        # 5. NLP Analysis
        try:
            nlp_engine = ProspectusNLP(
                custom_phrases=[
                    "Penawaran Umum", "Saham Baru", "Waran Seri I"
                ],
                flash_keywords=[
                    "modal kerja", "capex", "EBITDA", "PER", "PBV"
                ]
            )
            nlp_results = nlp_engine.analyze(
                text=text,
                docs_for_tfidf=[text],
                fuzzy_choices=["likuiditas", "tenaga medis", "covenant"]
            )
        except Exception as e:
            logger.exception("NLP analysis error")
            os.unlink(pdf_path)
            return jsonify({"error": "NLP analysis failed", "details": str(e)}), 500

        # 6. Summarization
        try:
            summarizer = ProspectusSummarizer(
                hf_model_name="facebook/bart-large-cnn",
                use_openai=False
            )
            summary = summarizer.summarize_hf(text)
        except Exception as e:
            logger.exception("Summarization error")
            os.unlink(pdf_path)
            return jsonify({"error": "Summarization failed", "details": str(e)}), 500

        # 7. Clean up temp file
        try:
            os.unlink(pdf_path)
        except OSError:
            logger.warning("Temporary file %s could not be removed", pdf_path)

        # 8. Build and return response
        response = {
            "metadata": metadata,
            "tables": tables,
            "parsed": parsed,
            "nlp": nlp_results,
            "summary": summary
        }
        return jsonify(response), 200
