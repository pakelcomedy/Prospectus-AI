from flask import Flask, request, jsonify, render_template, send_file
from fpdf import FPDF
from io import BytesIO
import os
import tempfile
import logging
import json
import torch                                  # for device detection
from typing import Any, Dict

from core.extractor import PDFExtractor
from core.parser    import ProspectusParser
from core.nlp       import ProspectusNLP
from core.summarizer import SummarizationEngine  # Engine yang integrasi Gemini + HuggingFace

# ----------------------------------------
# Logger Setup
# ----------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
)

# Pastikan direktori output ada
os.makedirs("output/", exist_ok=True)


def register_routes(app: Flask) -> None:
    @app.route("/", methods=["GET"])
    def index() -> Any:
        return render_template("index.html")

    @app.route("/health", methods=["GET"])
    def health_check() -> Any:
        return jsonify({"status": "ok"}), 200

    @app.route("/summarize", methods=["POST"])
    def summarize() -> Any:
        # 1) Validasi upload
        if "file" not in request.files:
            return jsonify({"error": "Missing 'file' field"}), 400
        pdf_file = request.files["file"]
        if not pdf_file.filename:
            return jsonify({"error": "No file selected"}), 400

        # 2) Simpan sementara
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            pdf_path = tmp.name
            pdf_file.save(pdf_path)
            tmp.close()
            logger.info("Saved uploaded PDF to %s", pdf_path)
        except Exception as e:
            logger.exception("Failed saving uploaded file")
            return jsonify({"error": "Failed to save file", "details": str(e)}), 500

        # 3) Ekstraksi
        try:
            extractor = PDFExtractor(
                input_pdf=pdf_path,
                output_dir="output/",
                force_ocr=False,
                ocr_languages="eng+ind",
                ocr_dpi=300,
                threads=4,
                auto_cleanup=False
            )
            extraction = extractor.extract()
            extractor.save_results()
            text     = extraction.get("text", "")
            tables   = extraction.get("tables", [])
            metadata = extraction.get("metadata", {})
        except Exception as e:
            logger.exception("Extraction error")
            os.unlink(pdf_path)
            return jsonify({"error": "Extraction failed", "details": str(e)}), 500

        # 4) Parsing
        try:
            parser = ProspectusParser()
            parsed = parser.parse_all(text)
        except Exception as e:
            logger.exception("Parsing error")
            os.unlink(pdf_path)
            return jsonify({"error": "Parsing failed", "details": str(e)}), 500

        # 5) NLP tambahan
        try:
            nlp_engine = ProspectusNLP(
                custom_phrases=["Penawaran Umum", "Saham Baru", "Waran Seri I"],
                flash_keywords=["modal kerja", "capex", "EBITDA", "PER", "PBV"]
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

        # 6) Summarization via Gemini → retry → fallback local
        try:
            api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBOkgDdN-NkjOegEwO_UcuTlgU2NloaJgs")
            engine = SummarizationEngine(api_key=api_key)
            # Terima satu paragraf ringkasan
            summary_text = engine.summarize(text)
        except Exception as e:
            logger.exception("Summarization error")
            os.unlink(pdf_path)
            return jsonify({"error": "Summarization failed", "details": str(e)}), 500

        # 7) Hapus file sementara
        try:
            os.unlink(pdf_path)
        except OSError:
            logger.warning("Temporary file %s could not be removed", pdf_path)

        # 8) Siapkan JSON response
        response_data: Dict[str, Any] = {
            "metadata": metadata,
            "tables": tables,
            "parsed": parsed,
            "nlp": nlp_results,
            "summary": summary_text
        }

        # Jika ingin JSON
        accept = request.headers.get("Accept", "")
        fmt = request.args.get("format", "").lower()
        if "application/json" in accept or fmt == "json":
            return jsonify(response_data), 200

        # Otherwise return PDF inline
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Prospektus Ringkasan AI", ln=1)
            pdf.ln(5)
            pdf.set_font("Arial", size=12)

            # Tulis ringkasan, safe encode Latin-1
            for line in summary_text.split("\n"):
                safe = line.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 8, safe)
                pdf.ln(1)

            # Tambah halaman untuk JSON lengkap
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Data Lengkap (JSON)", ln=1)
            pdf.set_font("Courier", size=8)
            json_str = json.dumps(response_data, ensure_ascii=False, indent=2)
            for jline in json_str.split("\n"):
                safe = jline.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 5, safe)

            # Output byte via dest="S"
            pdf_bytes = BytesIO(pdf.output(dest="S").encode('latin-1'))

            return send_file(
                pdf_bytes,
                mimetype="application/pdf",
                as_attachment=False,
                download_name="prospektus_summary.pdf"
            )
        except Exception as e:
            logger.exception("PDF generation failed")
            return jsonify({
                "error": "PDF generation failed",
                "details": str(e),
                **response_data
            }), 200


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")
    register_routes(app)
    return app


if __name__ == "__main__":
    # Pastikan GEMINI_API_KEY terset di env, atau fallback ke hardcoded
    if "GEMINI_API_KEY" not in os.environ:
        logger.warning("GEMINI_API_KEY not found in env; using fallback key.")
    app = create_app()
    app.run(debug=True)
