# core/extractor.py

import os
import sys
import json
import csv
import hashlib
import logging
from typing import List, Dict, Any, Optional
from multiprocessing.dummy import Pool as ThreadPool

import pdfplumber
import fitz   # PyMuPDF
from pdf2image import convert_from_path
import pytesseract

def setup_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        fmt = "[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
        h.setFormatter(logging.Formatter(fmt))
        logger.addHandler(h)
    logger.setLevel(logging.DEBUG)
    return logger

logger = setup_logger("PDFExtractor")


class PDFExtractor:
    """
    Extract text, tables, and metadata from a PDF with optional OCR,
    threaded per-page processing, and automatic cleanup.
    """
    def __init__(
        self,
        input_pdf: str,
        output_dir: str,
        force_ocr: bool = False,
        ocr_languages: str = "eng",
        ocr_dpi: int = 300,
        threads: int = 4,
        auto_cleanup: bool = False
    ):
        self.input_pdf    = input_pdf
        self.output_dir   = output_dir
        self.force_ocr    = force_ocr
        self.ocr_languages= ocr_languages
        self.ocr_dpi      = ocr_dpi
        self.threads      = threads
        self.auto_cleanup = auto_cleanup

        os.makedirs(self.output_dir, exist_ok=True)
        self.base_name   = os.path.splitext(os.path.basename(input_pdf))[0]
        self.result_data: Dict[str, Any] = {}
        self.logger = setup_logger("PDFExtractor")

    def compute_checksums(self) -> Dict[str, str]:
        md5 = hashlib.md5()
        sha256 = hashlib.sha256()
        with open(self.input_pdf, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
                sha256.update(chunk)
        return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}

    def extract_metadata(self) -> Dict[str, Any]:
        try:
            doc = fitz.open(self.input_pdf)
            meta = {k: v for k, v in doc.metadata.items() if v}
            doc.close()
            return meta
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return {}

    def _extract_text_plumber(self, pg: int) -> str:
        try:
            with pdfplumber.open(self.input_pdf) as pdf:
                return pdf.pages[pg-1].extract_text() or ""
        except:
            return ""

    def _extract_text_pymupdf(self, pg: int) -> str:
        try:
            doc = fitz.open(self.input_pdf)
            txt = doc.load_page(pg-1).get_text("text", sort=True)
            doc.close()
            return txt
        except:
            return ""

    def _ocr_page(self, pg: int) -> str:
        txt = ""
        try:
            images = convert_from_path(
                self.input_pdf,
                dpi=self.ocr_dpi,
                first_page=pg, last_page=pg
            )
            for img in images:
                txt += pytesseract.image_to_string(img, lang=self.ocr_languages)
        except Exception as e:
            self.logger.error(f"OCR failed on page {pg}: {e}")
        return txt

    def _extract_tables(self, pg: int) -> List[str]:
        paths: List[str] = []
        try:
            with pdfplumber.open(self.input_pdf) as pdf:
                tables = pdf.pages[pg-1].extract_tables()
            for idx, table in enumerate(tables, start=1):
                csv_path = os.path.join(
                    self.output_dir,
                    f"{self.base_name}_pg{pg}_tbl{idx}.csv"
                )
                with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                    writer = csv.writer(cf)
                    writer.writerows(table)
                paths.append(csv_path)
        except Exception:
            pass
        return paths

    def extract_page(self, pg: int) -> Dict[str, Any]:
        self.logger.debug(f"Extracting page {pg}")
        result = {"page": pg, "text": "", "tables": [], "method": None}
        txt = ""
        if not self.force_ocr:
            txt = self._extract_text_plumber(pg)
            if not txt.strip():
                txt = self._extract_text_pymupdf(pg)
        if self.force_ocr or not txt.strip():
            txt = self._ocr_page(pg)
            result["method"] = "ocr"
        else:
            result["method"] = "text"
            result["tables"] = self._extract_tables(pg)
        result["text"] = txt
        return result

    def extract(self) -> Dict[str, Any]:
        """
        Run full extraction:
          - metadata & checksums
          - threaded per-page text/table
          - aggregate top-level text & tables
          - optional cleanup
        """
        self.logger.info("Starting PDFExtractor.extract()")
        self.result_data["metadata"]  = self.extract_metadata()
        self.result_data["checksums"] = self.compute_checksums()

        # page count
        try:
            with pdfplumber.open(self.input_pdf) as pdf:
                n = len(pdf.pages)
        except:
            doc = fitz.open(self.input_pdf)
            n = doc.page_count
            doc.close()
        self.result_data["page_count"] = n

        # parallel extraction
        pool = ThreadPool(self.threads)
        pages = pool.map(self.extract_page, list(range(1, n+1)))
        pool.close(); pool.join()
        pages.sort(key=lambda x: x["page"])
        self.result_data["pages"] = pages

        # aggregate
        self.result_data["text"]   = "\n\n".join(p["text"] for p in pages)
        self.result_data["tables"] = [tbl for p in pages for tbl in p["tables"]]

        if self.auto_cleanup:
            try: os.remove(self.input_pdf)
            except: pass

        self.logger.info("PDFExtractor.extract() done")
        return self.result_data

    def save_results(self) -> None:
        """Write JSON + per-page txt + keep CSVs as-is."""
        # JSON
        jpath = os.path.join(self.output_dir, f"{self.base_name}_extract.json")
        with open(jpath, "w", encoding="utf-8") as jf:
            json.dump(self.result_data, jf, ensure_ascii=False, indent=2)
        self.logger.info(f"Wrote JSON: {jpath}")

        # per-page text
        for p in self.result_data["pages"]:
            tpath = os.path.join(
                self.output_dir, f"{self.base_name}_page_{p['page']}.txt"
            )
            with open(tpath, "w", encoding="utf-8") as tf:
                tf.write(p["text"])
        self.logger.info("Wrote per-page text files.")
