# parser.py

import re
import logging
from typing import Dict, Any, List, Optional
import spacy

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

logger = setup_logger("parser")

# ----------------------------------------
# ProspectusParser
# ----------------------------------------
class ProspectusParser:
    def __init__(self, nlp_model: str = "en_core_web_sm"):
        """
        Inisialisasi parser dengan spaCy NER untuk deteksi entitas perusahaan, orang, lokasi, dsb.
        """
        try:
            self.nlp = spacy.load(nlp_model)
            logger.info("Loaded spaCy model '%s'", nlp_model)
        except Exception as e:
            logger.error("Failed loading spaCy model: %s", e)
            self.nlp = None

    # -----------------------
    # Utility: regex find
    # -----------------------
    def _find_first(self, pattern: str, text: str, flags=0) -> Optional[str]:
        m = re.search(pattern, text, flags)
        if m:
            return m.group(1).strip()
        return None

    def _find_all(self, pattern: str, text: str, flags=0) -> List[str]:
        return [m.group(1).strip() for m in re.finditer(pattern, text, flags)]

    # -----------------------
    # 1. IPO Structure
    # -----------------------
    def parse_ipo(self, text: str) -> Dict[str, Any]:
        """
        Tangkap struktur IPO:
        - jumlah saham public & persen
        - waran & ratio
        - price range
        - target dana
        """
        data: Dict[str, Any] = {}

        # Saham public: "530.000.000 (…) (20,78%)"
        pub = self._find_first(
            r"(\d{1,3}(?:[.\d{3}]+))\s*\(.*?(\d{1,2},\d{2})%\)", text
        )
        if pub:
            parts = pub.split()[0].replace(".", "")
            data["saham_public"] = int(parts)
            data["saham_public_pct"] = float(pub.split()[1].replace(",", "."))
        # Waran: "265.000.000 (…) (13,12%)"
        waran = self._find_first(
            r"Waran Seri I\s+sebanyak-banyaknya\s+(\d{1,3}(?:[.\d{3}]+)).*?(\d{1,2},\d{2})%", text
        )
        if waran:
            num, pct = waran
            data["waran_qty"] = int(num.replace(".", ""))
            data["waran_pct"] = float(pct.replace(",", "."))

        # Price range: "Rp100,- sampai dengan Rp132,-"
        price = self._find_first(
            r"Rp\s*([\d\.]+)[^0-9]+Rp\s*([\d\.]+)", text
        )
        if price:
            lo, hi = price
            data["price_low"] = int(lo.replace(".", ""))
            data["price_high"] = int(hi.replace(".", ""))

        # Target dana: "Rp69.960.000.000"
        dana = self._find_first(
            r"sebanyak-banyaknya\s+Rp([\d\.]+)", text
        )
        if dana:
            data["target_dana"] = int(dana.replace(".", ""))

        logger.info("Parsed IPO structure: %s", data)
        return data

    # -----------------------
    # 2. Timeline & Dates
    # -----------------------
    def parse_timeline(self, text: str) -> Dict[str, str]:
        """
        Jadwal penting: TB of bookbuilding, effective date, penjatahan, listing.
        """
        dates: Dict[str, str] = {}
        # Bookbuilding: "Masa Penawaran Awal : 24 – 28 April 2025"
        bb = self._find_first(
            r"Masa Penawaran Awal\s*[:\-]\s*([\w\d\s–\-]+?\d{4})", text
        )
        if bb:
            dates["bookbuilding"] = bb
        # Efektif: "Tanggal Efektif : 30 April 2025"
        ef = self._find_first(
            r"Tanggal Efektif\s*[:\-]\s*(\d{1,2}\s+\w+\s+\d{4})", text
        )
        if ef:
            dates["effective"] = ef
        # Penjatahan
        pj = self._find_first(
            r"Tanggal Penjatahan\s*[:\-]\s*(\d{1,2}\s+\w+\s+\d{4})", text
        )
        if pj:
            dates["penjatahan"] = pj
        # Listing saham & waran: "Pencatatan Saham dan Waran …: 14 Mei 2025"
        li = self._find_first(
            r"Pencatatan Saham dan Waran.*?[:\-]\s*(\d{1,2}\s+\w+\s+\d{4})", text
        )
        if li:
            dates["listing"] = li

        logger.info("Parsed timeline: %s", dates)
        return dates

    # -----------------------
    # 3. Financial Ratios
    # -----------------------
    def parse_financial_ratios(self, text: str) -> Dict[str, float]:
        """
        Ambil PBV, PER, market cap proforma.
        """
        ratios: Dict[str, float] = {}

        # PBV: "PBV 4,9x"
        pbv = self._find_first(r"PBV\s*([\d,]+)x", text)
        if pbv:
            ratios["PBV"] = float(pbv.replace(",", "."))

        # PER: "PER 21 x"
        per = self._find_first(r"PER\s*([\d,]+)\s*[xX]", text)
        if per:
            ratios["PER"] = float(per.replace(",", "."))

        # Market cap: "Market cap Rp 312,7 M" or calculate: price_mid*shares
        mc = self._find_first(r"Market cap\s*Rp\s*([\d,\.]+)\s*M", text)
        if mc:
            ratios["market_cap_M"] = float(mc.replace(".", "").replace(",", "."))

        logger.info("Parsed financial ratios: %s", ratios)
        return ratios

    # -----------------------
    # 4. Company Profile & Segment
    # -----------------------
    def parse_business(self, text: str) -> Dict[str, Any]:
        """
        Tangkap ringkasan usaha: sektor, model franchise, item unik.
        """
        biz: Dict[str, Any] = {}
        # Cari kalimat 'Usaha restoran' atau 'Rumah Sakit'
        if "Rumah Sakit" in text:
            biz["sector"] = "Healthcare / Hospital"
        if "franchise" in text.lower():
            biz["model"] = "Franchise"
        # Jumlah outlet: "35 outlet", "28 outlet"
        outlets = self._find_all(r"(\d{1,3}) outlet", text, re.IGNORECASE)
        if outlets:
            biz["outlet_counts"] = [int(o) for o in outlets]

        logger.info("Parsed business profile: %s", biz)
        return biz

    # -----------------------
    # 5. Risk Factors
    # -----------------------
    def parse_risks(self, text: str) -> List[str]:
        """
        Extract risiko utama, misal: 'liquidity', 'tenaga medis'
        """
        risks: List[str] = []
        if "likuid" in text.lower():
            risks.append("Low liquidity")
        if "tenaga medis" in text.lower():
            risks.append("Medical staff availability")
        if "persaingan" in text.lower():
            risks.append("High competition")
        logger.info("Parsed risks: %s", risks)
        return risks

    # -----------------------
    # 6. Named Entity Extraction
    # -----------------------
    def parse_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Gunakan spaCy NER untuk dapatkan org, orang, lokasi.
        """
        ents: Dict[str, List[str]] = {}
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                ents.setdefault(ent.label_, []).append(ent.text)
        logger.info("Parsed entities counts: %s", {k: len(v) for k, v in ents.items()})
        return ents

    # -----------------------
    # 7. Wrapper parse_all
    # -----------------------
    def parse_all(self, text: str) -> Dict[str, Any]:
        """
        Jalankan semua parser dan kembalikan dict terstruktur.
        """
        result: Dict[str, Any] = {}
        result["ipo"] = self.parse_ipo(text)
        result["timeline"] = self.parse_timeline(text)
        result["ratios"] = self.parse_financial_ratios(text)
        result["business"] = self.parse_business(text)
        result["risks"] = self.parse_risks(text)
        result["entities"] = self.parse_entities(text)
        return result

# ----------------------------------------
# CLI / Example Usage
# ----------------------------------------
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Parse prospectus text into structured data")
    parser.add_argument("txt", help="Path to plain-text prospectus")
    parser.add_argument("-o", "--output", help="Output JSON file", default="parsed.json")
    args = parser.parse_args()

    # Load text
    with open(args.txt, encoding="utf8") as f:
        content = f.read()

    # Parse
    prospectus = ProspectusParser()
    parsed = prospectus.parse_all(content)

    # Save to JSON
    with open(args.output, "w", encoding="utf8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    logger.info("Parsing complete, output saved to %s", args.output)
