# nlp.py

import logging
import re
from typing import List, Dict, Any

import spacy                                               # spaCy for NER & PhraseMatcher :contentReference[oaicite:6]{index=6}
from spacy.matcher import PhraseMatcher
from rake_nltk import Rake                                 # RAKE for keyphrase extraction :contentReference[oaicite:7]{index=7}
from flashtext import KeywordProcessor                     # FlashText trie-based keyword spotting :contentReference[oaicite:8]{index=8}
from sklearn.feature_extraction.text import TfidfVectorizer # TF–IDF keyword scoring :contentReference[oaicite:9]{index=9}
from rapidfuzz import process as fuzzy_process             # RapidFuzz for fuzzy matching :contentReference[oaicite:10]{index=10}

def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = "[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
        h.setFormatter(logging.Formatter(fmt))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger("nlp")

class ProspectusNLP:
    def __init__(self,
                 spacy_model: str = "en_core_web_sm",
                 custom_phrases: List[str] = None,
                 flash_keywords: List[str] = None):
        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info("Loaded spaCy model: %s", spacy_model)
        except Exception as e:
            logger.error("spaCy load failed: %s", e)
            raise

        # PhraseMatcher for dictionary terms
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        if custom_phrases:
            patterns = [self.nlp.make_doc(text) for text in custom_phrases]
            self.phrase_matcher.add("CUSTOM", patterns)
            logger.info("Registered %d custom phrases", len(custom_phrases))

        # FlashText for blazing‐fast spotting
        self.flash = KeywordProcessor()
        if flash_keywords:
            for kw in flash_keywords:
                self.flash.add_keyword(kw)
            logger.info("Loaded %d flashtext keywords", len(flash_keywords))

        # RAKE for keyphrases
        self.rake = Rake()

        # TF–IDF vectorizer (we’ll fit on incoming docs dynamically)
        self.tfidf = TfidfVectorizer(ngram_range=(1,2),
                                     stop_words='english',
                                     max_features=50)

    # ---------------------------
    # 1) Named Entity Recognition
    # ---------------------------
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Returns spaCy NER dict: label → list of entity texts.
        """
        doc = self.nlp(text)
        ents: Dict[str, List[str]] = {}
        for ent in doc.ents:
            ents.setdefault(ent.label_, []).append(ent.text)
        logger.info("Extracted %d entities", len(doc.ents))
        return ents

    # ---------------------------
    # 2) Custom Phrase Matching
    # ---------------------------
    def extract_phrases(self, text: str) -> List[str]:
        """
        Returns list of matches from PhraseMatcher.
        """
        doc = self.nlp(text)
        matches = self.phrase_matcher(doc)
        phrases = [doc[start:end].text for _, start, end in matches]
        logger.info("Found %d phrase matches", len(phrases))
        return phrases

    # ---------------------------
    # 3) FlashText Keyword Spotting
    # ---------------------------
    def flash_keywords(self, text: str) -> List[str]:
        """
        Returns all flashtext keywords found.
        """
        found = self.flash.extract_keywords(text)
        logger.info("FlashText found %d keywords", len(found))
        return found

    # ---------------------------
    # 4) RAKE Keyphrase Extraction
    # ---------------------------
    def extract_keyphrases(self, text: str, max_phrases: int = 20) -> List[str]:
        """
        Uses RAKE to extract and rank keyphrases.
        """
        self.rake.extract_keywords_from_text(text)
        phrases = self.rake.get_ranked_phrases()[:max_phrases]
        logger.info("RAKE extracted %d keyphrases", len(phrases))
        return phrases

    # ---------------------------
    # 5) TF–IDF Keyword Ranking
    # ---------------------------
    def extract_tfidf_keywords(self, docs: List[str], top_n: int = 10) -> List[str]:
        """
        Fit TF–IDF on list of docs and return top_n scoring terms.
        """
        tfidf_matrix = self.tfidf.fit_transform(docs)
        feature_names = self.tfidf.get_feature_names_out()
        # Sum up scores per term across docs
        scores = tfidf_matrix.sum(axis=0).A1
        ranked = sorted(zip(feature_names, scores),
                        key=lambda x: x[1], reverse=True)
        keywords = [term for term, _ in ranked[:top_n]]
        logger.info("TF–IDF top %d terms", len(keywords))
        return keywords

    # ---------------------------
    # 6) Fuzzy Matching
    # ---------------------------
    def fuzzy_match(self, text: str, choices: List[str], limit: int = 5) -> List[str]:
        """
        Uses RapidFuzz to fuzzy‐match choices in text.
        Returns best scoring `limit` choices.
        """
        matches = fuzzy_process.extract(text, choices, limit=limit)
        # Each match is (choice, score, index)
        result = [m[0] for m in matches if m[1] > 70]  # threshold
        logger.info("Fuzzy matched %d choices", len(result))
        return result

    # ---------------------------
    # 7) Comprehensive API
    # ---------------------------
    def analyze(self, text: str, docs_for_tfidf: List[str] = None,
                fuzzy_choices: List[str] = None) -> Dict[str, Any]:
        """
        Runs full pipeline and returns structured dict.
        """
        result: Dict[str, Any] = {
            "entities": self.extract_entities(text),
            "phrases": self.extract_phrases(text),
            "flash": self.flash_keywords(text),
            "rake": self.extract_keyphrases(text),
        }
        if docs_for_tfidf:
            result["tfidf"] = self.extract_tfidf_keywords(docs_for_tfidf)
        if fuzzy_choices:
            result["fuzzy"] = self.fuzzy_match(text, fuzzy_choices)
        return result

if __name__ == "__main__":
    sample_text = open("data/sample.txt").read()
    nlp = ProspectusNLP(
        custom_phrases=["penawaran umum", "saham baru", "waran seri I"],
        flash_keywords=["modal kerja","capex","EBITDA"]
    )
    analysis = nlp.analyze(
        text=sample_text,
        docs_for_tfidf=[sample_text],  # or a corpus list
        fuzzy_choices=["likuiditas","persaingan","tenaga medis"]
    )
    import json; print(json.dumps(analysis, indent=2, ensure_ascii=False))
