"""
nlp.py - A comprehensive NLP module for production use.

Features:
- Keyword extraction
- Named entity extraction
- Text summarization
- Fuzzy matching
- Multi-language support
- Batch and parallel processing
- Robust error handling and logging
"""
# ========== Python Standard Library ==========
import os
import sys
import re
import logging
import multiprocessing
from pathlib import Path
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union

# ========== Third Party Libraries ==========
import spacy
from spacy.matcher import PhraseMatcher
from fuzzywuzzy import process as fuzzy_process
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
from flashtext import KeywordProcessor


try:
    import spacy
    from spacy.language import Language
except ImportError:
    raise ImportError("spaCy is required for this module. Please install spaCy (e.g., `pip install spacy`).")

# Attempt to import rapidfuzz for fuzzy matching
try:
    from rapidfuzz import fuzz, process
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

# Configure logging for the module
logger = logging.getLogger("nlp_module")
logger.setLevel(logging.DEBUG)  # Change to INFO or WARNING to reduce verbosity if desired
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
ch.setFormatter(formatter)
logger.addHandler(ch)


def load_spacy_model(
    model_name: Optional[str] = None,
    lang: Optional[str] = None,
    max_length: Optional[int] = None
) -> Language:
    """
    Load a spaCy language model, retrying on failure and falling back to a blank multilingual model.

    Parameters:
        model_name (Optional[str]): The name of the spaCy model to load (e.g., 'en_core_web_sm').
        lang (Optional[str]): The language code (e.g., 'en', 'de') if model_name is not provided.
        max_length (Optional[int]): If provided, will set nlp.max_length to this value or higher.

    Returns:
        spacy.language.Language: A loaded spaCy NLP model, or a blank multilingual model on failure.
    """
    nlp_model = None
    # Attempt to load specified model
    if model_name:
        try:
            logger.info(f"Loading spaCy model '{model_name}'")
            nlp_model = spacy.load(model_name)
            logger.info(f"Successfully loaded spaCy model '{model_name}'")
        except Exception as e:
            logger.error(f"Failed to load spaCy model '{model_name}': {e}")
            nlp_model = None

    # If not loaded yet, attempt to load based on language code
    if nlp_model is None and lang:
        lang_model_map = {
            'en': 'en_core_web_sm',
            'de': 'de_core_news_sm',
            'es': 'es_core_news_sm',
            'fr': 'fr_core_news_sm',
            'zh': 'zh_core_web_sm',
            'pt': 'pt_core_news_sm',
            'it': 'it_core_news_sm',
            'nl': 'nl_core_news_sm',
            'xx': None,  # blank multilingual
            # Add other mappings as needed
        }
        if lang in lang_model_map and lang_model_map[lang]:
            try_model = lang_model_map[lang]
            try:
                logger.info(f"Loading spaCy model for language '{lang}': '{try_model}'")
                nlp_model = spacy.load(try_model)
                logger.info(f"Successfully loaded spaCy model '{try_model}'")
            except Exception as e:
                logger.error(f"Failed to load spaCy model '{try_model}': {e}")
                nlp_model = None
        else:
            logger.warning(f"No specific model mapping for language '{lang}'. Will use blank multilingual model.")

    # Retry logic: if initial load failed, try once more
    if nlp_model is None and model_name:
        try:
            logger.info(f"Retrying to load spaCy model '{model_name}'")
            nlp_model = spacy.load(model_name)
            logger.info(f"Successfully loaded spaCy model '{model_name}' on retry")
        except Exception as e:
            logger.error(f"Retry failed for spaCy model '{model_name}': {e}")
            nlp_model = None

    # Final fallback to blank multilingual model
    if nlp_model is None:
        try:
            logger.warning("Falling back to blank multilingual spaCy model ('xx').")
            nlp_model = spacy.blank("xx")
        except Exception as e:
            logger.critical(f"Failed to create blank multilingual spaCy model: {e}")
            raise RuntimeError("Could not load any spaCy model.")

    # Set max_length to handle large texts if specified
    if max_length and hasattr(nlp_model, "max_length"):
        try:
            original_max = nlp_model.max_length
            nlp_model.max_length = max(original_max, max_length)
            logger.info(f"Set spaCy model max_length from {original_max} to {nlp_model.max_length}")
        except Exception as e:
            logger.error(f"Failed to set nlp.max_length: {e}")
    return nlp_model


def chunk_text(text: str, max_length: int, buffer: int = 50) -> List[str]:
    """
    Split text into chunks that are each at most max_length characters long.

    Attempts to break on whitespace to avoid splitting words.

    Parameters:
        text (str): The text to split.
        max_length (int): Maximum allowed length of each chunk.
        buffer (int): Look-back characters from max_length to find a safe split point.

    Returns:
        List[str]: A list of text chunks, each within the max_length limit.
    """
    if len(text) <= max_length:
        return [text]
    chunks: List[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + max_length, text_length)
        if end == text_length:
            chunks.append(text[start:end])
            break
        # Try to split at the last whitespace before the end
        split_pos = text.rfind(" ", start, end)
        if split_pos <= start:
            split_pos = text.find(" ", end)
        if split_pos == -1:
            # If no space found, force split
            split_pos = end
        chunks.append(text[start:split_pos].strip())
        start = split_pos
    logger.debug(f"Text split into {len(chunks)} chunks (original length {text_length}).")
    return chunks


class KeywordExtractor:
    """
    Extract keywords from a text using a combination of noun chunks and word frequency.
    """
    def __init__(
        self,
        model_name: Optional[str] = None,
        lang: Optional[str] = None,
        top_n: int = 10,
        max_length: int = 3_000_000
    ) -> None:
        """
        Initialize the KeywordExtractor.

        Parameters:
            model_name (Optional[str]): Specific spaCy model name to use (e.g., 'en_core_web_sm').
            lang (Optional[str]): Language code to select a model if model_name is not provided.
            top_n (int): Number of top keywords to extract.
            max_length (int): Max text length for spaCy processing.
        """
        self.top_n = top_n
        self.max_length = max_length
        try:
            self.nlp = load_spacy_model(model_name=model_name, lang=lang, max_length=self.max_length)
        except Exception as e:
            logger.error(f"KeywordExtractor: Error loading spaCy model: {e}")
            raise
        # Ensure nlp can handle large text
        if hasattr(self.nlp, "max_length"):
            self.nlp.max_length = max(self.nlp.max_length, self.max_length)

    def extract(self, text: str, top_n: Optional[int] = None) -> List[str]:
        """
        Extract keywords from a single text.

        Combines noun chunks (as phrases) and high-frequency words.

        Parameters:
            text (str): The text to analyze.
            top_n (Optional[int]): Override number of top keywords.

        Returns:
            List[str]: A list of extracted keywords/phrases.
        """
        if top_n is None:
            top_n = self.top_n
        if not text:
            logger.warning("KeywordExtractor: Empty text provided to extract().")
            return []
        # Handle long text by chunking
        texts = chunk_text(text, self.nlp.max_length)
        freq: Dict[str, int] = {}
        chunks: List[str] = []
        for chunk in texts:
            doc = self.nlp(chunk)
            # Extract noun chunks (if parser is available)
            if hasattr(doc, 'noun_chunks'):
                for np in doc.noun_chunks:
                    phrase = np.text.strip()
                    if phrase:
                        chunks.append(phrase)
            # Count word frequencies
            for token in doc:
                if token.is_alpha and not token.is_stop:
                    word = token.lemma_.lower()
                    freq[word] = freq.get(word, 0) + 1
        # Sort words by frequency
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, count in sorted_words[:top_n]]
        # Combine noun chunks and top words, ensure uniqueness preserving order
        keywords: List[str] = []
        for phrase in chunks:
            if all(phrase.lower() != existing.lower() for existing in keywords):
                keywords.append(phrase)
            if len(keywords) >= top_n:
                break
        for word in top_words:
            if word not in keywords:
                keywords.append(word)
            if len(keywords) >= top_n:
                break
        logger.info(f"KeywordExtractor: Extracted {len(keywords)} keywords (requested {top_n}).")
        return keywords[:top_n]

    def extract_batch(
        self,
        texts: List[str],
        top_n: Optional[int] = None,
        n_process: int = 1
    ) -> List[List[str]]:
        """
        Extract keywords from multiple texts.

        Parameters:
            texts (List[str]): List of texts to process.
            top_n (Optional[int]): Number of top keywords to extract per text.
            n_process (int): Number of parallel processes to use (if supported).

        Returns:
            List[List[str]]: List of keyword lists for each text.
        """
        if top_n is None:
            top_n = self.top_n
        results: List[List[str]] = []
        try:
            docs = list(self.nlp.pipe(texts, batch_size=50, n_process=n_process))
            for doc in docs:
                if not doc.text:
                    results.append([])
                    continue
                freq: Dict[str, int] = {}
                chunks: List[str] = []
                for np in doc.noun_chunks:
                    phrase = np.text.strip()
                    if phrase:
                        chunks.append(phrase)
                for token in doc:
                    if token.is_alpha and not token.is_stop:
                        word = token.lemma_.lower()
                        freq[word] = freq.get(word, 0) + 1
                sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
                top_words = [word for word, count in sorted_words[:top_n]]
                keywords: List[str] = []
                for phrase in chunks:
                    if all(phrase.lower() != existing.lower() for existing in keywords):
                        keywords.append(phrase)
                    if len(keywords) >= top_n:
                        break
                for word in top_words:
                    if word not in keywords:
                        keywords.append(word)
                    if len(keywords) >= top_n:
                        break
                results.append(keywords[:top_n])
        except Exception as e:
            logger.error(f"KeywordExtractor.extract_batch: spaCy pipe failed: {e}. Falling back to sequential processing.")
            for text in texts:
                results.append(self.extract(text, top_n=top_n))
        return results


class EntityExtractor:
    """
    Extract named entities from text using spaCy's NER.
    """
    def __init__(
        self,
        model_name: Optional[str] = None,
        lang: Optional[str] = None,
        max_length: int = 3_000_000,
        include_types: Optional[List[str]] = None
    ) -> None:
        """
        Initialize the EntityExtractor.

        Parameters:
            model_name (Optional[str]): Specific spaCy model name to use.
            lang (Optional[str]): Language code to select a model if model_name is not provided.
            max_length (int): Max text length for spaCy processing.
            include_types (Optional[List[str]]): List of entity labels to include (e.g., ['PERSON', 'ORG']).
                If None, all entity types are returned.
        """
        self.include_types = include_types
        self.max_length = max_length
        try:
            self.nlp = load_spacy_model(model_name=model_name, lang=lang, max_length=self.max_length)
        except Exception as e:
            logger.error(f"EntityExtractor: Error loading spaCy model: {e}")
            raise
        if hasattr(self.nlp, "max_length"):
            self.nlp.max_length = max(self.nlp.max_length, self.max_length)

    def extract(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities from a single text.

        Parameters:
            text (str): The text to analyze.

        Returns:
            List[Tuple[str, str]]: A list of (entity_text, entity_label) tuples.
        """
        if not text:
            logger.warning("EntityExtractor: Empty text provided to extract().")
            return []
        entities: List[Tuple[str, str]] = []
        # Handle long text by chunking
        texts = chunk_text(text, self.nlp.max_length)
        for chunk in texts:
            doc = self.nlp(chunk)
            for ent in doc.ents:
                if self.include_types is None or ent.label_ in self.include_types:
                    entities.append((ent.text, ent.label_))
        logger.info(f"EntityExtractor: Found {len(entities)} entities.")
        return entities

    def extract_batch(
        self,
        texts: List[str],
        n_process: int = 1
    ) -> List[List[Tuple[str, str]]]:
        """
        Extract entities from multiple texts.

        Parameters:
            texts (List[str]): List of texts to process.
            n_process (int): Number of parallel processes to use (if supported).

        Returns:
            List[List[Tuple[str, str]]]: List of entity lists for each text.
        """
        results: List[List[Tuple[str, str]]] = []
        try:
            docs = list(self.nlp.pipe(texts, batch_size=50, n_process=n_process))
            for doc in docs:
                ents: List[Tuple[str, str]] = []
                for ent in doc.ents:
                    if self.include_types is None or ent.label_ in self.include_types:
                        ents.append((ent.text, ent.label_))
                results.append(ents)
        except Exception as e:
            logger.error(f"EntityExtractor.extract_batch: spaCy pipe failed: {e}. Falling back to sequential processing.")
            for text in texts:
                results.append(self.extract(text))
        return results


class TextSummarizer:
    """
    Summarize text into a shorter version, focusing on top-ranking sentences by word importance.
    """
    def __init__(
        self,
        model_name: Optional[str] = None,
        lang: Optional[str] = None,
        num_sentences: int = 3,
        max_length: int = 3_000_000
    ) -> None:
        """
        Initialize the TextSummarizer.

        Parameters:
            model_name (Optional[str]): Specific spaCy model name to use.
            lang (Optional[str]): Language code to select a model if model_name is not provided.
            num_sentences (int): Default number of sentences in the summary.
            max_length (int): Max text length for spaCy processing.
        """
        self.num_sentences = num_sentences
        self.max_length = max_length
        try:
            self.nlp = load_spacy_model(model_name=model_name, lang=lang, max_length=self.max_length)
        except Exception as e:
            logger.error(f"TextSummarizer: Error loading spaCy model: {e}")
            raise
        if hasattr(self.nlp, "max_length"):
            self.nlp.max_length = max(self.nlp.max_length, self.max_length)
        # Attempt to get stop words set; fallback to English stop words if not found
        try:
            self.stop_words = self.nlp.Defaults.stop_words
        except Exception:
            try:
                from spacy.lang.en.stop_words import STOP_WORDS
                self.stop_words = set(STOP_WORDS)
            except Exception:
                self.stop_words = set()

    def summarize(self, text: str, num_sentences: Optional[int] = None) -> str:
        """
        Summarize a single text, returning the top 'num_sentences' important sentences.

        Parameters:
            text (str): The text to summarize.
            num_sentences (Optional[int]): Override the default number of sentences.

        Returns:
            str: The summarized text.
        """
        if num_sentences is None:
            num_sentences = self.num_sentences
        if not text:
            logger.warning("TextSummarizer: Empty text provided to summarize().")
            return ""
        # If text is very long, summarize in chunks and combine
        if len(text) > self.nlp.max_length:
            chunks = chunk_text(text, self.nlp.max_length)
            partial_summaries = [self.summarize(chunk, num_sentences) for chunk in chunks]
            combined_summary = " ".join(partial_summaries)
            # If still too long or too many sentences, summarize again
            if len(combined_summary) > self.nlp.max_length or combined_summary.count('.') > num_sentences:
                return self.summarize(combined_summary, num_sentences)
            return combined_summary

        doc = self.nlp(text)
        # Segment text into sentences
        sentences = [sent.text.strip() for sent in doc.sents]
        # Fallback if sentence segmentation is not available or only one sentence
        if len(sentences) <= 1:
            sentences = [sent.strip() for sent in text.split('.') if sent.strip()]
        if not sentences:
            logger.warning("TextSummarizer: No sentences found in text.")
            return text

        # Compute word frequencies for non-stopwords
        freq: Dict[str, int] = {}
        for token in doc:
            word = token.lemma_.lower()
            if token.is_alpha and word not in self.stop_words:
                freq[word] = freq.get(word, 0) + 1
        if not freq:
            # If no valid words for frequency, return first sentences
            return " ".join(sentences[:num_sentences])

        # Normalize frequencies by the maximum frequency
        max_freq = max(freq.values())
        for word in freq:
            freq[word] = freq[word] / max_freq

        # Score sentences
        sentence_scores: Dict[str, float] = {}
        for sent in sentences:
            for word in sent.split():
                word_l = word.lower().strip(".,!?")
                if word_l in freq:
                    sentence_scores[sent] = sentence_scores.get(sent, 0.0) + freq[word_l]

        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        selected_sentences = [s for s, score in top_sentences[:num_sentences]]
        if not selected_sentences:
            selected_sentences = sentences[:num_sentences]

        summary = " ".join(selected_sentences).strip()
        logger.info(f"TextSummarizer: Summary length {len(selected_sentences)} sentences (requested {num_sentences}).")
        return summary

    def summarize_batch(
        self,
        texts: List[str],
        num_sentences: Optional[int] = None,
        n_process: int = 1
    ) -> List[str]:
        """
        Summarize multiple texts.

        Parameters:
            texts (List[str]): List of texts to summarize.
            num_sentences (Optional[int]): Number of sentences for each summary.
            n_process (int): Number of parallel processes to use.

        Returns:
            List[str]: Summaries for each text.
        """
        if num_sentences is None:
            num_sentences = self.num_sentences
        results: List[str] = []
        if n_process and n_process > 1:
            try:
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(max_workers=n_process) as executor:
                    futures = [executor.submit(self.summarize, text, num_sentences) for text in texts]
                    for future in futures:
                        try:
                            results.append(future.result())
                        except Exception as e:
                            logger.error(f"TextSummarizer.summarize_batch: error in worker: {e}")
                            results.append("")
            except Exception as e:
                logger.error(f"TextSummarizer.summarize_batch: parallel execution failed: {e}.")
        if not results:
            for text in texts:
                results.append(self.summarize(text, num_sentences))
        return results


class FuzzyMatcher:
    """
    Perform fuzzy string matching between a query and a text.
    """
    def __init__(
        self,
        threshold: float = 80.0,
        limit: int = 5,
        model_name: Optional[str] = None,
        lang: Optional[str] = None,
        max_length: int = 1_000_000
    ) -> None:
        """
        Initialize the FuzzyMatcher.

        Parameters:
            threshold (float): Score threshold (0-100) to consider a match significant.
            limit (int): Maximum number of matches to return.
            model_name (Optional[str]): spaCy model name for sentence segmentation.
            lang (Optional[str]): Language code if needed.
            max_length (int): Max text length for spaCy processing.
        """
        self.threshold = threshold
        self.limit = limit
        self.has_rapidfuzz = _HAS_RAPIDFUZZ
        if not self.has_rapidfuzz:
            logger.warning("rapidfuzz not installed. FuzzyMatcher will use slower difflib fallback.")
        try:
            self.nlp = load_spacy_model(model_name=model_name, lang=lang, max_length=max_length)
        except Exception as e:
            logger.error(f"FuzzyMatcher: Error loading spaCy model: {e}")
            raise

    def match(self, query: str, text: str) -> List[Tuple[str, float]]:
        """
        Find fuzzy matches of query string within the text, returning matching fragments and scores.

        Parameters:
            query (str): The query string to match.
            text (str): The text to search within.

        Returns:
            List[Tuple[str, float]]: List of (matched_text, score) sorted by score descending.
        """
        if not query or not text:
            logger.warning("FuzzyMatcher: Empty query or text provided to match().")
            return []
        doc = self.nlp(text)
        candidates = [sent.text.strip() for sent in doc.sents] if doc.sents else [text]
        matches: List[Tuple[str, float]] = []
        if self.has_rapidfuzz:
            for sent in candidates:
                score = fuzz.partial_ratio(query.lower(), sent.lower())
                if score >= self.threshold:
                    matches.append((sent, score))
        else:
            from difflib import SequenceMatcher
            for sent in candidates:
                ratio = SequenceMatcher(None, query, sent).ratio()
                score = ratio * 100
                if score >= self.threshold:
                    matches.append((sent, score))
        matches = sorted(matches, key=lambda x: x[1], reverse=True)
        if self.limit:
            matches = matches[:self.limit]
        logger.info(f"FuzzyMatcher: Found {len(matches)} matches for query '{query}'.")
        return matches

    def match_batch(
        self,
        queries: List[str],
        text: str
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Perform fuzzy matching for multiple queries against the same text.

        Parameters:
            queries (List[str]): List of query strings.
            text (str): The text to search within.

        Returns:
            Dict[str, List[Tuple[str, float]]]: Mapping of query to list of (matched_text, score).
        """
        results: Dict[str, List[Tuple[str, float]]] = {}
        for query in queries:
            results[query] = self.match(query, text)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NLPPipeline Command-Line Interface for text analysis."
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands: analyze or batch')

    # Subparser for analyzing a single file
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single text file.')
    analyze_parser.add_argument('--file', '-f', required=True, help='Path to the text file to analyze.')
    analyze_parser.add_argument('--keywords', action='store_true', help='Perform keyword extraction.')
    analyze_parser.add_argument('--entities', action='store_true', help='Perform entity extraction.')
    analyze_parser.add_argument('--summary', action='store_true', help='Perform text summarization.')
    analyze_parser.add_argument('--fuzzy', nargs='*', help='Perform fuzzy matching for given queries (provide queries).')
    analyze_parser.add_argument('--lang', default='en', help='Language code (default: en).')
    analyze_parser.add_argument('--model', default=None, help='SpaCy model name to use (overrides lang).')
    analyze_parser.add_argument('--n_process', type=int, default=1, help='Number of processes for parallel processing.')
    analyze_parser.add_argument('--out', '-o', default=None, help='Optional output file (JSON format).')

    # Subparser for batch processing multiple files
    batch_parser = subparsers.add_parser('batch', help='Analyze all text files in a folder.')
    batch_parser.add_argument('--folder', '-d', required=True, help='Directory containing text files.')
    batch_parser.add_argument('--pattern', default='*.txt', help='File pattern to match (default: *.txt).')
    batch_parser.add_argument('--keywords', action='store_true', help='Perform keyword extraction.')
    batch_parser.add_argument('--entities', action='store_true', help='Perform entity extraction.')
    batch_parser.add_argument('--summary', action='store_true', help='Perform text summarization.')
    batch_parser.add_argument('--fuzzy', nargs='*', help='Perform fuzzy matching for given queries (provide queries).')
    batch_parser.add_argument('--lang', default='en', help='Language code (default: en).')
    batch_parser.add_argument('--model', default=None, help='SpaCy model name to use (overrides lang).')
    batch_parser.add_argument('--n_process', type=int, default=1, help='Number of processes for parallel processing.')
    batch_parser.add_argument('--out_dir', default=None, help='Optional output directory for results.')

    args = parser.parse_args()

    if args.command == 'analyze':
        file_path = Path(args.file)
        if not file_path.is_file():
            logger.error(f"File not found: {file_path}")
            sys.exit(1)
        text = file_path.read_text(encoding='utf-8')
        if args.keywords:
            ke = KeywordExtractor(model_name=args.model, lang=args.lang)
            keywords = ke.extract(text)
            print(f"Keywords ({file_path.name}): {keywords}")
        if args.entities:
            ee = EntityExtractor(model_name=args.model, lang=args.lang)
            entities = ee.extract(text)
            print(f"Entities ({file_path.name}): {entities}")
        if args.summary:
            ts = TextSummarizer(model_name=args.model, lang=args.lang)
            summary = ts.summarize(text)
            print(f"Summary ({file_path.name}): {summary}")
        if args.fuzzy:
            fm = FuzzyMatcher(model_name=args.model, lang=args.lang)
            for query in args.fuzzy:
                matches = fm.match(query, text)
                print(f"Fuzzy matches for '{query}' in {file_path.name}: {matches}")

    elif args.command == 'batch':
        dir_path = Path(args.folder)
        if not dir_path.is_dir():
            logger.error(f"Directory not found: {dir_path}")
            sys.exit(1)
        files = list(dir_path.glob(args.pattern))
        if not files:
            logger.warning(f"No files found in {dir_path} with pattern {args.pattern}.")
            sys.exit(0)
        # Initialize components if needed
        ke = KeywordExtractor(model_name=args.model, lang=args.lang) if args.keywords else None
        ee = EntityExtractor(model_name=args.model, lang=args.lang) if args.entities else None
        ts = TextSummarizer(model_name=args.model, lang=args.lang) if args.summary else None
        fm = FuzzyMatcher(model_name=args.model, lang=args.lang) if args.fuzzy else None
        for file_path in files:
            text = file_path.read_text(encoding='utf-8')
            print(f"Processing file: {file_path.name}")
            if ke:
                keywords = ke.extract(text)
                print(f"  Keywords: {keywords}")
            if ee:
                entities = ee.extract(text)
                print(f"  Entities: {entities}")
            if ts:
                summary = ts.summarize(text)
                print(f"  Summary: {summary}")
            if fm and args.fuzzy:
                for query in args.fuzzy:
                    matches = fm.match(query, text)
                    print(f"  Fuzzy matches for '{query}': {matches}")
        print("Batch processing completed.")
    else:
        parser.print_help()

class ProspectusNLP:
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        custom_phrases: Optional[List[str]] = None,
        flash_keywords: Optional[List[str]] = None
    ):
        """
        Initialize NLP pipelines:
        - spaCy NER & custom phrase matcher
        - FlashText keyword processor
        - RAKE for keyphrase extraction
        - TF-IDF vectorizer placeholder
        - Prepare for fuzzy matching
        """
        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info("Loaded spaCy model: %s", spacy_model)
        except Exception as e:
            logger.error("spaCy load failed: %s", e)
            raise

        # PhraseMatcher for dictionary phrases
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        if custom_phrases:
            patterns = [self.nlp.make_doc(text) for text in custom_phrases]
            self.phrase_matcher.add("CUSTOM_PHRASES", patterns)
            logger.info("Registered %d custom phrases", len(custom_phrases))

        # FlashText for fast keyword spotting
        self.flash = KeywordProcessor(case_sensitive=False)
        if flash_keywords:
            for kw in flash_keywords:
                self.flash.add_keyword(kw)
            logger.info("Loaded %d flashtext keywords", len(flash_keywords))

        # RAKE for keyphrase extraction
        self.rake = Rake()

        # TF-IDF vectorizer (to be fit at runtime)
        self.tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=50)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Run spaCy NER and return a dict: label -> list of entities.
        """
        ents: Dict[str, List[str]] = {}
        doc = self.nlp(text)
        for ent in doc.ents:
            ents.setdefault(ent.label_, []).append(ent.text)
        logger.info("Extracted %d entities", len(doc.ents))
        return ents

    def extract_phrases(self, text: str) -> List[str]:
        """
        Match custom phrases via PhraseMatcher.
        """
        doc = self.nlp(text)
        matches = self.phrase_matcher(doc)
        phrases = [doc[start:end].text for _, start, end in matches]
        logger.info("Found %d phrase matches", len(phrases))
        return phrases

    def flash_keywords(self, text: str) -> List[str]:
        """
        Return all FlashText keywords spotted in the text.
        """
        found = self.flash.extract_keywords(text)
        logger.info("FlashText found %d keywords", len(found))
        return found

    def extract_keyphrases(self, text: str, max_phrases: int = 20) -> List[str]:
        """
        Use RAKE to extract and rank keyphrases.
        """
        self.rake.extract_keywords_from_text(text)
        phrases = self.rake.get_ranked_phrases()[:max_phrases]
        logger.info("RAKE extracted %d keyphrases", len(phrases))
        return phrases

    def extract_tfidf_keywords(self, docs: List[str], top_n: int = 10) -> List[str]:
        """
        Fit TF-IDF on provided docs and return top terms.
        """
        tfidf_matrix = self.tfidf.fit_transform(docs)
        feature_names = self.tfidf.get_feature_names_out()
        scores = tfidf_matrix.sum(axis=0).A1
        ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
        keywords = [term for term, _ in ranked[:top_n]]
        logger.info("TF-IDF top %d terms", len(keywords))
        return keywords

    def fuzzy_match(self, text: str, choices: List[str], limit: int = 5, threshold: int = 70) -> List[str]:
        """
        Use RapidFuzz to fuzzy-match choices in text. Returns matches above threshold.
        """
        matches = fuzzy_process.extract(text, choices, limit=limit)
        result = [m[0] for m in matches if m[1] >= threshold]
        logger.info("Fuzzy matched %d choices", len(result))
        return result

    def analyze(
        self,
        text: str,
        docs_for_tfidf: Optional[List[str]] = None,
        fuzzy_choices: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run full pipeline: entities, phrases, flashtext, RAKE, TF-IDF, fuzzy.
        """
        result: Dict[str, Any] = {}
        try:
            result['entities'] = self.extract_entities(text)
            result['phrases']  = self.extract_phrases(text)
            result['flash']    = self.flash_keywords(text)
            result['rake']     = self.extract_keyphrases(text)
            if docs_for_tfidf:
                result['tfidf']  = self.extract_tfidf_keywords(docs_for_tfidf)
            if fuzzy_choices:
                result['fuzzy']  = self.fuzzy_match(text, fuzzy_choices)
        except Exception as e:
            logger.error("Error in analyze pipeline: %s", e)
        return result
