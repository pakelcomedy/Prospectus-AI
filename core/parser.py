#!/usr/bin/env python3
"""
Advanced Prospectus Parser

This script parses a financial prospectus text for IPO information,
timeline events, financial ratios, sector classification, business model,
risk factors, shareholder structure, and more, and outputs structured JSON.
"""
from typing import Dict, Any

import argparse
import json
import logging
import re
import sys

# Try importing spaCy
try:
    import spacy
except ImportError as e:
    print("spaCy is required. Please install spaCy (pip install spacy) and the English model (python -m spacy download en_core_web_sm).")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

def clean_and_normalize(text):
    """
    Perform basic text cleaning and normalization.
    """
    logging.info("Cleaning and normalizing text")
    # Replace smart quotes with straight quotes
    text = text.replace('’', "'").replace('“', '"').replace('”', '"')
    # Standardize newlines
    text = re.sub(r'\r\n|\r', '\n', text)
    # Collapse multiple newlines and spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def parse_ipo_structure(text):
    """
    Extract IPO structure details: public shares, warrants, price range, target raise.
    """
    logging.info("Parsing IPO structure")
    ipo_info = {}
    try:
        match = re.search(r'(?:public offering of|offering of)\s+([\d,\.]+\s*(?:million|billion|thousand|hundred)?\s*shares?)', text, re.IGNORECASE)
        if match:
            ipo_info['public_shares'] = match.group(1).strip(' .,')
        match = re.search(r'warrants\s*(?:at\s*\$?([\d,\.]+))?', text, re.IGNORECASE)
        if match:
            if match.group(1):
                ipo_info['warrant_price'] = match.group(1).strip(' .,')
            else:
                ipo_info['warrants'] = True
        match = re.search(r'price range\s*(?:is)?\s*\$?([\d,\.]+)\s*(?:to|-)\s*\$?([\d,\.]+)', text, re.IGNORECASE)
        if match:
            ipo_info['price_low'] = match.group(1).strip(' .,')
            ipo_info['price_high'] = match.group(2).strip(' .,')
        match = re.search(r'target(?:ing)?\s+(?:raising\s*)?\$?([\d,\.]+\s*(?:million|billion)?)', text, re.IGNORECASE)
        if match:
            ipo_info['target_raising'] = match.group(1).strip(' .,')
        match = re.search(r'fundraising\s*of\s*\$?([\d,\.]+\s*(?:million|billion)?)', text, re.IGNORECASE)
        if match:
            ipo_info['target_raising'] = match.group(1).strip(' .,')
    except Exception as e:
        logging.error("Error extracting IPO structure: %s", e)
    return ipo_info

def extract_timeline(text):
    """
    Extract key IPO timeline dates: bookbuilding, effective date, allotment, listing.
    """
    logging.info("Extracting timeline events")
    timeline = {}
    try:
        patterns = {
            'bookbuilding': r'book\s*building.*?(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
            'effective_date': r'effective\s+date.*?(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
            'allotment': r'allotment.*?(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
            'listing': r'listing.*?(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
        }
        for event, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                timeline[event] = match.group(1)
                logging.debug("Found %s date: %s", event, match.group(1))
    except Exception as e:
        logging.error("Error extracting timeline: %s", e)
    return timeline

def parse_financial_ratios(text):
    """
    Extract financial ratios: PBV, PER, Market Cap.
    """
    logging.info("Parsing financial ratios")
    ratios = {}
    try:
        match = re.search(r'P/?B\s+ratio.*?([\d\.]+)', text, re.IGNORECASE)
        if match:
            ratios['PBV'] = float(match.group(1))
        else:
            match = re.search(r'PBV\s*[:=]?\s*\$?([\d\.]+)', text, re.IGNORECASE)
            if match:
                ratios['PBV'] = float(match.group(1))
        match = re.search(r'P/?E\s+ratio.*?([\d\.]+)', text, re.IGNORECASE)
        if match:
            ratios['PER'] = float(match.group(1))
        else:
            match = re.search(r'PER\s*[:=]?\s*([\d\.]+)', text, re.IGNORECASE)
            if match:
                ratios['PER'] = float(match.group(1))
        # Market cap
        match = re.search(r'Market\s+Capitalization.*?([\d\.,]+\s*(?:million|billion)?)', text, re.IGNORECASE)
        if match:
            cap_text = match.group(1)
            cap_text = cap_text.replace(',', '')
            num_match = re.match(r'([\d\.]+)', cap_text)
            unit = None
            unit_match = re.search(r'(million|billion)', cap_text, re.IGNORECASE)
            if num_match:
                num = float(num_match.group(1))
                if unit_match:
                    unit = unit_match.group(1).lower()
                    if 'billion' in unit:
                        num *= 1e9
                    elif 'million' in unit:
                        num *= 1e6
                ratios['market_cap'] = num
            else:
                ratios['market_cap'] = cap_text
    except Exception as e:
        logging.error("Error parsing financial ratios: %s", e)
    return ratios

def classify_sector(text):
    """
    Classify business sector based on keywords.
    """
    logging.info("Classifying business sector")
    try:
        sectors = {
            'Technology': ['software', 'technology', 'internet', 'tech', 'hardware', 'semiconductor', 'it', 'telecom'],
            'Finance': ['bank', 'financial', 'insurance', 'fund', 'investment', 'asset', 'pension'],
            'Healthcare': ['healthcare', 'pharma', 'hospital', 'clinic', 'medical', 'biotech', 'vaccine', 'pharmaceutical'],
            'Energy': ['oil', 'gas', 'energy', 'petroleum', 'coal', 'mining', 'power'],
            'Consumer': ['retail', 'consumer', 'food', 'beverage', 'restaurant', 'fmcg', 'apparel', 'clothing', 'outlet'],
            'Industrial': ['manufacturing', 'construction', 'industrial', 'automotive', 'machinery', 'chemical'],
            'Real Estate': ['real estate', 'property', 'estate', 'construction'],
            'Utilities': ['electricity', 'utility', 'water', 'sewer', 'waste'],
            'Telecom': ['telecom', 'telecommunications', 'mobile', 'broadband'],
            'Agriculture': ['agri', 'farming', 'farm', 'agribusiness'],
            'Media': ['media', 'newspaper', 'television', 'radio', 'social media'],
            'Transportation': ['airlines', 'logistics', 'shipping', 'transportation', 'airport'],
            'Entertainment': ['entertainment', 'gaming', 'casino', 'movie', 'music'],
        }
        text_lower = text.lower()
        scores = {}
        for sector, keywords in sectors.items():
            for keyword in keywords:
                # use word boundary to avoid partial matches
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                    scores[sector] = scores.get(sector, 0) + 1
        if not scores:
            return 'Unknown'
        max_score = max(scores.values())
        top_sectors = [s for s, sc in scores.items() if sc == max_score]
        return top_sectors[0] if top_sectors else 'Unknown'
    except Exception as e:
        logging.error("Error classifying sector: %s", e)
        return 'Unknown'

def extract_business_model(text):
    """
    Extract business model description, outlet counts, unique features.
    """
    logging.info("Extracting business model and outlets")
    info = {}
    try:
        model_match = re.search(r'(Our\s+business\s+model\s+is.*?[.])', text, re.IGNORECASE)
        if model_match:
            info['model_description'] = model_match.group(1).strip()
        else:
            # fallback: first sentence
            first_sentence = re.split(r'\. ', text)
            if first_sentence:
                info['model_description'] = first_sentence[0].strip() + '.'
        outlets = {}
        match = re.search(r'([\d,]+)\s+stores?', text, re.IGNORECASE)
        if match:
            count = int(match.group(1).replace(',', ''))
            outlets['stores'] = count
        match = re.search(r'([\d,]+)\s+outlets?', text, re.IGNORECASE)
        if match:
            count = int(match.group(1).replace(',', ''))
            outlets['outlets'] = count
        if outlets:
            info['outlets'] = outlets
        unique_match = re.search(r'([^.]*unique[^.]*\.)', text, re.IGNORECASE)
        if unique_match:
            info['unique_features'] = unique_match.group(1).strip()
    except Exception as e:
        logging.error("Error extracting business model: %s", e)
    return info

def perform_ner(nlp, text):
    """
    Perform named entity recognition on the text.
    """
    logging.info("Performing Named Entity Recognition (NER)")
    entities = {}
    try:
        doc = nlp(text)
        for ent in doc.ents:
            label = ent.label_
            entities.setdefault(label, set()).add(ent.text)
        # Convert sets to sorted lists for consistency
        for label in entities:
            entities[label] = sorted(entities[label])
    except Exception as e:
        logging.error("Error during NER: %s", e)
    return entities

def extract_risk_factors(nlp, text):
    """
    Extract sentences related to risk factors.
    """
    logging.info("Extracting risk factor sentences")
    risk_sentences = []
    try:
        doc = nlp(text)
        for sent in doc.sents:
            if 'risk' in sent.text.lower():
                risk_sentences.append(sent.text.strip())
    except Exception as e:
        logging.error("Error extracting risk factors: %s", e)
    return risk_sentences

def score_risk(risk_texts):
    """
    Compute a simple risk score based on risk-related keywords.
    """
    logging.info("Scoring risk level")
    risk_words = ['risk', 'loss', 'uncertain', 'volatility', 'debt',
                  'bankruptcy', 'liability', 'default', 'competition',
                  'decline', 'regulatory', 'legal', 'environmental']
    score = 0
    try:
        for sent in risk_texts:
            text_lower = sent.lower()
            for word in risk_words:
                score += text_lower.count(word)
    except Exception as e:
        logging.error("Error scoring risk: %s", e)
    return score

def sentiment_of_risks(risk_texts):
    """
    Perform a simple sentiment analysis on risk factor text.
    """
    logging.info("Analyzing sentiment of risk sections")
    positive_words = ['opportunity', 'benefit', 'strong', 'growth', 'increase', 'improve', 'good', 'positive']
    negative_words = ['risk', 'loss', 'decline', 'problem', 'fail', 'weak', 'drop', 'negative', 'concern', 'threat']
    pos_count = neg_count = 0
    try:
        for sent in risk_texts:
            text_lower = sent.lower()
            for word in positive_words:
                pos_count += text_lower.count(word)
            for word in negative_words:
                neg_count += text_lower.count(word)
        if neg_count > pos_count:
            return 'Negative'
        elif pos_count > neg_count:
            return 'Positive'
        else:
            return 'Neutral'
    except Exception as e:
        logging.error("Error analyzing sentiment: %s", e)
        return 'Neutral'

def extract_shareholders(text):
    """
    Extract majority shareholder names and percentages.
    """
    logging.info("Extracting shareholder structure")
    shareholders = []
    try:
        match = re.search(r'major shareholders include (.+?)\.', text, re.IGNORECASE)
        if match:
            segment = match.group(1)
            parts = re.split(r'\s*and\s*|\s*,\s*', segment)
            for part in parts:
                part = part.strip()
                name_match = re.match(r'(.+)\s*\(?(\d{1,3}\.?\d*)%\s*\)?', part)
                if name_match:
                    name = name_match.group(1).strip()
                    percentage = name_match.group(2)
                    shareholders.append({'name': name, 'percentage': percentage})
        else:
            # fallback: find all occurrences of name (%)
            matches = re.findall(r'([A-Za-z][A-Za-z &,\.\'-]+)\s*\(\s*(\d{1,3}\.?\d*)%\s*\)', text)
            for name, perc in matches:
                shareholders.append({'name': name.strip(), 'percentage': perc})
    except Exception as e:
        logging.error("Error extracting shareholders: %s", e)
    return shareholders

def generate_summary(nlp, text, max_sentences=3):
    """
    Generate a short summary by selecting key sentences.
    """
    logging.info("Generating summary of prospectus")
    summary = ""
    try:
        doc = nlp(text)
        sentences = list(doc.sents)
        if not sentences:
            return summary
        # Score sentences by number of named entities
        sent_scores = []
        for sent in sentences:
            # count entities excluding dates, numbers
            ents = [ent for ent in sent.ents if ent.label_ not in ('DATE','CARDINAL','NUM','PERCENT','MONEY')]
            sent_scores.append((len(ents), sent.text.strip()))
        # Sort by score
        sent_scores.sort(reverse=True, key=lambda x: x[0])
        selected = []
        # Always include first sentence
        first_sent = sentences[0].text.strip()
        selected.append(first_sent)
        for score, sent_text in sent_scores:
            if sent_text != first_sent and sent_text not in selected:
                selected.append(sent_text)
            if len(selected) >= max_sentences:
                break
        summary = " ".join(selected[:max_sentences]).strip()
    except Exception as e:
        logging.error("Error generating summary: %s", e)
    return summary

def main():
    parser = argparse.ArgumentParser(description="Prospectus Parser CLI")
    parser.add_argument('-i', '--input', required=True, help='Path to input text file of prospectus')
    parser.add_argument('-o', '--output', required=True, help='Path to output JSON file')
    args = parser.parse_args()

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        logging.info("Input text loaded")
    except Exception as e:
        logging.error("Failed to read input file: %s", e)
        sys.exit(1)

    text = clean_and_normalize(raw_text)

    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        logging.error("spaCy model loading failed: %s", e)
        sys.exit(1)

    # Parse sections
    ipo_data = parse_ipo_structure(text)
    timeline = extract_timeline(text)
    ratios = parse_financial_ratios(text)
    sector = classify_sector(text)
    business_info = extract_business_model(text)
    entities = perform_ner(nlp, text)
    risk_sentences = extract_risk_factors(nlp, text)
    risk_score = score_risk(risk_sentences)
    risk_sentiment = sentiment_of_risks(risk_sentences)
    shareholders = extract_shareholders(text)
    summary = generate_summary(nlp, text)

    # Assemble output JSON
    output = {
        'ipo_structure': ipo_data,
        'timeline': timeline,
        'financial_ratios': ratios,
        'sector': sector,
        'business_model': business_info.get('model_description'),
        'outlets': business_info.get('outlets'),
        'unique_features': business_info.get('unique_features'),
        'entities': entities,
        'risk_factors': risk_sentences,
        'risk_score': risk_score,
        'risk_sentiment': risk_sentiment,
        'shareholders': shareholders,
        'summary': summary
    }

    # Write output JSON
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        logging.info("Output written to %s", args.output)
    except Exception as e:
        logging.error("Failed to write output file: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()

class ProspectusParser:
    """
    Wrapper class so we can import a single parser object.
    """
    def __init__(self, nlp_model: str = "en_core_web_sm"):
        self.nlp_model = nlp_model

    def parse_all(self, text: str) -> Dict[str, Any]:
        # you probably want to reuse one spaCy load rather than loading 5 times
        import spacy
        nlp = spacy.load(self.nlp_model)
        return {
            "ipo_structure":        parse_ipo_structure(text),
            "timeline":             extract_timeline(text),
            "financial_ratios":     parse_financial_ratios(text),
            "sector":               classify_sector(text),
            "business_model":       extract_business_model(text).get("model_description"),
            "outlets":              extract_business_model(text).get("outlets"),
            "unique_features":      extract_business_model(text).get("unique_features"),
            "entities":             perform_ner(nlp, text),
            "risk_factors":         extract_risk_factors(nlp, text),
            "risk_score":           score_risk(extract_risk_factors(nlp, text)),
            "risk_sentiment":       sentiment_of_risks(extract_risk_factors(nlp, text)),
            "shareholders":         extract_shareholders(text),
            "summary":              generate_summary(nlp, text),
        }