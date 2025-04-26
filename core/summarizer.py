# summarizer.py

import os
import logging
from typing import List, Optional, Dict, Any
from functools import lru_cache

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline as hf_pipeline,
    Pipeline,
)
import openai  # pip install openai
import tiktoken  # pip install tiktoken

# ----------------------------------------
# Logger Setup
# ----------------------------------------
def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger("summarizer")

# ----------------------------------------
# Helper: Tokenizer/Model Loader with Caching
# ----------------------------------------
@lru_cache(maxsize=4)
def load_hf_model(model_name: str) -> Pipeline:
    """
    Load a HuggingFace summarization pipeline.
    Cached to avoid reload penalties.
    """
    logger.info("Loading HF model '%s'...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = hf_pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=0 if torch.cuda.is_available() else -1,
    )
    logger.info("Model '%s' loaded, device='%s'", model_name,
                "cuda" if torch.cuda.is_available() else "cpu")
    return pipe

# ----------------------------------------
# Helper: OpenAI Chat Completion Summarizer
# ----------------------------------------
class OpenAISummarizer:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        openai.api_key = api_key
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(model)
        logger.info("Initialized OpenAI summarizer with model '%s'", model)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def summarize(self,
                  text: str,
                  prompt: Optional[str] = None,
                  max_tokens: int = 512,
                  temperature: float = 0.0) -> str:
        """
        Summarize with GPT via ChatCompletion. Automatically chunk if too long.
        """
        prompt = prompt or (
            "Please provide a concise, structured summary of the following text:"
        )
        # maximum context tokens minus output tokens
        max_ctx = self.tokenizer.max_token_limit if hasattr(self.tokenizer, 'max_token_limit') else 8192
        chunk_size = max_ctx - max_tokens - 200  # buffer

        # split into approximate chunks
        words = text.split()
        chunks = []
        current = []
        acc = 0
        for w in words:
            acc += len(self.tokenizer.encode(w))
            if acc > chunk_size:
                chunks.append(" ".join(current))
                current = [w]
                acc = len(self.tokenizer.encode(w))
            else:
                current.append(w)
        if current:
            chunks.append(" ".join(current))

        # summarize each chunk
        summaries = []
        for idx, chunk in enumerate(chunks):
            logger.info("OpenAI summarizer: processing chunk %d/%d", idx+1, len(chunks))
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": chunk}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            summ = resp.choices[0].message.content.strip()
            summaries.append(summ)

        # combine summaries
        final = "\n\n".join(summaries)
        return final

# ----------------------------------------
# Main Summarizer Class
# ----------------------------------------
class ProspectusSummarizer:
    def __init__(self,
                 hf_model_name: str = "facebook/bart-large-cnn",
                 use_openai: bool = False,
                 openai_api_key: Optional[str] = None,
                 openai_model: str = "gpt-4"):
        """
        - hf_model_name: HF model for local summarization
        - use_openai: whether to fallback or prefer OpenAI summarization
        """
        self.hf_model = load_hf_model(hf_model_name)
        self.use_openai = use_openai
        self.openai_summarizer = None
        if use_openai:
            if not openai_api_key:
                raise ValueError("OpenAI API key required if use_openai=True")
            self.openai_summarizer = OpenAISummarizer(openai_api_key, openai_model)

    def _chunk_text(self, text: str, max_tokens: int = 1024) -> List[str]:
        """
        Chunk text by sentences to fit HF model constraints.
        """
        tokenizer = self.hf_model.tokenizer
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current = ""
        for sent in sentences:
            tok = tokenizer.encode(current + " " + sent, return_tensors="pt")
            if tok.size(1) > max_tokens:
                chunks.append(current.strip())
                current = sent
            else:
                current = (current + " " + sent).strip()
        if current:
            chunks.append(current)
        logger.info("Text chunked into %d parts for HF model", len(chunks))
        return chunks

    def summarize_hf(self,
                     text: str,
                     chunk_max_tokens: int = 1024,
                     summary_max_length: int = 150,
                     summary_min_length: int = 50,
                     **kwargs) -> str:
        """
        Summarize using HuggingFace model with chunking.
        """
        chunks = self._chunk_text(text, max_tokens=chunk_max_tokens)
        summaries = []
        for idx, chunk in enumerate(chunks):
            logger.info("HF summarizer: chunk %d/%d", idx+1, len(chunks))
            summ = self.hf_model(
                chunk,
                max_length=summary_max_length,
                min_length=summary_min_length,
                do_sample=False,
                **kwargs
            )[0]['summary_text']
            summaries.append(summ.strip())
        return "\n\n".join(summaries)

    def summarize(self,
                  text: str,
                  method: str = "auto",
                  **kwargs) -> str:
        """
        method: 'hf', 'openai', or 'auto'
        """
        if method == "hf":
            return self.summarize_hf(text, **kwargs)
        if method == "openai":
            if not self.openai_summarizer:
                raise RuntimeError("OpenAI summarizer not initialized")
            return self.openai_summarizer.summarize(text, **kwargs)
        # auto: prefer OpenAI if available
        if self.use_openai:
            try:
                return self.summarize_openai(text, **kwargs)
            except Exception as e:
                logger.warning("OpenAI summarization failed: %s; falling back to HF", e)
        return self.summarize_hf(text, **kwargs)

# ----------------------------------------
# CLI / Example Usage
# ----------------------------------------
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Summarize prospectus text")
    parser.add_argument("txt", help="Path to plain-text file")
    parser.add_argument("-o", "--output", help="Output JSON file", default="summary.json")
    parser.add_argument("--method", choices=["hf", "openai", "auto"], default="auto")
    parser.add_argument("--hf_model", default="facebook/bart-large-cnn")
    parser.add_argument("--openai_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--openai_model", default="gpt-4")
    args = parser.parse_args()

    text = open(args.txt, encoding="utf8").read()
    summarizer = ProspectusSummarizer(
        hf_model_name=args.hf_model,
        use_openai=(args.method in ["openai", "auto"]),
        openai_api_key=args.openai_key,
        openai_model=args.openai_model
    )
    summary = summarizer.summarize(text, method=args.method)
    with open(args.output, "w", encoding="utf8") as f:
        json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)
    logger.info("Summary saved to %s", args.output)