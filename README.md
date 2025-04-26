```
prospectus_ai/
├── app/
│   ├── templates/
│   │   └── index.html          # Frontend HTML page
│   ├── main.py                 # Entry point (Flask app creation)
│   └── routes.py               # API endpoint logic (e.g., /summarize)
│
├── core/
│   ├── extractor.py            # PDF text extraction (pdfplumber, fitz)
│   ├── parser.py               # Rule-based parsing and regex for numbers/ratios
│   ├── nlp.py                  # Named Entity Recognition & keyword spotting
│   └── summarizer.py           # Automatic summarization (transformers, OpenAI)
│
├── .env                        # Environment variables (API keys, secrets)
├── .gitattributes              # Git attributes (optional, for LFS or text settings)
├── .gitignore                  # Files/folders to ignore in git
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── run.py                      # Production entry point (import from app.main)
```

---

| Modul         | Fungsi                                                                 |
|---------------|------------------------------------------------------------------------|
| `extractor.py`| Ambil teks dari PDF (multi-page)                                       |
| `parser.py`   | Ambil angka penting: jumlah saham, dana, PBV, PER, % kepemilikan       |
| `nlp.py`      | NER (spaCy) + keyword search (misal: “modal kerja”, “capex”)          |
| `summarizer.py`| Gunakan model BART, T5, atau OpenAI buat ringkasan (bisa per bab)    |
| `routes.py`   | Endpoint `/upload` → `extract + parse + summarize` → return JSON      |

---

1. User upload PDF
2. `extractor.py` ambil teks semua halaman
3. `parser.py` cari angka rasio, struktur IPO, harga, jumlah saham
4. `nlp.py` cari nama entitas: perusahaan, pengendali, RS, lokasi, dll
5. `summarizer.py` → ringkas dokumen jadi 1 paragraf atau per bagian
6. JSON output siap: bisa disimpan, dikirim ke front-end, atau dikaji lanjut

---
