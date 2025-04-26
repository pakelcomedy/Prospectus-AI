```
prospectus_ai/
├── app/
│   ├── main.py               # Entry point (Flask / FastAPI API handler)
│   ├── routes.py             # Endpoint logic (misal /summarize)
│   └── config.py             # API keys, model paths, env vars
│
├── core/
│   ├── extractor.py          # Ekstraksi teks PDF (pdfplumber / fitz)
│   ├── parser.py             # Rule-based parsing dan regex angka/rasio
│   ├── nlp.py                # Named Entity Recognition + keyword spotting
│   ├── summarizer.py         # Ringkasan otomatis (transformers / OpenAI)
│
├── data/
│   ├── sample.pdf            # Contoh dokumen PDF prospektus
│   └── output.json           # Output hasil ekstraksi/summarization
│
├── models/
│   └── ner_model/            # Optional: model spaCy custom untuk keuangan
│
├── tests/
│   ├── test_extractor.py
│   ├── test_parser.py
│   ├── test_summarizer.py
│
├── requirements.txt
├── .env                      # API Key / secrets
└── README.md
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
