"""Build the written analysis report (Markdown, English + Indonesian headings)
from real evaluation outputs.  Manual annotations are read from
``report/annotations.json`` (``{"<sentence id>": {"phenomenon": "...", "analysis": "..."}}``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .mt_eval import Scores

PHENOMENA = """\
| Phenomenon | English -> Indonesian illustration |
|---|---|
| Structural disparity / D-M vs M-D | EN modifier-head *"the new red car"* vs ID head-modifier *"mobil merah baru"*; EN possessive *"John's book"* vs ID *"buku John"*; compound nouns *"city council member"* -> *"anggota dewan kota"* (order reverses) |
| Lack of tense / aspect morphology | EN *"he walked / has walked"* -> ID *"dia berjalan / sudah berjalan"* (aspect carried by adverbs *sudah, telah, sedang*; wrong choice = fluent but different meaning) |
| Voice and affixation | ID active *me-* / passive *di-* / *ter-* / *ke-an*; MT may pick the wrong voice or mis-affix (*mem-pengaruhi* vs *dipengaruhi*) |
| Polysemy / lexical ambiguity | *"bank"* (bank / tepi sungai), *"run"* (berlari / menjalankan / mencalonkan diri), *"party"* (pesta / partai) |
| Lexical gap | no one-word ID equivalent (*"brunch"*, *"inkjet"*, kinship/institution terms) -> borrowing, paraphrase or omission |
| Pronoun / politeness system | EN *"you"* -> ID *Anda / kamu / kalian*; inclusive vs exclusive *kita / kami* (EN has only *we*) |
| Plurality | EN plural *-s* vs ID reduplication (*buku-buku*) or numerals/quantifiers (*banyak buku*) |
| Classifiers | *"three people"* -> *"tiga orang"*, *"two cars"* -> *"dua buah mobil"* |
| Named entities, numbers, units | copied, wrongly translated, or dropped; ID uses *"."* for thousands and *","* for decimals |
| Fluency/register | formal vs informal (baku vs tidak baku), calques from English |
"""


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, float) else str(v).replace("|", "\\|").replace("\n", " "))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def build_report(model_scores: dict[str, Scores], model_sentences: dict[str, pd.DataFrame],
                 worst: dict[str, pd.DataFrame], n_sentences: int, seed: int,
                 ci: dict[str, dict[str, tuple[float, float, float]]] | None = None,
                 annotations_path: str | Path = "report/annotations.json",
                 dry_run: bool = False) -> str:
    ann = {}
    p = Path(annotations_path)
    if p.exists():
        ann = json.loads(p.read_text(encoding="utf-8"))

    L = []
    L.append("# Laporan Tugas Kelompok 2 — Evaluasi Otomatis & Analisis Error MT (EN→ID)\n")
    if dry_run:
        L.append("> ⚠️ **DRY RUN** — the hypotheses were synthetic (corrupted references), NOT real MT output. "
                 "Re-run with a real model before submitting.\n")
    L.append("## 1. Metodologi\n")
    L.append(f"* **Dataset**: FLORES-200 parallel EN–ID (2009 pairs available); **{n_sentences}** pairs randomly sampled (seed={seed}).")
    L.append("* **Models**: " + ", ".join(model_scores.keys()) + " (pre-trained, beam search = 4).")
    L.append("* **Metrics**: BLEU-1…BLEU-4 (cumulative, sacreBLEU, 13a tokenisation, no smoothing at corpus level), chrF (char 6-gram, β=2), chrF++ (+ word bigrams), TER (lower = better), "
             "plus per-sentence BLEU/chrF/TER for the error analysis.")
    L.append("* **Architecture recap (Bab 13)**: the models are Transformer encoder–decoder NMT systems. The encoder builds contextual representations of the English "
             "sentence; the decoder generates Indonesian tokens autoregressively, attending to encoder states (cross-attention) at each step.\n")

    L.append("## 2. Hasil Metrik Korpus\n")
    names = list(model_scores)
    merged = None
    for n in names:
        t = model_scores[n].table[["metric", "score"]].rename(columns={"score": n})
        merged = t if merged is None else merged.merge(t, on="metric")
    L.append(_md_table(merged) + "\n")
    L.append("TER is an error rate — **lower is better**; every other metric is higher-is-better.\n")
    if ci:
        L.append("### 95 % bootstrap confidence intervals\n")
        rows = []
        for n, d in ci.items():
            for m, (v, lo, hi) in d.items():
                rows.append({"model": n, "metric": m, "score": v, "CI low": lo, "CI high": hi})
        L.append(_md_table(pd.DataFrame(rows)) + "\n")

    L.append("## 3. Pengaruh Panjang n-gram terhadap BLEU\n")
    for n in names:
        t = model_scores[n].table.set_index("metric")["score"]
        pr = model_scores[n].precisions
        b = [t[f"BLEU-{k}"] for k in range(1, 5)]
        L.append(f"**{n}**: BLEU-1 = {b[0]:.2f}, BLEU-2 = {b[1]:.2f}, BLEU-3 = {b[2]:.2f}, BLEU-4 = {b[3]:.2f} "
                 f"(drop 1→4: {b[0]-b[3]:.2f} points, {100*(b[0]-b[3])/b[0] if b[0] else 0:.1f} % relative). "
                 f"Individual precisions: " + ", ".join(f"{k+1}-gram {v:.1f} %" for k, v in enumerate(pr['precision (%)'])) +
                 f". Brevity penalty = {pr.attrs.get('bp', float('nan')):.3f}.\n")
    L.append("**Interpretation.** BLEU-n is the geometric mean of the 1…n-gram precisions times the brevity penalty, so adding higher orders can only add "
             "stricter factors. Unigram precision rewards lexical choice alone; bigram–4-gram precision additionally requires the *same local word order*. "
             "Because English and Indonesian differ in modifier–head order, relative-clause marking (*yang*), tense/aspect expression and affixation, "
             "matches of long n-grams are much rarer, hence the monotone decrease. A large gap between BLEU-1 and BLEU-4 signals correct vocabulary "
             "with different ordering/phrasing — which can be a *valid* alternative translation or a real word-order error.\n")
    L.append("![n-gram effect](../results/ngram_effect.png)\n")

    L.append("## 4. Analisis Error Kualitatif\n")
    L.append("Selection: the lowest-scoring sentences by rank-average of sentence-BLEU and sentence-chrF (sentences with ≥ 6 words). "
             "The *auto-flags* are heuristics; the *phenomenon / analysis* columns are the human linguistic judgement "
             "(edit `report/annotations.json` or the notebook cell, then rebuild the report).\n")
    L.append("### 4.1 Linguistic phenomena checklist\n")
    L.append(PHENOMENA)
    for n in names:
        L.append(f"### 4.2 Lowest-scoring sentences — {n}\n")
        w = worst[n]
        for _, r in w.iterrows():
            a = ann.get(str(r["id"]), {})
            L.append(f"**#{r['id']}** — sBLEU {r['sBLEU']:.1f} · sChrF {r['sChrF']:.1f} · sTER {r['sTER']:.1f}\n")
            L.append(f"* EN (source): {r['source']}")
            L.append(f"* ID (reference): {r['reference']}")
            L.append(f"* ID (MT): {r['hypothesis']}")
            L.append(f"* Auto-flags: {r['flags'] or '—'}")
            L.append(f"* **Phenomenon**: {a.get('phenomenon', '_TODO — annotate_')}")
            L.append(f"* **Analysis**: {a.get('analysis', '_TODO — explain what differs and why it is (or is not) a real error_')}\n")

    L.append("## 5. Keterbatasan BLEU dibanding Kualitas Terjemahan Aktual\n")
    L.append("""\
1. **Surface matching only.** BLEU counts exact n-gram overlap; synonyms (*mobil/kendaraan*, *kediaman/rumah dinas*), paraphrases and valid reorderings get zero credit. A correct translation can score near 0 when it differs from the single reference.
2. **Single reference.** FLORES provides one human translation, but a sentence has many acceptable Indonesian renderings (active/passive, *yang* relative clauses, formal/informal vocabulary). BLEU measures similarity to *that* translator's choices, not adequacy.
3. **Morphology-blind.** Indonesian affixation (*me-, ber-, di-, -kan, -an*) makes a one-letter difference a total mismatch for BLEU (*mengumumkan* vs *diumumkan*), although chrF partially credits it — the reason chrF correlates better with human judgement for morphologically rich languages.
4. **Insensitive to error severity.** Dropping a negation (*tidak*), changing a number, or mistranslating a named entity costs one token, yet destroys meaning; a harmless synonym costs the same.
5. **Fluency and word order are weakly modelled.** Only local (≤ 4-gram) order is checked; global structure, agreement and register are ignored. A scrambled sentence with the right words keeps a decent BLEU-1.
6. **Tokenisation and corpus-level dependence.** Scores depend on the tokenizer/smoothing, are unreliable per sentence (many zero 4-gram counts), and are meaningful mainly as corpus-level comparisons with confidence intervals.
7. **Brevity penalty is crude.** It punishes short output but does not tell *what* is missing.

**Complementary metrics.** chrF/chrF++ (character-level, tolerant to affixes), TER (edit effort ≈ post-editing cost, interpretable), and embedding-based metrics (BERTScore, COMET) capture meaning better but still need human evaluation for adequacy/fluency.
""")
    L.append("## 6. Kesimpulan\n")
    L.append("_Write 4–6 sentences here: which model scored higher on which metric, how BLEU fell from n=1 to n=4, the dominant error types you found "
             "(e.g. word order, polysemy, lexical gaps, named entities), and why BLEU alone is insufficient._\n")
    return "\n".join(L)
