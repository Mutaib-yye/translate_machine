"""Generates notebooks/MT_Evaluation_EN_ID.ipynb (run: python notebooks/build_notebook.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip("\n")))
M = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))

M("""
# Tugas Kelompok 2 — Evaluasi Otomatis & Analisis Error MT (EN → ID)

Pipeline: **FLORES-200 EN–ID pairs → pre-trained NMT (MarianMT / NLLB) → BLEU-1…4, chrF, chrF++, TER (+BERTScore) → n-gram analysis → worst-sentence error analysis → report**.

Run top to bottom. On **Google Colab** use *Runtime → GPU* and run the setup cell first.
""")
M("## 0. Setup")
C("""
import sys, os, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and not os.path.exists("translate_machine"):
    subprocess.run(["git", "clone", "https://github.com/Mutaib-yye/translate_machine"], check=True)
    os.chdir("translate_machine")
elif os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")                      # run from the repo root
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
sys.path.insert(0, os.getcwd())
print("cwd:", os.getcwd())
""")
M("## 1. Configuration")
C("""
import os
N_SENTENCES = 200          # prof: every member may choose; 0 = use all 2009 pairs
SEED        = 42
MODELS      = ["marian", "nllb"]   # "marian" = Helsinki-NLP/opus-mt-en-id, "nllb" = facebook/nllb-200-distilled-600M
NUM_BEAMS   = 4
BATCH_SIZE  = 16
WORST_K     = 10           # >= 5 required by the assignment
USE_BERTSCORE = False      # extra metric (downloads xlm-roberta); set True for the bonus
BOOTSTRAP   = 300          # resamples for 95% confidence intervals (0 = skip)
DRY_RUN     = os.environ.get("MT_DRY_RUN") == "1"   # offline smoke test only -> NOT real results
OUT = "results_dryrun" if DRY_RUN else "results"
os.makedirs(OUT, exist_ok=True); os.makedirs("report", exist_ok=True)
""")
M("## 2. Data — EN→ID parallel sentences (FLORES-200)\nMojibake in the references (e.g. `BarÃ§a`) is repaired on load.")
C("""
import pandas as pd
from src import mt_eval as m
pd.set_option("display.max_colwidth", 140, "display.width", 200)
full = m.load_pairs()
df = m.sample_pairs(full, N_SENTENCES, SEED)
print(f"{len(full)} pairs available, using {len(df)}")
srcs, refs = df.source.tolist(), df.reference.tolist()
df.head()
""")
M("## 3. Translate with pre-trained NMT models\nBoth are Transformer **encoder–decoder** models with cross-attention (Bab 13).")
C("""
hyps = {}
for key in (["dry-run"] if DRY_RUN else MODELS):
    label = key if DRY_RUN else m.MODELS[key]["label"]
    hyps[label] = m.dry_run_translate(refs) if DRY_RUN else m.translate(srcs, key, BATCH_SIZE, NUM_BEAMS)
    pd.DataFrame({"id": df.id, "source": srcs, "reference": refs, "hypothesis": hyps[label]}).to_csv(
        f"{OUT}/translations_{key.replace('/', '_')}.csv", index=False)
{k: v[:2] for k, v in hyps.items()}
""")
M("""
## 4. Corpus-level metrics: BLEU-1…BLEU-4, chrF, chrF++, TER
* **BLEU-n** = BP · exp(Σ_{k=1..n} (1/n)·log p_k), where p_k is the modified k-gram precision.
* **chrF** = F-β (β=2) over character 1–6-grams (recall-weighted); **chrF++** adds word 1–2-grams.
* **TER** = (insertions + deletions + substitutions + shifts) / reference length — **lower is better**.
""")
C("""
scores = {k: m.corpus_scores(h, refs, bertscore=USE_BERTSCORE) for k, h in hyps.items()}
summary = None
for k, sc in scores.items():
    t = sc.table[["metric", "score"]].rename(columns={"score": k})
    summary = t if summary is None else summary.merge(t, on="metric")
    sc.table.to_csv(f"{OUT}/corpus_scores_{k.replace(' ', '_')}.csv", index=False)
summary.round(2)
""")
C("""
if BOOTSTRAP:
    ci = {k: {mt: m.bootstrap_ci(h, refs, mt, BOOTSTRAP, SEED) for mt in ("BLEU-1", "BLEU-4", "chrF", "TER")}
          for k, h in hyps.items()}
    display(pd.DataFrame([{"model": k, "metric": mt, "score": v, "CI low": lo, "CI high": hi}
                          for k, d in ci.items() for mt, (v, lo, hi) in d.items()]).round(2))
else:
    ci = None
""")
M("## 5. Effect of n-gram length on BLEU")
C("""
cmp_ = pd.DataFrame({k: [s.table.set_index("metric").loc[f"BLEU-{n}", "score"] for n in range(1, 5)]
                     for k, s in scores.items()}, index=[f"BLEU-{n}" for n in range(1, 5)])
cmp_["drop vs BLEU-1 (first model)"] = cmp_.iloc[:, 0] - cmp_.iloc[0, 0]
display(cmp_.round(2))
for k, s in scores.items():
    print(k, "| individual precisions:", s.precisions["precision (%)"].round(1).tolist(),
          "| BP =", round(s.precisions.attrs["bp"], 3))
fig = m.plot_ngram_effect(scores, f"{OUT}/ngram_effect.png")
""")
M("""
**Reading the curve.** Cumulative BLEU falls monotonically from n=1 to n=4: unigram matches only need the right *words*, whereas 3–4-gram matches also need the right *local order and inflection*. 
English→Indonesian differs in modifier–head order, relative clauses (*yang*), affixation and tense/aspect expression, so long n-gram matches are rarer even when the translation is acceptable.
""")
M("## 6. Sentence-level scores & the lowest-scoring sentences (≥ 5 required)")
C("""
sents = {k: m.sentence_scores(srcs, h, refs, df.id.tolist()) for k, h in hyps.items()}
worst = {k: m.worst_sentences(s, WORST_K) for k, s in sents.items()}
for k in sents:
    sents[k].to_csv(f"{OUT}/sentence_scores_{k.replace(' ', '_')}.csv", index=False)
    worst[k].to_csv(f"{OUT}/worst_sentences_{k.replace(' ', '_')}.csv", index=False)
    print("=" * 30, k)
    display(worst[k][["id", "sBLEU", "sChrF", "sTER", "flags"]].round(1))
""")
C("""
# Read each triple carefully: source / reference / hypothesis
first = list(worst)[0]
for _, r in worst[first].iterrows():
    print(f"#{r['id']}  sBLEU={r['sBLEU']:.1f}  sChrF={r['sChrF']:.1f}  TER={r['sTER']:.1f}")
    print("  EN :", r["source"]); print("  REF:", r["reference"]); print("  MT :", r["hypothesis"])
    print("  auto-flags:", r["flags"] or "-"); print()
""")
M("""
### 6.1 Manual linguistic annotation (YOUR part)
Fill the dictionary below for **at least 5** sentences, using the checklist:
*structural disparity / D-M vs M-D word order · polysemy · lexical gap · tense–aspect & voice (me-/di-) · plurality (reduplication) · pronoun/politeness · named entity & number · acceptable synonym (BLEU too harsh)*.
The key is the sentence `id` printed above.
""")
C("""
import json, pathlib
annotations = {
    # "123": {"phenomenon": "Structural disparity (M-D vs D-M)",
    #         "analysis": "EN 'the newly elected committee chairman' ... ID reference puts the head noun first ... MT kept English order ..."},
}
p = pathlib.Path("report/annotations.json")
if annotations:
    p.write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")
print("annotations saved:", len(annotations))
""")
M("## 7. Where BLEU fails: sentences with low BLEU but high chrF")
C("""
for k, s in sents.items():
    ex = m.limitation_examples(s, 5)["low_bleu_ok_chrf"]
    print("=" * 30, k)
    for _, r in ex.iterrows():
        print(f"#{r['id']} sBLEU={r['sBLEU']:.1f} sChrF={r['sChrF']:.1f}\\n  REF: {r['reference']}\\n  MT : {r['hypothesis']}\\n")
""")
C("""
for k, s in sents.items():
    m.plot_score_distribution(s, f"{OUT}/score_dist_{k.replace(' ', '_')}.png", k)
print("corr(sBLEU, sChrF):", {k: round(s["sBLEU"].corr(s["sChrF"]), 3) for k, s in sents.items()})
""")
M("## 8. Build the written report (Markdown)")
C("""
from src.report import build_report
md = build_report(scores, sents, worst, len(df), SEED, ci, dry_run=DRY_RUN)
rp = f"{OUT}/laporan_analisis.md" if DRY_RUN else "report/laporan_analisis.md"
open(rp, "w", encoding="utf-8").write(md)
print("report written to", rp)
""")

nbf.write(nb, "notebooks/MT_Evaluation_EN_ID.ipynb")
print("ok")

# ---- appended: export cell -------------------------------------------------
nb2 = nbf.read("notebooks/MT_Evaluation_EN_ID.ipynb", 4)
nb2.cells.append(nbf.v4.new_markdown_cell("## 9. Export results (to send for the final report)\nZips `results/` and `report/` so you can download them and upload them to the Claude chat."))
nb2.cells.append(nbf.v4.new_code_cell(
    "import shutil\n"
    "shutil.make_archive('mt_results', 'zip', root_dir='.', base_dir=OUT)\n"
    "print('created mt_results.zip with', OUT, '- download it from the Colab file panel (left sidebar > Files > mt_results.zip)')\n"
    "try:\n"
    "    from google.colab import files; files.download('mt_results.zip')\n"
    "except Exception:\n"
    "    pass"))
nbf.write(nb2, "notebooks/MT_Evaluation_EN_ID.ipynb")
