# EN→ID Machine Translation: Automatic Evaluation & Error Analysis
Tugas Kelompok 2 — Bab 13 (Machine Translation): NMT encoder–decoder/Transformer, BLEU, chrF, TER.

**What it does:** takes FLORES-200 EN–ID sentence pairs → translates with pre-trained **MarianMT** (`Helsinki-NLP/opus-mt-en-id`) and **NLLB** (`facebook/nllb-200-distilled-600M`) → computes **BLEU-1…BLEU-4, chrF, chrF++, TER** (+ optional BERTScore, bootstrap 95 % CIs) → compares n-gram orders → ranks the worst sentences with linguistic error flags → generates the written report.

## Repo layout
| Path | Purpose |
|---|---|
| `notebooks/MT_Evaluation_EN_ID.ipynb` | **Main deliverable** — run top to bottom |
| `src/mt_eval.py` | data loading (+ mojibake repair), translation, metrics, sentence scores, error flags |
| `src/report.py` | builds `report/laporan_analisis.md` from real results + your annotations |
| `run_evaluation.py` | same pipeline as a CLI |
| `data/FLORES200_EN_ID_2009_pairs.txt` | 2009 `EN<TAB>ID` pairs |
| `tests/` | metric sanity tests (`pytest`) |
| `results/` | CSV + PNG outputs (created when you run it) |
| `report/` | `laporan_analisis.md` (generated) and `annotations.json` (yours) |

## Run it
### Option A — Google Colab (easiest, free GPU)
1. Upload/open `notebooks/MT_Evaluation_EN_ID.ipynb` (File → Open → GitHub → `Mutaib-yye/translate_machine`).
2. *Runtime → Change runtime type → GPU*.
3. Run all cells (the first cell clones the repo and installs requirements).

### Option B — local
```bash
git clone https://github.com/Mutaib-yye/translate_machine && cd translate_machine
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/MT_Evaluation_EN_ID.ipynb
# or CLI:
python run_evaluation.py --n 200 --models marian nllb --bertscore
```
First run downloads the models (~300 MB Marian, ~2.4 GB NLLB). CPU works; 200 sentences take a few minutes per model.

## Choosing the sample size
Prof allows each member to choose: change `N_SENTENCES` in the notebook (or `--n`). `0` = all 2009. Keep `SEED` the same if you want to compare with teammates on identical sentences, or change it for a different sample.

## What YOU must do after running (≈30–45 min)
1. Open the *worst sentences* output (notebook section 6) and read each EN / reference / MT triple.
2. Fill `annotations` in section 6.1 for **≥ 5 sentences**: phenomenon (word order D-M vs M-D, polysemy, lexical gap, tense/aspect, voice me-/di-, plurality, named entity/number, or "valid synonym — BLEU too harsh") and a 2–4 sentence explanation. Run the cell, then re-run section 8.
3. Open `report/laporan_analisis.md`, write the **Kesimpulan** (section 6), and export to PDF/Word (e.g. VS Code “Markdown PDF”, Pandoc, or paste into Word).
4. Submit: the notebook (with outputs saved) + the report.

## Tests
```bash
python -m pytest -q tests
```
`MT_DRY_RUN=1` (or `python run_evaluation.py --dry-run`) runs the pipeline with synthetic hypotheses to test everything offline — those numbers are **not** MT results and go to `results_dryrun/`.

## Metric notes
* BLEU-n: cumulative (sacreBLEU, `max_ngram_order=n`, 13a tokenisation); individual n-gram precisions and brevity penalty are also reported.
* chrF: character 6-grams, β=2; chrF++: adds word bigrams. TER: **lower is better**.
* Per-sentence scores use smoothed sentence-BLEU / chrF / TER — noisy, used only to find error candidates.

## Use of AI assistance
An AI assistant (Claude, by Anthropic) was used while preparing this work, for writing the evaluation code and notebook, structuring the report and drafting parts of its text. The experiments were run by the author on Google Colab.

## Author
Mutaib Bin Mehraj Rather — NPM 250820701100010
