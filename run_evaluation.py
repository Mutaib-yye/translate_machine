#!/usr/bin/env python
"""CLI: translate a sample of the EN-ID FLORES pairs and compute all metrics.

Examples
--------
python run_evaluation.py --n 200 --models marian nllb --bertscore
python run_evaluation.py --n 0                      # all 2009 pairs
python run_evaluation.py --dry-run --n 100          # offline smoke test (no model download)
"""
import argparse
import json
from pathlib import Path

from src import mt_eval as m
from src.report import build_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of sentence pairs (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=["marian", "nllb"], choices=list(m.MODELS))
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--worst-k", type=int, default=10)
    ap.add_argument("--bertscore", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=300, help="bootstrap resamples (0 = skip)")
    ap.add_argument("--dry-run", action="store_true", help="synthetic hypotheses, no model download")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out or ("results_dryrun" if a.dry_run else "results"))
    out.mkdir(exist_ok=True, parents=True)
    Path("report").mkdir(exist_ok=True)

    df = m.sample_pairs(m.load_pairs(), a.n, a.seed)
    print(f"Using {len(df)} sentence pairs")
    refs, srcs = df.reference.tolist(), df.source.tolist()

    scores, sents, worst, ci = {}, {}, {}, {}
    for key in (["dry-run"] if a.dry_run else a.models):
        hyps = m.dry_run_translate(refs) if a.dry_run else m.translate(srcs, key, a.batch_size, a.beams)
        label = key if a.dry_run else m.MODELS[key]["label"]
        sc = m.corpus_scores(hyps, refs, bertscore=a.bertscore)
        ss = m.sentence_scores(srcs, hyps, refs, df.id.tolist())
        scores[label], sents[label] = sc, ss
        worst[label] = m.worst_sentences(ss, a.worst_k)
        if a.bootstrap:
            ci[label] = {mt: m.bootstrap_ci(hyps, refs, mt, a.bootstrap, a.seed)
                         for mt in ("BLEU-1", "BLEU-4", "chrF", "TER")}
        tag = key.replace("/", "_")
        ss.to_csv(out / f"sentence_scores_{tag}.csv", index=False)
        worst[label].to_csv(out / f"worst_sentences_{tag}.csv", index=False)
        sc.table.to_csv(out / f"corpus_scores_{tag}.csv", index=False)
        print(f"\n== {label} ==\n{sc.table[['metric','score']].to_string(index=False)}")

    m.plot_ngram_effect(scores, out / "ngram_effect.png")
    md = build_report(scores, sents, worst, len(df), a.seed, ci or None, dry_run=a.dry_run)
    rp = out / "laporan_analisis.md" if a.dry_run else Path("report/laporan_analisis.md")
    rp.write_text(md, encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps(vars(a), indent=2))
    print(f"\nSaved results to {out}/ and report to {rp}")


if __name__ == "__main__":
    main()
