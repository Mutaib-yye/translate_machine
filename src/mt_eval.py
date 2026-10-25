"""Core library for the EN->ID machine-translation evaluation project.

Everything the notebook and the CLI do lives here so that both stay thin:

* data loading (+ repair of mojibake such as ``BarÃ§a`` -> ``Barça``)
* translation with MarianMT or NLLB (HuggingFace)
* corpus metrics: BLEU-1..4 (cumulative), individual n-gram precisions,
  chrF, chrF++, TER, optional BERTScore
* bootstrap confidence intervals
* sentence-level scoring + selection of the worst sentences
* heuristic error flags that help the *manual* linguistic annotation
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF, TER

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "FLORES200_EN_ID_2009_pairs.txt"

MODELS = {
    "marian": {
        "hf_name": "Helsinki-NLP/opus-mt-en-id",
        "label": "MarianMT (opus-mt-en-id)",
    },
    "nllb": {
        "hf_name": "facebook/nllb-200-distilled-600M",
        "label": "NLLB-200 distilled 600M",
        "src_lang": "eng_Latn",
        "tgt_lang": "ind_Latn",
    },
}

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
_MOJIBAKE_HINT = re.compile(r"[ÃÂ][\x80-\xbf\u0080-ÿ€‚ƒ„…†‡ˆ‰Š‹ŒŽ‘’“”•–—˜™š›œžŸ]|â€")


def fix_mojibake(text: str) -> str:
    """Repair UTF-8 text that was wrongly decoded as cp1252/latin-1.

    The FLORES file contains e.g. ``BarÃ§a`` (should be ``Barça``) and
    ``SÃ£o`` (should be ``São``).  Lines without such patterns are untouched.
    """
    if not _MOJIBAKE_HINT.search(text):
        return text
    for enc in ("cp1252", "latin-1"):
        try:
            return text.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    # Fall back to repairing only the broken fragments.
    def _frag(m: re.Match) -> str:
        for enc in ("cp1252", "latin-1"):
            try:
                return m.group(0).encode(enc).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        return m.group(0)

    return re.sub(r"[ÃÂâ][^\s]{1,2}", _frag, text)


def load_pairs(path: str | Path = DATA_FILE) -> pd.DataFrame:
    """Read the tab-separated EN<TAB>ID file into a DataFrame (id, source, reference)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            src, ref = parts[0].strip(), "\t".join(parts[1:]).strip()
            rows.append((i, fix_mojibake(src), fix_mojibake(ref)))
    return pd.DataFrame(rows, columns=["id", "source", "reference"])


def sample_pairs(df: pd.DataFrame, n: int | None, seed: int = 42) -> pd.DataFrame:
    """Random reproducible sample of ``n`` pairs (``None`` / ``<=0`` = all)."""
    if not n or n <= 0 or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).sort_values("id").reset_index(drop=True)


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------
def translate(sources: list[str], model_key: str = "marian", batch_size: int = 16,
              num_beams: int = 4, max_length: int = 256, device: str | None = None,
              progress: bool = True) -> list[str]:
    """Translate English sentences into Indonesian with a pre-trained NMT model."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    cfg = MODELS[model_key]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model_key == "nllb":
        tok = AutoTokenizer.from_pretrained(cfg["hf_name"], src_lang=cfg["src_lang"])
    else:
        tok = AutoTokenizer.from_pretrained(cfg["hf_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["hf_name"]).to(device).eval()

    gen_kwargs = dict(num_beams=num_beams, max_length=max_length)
    if model_key == "nllb":
        gen_kwargs["forced_bos_token_id"] = tok.convert_tokens_to_ids(cfg["tgt_lang"])

    out: list[str] = []
    batches = range(0, len(sources), batch_size)
    if progress:
        from tqdm.auto import tqdm
        batches = tqdm(batches, desc=f"Translating [{model_key}]")
    for i in batches:
        batch = sources[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, **gen_kwargs)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return [h.strip() for h in out]


def dry_run_translate(references: list[str], seed: int = 0) -> list[str]:
    """OFFLINE TEST ONLY: corrupt the references (drop/swap words) to mimic MT noise.

    Used to smoke-test the pipeline when HuggingFace cannot be reached.  The
    resulting scores are *not* machine-translation results.
    """
    rng = random.Random(seed)
    outs = []
    for ref in references:
        w = ref.split()
        w = [x for x in w if rng.random() > 0.12]
        for _ in range(max(1, len(w) // 8)):
            if len(w) > 3:
                i = rng.randrange(len(w) - 1)
                w[i], w[i + 1] = w[i + 1], w[i]
        outs.append(" ".join(w))
    return outs


# --------------------------------------------------------------------------
# Corpus-level metrics
# --------------------------------------------------------------------------
@dataclass
class Scores:
    table: pd.DataFrame          # one row per metric
    precisions: pd.DataFrame     # individual n-gram precisions + BP


def _bleu_n(hyps, refs, n: int) -> BLEU:
    bleu = BLEU(max_ngram_order=n, effective_order=False)
    return bleu.corpus_score(hyps, [refs])


def corpus_scores(hyps: list[str], refs: list[str], bertscore: bool = False) -> Scores:
    """BLEU-1..4 (cumulative), chrF, chrF++, TER (+ optional BERTScore)."""
    rows = []
    for n in range(1, 5):
        rows.append((f"BLEU-{n}", _bleu_n(hyps, refs, n).score,
                     "higher=better; geometric mean of 1..n-gram precisions x BP"))
    rows.append(("chrF", CHRF().corpus_score(hyps, [refs]).score,
                 "higher=better; char n-gram F-score (n=6, beta=2)"))
    rows.append(("chrF++", CHRF(word_order=2).corpus_score(hyps, [refs]).score,
                 "higher=better; chrF + word n-grams up to 2"))
    rows.append(("TER", TER().corpus_score(hyps, [refs]).score,
                 "LOWER=better; edit rate (ins/del/sub/shift) per reference word"))
    if bertscore:
        try:
            from bert_score import score as bs
            _, _, f1 = bs(hyps, refs, lang="id", verbose=False)
            rows.append(("BERTScore-F1", float(f1.mean()) * 100,
                         "higher=better; contextual-embedding similarity"))
        except Exception as e:  # pragma: no cover - optional dependency
            print(f"[warn] BERTScore skipped: {e}")

    b4 = _bleu_n(hyps, refs, 4)
    prec = pd.DataFrame({
        "n-gram": ["1-gram", "2-gram", "3-gram", "4-gram"],
        "precision (%)": list(b4.precisions),
    })
    prec.attrs["bp"] = b4.bp
    prec.attrs["sys_len"] = b4.sys_len
    prec.attrs["ref_len"] = b4.ref_len
    table = pd.DataFrame(rows, columns=["metric", "score", "note"])
    return Scores(table=table, precisions=prec)


def bootstrap_ci(hyps, refs, metric: str = "BLEU-4", n_resamples: int = 500,
                 seed: int = 42, alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap (resampling sentences) confidence interval for one corpus metric."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(hyps))

    def score(sel):
        h = [hyps[i] for i in sel]
        r = [refs[i] for i in sel]
        if metric.startswith("BLEU-"):
            return _bleu_n(h, r, int(metric[-1])).score
        if metric == "chrF":
            return CHRF().corpus_score(h, [r]).score
        if metric == "chrF++":
            return CHRF(word_order=2).corpus_score(h, [r]).score
        if metric == "TER":
            return TER().corpus_score(h, [r]).score
        raise ValueError(metric)

    vals = [score(rng.choice(idx, size=len(idx), replace=True)) for _ in range(n_resamples)]
    return score(idx), float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


# --------------------------------------------------------------------------
# Sentence-level scores + error candidates
# --------------------------------------------------------------------------
_EN_STOP = set("the of and to in is are was were be been that this with for on as by at from it its "
               "which who has have had will would can could their his her they he she we you not "
               "but or an a".split())
_ID_STOP = set("yang dan di ke dari untuk dengan pada adalah itu ini dalam tidak akan juga atau "
               "oleh karena sebagai telah dapat bisa mereka kami kita saya".split())
_num = re.compile(r"\d+(?:[.,]\d+)*")
_tokre = re.compile(r"\w+", re.UNICODE)


def _tok(s: str) -> list[str]:
    return _tokre.findall(s.lower())


def sentence_scores(sources, hyps, refs, ids=None) -> pd.DataFrame:
    """Per-sentence BLEU (smoothed), chrF, TER, individual BLEU-1/BLEU-4 + length ratio."""
    sb = BLEU(effective_order=True)
    sb1 = BLEU(max_ngram_order=1, effective_order=True)
    sb4 = BLEU(max_ngram_order=4, effective_order=True)
    ch, ter = CHRF(), TER()
    rows = []
    for k, (s, h, r) in enumerate(zip(sources, hyps, refs)):
        rl, hl = len(_tok(r)), len(_tok(h))
        rows.append({
            "id": ids[k] if ids is not None else k + 1,
            "source": s, "reference": r, "hypothesis": h,
            "sBLEU": sb.sentence_score(h, [r]).score,
            "sBLEU-1": sb1.sentence_score(h, [r]).score,
            "sBLEU-4": sb4.sentence_score(h, [r]).score,
            "sChrF": ch.sentence_score(h, [r]).score,
            "sTER": ter.sentence_score(h, [r]).score,
            "len_ratio": (hl / rl) if rl else float("nan"),
        })
    return pd.DataFrame(rows)


def error_flags(row: pd.Series) -> list[str]:
    """Cheap heuristics that *suggest* which linguistic phenomenon to look for.

    These are hints for the human annotator — never a substitute for reading the
    source/reference/hypothesis triple.
    """
    flags = []
    s, h, r = row["source"], row["hypothesis"], row["reference"]
    # numbers dropped / altered
    sn, hn = set(_num.findall(s)), set(_num.findall(h))
    if sn and not sn <= hn:
        flags.append("NUMBER/UNIT mismatch")
    # untranslated English left in output
    ht = _tok(h)
    en_left = [t for t in ht if t in _EN_STOP]
    if len(en_left) >= 2:
        flags.append("UNTRANSLATED English words (code-switching)")
    # omission / addition
    lr = row["len_ratio"]
    if lr == lr and lr < 0.75:
        flags.append("OMISSION (hypothesis much shorter)")
    elif lr == lr and lr > 1.3:
        flags.append("ADDITION/over-translation (hypothesis much longer)")
    # capitalised source tokens (proper nouns) missing from hypothesis
    caps = [w for w in re.findall(r"\b[A-Z][\w'-]+", s)[1:] if len(w) > 2]
    if caps and sum(1 for w in caps if w.lower() not in h.lower()) / len(caps) > 0.4:
        flags.append("NAMED ENTITY / term translated or lost")
    # word order: unigram ok but higher n-grams collapse
    if row["sBLEU-1"] > 55 and row["sBLEU-4"] < 0.5 * row["sBLEU-1"]:
        flags.append("WORD ORDER / structure (good unigrams, poor 4-grams) -> check D-M vs M-D")
    # likely lexical choice (low chrF but similar length, no other flag)
    if not flags and row["sChrF"] < 45:
        flags.append("LEXICAL CHOICE (polysemy / synonym / lexical gap?) -> read carefully")
    # reference-side hint: Indonesian affix/reduplication loss
    if re.search(r"\b(\w+)-\1\b", r) and not re.search(r"\b(\w+)-\1\b", h):
        flags.append("REDUPLICATION in reference missing in hypothesis (plurality)")
    return flags


def worst_sentences(df: pd.DataFrame, k: int = 10, min_words: int = 6) -> pd.DataFrame:
    """Pick the k lowest-scoring sentences (rank-average of sBLEU and sChrF)."""
    d = df.copy()
    d = d[d["source"].str.split().str.len() >= min_words]
    d["rank"] = d["sBLEU"].rank() + d["sChrF"].rank()
    d = d.sort_values("rank").head(k).copy()
    d["flags"] = d.apply(lambda r: "; ".join(error_flags(r)), axis=1)
    return d.drop(columns="rank").reset_index(drop=True)


def limitation_examples(df: pd.DataFrame, k: int = 5) -> dict[str, pd.DataFrame]:
    """Cases that expose BLEU's blind spots (for the 'limitations' section).

    * ``low_bleu_ok_chrf``  : lots of surface mismatch but char-level overlap — often valid paraphrase/inflection
    * ``high_bleu_but_ter`` : good n-gram match but still a large edit rate
    """
    d = df[df["source"].str.split().str.len() >= 6].copy()
    d["gap"] = d["sChrF"] - d["sBLEU"]
    low = d.sort_values("gap", ascending=False).head(k)
    d["gap2"] = d["sTER"] - (100 - d["sBLEU"])
    return {"low_bleu_ok_chrf": low.reset_index(drop=True)}


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def plot_ngram_effect(results: dict[str, Scores], out: str | Path | None = None):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, sc in results.items():
        t = sc.table.set_index("metric")["score"]
        axes[0].plot([1, 2, 3, 4], [t[f"BLEU-{n}"] for n in range(1, 5)], marker="o", label=name)
        axes[1].plot([1, 2, 3, 4], sc.precisions["precision (%)"], marker="s", label=name)
    axes[0].set(title="Cumulative BLEU-n", xlabel="max n-gram order n", ylabel="score")
    axes[1].set(title="Individual n-gram precision", xlabel="n-gram order", ylabel="precision (%)")
    for a in axes:
        a.set_xticks([1, 2, 3, 4]); a.grid(alpha=.3); a.legend()
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return fig


def plot_score_distribution(df: pd.DataFrame, out: str | Path | None = None, title: str = ""):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df["sChrF"], bins=20); ax[0].set(title=f"Sentence chrF {title}", xlabel="chrF")
    ax[1].scatter(df["sBLEU"], df["sChrF"], s=12, alpha=.6)
    ax[1].set(title="sBLEU vs sChrF", xlabel="sentence BLEU", ylabel="sentence chrF")
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return fig
