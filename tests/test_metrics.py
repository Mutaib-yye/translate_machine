import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import mt_eval as m


def test_mojibake_fixed():
    assert m.fix_mojibake("BarÃ§a") == "Barça"
    assert m.fix_mojibake("SÃ£o Paulo") == "São Paulo"
    assert m.fix_mojibake("normal text") == "normal text"


def test_load_pairs():
    df = m.load_pairs()
    assert len(df) == 2009
    assert not df["reference"].str.contains("Ã").any()


def test_perfect_translation():
    refs = ["saya makan nasi goreng hari ini", "dia pergi ke pasar"]
    sc = m.corpus_scores(refs, refs).table.set_index("metric")["score"]
    for k in ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "chrF", "chrF++"):
        assert sc[k] > 99.9
    assert sc["TER"] < 0.1


def test_bleu_decreases_with_n_on_noisy_output():
    df = m.load_pairs().head(150)
    hyps = m.dry_run_translate(df.reference.tolist())
    sc = m.corpus_scores(hyps, df.reference.tolist()).table.set_index("metric")["score"]
    assert sc["BLEU-1"] > sc["BLEU-2"] > sc["BLEU-3"] > sc["BLEU-4"]


def test_worst_sentences():
    df = m.load_pairs().head(60)
    hyps = m.dry_run_translate(df.reference.tolist())
    ss = m.sentence_scores(df.source, hyps, df.reference, df.id)
    w = m.worst_sentences(ss, k=5)
    assert len(w) == 5 and "flags" in w
