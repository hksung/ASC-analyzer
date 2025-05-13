import pytest
import os
import sys

# Add project root to sys.path so tests can find asc_analyzer
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from asc_analyzer.core import (
    fullExtractSent, fullExtractDoc, ascExtractDoc,
    processText, indexCalc, indexCalcFull, writeCsv
)
import json

def test_fullExtractSent_simple():
    sent = "Hello world."
    tokens = fullExtractSent(sent)
    # Expect at least token entries for 'Hello' and 'world'
    texts = [tok[1] for tok in tokens]
    assert "Hello" in texts
    assert "world" in texts


def test_fullExtractDoc_simple():
    text = "This is one. This is two."
    docs = fullExtractDoc(text)
    # Two sentences
    assert len(docs) == 2
    # First sentence should have tokens
    assert isinstance(docs[0], list)
    assert len(docs[0]) > 0


def test_ascExtractDoc_structure():
    text = "Run fast."
    # Mock freq and soa dicts with minimal entries
    ascFreqDict = {
        "lemmaFreq": {"run": 1, "fast": 1},
        "ascFreqD": {"X": 1},
        "ascLemmaFreqD": {"run_X": 1}
    }
    ascSoaDict = {"mi": {}, "tscore": {}, "deltap_lemma_cue": {}, "deltap_structure_cue": {}}
    output = ascExtractDoc(text, ascFreqDict, ascSoaDict)
    # Should be list of sentences
    assert isinstance(output, list)
    for sent in output:
        assert isinstance(sent, list)
    # Ensure each token entry has 4 fields at least
    assert len(output[0][0]) >= 4


def test_processText_and_indexCalc():
    # Create a small ascDict for testing indexCalc
    ascDict = {"lemmas": ["a", "b", "a"],
               "ascs": ["X", "Y", "X"],
               "asc+lemmas": ["a_X", "b_Y", "a_X"]}
    freqD = {"lemmaFreq": {"a": 2, "b": 1},
             "ascFreqD": {"X": 2, "Y": 1},
             "ascLemmaFreqD": {"a_X": 2, "b_Y": 1}}
    ascD = {"mi": {"a_X": 0.5, "b_Y": 0.3},
            "tscore": {"a_X": 1.0, "b_Y": 0.6},
            "deltap_lemma_cue": {},
            "deltap_structure_cue": {}}
    idx = indexCalc(ascDict, freqD, ascD)
    # Check that raw lists are not in output keys
    for raw_key in ["lemmas", "ascs", "asc+lemmas"]:
        assert raw_key not in idx
    # Check computed metrics
    assert idx["clauseCount"] == 3
    assert pytest.approx(idx["mvTTR"], rel=1e-3) == len(set(["a", "b", "a"])) / 3
    assert idx.get("X_Prop") == pytest.approx(2/3)


def test_indexCalcFull_and_writeCsv(tmp_path):
    # Create dummy text files
    d = tmp_path / "texts"
    d.mkdir()
    file1 = d / "a.txt"
    file1.write_text("Hello world.")
    file2 = d / "b.txt"
    file2.write_text("Test sentence.")
    # Dummy freq and soa dicts with empty metrics
    freqD = {"lemmaFreq": {}, "ascFreqD": {}, "ascLemmaFreqD": {}}
    ascD = {"mi": {}, "tscore": {}, "deltap_lemma_cue": {}, "deltap_structure_cue": {}}
    # Run batch index computation
    results = indexCalcFull([str(file1), str(file2)], freqD, ascD)
    assert set(results.keys()) == {"a.txt", "b.txt"}
    # Write to CSV
    out = tmp_path / "out.csv"
    writeCsv(results, list(next(iter(results.values())).keys()), str(out))
    assert out.exists()
    content = out.read_text()
    # Header should start with filename
    assert content.splitlines()[0].startswith("filename,")

if __name__ == "__main__":
    pytest.main()
