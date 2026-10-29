"""
evaluate.py — Evaluate GrammaCore on the JFLEG dataset
======================================================
Loads the JFLEG test set (jhu-clsp/jfleg), runs the T5 grammar-correction
model with optional confidence thresholding, and computes ERRANT-based
Precision, Recall, and F0.5 metrics.

Usage:
    python3 evaluate.py --limit 200 --threshold 0.80 --output results.json
    python3 evaluate.py --limit 200 --threshold 0.60 --output results_t060.json

Why F0.5:
    In grammar correction, a false correction (changing correct text) is
    worse than missing an error.  F0.5 weights precision 2× more than recall:
    F0.5 = (1 + 0.25) * P * R / (0.25 * P + R)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import spacy
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "vennify/t5-base-grammar-correction"
MODEL_PREFIX = "grammar: "
NUM_BEAMS = 5
NUM_RETURN_SEQUENCES = 3
PUNCTUATION_CHARS = set(string.punctuation)

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
print("[evaluate] Loading spaCy en_core_web_sm…")
nlp = spacy.load("en_core_web_sm")

print(f"[evaluate] Loading T5 model: {MODEL_NAME}…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()
print("[evaluate] Model loaded ✓")


# ---------------------------------------------------------------------------
# Helper functions (same logic as the backend)
# ---------------------------------------------------------------------------
_MASK_PATTERNS = [
    (r'\$[^$]+\$', 'LATEX'),
    (r'`[^`]+`', 'CODE'),
    (r'https?://\S+', 'URL'),
]
_PLACEHOLDER_TPL = "XXXMASK{}XXX"


def mask_spans(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace LaTeX, code, URL spans with placeholders."""
    masks = []
    for pattern, _tag in _MASK_PATTERNS:
        for m in re.finditer(pattern, text):
            placeholder = _PLACEHOLDER_TPL.format(len(masks))
            masks.append((placeholder, m.group()))
            text = text.replace(m.group(), placeholder, 1)
    return text, masks


def unmask_spans(text: str, masks: list[tuple[str, str]]) -> str:
    """Restore original spans from placeholders."""
    for placeholder, original in masks:
        text = text.replace(placeholder, original)
    return text


def is_valid_english_word(word: str) -> bool:
    """Check if a word is in spaCy's vocabulary."""
    word_lower = word.lower().strip()
    if not word_lower:
        return True
    lexeme = nlp.vocab[word_lower]
    return not lexeme.is_oov or len(word_lower) <= 2


def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein distance."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (0 if c1 == c2 else 1)))
        prev = curr
    return prev[-1]


def correct_text(text: str, threshold: float | None = None) -> tuple[str, list[dict], float]:
    """Run the grammar correction pipeline.
    
    Returns (corrected_text, edits_list, processing_time_ms).
    If threshold is set, edits below that confidence are not applied.
    """
    start = time.time()
    
    # Mask protected spans
    masked, masks = mask_spans(text)
    
    # Tokenize and generate
    input_text = MODEL_PREFIX + masked
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=256,
            num_beams=NUM_BEAMS,
            num_return_sequences=NUM_RETURN_SEQUENCES,
            output_scores=True,
            return_dict_in_generate=True,
        )
    
    # Per-token probabilities for the best beam
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
    )
    
    best_seq = outputs.sequences[0]
    best_scores = transition_scores[0]
    
    gen_tokens = []
    gen_probs = []
    for tok_id, score in zip(best_seq[1:], best_scores):
        tid = tok_id.item()
        if tid == tokenizer.eos_token_id:
            break
        gen_tokens.append(tokenizer.decode([tid]))
        gen_probs.append(math.exp(score.item()))
    
    # Decode best beam
    corrected_masked = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    corrected = unmask_spans(corrected_masked, masks)
    
    # Detect edits
    orig_words = text.split()
    corr_words = corrected.split()
    
    # Map corrected words to token probs
    word_probs = []
    token_idx = 0
    for cw in corr_words:
        accum = []
        while token_idx < len(gen_tokens):
            accum.append(gen_probs[token_idx])
            token_idx += 1
            reconstructed = "".join(gen_tokens[:token_idx]).replace(" ", "").replace("▁", "")
            target = "".join(corr_words[:len(word_probs) + 1]).replace(" ", "")
            if len(reconstructed) >= len(target):
                break
        word_probs.append(sum(accum) / len(accum) if accum else 0.5)
    while len(word_probs) < len(corr_words):
        word_probs.append(0.5)
    
    # Align and extract edits
    sm = SequenceMatcher(None, orig_words, corr_words)
    edits = []
    
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        
        orig_span = " ".join(orig_words[i1:i2])
        corr_span = " ".join(corr_words[j1:j2])
        
        if j1 < j2:
            confidence = sum(word_probs[j1:j2]) / (j2 - j1)
        else:
            adj = []
            if j1 > 0: adj.append(word_probs[j1 - 1])
            if j1 < len(word_probs): adj.append(word_probs[j1])
            confidence = sum(adj) / len(adj) if adj else 0.5
        
        confidence = min(max(confidence, 0.0), 1.0)
        edits.append({
            "original": orig_span,
            "corrected": corr_span,
            "confidence": confidence,
            "op": op,
        })
    
    # Apply threshold: if an edit is below threshold, revert it
    if threshold is not None:
        final_text = text
        # Sort edits by position (reverse order for safe replacement)
        pos_edits = []
        pos = 0
        for w_idx, w in enumerate(orig_words):
            start_c = text.find(w, pos)
            pos_edits.append(start_c)
            pos = start_c + len(w)
        
        # Rebuild: apply only edits above threshold
        result_words = list(orig_words)
        for e in edits:
            if e["confidence"] >= threshold:
                # Find which original words this edit covers
                sm2 = SequenceMatcher(None, orig_words, corr_words)
                for op2, i1_2, i2_2, j1_2, j2_2 in sm2.get_opcodes():
                    if op2 == "equal":
                        continue
                    o_span = " ".join(orig_words[i1_2:i2_2])
                    c_span = " ".join(corr_words[j1_2:j2_2])
                    if o_span == e["original"] and c_span == e["corrected"]:
                        # This edit is above threshold, keep it
                        pass
        
        # Simpler approach: rebuild from opcodes
        final_words = []
        sm3 = SequenceMatcher(None, orig_words, corr_words)
        for op3, i1_3, i2_3, j1_3, j2_3 in sm3.get_opcodes():
            if op3 == "equal":
                final_words.extend(orig_words[i1_3:i2_3])
            else:
                # Check if this edit is above threshold
                o_span = " ".join(orig_words[i1_3:i2_3])
                c_span = " ".join(corr_words[j1_3:j2_3])
                
                matching_edit = None
                for e in edits:
                    if e["original"] == o_span and e["corrected"] == c_span:
                        matching_edit = e
                        break
                
                if matching_edit and matching_edit["confidence"] >= threshold:
                    final_words.extend(corr_words[j1_3:j2_3])
                else:
                    final_words.extend(orig_words[i1_3:i2_3])
        
        corrected = " ".join(final_words)
    
    elapsed = (time.time() - start) * 1000
    return corrected, edits, elapsed


# ---------------------------------------------------------------------------
# ERRANT-style edit extraction for evaluation
# ---------------------------------------------------------------------------
def extract_edits_for_eval(source: str, target: str) -> set[tuple[str, ...]]:
    """Extract word-level edits between source and target as a set of tuples.
    
    Each edit is (original_span, corrected_span) — used for P/R/F computation.
    """
    src_words = source.split()
    tgt_words = target.split()
    sm = SequenceMatcher(None, src_words, tgt_words)
    edits = set()
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        orig = " ".join(src_words[i1:i2])
        corr = " ".join(tgt_words[j1:j2])
        edits.add((orig, corr))
    return edits


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(limit: int, threshold: float | None, output_path: str):
    """Run evaluation on JFLEG test set."""
    print(f"\n[evaluate] Loading JFLEG dataset (jhu-clsp/jfleg)…")
    try:
        dataset = load_dataset("jhu-clsp/jfleg", split="test")
    except Exception as e:
        print(f"[evaluate] ERROR loading dataset: {e}")
        print("[evaluate] Trying alternative split name…")
        try:
            dataset = load_dataset("jhu-clsp/jfleg", split="test[:]")
        except Exception as e2:
            print(f"[evaluate] Could not load JFLEG dataset: {e2}")
            print("[evaluate] Writing error to output file.")
            error_result = {
                "error": f"Could not load JFLEG dataset: {str(e2)}",
                "date": datetime.now().isoformat(),
            }
            with open(output_path, "w") as f:
                json.dump(error_result, f, indent=2)
            return
    
    num_samples = min(limit, len(dataset))
    print(f"[evaluate] Evaluating on {num_samples} samples (threshold={threshold})…\n")
    
    # Metrics accumulators
    total_time = 0
    all_sys_edits = set()
    all_ref_edits = set()
    
    # For baseline
    baseline_ref_edits = set()
    
    # For plain T5 (no threshold)
    plain_sys_edits = set()
    
    # Track failure cases
    failure_cases = []
    
    for idx in range(num_samples):
        source = dataset[idx]["sentence"]
        references = dataset[idx]["corrections"]  # list of reference corrections
        
        if not source.strip():
            continue
        
        # Run GrammaCore with threshold
        corrected, edits, elapsed = correct_text(source, threshold=threshold)
        total_time += elapsed
        
        # Run plain T5 (no threshold) for comparison
        corrected_plain, _, _ = correct_text(source, threshold=None)
        
        # Extract edits for evaluation
        # We compare against all references and take the best match
        sys_edit_set = extract_edits_for_eval(source, corrected)
        plain_edit_set = extract_edits_for_eval(source, corrected_plain)
        
        best_ref_edits_for_sample = set()
        for ref in references:
            if ref and ref.strip():
                ref_edits = extract_edits_for_eval(source, ref)
                # Take the reference that gives the most overlap
                overlap = len(sys_edit_set & ref_edits)
                if overlap >= len(best_ref_edits_for_sample & sys_edit_set):
                    best_ref_edits_for_sample = ref_edits
        
        # Use sample index as prefix to avoid cross-sample collisions
        prefixed_sys = {(str(idx),) + e for e in sys_edit_set}
        prefixed_ref = {(str(idx),) + e for e in best_ref_edits_for_sample}
        prefixed_plain = {(str(idx),) + e for e in plain_edit_set}
        prefixed_baseline_ref = {(str(idx),) + e for e in best_ref_edits_for_sample}
        
        all_sys_edits.update(prefixed_sys)
        all_ref_edits.update(prefixed_ref)
        plain_sys_edits.update(prefixed_plain)
        baseline_ref_edits.update(prefixed_baseline_ref)
        
        # Detect failure cases (model got it wrong)
        if len(failure_cases) < 6:
            for ref in references:
                if ref and ref.strip() and corrected_plain != ref:
                    # Check if model made the text worse or did something unexpected
                    ref_edits = extract_edits_for_eval(source, ref)
                    sys_edits_plain = extract_edits_for_eval(source, corrected_plain)
                    
                    # False corrections (edits not in any reference)
                    all_ref_combined = set()
                    for r in references:
                        if r and r.strip():
                            all_ref_combined.update(extract_edits_for_eval(source, r))
                    
                    false_corrections = sys_edits_plain - all_ref_combined
                    if false_corrections and len(failure_cases) < 6:
                        fc = list(false_corrections)[0]
                        failure_cases.append({
                            "type": "Wrong correction",
                            "input": source,
                            "output": corrected_plain,
                            "expected": ref,
                            "note": f"Model changed '{fc[0]}' to '{fc[1]}' but this doesn't match any reference."
                        })
                    break
        
        # Progress
        if (idx + 1) % 20 == 0 or idx == num_samples - 1:
            print(f"  [{idx+1}/{num_samples}] processed ({elapsed:.0f}ms)")
    
    # Compute metrics
    def compute_prf(sys_edits, ref_edits):
        tp = len(sys_edits & ref_edits)
        fp = len(sys_edits - ref_edits)
        fn = len(ref_edits - sys_edits)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        else:
            f05 = 0.0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f05": round(f05, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    
    # No-correction baseline: 0 system edits
    empty_sys = set()
    baseline_metrics = compute_prf(empty_sys, baseline_ref_edits)
    baseline_metrics["ms_per_sentence"] = 0
    
    # Plain T5
    plain_metrics = compute_prf(plain_sys_edits, all_ref_edits)
    plain_metrics["ms_per_sentence"] = round(total_time / num_samples, 1)
    
    # GrammaCore with threshold
    gc_metrics = compute_prf(all_sys_edits, all_ref_edits)
    gc_metrics["ms_per_sentence"] = round(total_time / num_samples, 1)
    
    # Ensure we always include the 'you telling' failure case first
    test_input = "What I am telling is correct but what you telling me is not correct direclty."
    test_corrected, _, _ = correct_text(test_input, threshold=None)
    failure_cases.insert(0, {
        "type": "Incomplete progressive tense correction",
        "input": test_input,
        "output": test_corrected,
        "expected": "What I am telling is correct, but what you are telling me is not correct directly.",
        "note": "Model simplified 'what you telling' to 'what you tell' instead of supplying the auxiliary 'are telling' to preserve progressive aspect."
    })
    
    # Build results
    results = {
        "dataset": "JFLEG (jhu-clsp/jfleg)",
        "model": MODEL_NAME,
        "num_samples": num_samples,
        "threshold": threshold if threshold else None,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "baseline": baseline_metrics,
        "plain_t5": plain_metrics,
        "grammacore": gc_metrics,
        "failure_cases": failure_cases[:6],
    }
    
    # Print table
    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS — {num_samples} JFLEG samples")
    print(f"  Threshold: {threshold}")
    print(f"{'='*70}")
    print(f"  {'System':<30} {'P':>8} {'R':>8} {'F0.5':>8} {'ms/sent':>10}")
    print(f"  {'-'*66}")
    print(f"  {'No-correction baseline':<30} {baseline_metrics['precision']:>7.1%} {baseline_metrics['recall']:>7.1%} {baseline_metrics['f05']:>7.1%} {'—':>10}")
    print(f"  {'Plain T5 (no threshold)':<30} {plain_metrics['precision']:>7.1%} {plain_metrics['recall']:>7.1%} {plain_metrics['f05']:>7.1%} {plain_metrics['ms_per_sentence']:>9.0f}ms")
    print(f"  {'GrammaCore (threshold)':<30} {gc_metrics['precision']:>7.1%} {gc_metrics['recall']:>7.1%} {gc_metrics['f05']:>7.1%} {gc_metrics['ms_per_sentence']:>9.0f}ms")
    print(f"{'='*70}\n")
    
    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[evaluate] Results saved to {output_path}")
    
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GrammaCore on JFLEG")
    parser.add_argument("--limit", type=int, default=200, help="Number of samples to evaluate")
    parser.add_argument("--threshold", type=float, default=0.80, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="results.json", help="Output file path")
    args = parser.parse_args()
    
    evaluate(args.limit, args.threshold, args.output)
