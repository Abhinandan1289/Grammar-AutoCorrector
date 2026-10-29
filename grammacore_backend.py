"""
GrammaCore Backend — Explainable Grammar AutoCorrector
=====================================================
FastAPI server that wraps the HuggingFace T5-base grammar-correction model,
adds per-edit confidence scoring, ERRANT-style edit categorisation, and
explainable annotations.

Key design choices (for viva):
  • We use "vennify/t5-base-grammar-correction" with the prefix "grammar: ".
  • Per-edit confidence is computed from the model's per-token log-probabilities
    (via `compute_transition_scores`).  We average the probabilities of the
    generated tokens that correspond to a changed span so that *different*
    edits in the same sentence receive *different* confidence values.
  • Categories (grammar, spelling, punctuation, style, redundancy, word_choice)
    are assigned by heuristic rules *after* ERRANT-style alignment.
  • Span masking protects LaTeX ($…$), inline code (`…`), and URLs so that
    the model cannot alter them.
"""

from __future__ import annotations

import json
import math
import os
import re
import string
import time
import traceback
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

import numpy as np
import spacy
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "vennify/t5-base-grammar-correction"
MODEL_PREFIX = "grammar: "
MAX_INPUT_LENGTH = 2000          # characters
NUM_BEAMS = 5                    # beam search width
NUM_RETURN_SEQUENCES = 3         # how many beam candidates we keep
STATIC_DIR = Path(__file__).parent / "static"

# Punctuation characters for category detection
PUNCTUATION_CHARS = set(string.punctuation)

# ---------------------------------------------------------------------------
# Lifespan: load model + spaCy at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app):
    """Load T5 grammar-correction model and spaCy English model at startup."""
    global tokenizer, model, model_loaded, nlp
    try:
        print("[GrammaCore] Loading spaCy en_core_web_sm …")
        nlp = spacy.load("en_core_web_sm")
        print("[GrammaCore] Loading T5 model …")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model.eval()
        model_loaded = True
        print("[GrammaCore] Model loaded successfully ✓")
    except Exception as exc:
        print(f"[GrammaCore] FAILED to load model: {exc}")
        traceback.print_exc()
    yield


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="GrammaCore API",
    description="Explainable Grammar AutoCorrector",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------
nlp: spacy.language.Language | None = None
tokenizer = None
model = None
model_loaded = False


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    """POST body for /api/v1/analyze."""
    text: str = Field(..., max_length=MAX_INPUT_LENGTH)
    confidence_threshold: float = Field(0.80, ge=0.0, le=1.0)
    num_beams: int = Field(NUM_BEAMS, ge=1, le=10)


class EditInfo(BaseModel):
    """One atomic edit returned by the analysis endpoint."""
    original: str
    corrected: str
    category: str
    explanation: str
    confidence: float
    start_char: int
    end_char: int
    is_suggestion: bool = False   # True when below threshold
    alternatives: list[str] = []


class AnalyzeResponse(BaseModel):
    """Full response from /api/v1/analyze."""
    original_text: str
    corrected_text: str
    edits: list[EditInfo]
    overall_confidence: float
    processing_time_ms: float
    beam_candidates: list[str] = []


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return empty 204 for favicon to prevent 404 in browsers."""
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Return health status including whether the model is loaded."""
    return {"status": "ok", "model_loaded": model_loaded}


# ---------------------------------------------------------------------------
# Span-masking helpers (LaTeX, inline code, URLs)
# ---------------------------------------------------------------------------
# These regexes identify spans that should be passed through unchanged.
_MASK_PATTERNS = [
    (r'\$[^$]+\$', 'LATEX'),                      # LaTeX math $…$
    (r'`[^`]+`', 'CODE'),                          # inline code `…`
    (r'https?://\S+', 'URL'),                      # URLs
]

_PLACEHOLDER_TPL = "XXXMASK{}XXX"


def _mask_spans(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace protected spans with numbered placeholders.
    
    Returns (masked_text, list_of_(placeholder, original) pairs).
    Protected spans: LaTeX ($...$), inline code (`...`), and URLs.
    This ensures the model cannot alter these spans.
    """
    masks: list[tuple[str, str]] = []
    for pattern, _tag in _MASK_PATTERNS:
        for m in re.finditer(pattern, text):
            placeholder = _PLACEHOLDER_TPL.format(len(masks))
            masks.append((placeholder, m.group()))
            text = text.replace(m.group(), placeholder, 1)
    return text, masks


def _unmask_spans(text: str, masks: list[tuple[str, str]]) -> str:
    """Restore original spans from placeholders (byte-for-byte)."""
    for placeholder, original in masks:
        text = text.replace(placeholder, original)
    return text


# ---------------------------------------------------------------------------
# Per-edit confidence from token log-probabilities
# ---------------------------------------------------------------------------
def _compute_per_token_probs(
    input_ids: torch.Tensor,
    model_obj,
) -> tuple[list[str], list[float], list[int]]:
    """Run model.generate with output_scores=True and return per-token
    probabilities for the top beam.

    Returns:
        tokens  – list of decoded token strings
        probs   – list of token probabilities (0–1)
        tok_ids – list of token ids
    
    Method: We call model.generate with output_scores=True, then use
    compute_transition_scores to get the log-probabilities assigned to each
    generated token.  Converting via exp() gives probabilities.
    """
    with torch.no_grad():
        outputs = model_obj.generate(
            input_ids,
            max_length=256,
            num_beams=NUM_BEAMS,
            num_return_sequences=NUM_RETURN_SEQUENCES,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # compute_transition_scores gives log-probs for each generated token
    # for each returned sequence.  We take the first (best) beam.
    transition_scores = model_obj.compute_transition_scores(
        outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
    )

    # First sequence (best beam)
    best_seq = outputs.sequences[0]
    best_scores = transition_scores[0]

    tokens: list[str] = []
    probs: list[float] = []
    tok_ids: list[int] = []

    for i, (tok_id, score) in enumerate(zip(best_seq[1:], best_scores)):  # skip <pad>
        tid = tok_id.item()
        if tid == tokenizer.eos_token_id:
            break
        token_str = tokenizer.decode([tid])
        prob = math.exp(score.item())
        tokens.append(token_str)
        probs.append(prob)
        tok_ids.append(tid)

    return tokens, probs, tok_ids


def _align_and_score_edits(
    original: str,
    corrected: str,
    gen_tokens: list[str],
    gen_probs: list[float],
) -> list[dict]:
    """Align original and corrected text word-by-word, detect edits, and
    assign per-edit confidence from the generated token probabilities.

    Per-edit confidence method:
    We reconstruct the corrected text from its generated tokens and match
    each corrected word to the token range that produced it.  For each edit
    (changed span), we take the *mean probability* of the tokens that
    correspond to the new word(s).  This gives different edits different
    confidence values, because different parts of the output have different
    token-level certainties.
    """
    orig_words = original.split()
    corr_words = corrected.split()

    # Map each corrected word to its approximate token probability range
    # by greedily assigning tokens to words.
    word_probs: list[float] = []
    token_idx = 0
    reconstructed = ""
    current_word_prob_accum: list[float] = []

    for cw in corr_words:
        current_word_prob_accum = []
        while token_idx < len(gen_tokens):
            tok = gen_tokens[token_idx]
            p = gen_probs[token_idx]
            current_word_prob_accum.append(p)
            reconstructed += tok
            token_idx += 1
            # Check if we've accumulated enough tokens for this word
            stripped = reconstructed.replace(" ", "").replace("▁", "")
            target = "".join(corr_words[:len(word_probs) + 1]).replace(" ", "")
            if len(stripped) >= len(target):
                break
        avg_p = sum(current_word_prob_accum) / len(current_word_prob_accum) if current_word_prob_accum else 0.5
        word_probs.append(avg_p)

    # Fill any remaining words with default probability
    while len(word_probs) < len(corr_words):
        word_probs.append(0.5)

    # Use SequenceMatcher to find edits
    edits = []
    sm = SequenceMatcher(None, orig_words, corr_words)
    char_offset = 0  # track character position in original text

    # Pre-compute character offsets for each original word
    orig_word_char_starts: list[int] = []
    pos = 0
    for w in orig_words:
        idx = original.find(w, pos)
        orig_word_char_starts.append(idx)
        pos = idx + len(w)

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue

        orig_span = " ".join(orig_words[i1:i2])
        corr_span = " ".join(corr_words[j1:j2])

        # Character positions in the original text
        if i1 < len(orig_word_char_starts):
            start_char = orig_word_char_starts[i1]
        else:
            start_char = len(original)

        if i2 > 0 and (i2 - 1) < len(orig_word_char_starts):
            end_char = orig_word_char_starts[i2 - 1] + len(orig_words[i2 - 1])
        elif i1 < len(orig_word_char_starts):
            end_char = orig_word_char_starts[i1]
        else:
            end_char = len(original)

        # Per-edit confidence: average of token probabilities for the
        # corrected words in this span.  For deletions (no replacement words),
        # use the average of surrounding word probabilities.
        if j1 < j2:
            edit_probs = word_probs[j1:j2]
            confidence = sum(edit_probs) / len(edit_probs)
        else:
            # Deletion — use adjacent tokens' probs
            adj_probs = []
            if j1 > 0:
                adj_probs.append(word_probs[j1 - 1])
            if j1 < len(word_probs):
                adj_probs.append(word_probs[j1])
            confidence = sum(adj_probs) / len(adj_probs) if adj_probs else 0.5

        confidence = round(min(max(confidence, 0.0), 1.0), 4)

        edits.append({
            "original": orig_span,
            "corrected": corr_span,
            "start_char": start_char,
            "end_char": end_char,
            "confidence": confidence,
            "op": op,
        })

    return edits


# ---------------------------------------------------------------------------
# Edit categorisation & explanation
# ---------------------------------------------------------------------------
def _is_valid_english_word(word: str) -> bool:
    """Check if a word is in spaCy's vocabulary (reasonable proxy for
    'is a valid English word').  Falls back to True if spaCy is unavailable.
    """
    if nlp is None:
        return True
    # spaCy's vocab.has_vector or lexeme.is_oov
    word_lower = word.lower().strip()
    if not word_lower:
        return True
    lexeme = nlp.vocab[word_lower]
    # A word is considered valid if it's not out-of-vocabulary
    # We also check common short words that spaCy might not know
    return not lexeme.is_oov or len(word_lower) <= 2


def _edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            ))
        prev_row = curr_row
    return prev_row[-1]


def _is_only_punctuation(text: str) -> bool:
    """Check if the text consists only of punctuation characters and spaces."""
    return all(c in PUNCTUATION_CHARS or c.isspace() for c in text) and len(text.strip()) > 0


def _categorize_edit(orig: str, corr: str, op: str) -> tuple[str, str]:
    """Determine the category and generate a specific explanation for an edit.

    Categories:
      - spelling:     original is not a valid word, replacement is within edit distance ≤ 2
      - punctuation:  edit only involves punctuation characters
      - grammar:      verb form, article, preposition, agreement changes
      - word_choice:  a valid word is replaced by a different valid word
      - redundancy:   a real word is deleted and meaning is repeated
      - style:        rewording that doesn't change meaning
    
    Returns (category, explanation) with explanations that quote actual words.
    """
    orig_stripped = orig.strip()
    corr_stripped = corr.strip()

    # --- Punctuation ---
    # If the change only involves punctuation (insert, delete, or replace punctuation)
    if not orig_stripped and _is_only_punctuation(corr_stripped):
        # Insertion of punctuation
        return "punctuation", f"Add '{corr_stripped}' here for correct punctuation."
    if _is_only_punctuation(orig_stripped) and not corr_stripped:
        # Deletion of punctuation
        return "punctuation", f"Remove '{orig_stripped}' — it is not needed here."
    if _is_only_punctuation(orig_stripped) and _is_only_punctuation(corr_stripped):
        return "punctuation", f"Replace '{orig_stripped}' with '{corr_stripped}' for correct punctuation."

    # Check if the difference between orig and corr is only punctuation
    orig_alpha = re.sub(r'[^\w\s]', '', orig_stripped)
    corr_alpha = re.sub(r'[^\w\s]', '', corr_stripped)
    if orig_alpha == corr_alpha and orig_stripped != corr_stripped:
        return "punctuation", f"Change punctuation from '{orig_stripped}' to '{corr_stripped}'."

    # --- Spelling ---
    # Original is not a valid English word, replacement is close
    if orig_stripped and corr_stripped:
        orig_words_list = orig_stripped.split()
        corr_words_list = corr_stripped.split()
        if len(orig_words_list) == 1 and len(corr_words_list) == 1:
            ow = orig_words_list[0]
            cw = corr_words_list[0]
            # Strip trailing punctuation for comparison
            ow_clean = ow.rstrip(string.punctuation)
            cw_clean = cw.rstrip(string.punctuation)
            if ow_clean and cw_clean and not _is_valid_english_word(ow_clean) and _edit_distance(ow_clean.lower(), cw_clean.lower()) <= 2:
                return "spelling", f"'{ow_clean}' is a misspelling of '{cw_clean}'."

    # --- Multi-word replacement with spelling errors ---
    # If the replacement is a subset of the original (words deleted) and one
    # of the deleted words is a misspelling, categorize as spelling.
    if orig_stripped and corr_stripped:
        orig_words_list = orig_stripped.split()
        corr_words_list = corr_stripped.split()
        if len(orig_words_list) > len(corr_words_list):
            # Some words were deleted — check if any deleted word is misspelled
            # Strip punctuation from words for comparison
            orig_clean = [w.rstrip(string.punctuation) for w in orig_words_list]
            corr_clean = [w.rstrip(string.punctuation) for w in corr_words_list]
            deleted_words = []
            for ow in orig_clean:
                if ow and ow not in corr_clean:
                    deleted_words.append(ow)
            misspelled_deleted = [w for w in deleted_words if w and not _is_valid_english_word(w)]
            if misspelled_deleted:
                return "spelling", f"Remove misspelled word '{misspelled_deleted[0]}' — it appears to be a typo."

    # --- Deletion ---
    if op == "delete" or (not corr_stripped and orig_stripped):
        orig_words_list = orig_stripped.split()
        # If deleting a misspelled word, it's not redundancy
        for w in orig_words_list:
            w_clean = w.rstrip(string.punctuation)
            if w_clean and not _is_valid_english_word(w_clean):
                return "spelling", f"Remove misspelled word '{w_clean}'."
        # True redundancy: deleting a real word because the meaning is repeated
        if all(_is_valid_english_word(w.rstrip(string.punctuation)) for w in orig_words_list if w.rstrip(string.punctuation)):
            return "redundancy", f"Remove '{orig_stripped}' — it is redundant in this context."
        return "grammar", f"Remove '{orig_stripped}' for grammatical correctness."

    # --- Insertion ---
    if op == "insert" or (not orig_stripped and corr_stripped):
        # Check what kind of word is inserted
        corr_words_list = corr_stripped.split()
        # Common grammar insertions: articles, auxiliaries, prepositions
        grammar_words = {"a", "an", "the", "is", "are", "am", "was", "were",
                         "be", "been", "being", "have", "has", "had", "do",
                         "does", "did", "will", "would", "shall", "should",
                         "can", "could", "may", "might", "must", "to", "of",
                         "in", "on", "at", "for", "with", "by", "from"}
        if all(w.lower() in grammar_words for w in corr_words_list):
            return "grammar", f"Insert '{corr_stripped}' for grammatical correctness."
        return "grammar", f"Add '{corr_stripped}' here to complete the sentence structure."

    # --- Replacement ---
    if orig_stripped and corr_stripped:
        # Grammar: verb forms, articles, prepositions
        grammar_words = {"a", "an", "the", "is", "are", "am", "was", "were",
                         "be", "been", "being", "have", "has", "had", "do",
                         "does", "did", "will", "would", "shall", "should",
                         "can", "could", "may", "might", "must"}

        orig_lower = orig_stripped.lower()
        corr_lower = corr_stripped.lower()

        # Check if it involves verb/article/preposition changes
        orig_words_set = set(orig_lower.split())
        corr_words_set = set(corr_lower.split())
        if orig_words_set & grammar_words or corr_words_set & grammar_words:
            return "grammar", f"Change '{orig_stripped}' to '{corr_stripped}' for correct grammar."

        # Word choice: both are valid words
        if (_is_valid_english_word(orig_stripped) and _is_valid_english_word(corr_stripped)):
            return "word_choice", f"Replace '{orig_stripped}' with '{corr_stripped}' for better word choice."

        # Default for replacements
        return "grammar", f"Change '{orig_stripped}' to '{corr_stripped}' for correctness."

    return "grammar", f"Change '{orig_stripped}' to '{corr_stripped}'."


# ---------------------------------------------------------------------------
# Extract alternatives from beam candidates
# ---------------------------------------------------------------------------
def _extract_alternatives(
    edit: dict,
    beam_candidates: list[str],
    best_corrected: str,
) -> list[str]:
    """For a given edit, check other beam candidates to find genuinely
    different corrections for the same span.  Returns only real, distinct
    alternatives (no filler).
    
    We align each beam candidate against the original text and check if
    the correction for the same original span differs.
    """
    orig_span = edit["original"]
    best_span = edit["corrected"]
    alternatives = set()

    for candidate in beam_candidates[1:]:  # skip best (index 0)
        if candidate == best_corrected:
            continue
        # Find what this candidate does with the same region
        cand_words = candidate.split()
        best_words = best_corrected.split()
        orig_words = orig_span.split() if orig_span else []

        # Simple heuristic: find the original span in the original text
        # and see what this candidate has in that position
        sm = SequenceMatcher(None, best_words, cand_words)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != "equal":
                cand_span = " ".join(cand_words[j1:j2])
                best_sub = " ".join(best_words[i1:i2])
                # Check if this difference relates to our edit
                if best_sub == best_span or orig_span in best_sub:
                    if cand_span and cand_span != best_span and cand_span != orig_span:
                        alternatives.add(cand_span)

    return list(alternatives)[:3]  # max 3 alternatives


# ---------------------------------------------------------------------------
# Main analysis endpoint
# ---------------------------------------------------------------------------
@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Analyze input text for grammar errors and return corrections with
    per-edit confidence, categories, and explanations.
    
    Pipeline:
    1. Validate input (length check).
    2. Mask protected spans (LaTeX, code, URLs).
    3. Run T5 model with beam search + output_scores.
    4. Unmask protected spans in all outputs.
    5. Align original vs corrected to detect edits.
    6. Score each edit using per-token probabilities.
    7. Categorize each edit and generate explanations.
    8. Mark edits below threshold as suggestions.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model is still loading. Please try again shortly.")

    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Input text cannot be empty.")

    if len(request.text) > MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Input text exceeds maximum length of {MAX_INPUT_LENGTH} characters.",
        )

    start_time = time.time()

    # Step 1: Mask protected spans
    masked_text, masks = _mask_spans(request.text)

    # Step 2: Prepare input for the model
    input_text = MODEL_PREFIX + masked_text
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids

    # Step 3: Generate with scores for per-token confidence
    gen_tokens, gen_probs, gen_tok_ids = _compute_per_token_probs(input_ids, model)

    # Step 4: Decode all beam candidates
    with torch.no_grad():
        beam_output = model.generate(
            input_ids,
            max_length=256,
            num_beams=NUM_BEAMS,
            num_return_sequences=min(NUM_RETURN_SEQUENCES, NUM_BEAMS),
        )
    beam_candidates_masked = [
        tokenizer.decode(seq, skip_special_tokens=True)
        for seq in beam_output
    ]

    # Step 5: Unmask all outputs
    corrected_text = _unmask_spans(
        tokenizer.decode(
            tokenizer(
                "".join(gen_tokens), return_tensors="pt"
            ).input_ids[0] if False else  # noqa — we already have the decoded text
            beam_output[0],
            skip_special_tokens=True,
        ),
        masks,
    )
    # Actually use the best beam candidate for the corrected text
    corrected_text = _unmask_spans(beam_candidates_masked[0], masks)
    beam_candidates = [_unmask_spans(c, masks) for c in beam_candidates_masked]

    # Step 6: Align and score edits
    raw_edits = _align_and_score_edits(
        request.text, corrected_text, gen_tokens, gen_probs
    )

    # Step 7: Categorize, explain, extract alternatives
    edits: list[EditInfo] = []
    for e in raw_edits:
        category, explanation = _categorize_edit(e["original"], e["corrected"], e["op"])
        alternatives = _extract_alternatives(e, beam_candidates, corrected_text)

        is_suggestion = e["confidence"] < request.confidence_threshold

        edits.append(EditInfo(
            original=e["original"],
            corrected=e["corrected"],
            category=category,
            explanation=explanation,
            confidence=round(e["confidence"] * 100, 1),  # as percentage
            start_char=e["start_char"],
            end_char=e["end_char"],
            is_suggestion=is_suggestion,
            alternatives=alternatives,
        ))

    # Overall confidence: average of all edit confidences (or 100 if no edits)
    if edits:
        overall_confidence = round(sum(e.confidence for e in edits) / len(edits), 1)
    else:
        overall_confidence = 100.0

    processing_time = round((time.time() - start_time) * 1000, 1)

    return AnalyzeResponse(
        original_text=request.text,
        corrected_text=corrected_text,
        edits=edits,
        overall_confidence=overall_confidence,
        processing_time_ms=processing_time,
        beam_candidates=beam_candidates,
    )


# ---------------------------------------------------------------------------
# Serve results.json
# ---------------------------------------------------------------------------
@app.get("/results.json")
async def get_results():
    """Serve results.json if it exists."""
    results_path = Path(__file__).parent / "results.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="results.json not found. Run evaluate.py first.")


@app.get("/results_t060.json")
async def get_results_t060():
    """Serve results_t060.json (threshold=0.60 evaluation) if it exists."""
    results_path = Path(__file__).parent / "results_t060.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="results_t060.json not found.")


# ---------------------------------------------------------------------------
# Serve the frontend  (must be ABOVE __main__ to not be dead code)
# ---------------------------------------------------------------------------
# Mount static files for CSS/JS/images if any
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the single-page frontend at the root URL."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>GrammaCore</h1><p>Frontend not found.</p>")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "grammacore_backend:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
    )
