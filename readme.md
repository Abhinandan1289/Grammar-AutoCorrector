# GrammaCore — Explainable Grammar AutoCorrector
**NLP Course Project**

GrammaCore is an explainable grammar auto-correction web application powered by HuggingFace's `vennify/t5-base-grammar-correction` model and spaCy. Unlike black-box correction systems, GrammaCore breaks down every change into fine-grained atomic edits, assigns individual token-calibrated confidence scores, provides linguistic category classifications, and generates human-readable explanations quoting the exact spans changed.

---

## Architecture & Project Structure

```
.
|-----index.html          # Single-page UI (Live Corrector, How It Works, Results)
├── grammacore_backend.py   # FastAPI backend serving API & UI at "/"
├── evaluate.py             # Evaluation script for JFLEG benchmark (P, R, F0.5)
├── test_grammacore.py      # Automated test suite (pytest)
├── results.json            # Real JFLEG evaluation metrics (threshold 0.80)
├── results_t060.json       # Real JFLEG evaluation metrics (threshold 0.60)
├── requirements.txt        # Frozen dependencies
└── readme.md               # Documentation & viva preparation
```

### File Breakdown
1. **`grammacore_backend.py`**:
   - Loads `vennify/t5-base-grammar-correction` with prefix `"grammar: "` and spaCy `en_core_web_sm` via FastAPI lifespan handler.
   - Preserves LaTeX math (`$...$`), inline code (`` `...` ``), and URLs using reversible regex span masking.
   - Extracts beam search candidates and computes per-edit token log-probabilities via `compute_transition_scores()`.
   - Identifies word-level alignment diffs using `SequenceMatcher`.
   - Heuristically categorises edits into `grammar`, `spelling`, `punctuation`, `word_choice`, `redundancy`, and `style`.
   - Exposes `/health`, `/api/v1/analyze`, `/results.json`, `/results_t060.json`, and serves `/` with `static/index.html`.

2. **`index.html`**:
   - Single-page frontend using relative API paths (`/health`, `/api/v1/analyze`, `/results.json`, `/results_t060.json`).
   - Dynamic sample sentence buttons count (10 sample buttons).
   - Interactive diff viewer highlighting insertions (green) and deletions (red).
   - Edit cards with per-edit confidence bars, category badges, explanations, real alternative candidates, and Accept/Reject buttons.
   - Real-time Category breakdown chart.
   - "How It Works" tab detailing the pipeline stages.
   - "Results & Analysis" tab dynamically rendering real evaluation metrics, trade-off analysis, real failure cases, and limitations.

3. **`evaluate.py`**:
   - Loads the official JFLEG benchmark (`jhu-clsp/jfleg`, test split).
   - Evaluates:
     - No-correction baseline
     - Plain T5 (no thresholding)
     - GrammaCore pipeline with configurable confidence threshold (e.g. 0.80 and 0.60)
   - Computes Precision, Recall, and $F_{0.5}$ score (weighting precision 2× over recall).
   - Discovers and exports real failure cases and latency numbers.

4. **`test_grammacore.py`**:
   - Complete unit and integration test suite covering input validation, span masking, spelling categorisation, punctuation handling, distinct per-edit confidences, and health endpoint status.

---

## Installation & Setup

### Prerequisites
- Linux (Fedora / Ubuntu / Debian) or macOS / WSL2
- Python 3.10+ (tested on Python 3.14)

### 1. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

---

## Running the Application

Port 8000 is reserved on this workstation. Run GrammaCore on **port 8001**:

```bash
source .venv/bin/activate
uvicorn grammacore_backend:app --host 127.0.0.1 --port 8001 --reload
```

Open your browser and navigate to:
```
http://127.0.0.1:8001
```

---

## Running the Tests

To run the automated test suite with full verbose output:

```bash
source .venv/bin/activate
pytest test_grammacore.py -v
```

---

## Running Evaluation on JFLEG Benchmark

Run evaluation on 200 JFLEG test sentences at threshold 0.80 and 0.60:

```bash
source .venv/bin/activate

# Threshold 0.80 (Default precision-focused)
python3 evaluate.py --limit 200 --threshold 0.80 --output results.json

# Threshold 0.60 (Higher recall exploration)
python3 evaluate.py --limit 200 --threshold 0.60 --output results_t060.json
```

---

## Core Features & Fixes

1. **Relative Frontend Wiring**:
   - Removed all hardcoded `http://127.0.0.1:8000` base URLs. All calls use relative paths (`/health`, `/api/v1/analyze`, `/results.json`).
   - Serves `static/index.html` directly from `/`.
2. **True Per-Edit Confidence**:
   - Uses `model.compute_transition_scores` to calculate token-level probabilities $\exp(\text{score})$ for each generated token in the output sequence.
   - Maps words in changed spans back to their constituent subword token probabilities, producing unique, calibrated confidence scores for each edit.
3. **Punctuation Category**:
   - Dedicated category for punctuation insertions, deletions, and replacements (e.g. comma insertion between independent clauses).
4. **Spelling vs Redundancy Disambiguation**:
   - Word validity verified against spaCy vocabulary and Levenshtein edit distance ($\le 2$).
   - Deletions of misspelled non-words (such as `direclty`) are categorized under `spelling` rather than falsely labeled as `redundancy`.
5. **Span Masking**:
   - Protects LaTeX expressions (`$E = mc^2$`), inline code (`` `print()` ``), and web links (`https://...`) so they survive generation untouched.
6. **Real Alternative Candidates**:
   - Extracted from diverse beam search paths (`num_beams=5`, `num_return_sequences=3`). If no genuine alternative exists for a span, the alternatives section is cleanly omitted.

---

## Viva Preparation Guide

### 1. Why $F_{0.5}$ instead of standard $F_1$ or Accuracy?
In Grammatical Error Correction (GEC), false positives (incorrectly altering already grammatical text) are much more damaging to user trust and meaning than false negatives (missing an error). $F_{0.5}$ weights Precision twice as heavily as Recall ($\beta = 0.5$ in $F_\beta$):
$$F_{0.5} = \frac{(1 + 0.5^2) \cdot P \cdot R}{(0.5^2 \cdot P) + R} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

### 2. Why Span Masking?
Sequence-to-sequence models trained on natural language frequently hallucinate or corrupt code syntax, mathematical expressions, or URLs (e.g. converting `print()` into natural text or modifying URL path tokens). By masking these substrings with inert tokens before inference and restoring them afterwards, we guarantee 100% byte-for-byte fidelity.

### 3. How is Per-Edit Confidence Computed?
During beam search generation, Hugging Face `generate(..., output_scores=True, return_dict_in_generate=True)` collects logits at each decoding step. `compute_transition_scores()` normalizes these logits into per-token log probabilities. We convert them via $\exp(s)$ to probabilities in $[0, 1]$ and compute the mean token probability corresponding to the specific tokens that constitute each edit span.

### 4. Why use a Confidence Threshold?
A confidence threshold allows users or automated pipelines to filter out low-confidence model hallucinations. Edits below the threshold (e.g. 0.80) are classified as "suggestions" rather than automatically accepted changes, maximizing precision while still offering helpful guidance.

### 5. What is ERRANT and how does it evaluate GEC?
ERRANT (Enhanced Reference-based Annotation and Representation of Text) aligns original and corrected sentences using linguistically informed alignment and rule-based error classification. In evaluation, it compares system edit tuples $(start, end, correction)$ against gold reference edits, enabling precise calculation of true positives, false positives, and false negatives.

### 6. Limitations of T5-Base for GEC
- **Model Capacity (220M parameters)**: T5-base has limited context understanding compared to 7B+ models and can produce unnatural simplifications (e.g., "you telling" $\to$ "you tell" instead of "you are telling").
- **CPU Latency**: Generating 5 beams with score extraction takes 500–2000 ms per sentence on CPU.
- **Single reference bias**: Standard benchmarks like JFLEG provide up to 4 human references, but many other valid paraphrases exist that might be penalized as false positives.

---

## Deploying to Hugging Face Spaces (Docker SDK)

GrammaCore can be deployed to a free CPU tier on [Hugging Face Spaces](https://huggingface.co/spaces) using Docker:

1. **Create a New Space**:
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - Space Name: `grammacore` (or your choice).
   - License: `mit` or `apache-2.0`.
   - Space SDK: Select **Docker** (Blank template).
   - Hardware: **CPU Basic** (Free).

2. **Clone the Space Repository**:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/grammacore
   cd grammacore
   ```

3. **Copy Project Files into the Space**:
   ```bash
   cp -r /path/to/project/{Dockerfile,grammacore_backend.py,static,requirements.txt,results.json,results_t060.json} .
   ```

4. **Commit and Push**:
   ```bash
   git add .
   git commit -m "Deploy GrammaCore to Hugging Face Spaces"
   git push
   ```

5. **Build & Live**:
   Hugging Face Spaces will automatically trigger the Docker build, pre-download the model weights during image building, and expose the application on port `7860`.

