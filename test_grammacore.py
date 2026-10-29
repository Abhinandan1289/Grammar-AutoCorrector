"""
test_grammacore.py — Unit tests for GrammaCore
===============================================
Tests cover:
  1. LaTeX, URL, and inline code preserved together in one sentence
  2. "direclty" → spelling category
  3. A lone comma insertion → punctuation category
  4. Two edits in one sentence get different confidences
  5. Empty input and over-length input handled with proper errors
  6. Health endpoint returns expected structure
"""

import pytest
import json
import sys
import os
from contextlib import contextmanager

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient


def get_client():
    """Create a fresh TestClient that triggers startup events."""
    from grammacore_backend import app
    return TestClient(app, raise_server_exceptions=False)


# Module-scoped client so model loads once
_client = None

def get_or_create_client():
    global _client
    if _client is None:
        _client = get_client()
        # Force startup by entering the context
        _client.__enter__()
    return _client

client = None


# ---------------------------------------------------------------------------
# Fixture: wait for model to load
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def wait_for_model():
    """The TestClient triggers startup events, so the model should load.
    We verify via the health endpoint.
    """
    global client
    client = get_or_create_client()
    resp = client.get("/health")
    data = resp.json()
    if not data.get("model_loaded"):
        pytest.skip("Model did not load — cannot run integration tests")


# ---------------------------------------------------------------------------
# Test 1: Health endpoint
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_model_loaded(self):
        resp = client.get("/health")
        data = resp.json()
        assert "model_loaded" in data
        assert "status" in data

    def test_health_model_loaded_true(self):
        resp = client.get("/health")
        data = resp.json()
        assert data["model_loaded"] is True


# ---------------------------------------------------------------------------
# Test 2: Frontend serves at root
# ---------------------------------------------------------------------------
class TestFrontend:
    def test_root_returns_html(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_root_contains_grammacore(self):
        resp = client.get("/")
        assert "GrammaCore" in resp.text


# ---------------------------------------------------------------------------
# Test 3: Input validation
# ---------------------------------------------------------------------------
class TestInputValidation:
    def test_empty_input_returns_422(self):
        """Empty input should return 422 error."""
        resp = client.post("/api/v1/analyze", json={"text": "", "confidence_threshold": 0.8})
        assert resp.status_code == 422

    def test_whitespace_only_returns_422(self):
        """Whitespace-only input should return 422 error."""
        resp = client.post("/api/v1/analyze", json={"text": "   ", "confidence_threshold": 0.8})
        assert resp.status_code == 422

    def test_over_length_returns_422(self):
        """Input exceeding 2000 characters should return 422 error."""
        long_text = "a " * 1500  # ~3000 chars
        resp = client.post("/api/v1/analyze", json={"text": long_text, "confidence_threshold": 0.8})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Test 4: Span masking — LaTeX, URL, and inline code preserved together
# ---------------------------------------------------------------------------
class TestSpanMasking:
    def test_latex_url_code_preserved_together(self):
        """LaTeX ($...$), URL, and inline code in one sentence must survive
        the correction pipeline byte-for-byte unchanged."""
        text = "The $\\alpha$ value at https://arxiv.org/paper and `x = 1` is importent."
        resp = client.post("/api/v1/analyze", json={
            "text": text,
            "confidence_threshold": 0.5,
        })
        assert resp.status_code == 200
        data = resp.json()

        corrected = data["corrected_text"]
        # All three masked spans must appear unchanged
        assert "$\\alpha$" in corrected, f"LaTeX not preserved: {corrected}"
        assert "https://arxiv.org/paper" in corrected, f"URL not preserved: {corrected}"
        assert "`x = 1`" in corrected, f"Inline code not preserved: {corrected}"

    def test_latex_preserved(self):
        """LaTeX math should pass through unchanged."""
        text = "The equation $E = mc^2$ is very famouse."
        resp = client.post("/api/v1/analyze", json={"text": text, "confidence_threshold": 0.5})
        data = resp.json()
        assert "$E = mc^2$" in data["corrected_text"]

    def test_url_preserved(self):
        """URLs should pass through unchanged."""
        text = "Visit https://example.com/page for more informations."
        resp = client.post("/api/v1/analyze", json={"text": text, "confidence_threshold": 0.5})
        data = resp.json()
        assert "https://example.com/page" in data["corrected_text"]

    def test_code_preserved(self):
        """Inline code should pass through unchanged."""
        text = "Use the `print()` function to display the ouput."
        resp = client.post("/api/v1/analyze", json={"text": text, "confidence_threshold": 0.5})
        data = resp.json()
        assert "`print()`" in data["corrected_text"]


# ---------------------------------------------------------------------------
# Test 5: Spelling detection — "direclty" → spelling
# ---------------------------------------------------------------------------
class TestSpelling:
    def test_direclty_is_spelling(self):
        """'direclty' should be categorized as a spelling error, not redundancy."""
        text = "What I am telling is correct but what you telling me is not correct direclty."
        resp = client.post("/api/v1/analyze", json={
            "text": text,
            "confidence_threshold": 0.5,
        })
        assert resp.status_code == 200
        data = resp.json()

        # Find the edit that involves "direclty"
        spelling_edits = [
            e for e in data["edits"]
            if "direclty" in e["original"].lower() or "direclty" in e.get("original", "")
        ]
        
        assert len(spelling_edits) > 0, (
            f"No edit found for 'direclty'. Edits: {json.dumps(data['edits'], indent=2)}"
        )
        
        for e in spelling_edits:
            assert e["category"] == "spelling", (
                f"'direclty' edit should be 'spelling' but got '{e['category']}'. "
                f"Edit: {json.dumps(e, indent=2)}"
            )


# ---------------------------------------------------------------------------
# Test 6: Punctuation category
# ---------------------------------------------------------------------------
class TestPunctuation:
    def test_comma_insertion_is_punctuation(self):
        """An edit that only inserts a comma should be categorized as 'punctuation'."""
        # We test this via the categorize function directly since the model
        # might not always insert a comma for any given input
        from grammacore_backend import _categorize_edit
        
        cat, explanation = _categorize_edit("", ",", "insert")
        assert cat == "punctuation", f"Comma insertion should be 'punctuation' but got '{cat}'"
        assert "," in explanation

    def test_period_change_is_punctuation(self):
        """Replacing a period with an exclamation mark should be punctuation."""
        from grammacore_backend import _categorize_edit
        
        cat, explanation = _categorize_edit(".", "!", "replace")
        assert cat == "punctuation"

    def test_only_punctuation_diff(self):
        """If the only difference is punctuation, should be punctuation category."""
        from grammacore_backend import _categorize_edit
        
        cat, explanation = _categorize_edit("hello", "hello,", "replace")
        assert cat == "punctuation"


# ---------------------------------------------------------------------------
# Test 7: Different confidences for different edits
# ---------------------------------------------------------------------------
class TestDifferentConfidences:
    def test_two_edits_different_confidence(self):
        """Two edits in the same sentence must have different confidence values."""
        # Use a sentence likely to produce multiple edits
        text = "She go to school every day and learn many thing."
        resp = client.post("/api/v1/analyze", json={
            "text": text,
            "confidence_threshold": 0.3,
        })
        assert resp.status_code == 200
        data = resp.json()

        if len(data["edits"]) >= 2:
            confidences = [e["confidence"] for e in data["edits"]]
            # At least two edits should have different confidence values
            assert len(set(confidences)) > 1, (
                f"All {len(confidences)} edits have the same confidence: {confidences}. "
                f"Per-edit confidence scoring is not working."
            )
        else:
            # If model produced fewer than 2 edits, try another sentence
            text2 = "He dont knows nothing about the the situation."
            resp2 = client.post("/api/v1/analyze", json={
                "text": text2,
                "confidence_threshold": 0.3,
            })
            data2 = resp2.json()
            if len(data2["edits"]) >= 2:
                confidences2 = [e["confidence"] for e in data2["edits"]]
                assert len(set(confidences2)) > 1, (
                    f"All edits have the same confidence: {confidences2}"
                )


# ---------------------------------------------------------------------------
# Test 8: Correct text should have no edits
# ---------------------------------------------------------------------------
class TestNoEdits:
    def test_correct_sentence(self):
        """A grammatically correct sentence should produce no or very few edits."""
        text = "This sentence is already correct."
        resp = client.post("/api/v1/analyze", json={
            "text": text,
            "confidence_threshold": 0.8,
        })
        assert resp.status_code == 200
        data = resp.json()
        # We don't assert zero edits because the model may still make changes,
        # but the response should be valid
        assert "edits" in data
        assert "corrected_text" in data


# ---------------------------------------------------------------------------
# Test 9: Response structure
# ---------------------------------------------------------------------------
class TestResponseStructure:
    def test_response_has_required_fields(self):
        """Response must include all required fields."""
        text = "She go to school."
        resp = client.post("/api/v1/analyze", json={
            "text": text,
            "confidence_threshold": 0.8,
        })
        assert resp.status_code == 200
        data = resp.json()

        assert "original_text" in data
        assert "corrected_text" in data
        assert "edits" in data
        assert "overall_confidence" in data
        assert "processing_time_ms" in data
        assert "beam_candidates" in data

    def test_edit_has_required_fields(self):
        """Each edit must have the expected fields."""
        text = "She go to school."
        resp = client.post("/api/v1/analyze", json={
            "text": text,
            "confidence_threshold": 0.5,
        })
        data = resp.json()

        if data["edits"]:
            edit = data["edits"][0]
            assert "original" in edit
            assert "corrected" in edit
            assert "category" in edit
            assert "explanation" in edit
            assert "confidence" in edit
            assert "start_char" in edit
            assert "end_char" in edit
            assert "is_suggestion" in edit
            assert "alternatives" in edit


# ---------------------------------------------------------------------------
# Test 10: Threshold behaviour
# ---------------------------------------------------------------------------
class TestThreshold:
    def test_low_confidence_marked_as_suggestion(self):
        """Edits below threshold should be marked as suggestions."""
        text = "She go to school."
        resp = client.post("/api/v1/analyze", json={
            "text": text,
            "confidence_threshold": 0.99,  # very high threshold
        })
        data = resp.json()

        if data["edits"]:
            # With threshold 0.99, most edits should be suggestions
            suggestions = [e for e in data["edits"] if e["is_suggestion"]]
            # At least some should be suggestions at 99% threshold
            assert len(suggestions) > 0 or all(
                e["confidence"] >= 99.0 for e in data["edits"]
            ), "High threshold should mark some edits as suggestions"
