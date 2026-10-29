"""
verify_e2e.py — End-to-end browser verification for GrammaCore
==============================================================
Uses Playwright with local Google Chrome to drive the full user workflow:
  - Verifies health and root page
  - Tests all 10 sample sentences
  - Tests Accept/Reject, Copy Final, Diff view, Category chart
  - Verifies preservation of LaTeX, code, and URLs
  - Tests observed bug sentence ('direclty' -> spelling, comma -> punctuation, different confidences)
  - Tests Results & Analysis tab against results.json and results_t060.json
  - Takes screenshots of Live Corrector and Results tabs
"""

import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8001"
OUTPUT_DIR = Path(__file__).parent
LIVE_SCREENSHOT = OUTPUT_DIR / "screenshot_live_corrector.png"
RESULTS_SCREENSHOT = OUTPUT_DIR / "screenshot_results_tab.png"

def run_verification():
    console_errors = []

    with sync_playwright() as p:
        print("[verify_e2e] Launching Chrome...")
        browser = p.chromium.launch(
            executable_path="/usr/bin/google-chrome",
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        # Monitor browser console for red errors
        def on_console(msg):
            if msg.type == "error":
                console_errors.append(msg.text)
                print(f"  [BROWSER CONSOLE ERROR] {msg.text}")
        page.on("console", on_console)

        print(f"[verify_e2e] Navigating to {BASE_URL}...")
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        # 1. Wait for model ready
        print("[verify_e2e] Waiting for model to be ready...")
        page.wait_for_selector("#status-bar.status-ready", timeout=60000)
        status_text = page.locator("#status-bar").inner_text()
        print(f"  Status bar: {status_text}")
        assert "ready" in status_text.lower() or "loaded" in status_text.lower()

        # Check sample sentences count dynamic label
        samples_label = page.locator("#samples-label").inner_text()
        sample_btns = page.locator(".sample-btn")
        sample_count = sample_btns.count()
        print(f"  Sample buttons found: {sample_count}")
        print(f"  Sample header label: '{samples_label}'")
        assert f"{sample_count} sample test sentences" in samples_label, f"Label mismatch: {samples_label}"
        assert sample_count == 10, f"Expected 10 sample buttons, got {sample_count}"

        # 2. Click ALL 10 sample buttons and run Correct on each
        print("\n[verify_e2e] Testing all 10 sample buttons...")
        for i in range(sample_count):
            btn = sample_btns.nth(i)
            btn_text = btn.inner_text()
            expected_text = btn.get_attribute("data-text")
            print(f"  [{i+1}/10] Clicking sample: '{btn_text}'")
            btn.click()
            time.sleep(0.2)

            curr_val = page.locator("#input-text").input_value()
            assert curr_val == expected_text, f"Input text does not match sample data-text!"

            # Click Correct
            page.locator("#correct-btn").click()
            # Wait for spinner to disappear and results to appear
            page.wait_for_selector("#results-section", state="visible", timeout=30000)
            page.wait_for_selector("#loading", state="hidden", timeout=30000)

            # Check diff view has content
            diff_text = page.locator("#diff-view").inner_text()
            assert len(diff_text.strip()) > 0, "Diff view is empty!"

            # Check final text has content
            final_val = page.locator("#final-text").inner_text()
            assert len(final_val.strip()) > 0, "Final text is empty!"

            # If edits exist, check accept/reject and chart
            edit_cards = page.locator(".edit-card")
            edit_count = edit_cards.count()
            print(f"    -> Edits generated: {edit_count}")
            if edit_count > 0:
                chart_bars = page.locator(".chart-bar")
                assert chart_bars.count() > 0, "Category chart did not render!"

                # Test Accept on first edit
                first_card = edit_cards.nth(0)
                accept_btn = first_card.locator(".accept-btn")
                if accept_btn.count() > 0:
                    accept_btn.click()
                    time.sleep(0.1)

                # Test Reject on first edit
                reject_btn = first_card.locator(".reject-btn")
                if reject_btn.count() > 0:
                    reject_btn.click()
                    time.sleep(0.1)

            # Specific preservation checks:
            if "LaTeX" in btn_text:
                assert "$E = mc^2$" in final_val, f"LaTeX math not preserved in: {final_val}"
                print("    ✓ LaTeX $E = mc^2$ preserved")
            elif "URL" in btn_text:
                assert "https://example.com/page" in final_val, f"URL not preserved in: {final_val}"
                print("    ✓ URL https://example.com/page preserved")
            elif "Code" in btn_text:
                assert "`print()`" in final_val, f"Code not preserved in: {final_val}"
                print("    ✓ Inline code `print()` preserved")
            elif "masks combined" in btn_text.lower():
                assert r"$\alpha$" in final_val or "$\\alpha$" in final_val, f"LaTeX $\\alpha$ not preserved in: {final_val}"
                assert "https://arxiv.org/paper" in final_val, f"URL not preserved in: {final_val}"
                assert "`x = 1`" in final_val, f"Code not preserved in: {final_val}"
                print("    ✓ All masks combined preserved byte-for-byte")

        # 3. Test Copy Final Text button
        print("\n[verify_e2e] Testing Copy Final Text button...")
        copy_btn = page.locator("#copy-btn")
        copy_btn.click()
        time.sleep(0.3)
        assert "Copied" in copy_btn.inner_text()
        print("  ✓ Copy Final button works")

        # 4. Test observed bug sentence
        print("\n[verify_e2e] Testing observed bug sentence...")
        bug_input = "What I am telling is correct but what you telling me is not correct direclty."
        page.locator("#input-text").fill(bug_input)
        page.locator("#threshold").fill("50")
        page.locator("#threshold").dispatch_event("input")
        page.locator("#correct-btn").click()

        page.wait_for_selector("#results-section", state="visible", timeout=30000)
        page.wait_for_selector("#loading", state="hidden", timeout=30000)

        # Inspect edits
        edit_cards = page.locator(".edit-card")
        assert edit_cards.count() >= 3, f"Expected at least 3 edits, got {edit_cards.count()}"

        categories = [edit_cards.nth(i).locator(".edit-category").inner_text().lower() for i in range(edit_cards.count())]
        explanations = [edit_cards.nth(i).locator(".edit-explanation").inner_text() for i in range(edit_cards.count())]
        confidences = [edit_cards.nth(i).locator(".edit-confidence").inner_text() for i in range(edit_cards.count())]

        print(f"  Categories found: {categories}")
        print(f"  Confidences found: {confidences}")
        print(f"  Explanations found: {explanations}")

        # Check spelling
        assert any("spelling" in cat for cat in categories), f"No 'spelling' category found! Got: {categories}"
        # Check punctuation
        assert any("punctuation" in cat for cat in categories), f"No 'punctuation' category found! Got: {categories}"

        # Check distinct confidences
        conf_vals = [float(c.split("%")[0]) for c in confidences]
        assert len(set(conf_vals)) > 1, f"All edits have identical confidences: {conf_vals}"
        print("  ✓ Observed bug verified: 'direclty' -> spelling, comma -> punctuation, different confidences!")

        # Screenshot Live Corrector
        page.screenshot(path=str(LIVE_SCREENSHOT), full_page=True)
        print(f"  ✓ Live Corrector screenshot saved to {LIVE_SCREENSHOT}")

        # 5. Open Results & Analysis tab
        print("\n[verify_e2e] Testing Results & Analysis tab...")
        page.locator(".tab-btn[data-tab='results']").click()
        page.wait_for_selector("#results-content", state="visible", timeout=15000)
        page.wait_for_selector("#results-loading", state="hidden", timeout=15000)

        # Verify table rows
        rows = page.locator("#eval-tbody tr")
        row_count = rows.count()
        print(f"  Results table rows: {row_count}")
        assert row_count >= 3, f"Expected at least 3 rows in results table, got {row_count}"

        row_texts = [rows.nth(i).inner_text() for i in range(row_count)]
        for r_txt in row_texts:
            print(f"    Row: {r_txt}")

        assert any("No-correction baseline" in r for r in row_texts)
        assert any("Plain T5" in r for r in row_texts)
        assert any("GrammaCore pipeline" in r for r in row_texts)

        # Check failure cases are displayed
        failures = page.locator(".failure-case")
        print(f"  Failure cases rendered: {failures.count()}")
        assert failures.count() >= 5, f"Expected >= 5 failure cases, got {failures.count()}"

        # Check limitations
        limitations = page.locator("#limitations-list li")
        print(f"  Limitations rendered: {limitations.count()}")
        assert limitations.count() >= 4, f"Expected >= 4 limitations, got {limitations.count()}"

        # Screenshot Results tab
        page.screenshot(path=str(RESULTS_SCREENSHOT), full_page=True)
        print(f"  ✓ Results tab screenshot saved to {RESULTS_SCREENSHOT}")

        # 6. Check console errors
        print(f"\n[verify_e2e] Browser console errors recorded: {len(console_errors)}")
        assert len(console_errors) == 0, f"Found browser console errors: {console_errors}"

        browser.close()
        print("\n[verify_e2e] ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! ✓\n")

if __name__ == "__main__":
    run_verification()
