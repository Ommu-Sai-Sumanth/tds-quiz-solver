"""
FastAPI quiz solver endpoint for the TDS LLM Analysis project.

Features:
- Verifies incoming JSON and secret
- Uses Playwright headless browser to fetch and render JS pages
- Extracts a submit URL from the quiz page (attempts common patterns)
- Downloads linked files (PDF/CSV/JSON) and tries to extract numeric/string answers
- Posts answer JSON back to the provided submit URL
- Loops if a new URL is returned, within a 3-minute window

NOTES/ASSUMPTIONS:
- Deploy behind HTTPS. Set environment variable QUIZ_SECRET to your chosen secret.
- This is a template: adapt parsing logic to match the specific quiz types you encounter.
- Install requirements: fastapi, uvicorn, httpx, playwright, pandas, PyPDF2, python-multipart
  and run `playwright install` during deployment build.

"""

import os
import re
import time
import json
import base64
import asyncio
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx

# Playwright imports
from playwright.async_api import async_playwright

# PDF parsing
import io
from PyPDF2 import PdfReader
import pandas as pd

# Config
QUIZ_SECRET = os.environ.get("QUIZ_SECRET", "replace-with-your-secret")
MAX_TOTAL_SECONDS = 180  # 3 minutes
PAGE_LOAD_TIMEOUT = 30_000  # ms

app = FastAPI(title="TDS Quiz Solver Endpoint")


def validate_payload(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    required = ["email", "secret", "url"]
    for r in required:
        if r not in payload:
            raise HTTPException(status_code=400, detail=f"Missing field: {r}")


@app.post("/")
async def handle_quiz(req: Request):
    try:
        payload = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    validate_payload(payload)
    email = payload.get("email")
    secret = payload.get("secret")
    start_url = payload.get("url")

    if secret != QUIZ_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # Start timer
    t0 = time.time()

    result = {
        "email": email,
        "status": "started",
        "steps": []
    }

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        current_url = start_url
        last_response = None

        while True:
            elapsed = time.time() - t0
            if elapsed > MAX_TOTAL_SECONDS:
                result["status"] = "timeout"
                break

            try:
                await page.goto(current_url, timeout=PAGE_LOAD_TIMEOUT)
                # Wait for network and rendering
                await page.wait_for_load_state("networkidle", timeout=PAGE_LOAD_TIMEOUT)
            except Exception as e:
                result["steps"].append({"url": current_url, "error": f"load_failed: {e}"})
                break

            # Try to extract useful content
            page_content = await page.content()
            text_content = await page.inner_text("body")
            result["steps"].append({"url": current_url, "body_preview": text_content[:1000]})

            # Attempt to find a submit URL on the page (common patterns)
            submit_url = await find_submit_url(page, text_content)

            # Try to solve the quiz on the page
            answer_payload = None
            try:
                answer_payload = await solve_quiz_from_page(page, text_content, email, secret)
            except Exception as e:
                result["steps"].append({"url": current_url, "solve_error": str(e)})

            if submit_url and answer_payload is not None:
                # Post the answer
                try:
                    async with httpx.AsyncClient(timeout=30) as client:
                        r = await client.post(submit_url, json=answer_payload)
                        last_response = r
                except Exception as e:
                    result["steps"].append({"post_error": str(e)})
                    break

                try:
                    resp_json = r.json()
                except Exception:
                    resp_json = {"status_text": r.text}

                result["steps"].append({"posted_to": submit_url, "answer": answer_payload, "resp": resp_json})

                # Check response for correctness and next url
                if isinstance(resp_json, dict) and resp_json.get("url"):
                    # move to next URL
                    current_url = resp_json.get("url")
                    # allow further loop iterations until timeout
                    continue
                else:
                    # final response
                    result["status"] = "finished"
                    result["final_response"] = resp_json
                    break
            else:
                # No submit URL or no answer. Stop.
                result["status"] = "no_submit_or_no_answer"
                break

        await browser.close()

    return JSONResponse(content=result)


async def find_submit_url(page, text_content: str) -> Optional[str]:
    # Heuristics: look for action or API endpoints in forms or direct examples in text
    # Check for form actions
    try:
        forms = await page.query_selector_all("form")
        for form in forms:
            action = await form.get_attribute("action")
            if action:
                if action.startswith("/"):
                    base = page.url
                    action = httpx.URL(base).join(action).human_repr()
                return action
    except Exception:
        pass

    # Search text for obvious URLs
    m = re.search(r"https?://[\w\-._~:/?#[\]@!$&'()*+,;=%]+", text_content)
    if m:
        return m.group(0)
    return None


async def solve_quiz_from_page(page, text_content: str, email: str, secret: str) -> Optional[dict]:
    """A flexible solver that handles a few common patterns. Extend this for your quiz types.

    Strategies implemented here (basic examples):
    - If page contains a base64-encoded JSON inside atob(...) (like sample), decode and use its 'answer'
    - If page links to a CSV/JSON/PDF file, download and compute simple aggregates
    - If the page contains an explicit instruction and a submit URL, craft the answer
    """
    # 1) detect base64 JSON inside page scripts
    text = text_content
    atob_matches = re.findall(r"atob\(`([A-Za-z0-9+/=\n]+)`\)", page.content or "")
    if not atob_matches:
        # try searching in page text too
        atob_matches = re.findall(r"atob\('\s*([A-Za-z0-9+/=\\n]+)\s*'\)", text)

    if atob_matches:
        for b64 in atob_matches:
            try:
                decoded = base64.b64decode(b64)
                s = decoded.decode("utf-8", errors="ignore")
                # if it looks like JSON
                jmatch = re.search(r"\{.*\}", s, flags=re.DOTALL)
                if jmatch:
                    j = json.loads(jmatch.group(0))
                    # If it contains 'answer' or instructions
                    if "answer" in j:
                        return j
                    # fallback: prepare payload per sample
                    return {"email": email, "secret": secret, "url": page.url, "answer": j.get("answer")}
            except Exception:
                continue

    # 2) Look for links to files (pdf/csv/json)
    links = await page.query_selector_all("a")
    hrefs = []
    for a in links:
        href = await a.get_attribute("href")
        if href:
            hrefs.append(href)

    for href in hrefs:
        if href.lower().endswith(".csv") or ".csv?" in href.lower():
            dv = await download_and_sum_csv(href, page)
            if dv is not None:
                return {"email": email, "secret": secret, "url": page.url, "answer": dv}
        if href.lower().endswith(".pdf"):
            # Attempt to download and perform PDF-based tasks
            pdf_result = await download_and_process_pdf(href, page)
            if pdf_result is not None:
                return {"email": email, "secret": secret, "url": page.url, "answer": pdf_result}

    # 3) As a last resort, try to find explicit numeric question in the text
    # e.g. "What is the sum of the \"value\" column" => we can't compute without file
    # Return None to indicate we couldn't craft an answer
    return None


async def download_and_sum_csv(href: str, page) -> Optional[object]:
    # Resolve relative URLs
    if href.startswith("/"):
        base = page.url
        href = httpx.URL(base).join(href).human_repr()

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.get(href)
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content))
            # heuristic: if column named 'value' exists, sum it
            if "value" in df.columns:
                s = float(df["value"].sum())
                return s
            # otherwise try numeric columns
            numeric = df.select_dtypes(include=["number"]).sum(numeric_only=True)
            if len(numeric) == 1:
                return float(numeric.iloc[0])
            # otherwise return first aggregate as fallback
            return numeric.to_dict()
        except Exception:
            return None


async def download_and_process_pdf(href: str, page) -> Optional[object]:
    if href.startswith("/"):
        base = page.url
        href = httpx.URL(base).join(href).human_repr()
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.get(href)
            r.raise_for_status()
            reader = PdfReader(io.BytesIO(r.content))
            # Basic heuristic: look for tables by text and find a column named 'value'
            full_text_pages = [\n                    p.extract_text() or "" for p in reader.pages\n            ]
            # Example heuristic: if question asked for 'sum of the "value" column in the table on page 2',
            # parse page 2
            if len(full_text_pages) >= 2:
                page2 = full_text_pages[1]
                # find numbers
                nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", page2)
                nums = [float(n) for n in nums]
                return sum(nums)
            else:
                # fallback: sum all numbers across PDF
                nums = []
                for t in full_text_pages:
                    found = re.findall(r"[-+]?[0-9]*\.?[0-9]+", t)
                    nums.extend([float(x) for x in found])
                if nums:
                    return sum(nums)
            return None
        except Exception:
            return None


# If running as main for local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
