import os
import json
import re
import time
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from google.genai import types

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

app = FastAPI(
    title="AI Hallucination Detector",
    description="Checks claims against web search evidence and Gemini analysis.",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class AnalysisRequest(BaseModel):
    text: str
    context: Optional[str] = None


class ClaimResult(BaseModel):
    claim: str
    verdict: str
    confidence: int
    explanation: str
    is_hallucination: bool
    sources: list[dict]


class AnalysisResponse(BaseModel):
    original_text: str
    hallucination_percentage: float
    overall_verdict: str
    claims: list[ClaimResult]
    summary: str
    analysis_time: float



def search_web(query: str) -> list[dict]:
    """Search for relevant evidence using Tavily's advanced search."""
    search_query = (
        f'{query} factual verification evidence reliable sources'
    )

    response = requests.post(
        "https://api.tavily.com/search",
        headers={
            "Authorization": f"Bearer {TAVILY_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "query": search_query,
            "search_depth": "basic",
            "max_results": 6,
            "include_answer": False,
            "include_raw_content": False,
        },
        timeout=30,
    )

    if response.status_code != 200:
        print("Tavily error:", response.status_code, response.text[:500])
        raise HTTPException(
            status_code=502,
            detail=(
                "Web search failed. Check your Tavily API key "
                "and account limits."
            ),
        )

    data = response.json()
    results = data.get("results", [])
    cleaned = []

    query_words = {
        word.lower()
        for word in re.findall(r"\b[a-zA-Z]{4,}\b", query)
    }

    for item in results:
        title = item.get("title") or "Web result"
        url = item.get("url") or ""
        content = item.get("content") or ""

        if not url or not content.strip():
            continue

        # Prefer sources whose title or snippet overlaps with the claim.
        searchable_text = f"{title} {content}".lower()
        overlap = sum(
            1 for word in query_words if word in searchable_text
        )
        relevance = overlap / max(len(query_words), 1)

        # Keep reasonably relevant results, but don't discard everything
        # when the claim is unusual or difficult to search.
        score = item.get("score", 0)
        if relevance < 0.10 and score < 0.25:
            continue

        cleaned.append({
            "title": title,
            "url": url,
            "content": content[:1800],
            "relevance_score": round(float(score), 3),
        })

    # Put the stronger search matches first.
    cleaned.sort(
        key=lambda item: item["relevance_score"],
        reverse=True,
    )

    return cleaned[:5]



def gemini_json(client, prompt, retries=4):
    """Call Gemini, retry when it is busy, and fall back to a second model."""
    models = [os.getenv("GEMINI_MODEL", "gemini-3.8-flash"), "gemini-2.5-flash"]
    last = None
    for attempt in range(retries):
        for model in models:
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (r.text or "").strip())
                return json.loads(text)
            except Exception as e:
                msg = str(e)
                if any(k in msg for k in ("503", "429", "UNAVAILABLE", "overloaded", "RESOURCE_EXHAUSTED")):
                    print(f"Gemini busy ({model}), attempt {attempt + 1}/{retries}")
                    last = e
                    continue
                raise
        time.sleep(2 ** attempt)
    raise last


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest):
    started = time.time()

    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is missing. Add GOOGLE_API_KEY to your .env file."
        )
    if not TAVILY_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Tavily API key is missing. Add TAVILY_API_KEY to your .env file."
        )

    text = request.text.strip()
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Please enter at least 20 characters.")
    if len(text) > 12000:
        raise HTTPException(status_code=400, detail="Please keep the text under 12,000 characters.")

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)

        # First ask Gemini to identify individual factual claims.
        extract_prompt = f"""
Identify up to 8 distinct factual claims in the text below that can be checked using public sources.
Do not treat opinions or personal preferences as factual claims.
Return ONLY valid JSON in this format:
{{"claims": ["claim one", "claim two"]}}

Optional topic/context: {request.context or "Not provided"}

Text:
{text}
"""
        extracted = gemini_json(client, extract_prompt)
        raw_claims = extracted.get("claims", [])
        raw_claims = [str(c).strip() for c in raw_claims if str(c).strip()][:8]

        if not raw_claims:
            return AnalysisResponse(
                original_text=text,
                hallucination_percentage=0,
                overall_verdict="NO_CLAIMS_FOUND",
                claims=[],
                summary="No clear factual claims were identified. Try entering a passage containing specific facts, dates, figures, or statements.",
                analysis_time=round(time.time() - started, 2),
            )

        # Search each claim separately so each result has its own evidence.
        evidence_by_claim = []
        for claim in raw_claims:
            sources = search_web(claim)
            evidence_by_claim.append({"claim": claim, "sources": sources})

        verification_prompt = f"""
You are carefully checking factual claims against retrieved web search evidence.
The evidence may be incomplete, misleading, outdated, or low quality. Do not assume a claim is true just because a search result says so.
Use only these verdicts:
- SUPPORTED: the provided evidence supports the claim
- CONTRADICTED: reliable evidence directly conflicts with the claim
- UNVERIFIED: the evidence is insufficient, ambiguous, or not reliable enough

Rules:
- Never invent evidence or citations.
- If the evidence does not clearly settle the claim, use UNVERIFIED.
- "is_hallucination" must be true only for CONTRADICTED claims. UNVERIFIED does not automatically mean false.
- Confidence is your confidence in this evidence-based verdict, 0 to 100.
- Keep explanations concise and grounded in the evidence.
- Return ONLY JSON matching this structure:
{{
  "claims": [
    {{
      "claim": "claim text",
      "verdict": "SUPPORTED|CONTRADICTED|UNVERIFIED",
      "confidence": 0,
      "explanation": "brief explanation based on the evidence",
      "is_hallucination": false
    }}
  ],
  "summary": "2-3 sentence summary"
}}

Original text:
{text}

Retrieved evidence for each claim:
{json.dumps(evidence_by_claim, ensure_ascii=False)}
"""
        verified = gemini_json(client, verification_prompt)
        verified_claims = verified.get("claims", [])

        # Attach source links to the matching claim.
        results = []
        for index, item in enumerate(verified_claims[:8]):
            claim_text = str(item.get("claim", raw_claims[index] if index < len(raw_claims) else "Claim"))
            matched = next(
                (e for e in evidence_by_claim if e["claim"].lower() == claim_text.lower()),
                evidence_by_claim[index] if index < len(evidence_by_claim) else {"sources": []}
            )
            verdict = str(item.get("verdict", "UNVERIFIED")).upper()
            if verdict not in {"SUPPORTED", "CONTRADICTED", "UNVERIFIED"}:
                verdict = "UNVERIFIED"
            sources = [
                {"title": s["title"], "url": s["url"], "content": s["content"]}
                for s in matched["sources"] if s.get("url")
            ]
            results.append(ClaimResult(
                claim=claim_text,
                verdict=verdict,
                confidence=max(0, min(100, int(item.get("confidence", 50)))),
                explanation=str(item.get("explanation", "Evidence was reviewed.")),
                is_hallucination=(verdict == "CONTRADICTED"),
                sources=sources,
            ))

        contradicted = sum(1 for c in results if c.verdict == "CONTRADICTED")
        percentage = round((contradicted / len(results)) * 100, 1) if results else 0
        if percentage == 0:
            overall = "NO_CONTRADICTIONS_FOUND"
        elif percentage < 35:
            overall = "SOME_CONTRADICTIONS_FOUND"
        else:
            overall = "MANY_CONTRADICTIONS_FOUND"

        return AnalysisResponse(
            original_text=text,
            hallucination_percentage=percentage,
            overall_verdict=overall,
            claims=results,
            summary=str(verified.get("summary", "Analysis complete. Please review the linked evidence yourself.")),
            analysis_time=round(time.time() - started, 2),
        )

    except HTTPException:
        raise
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="The AI returned an unexpected response. Please try again.")
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="Could not reach the web search service. Check your internet connection.")
    except Exception as exc:
        error_text = str(exc)
        print("Analysis error:", repr(exc))
        if any(k in error_text for k in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
            raise HTTPException(
                status_code=503,
                detail="The AI service is temporarily busy. Please try again shortly."
            )
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Check the backend terminal for details."
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)