# ============================================================
# src/document_analyzer.py
# Structured analysis — works with Groq OR Ollama
# ============================================================

import json
import logging
import re
from typing import Dict

from langchain_core.prompts import PromptTemplate
from config.settings import DEFAULT_MODEL, LLM_CONTEXT_WINDOW, GROQ_API_KEY, USE_GROQ, GROQ_MODEL

logger = logging.getLogger(__name__)


def _build_llm():
    if USE_GROQ and GROQ_API_KEY:
        from langchain_groq import ChatGroq
        return ChatGroq(model=GROQ_MODEL, temperature=0.0, max_tokens=1024, groq_api_key=GROQ_API_KEY)
    else:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=DEFAULT_MODEL, temperature=0.0, num_predict=1024, num_ctx=LLM_CONTEXT_WINDOW)


class DMEDocumentAnalyzer:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.llm = _build_llm()

    def analyze_claim(self, claim_text: str) -> Dict:
        prompt = PromptTemplate(
            input_variables=["claim_text"],
            template="""Extract billing information from the claim. Return ONLY valid JSON, no markdown.

{{
  "patient_name": null, "patient_dob": null, "patient_id": null,
  "date_of_service": null, "hcpcs_codes": [], "diagnosis_codes": [],
  "modifiers": [], "provider_name": null, "provider_npi": null,
  "billing_amount": null, "payer": null, "prior_auth_number": null,
  "potential_issues": []
}}

Claim:
{claim_text}

JSON:""",
        )
        raw = self.llm.invoke(prompt.format(claim_text=claim_text[:3000]))
        if hasattr(raw, "content"):
            raw = raw.content
        return self._parse_json(raw)

    def predict_denial_risk(self, claim_data: Dict) -> Dict:
        risk_score, factors = 0, []
        hcpcs = claim_data.get("hcpcs_codes", [])

        for code in hcpcs:
            if code.upper() in {"E0470", "E0471", "K0005", "K0856", "E1390"}:
                factors.append(f"High-scrutiny code: {code}")
                risk_score += 20

        if not claim_data.get("diagnosis_codes"):
            factors.append("No diagnosis codes")
            risk_score += 30
        if not claim_data.get("provider_npi"):
            factors.append("Missing provider NPI")
            risk_score += 30
        if not claim_data.get("modifiers"):
            for c in hcpcs:
                if c.upper() in {"E0601", "E0470", "E0471", "E1390"}:
                    factors.append(f"No modifier on rental code {c}")
                    risk_score += 15

        risk_score = min(risk_score, 100)
        level = "HIGH" if risk_score >= 50 else "MEDIUM" if risk_score >= 25 else "LOW"
        return {
            "risk_score": risk_score, "risk_level": level,
            "risk_factors": factors,
            "recommendation": (
                "Hold for review" if level == "HIGH"
                else "Review then submit" if level == "MEDIUM"
                else "Ready for submission"
            ),
        }

    def extract_cmn_fields(self, cmn_text: str) -> Dict:
        prompt = PromptTemplate(
            input_variables=["cmn_text"],
            template="""Extract fields from this CMN. Return ONLY valid JSON, no markdown.

{{
  "form_type": null, "patient_name": null, "hic_number": null,
  "date_of_order": null, "equipment_requested": null, "primary_diagnosis": null,
  "physician_name": null, "physician_npi": null, "physician_signature_date": null,
  "initial_or_recertification": null, "length_of_need": null, "clinical_summary": null
}}

CMN:
{cmn_text}

JSON:""",
        )
        raw = self.llm.invoke(prompt.format(cmn_text=cmn_text[:3000]))
        if hasattr(raw, "content"):
            raw = raw.content
        return self._parse_json(raw)

    def _parse_json(self, raw: str) -> Dict:
        try:
            clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.warning(f"JSON parse failed: {e}")
        return {"error": "Could not parse", "raw": raw[:300]}
