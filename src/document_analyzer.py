# ============================================================
# src/document_analyzer.py
# Structured analysis of DME billing documents using LLM
# Extracts fields, predicts denial risk, parses CMN forms
# ============================================================

import json
import logging
import re
from typing import Dict, List

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from config.settings import DEFAULT_MODEL, LLM_CONTEXT_WINDOW

logger = logging.getLogger(__name__)


class DMEDocumentAnalyzer:
    """
    Performs structured analysis of DME billing documents.
    Uses zero-temperature LLM calls for consistent, deterministic extraction.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.0,        # Deterministic — critical for data extraction
            num_predict=1024,
            num_ctx=LLM_CONTEXT_WINDOW,
        )

    # ── Claim Field Extraction ───────────────────────────────

    def analyze_claim(self, claim_text: str) -> Dict:
        """
        Extract structured fields from a claim document.
        Returns a dict with patient info, codes, and potential issues.
        """
        prompt = PromptTemplate(
            input_variables=["claim_text"],
            template="""Extract billing information from the following claim document.
Return ONLY a valid JSON object — no markdown, no explanation.

Fields to extract (use null if not found):
{{
  "patient_name": null,
  "patient_dob": null,
  "patient_id": null,
  "date_of_service": null,
  "hcpcs_codes": [],
  "diagnosis_codes": [],
  "modifiers": [],
  "provider_name": null,
  "provider_npi": null,
  "billing_amount": null,
  "payer": null,
  "prior_auth_number": null,
  "potential_issues": []
}}

Claim document:
{claim_text}

JSON:""",
        )

        raw = self.llm.invoke(prompt.format(claim_text=claim_text[:3000]))
        return self._parse_json_response(raw)

    # ── Denial Risk Prediction ───────────────────────────────

    def predict_denial_risk(self, claim_data: Dict) -> Dict:
        """
        Rule-based denial risk scoring.
        Returns risk level (LOW / MEDIUM / HIGH) and specific risk factors.
        """
        risk_score = 0
        factors = []

        hcpcs_codes = claim_data.get("hcpcs_codes", [])
        dx_codes = claim_data.get("diagnosis_codes", [])

        # High-scrutiny codes
        high_risk = {"E0470", "E0471", "K0005", "K0856", "E1390", "E0260"}
        for code in hcpcs_codes:
            if code.upper() in high_risk:
                factors.append(f"High-scrutiny code: {code} — extra documentation required")
                risk_score += 20

        # Missing diagnosis codes
        if not dx_codes:
            factors.append("No diagnosis codes — required for all claims")
            risk_score += 30

        # Missing NPI
        if not claim_data.get("provider_npi"):
            factors.append("Missing provider NPI — will cause immediate rejection")
            risk_score += 30

        # No modifiers on rental equipment
        if not claim_data.get("modifiers"):
            for code in hcpcs_codes:
                if code.upper() in {"E0601", "E0470", "E0471", "E1390"}:
                    factors.append(
                        f"No modifier on rental code {code} — RR or KX likely required"
                    )
                    risk_score += 15

        # No prior auth on high-auth codes
        if not claim_data.get("prior_auth_number"):
            for code in hcpcs_codes:
                if code.upper() in {"E0470", "E0471", "K0005"}:
                    factors.append(f"No prior auth number for {code} — usually required")
                    risk_score += 20

        risk_score = min(risk_score, 100)

        if risk_score >= 50:
            level = "HIGH"
            recommendation = "Hold for documentation review before submission"
        elif risk_score >= 25:
            level = "MEDIUM"
            recommendation = "Review risk factors then submit"
        else:
            level = "LOW"
            recommendation = "Appears ready for submission"

        return {
            "risk_score": risk_score,
            "risk_level": level,
            "risk_factors": factors,
            "recommendation": recommendation,
        }

    # ── CMN Extraction ───────────────────────────────────────

    def extract_cmn_fields(self, cmn_text: str) -> Dict:
        """Extract key fields from a Certificate of Medical Necessity."""
        prompt = PromptTemplate(
            input_variables=["cmn_text"],
            template="""Extract fields from this Certificate of Medical Necessity (CMN).
Return ONLY valid JSON — no markdown, no explanation.

{{
  "form_type": null,
  "patient_name": null,
  "hic_number": null,
  "date_of_order": null,
  "equipment_requested": null,
  "primary_diagnosis": null,
  "physician_name": null,
  "physician_npi": null,
  "physician_signature_date": null,
  "initial_or_recertification": null,
  "length_of_need": null,
  "clinical_summary": null
}}

CMN text:
{cmn_text}

JSON:""",
        )

        raw = self.llm.invoke(prompt.format(cmn_text=cmn_text[:3000]))
        return self._parse_json_response(raw)

    # ── Helpers ──────────────────────────────────────────────

    def _parse_json_response(self, raw: str) -> Dict:
        """Safely parse JSON from LLM response, handling markdown fences."""
        try:
            # Strip markdown code fences if present
            clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            # Find the outermost JSON object
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"JSON parse failed: {e}")
        return {"error": "Could not parse structured data", "raw": raw[:500]}
