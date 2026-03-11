# ============================================================
# src/billing_tools.py
# Custom LangChain tools for DME billing workflows
# Tools are callable by the LangChain agent automatically
# ============================================================

import re
import json
import logging
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Tool 1: HCPCS Code Lookup ────────────────────────────────

class HCPCSLookupInput(BaseModel):
    code: str = Field(description="HCPCS or CPT code to look up, e.g. E0601 or K0001")


class HCPCSLookupTool(BaseTool):
    """Look up HCPCS codes from a built-in reference database."""

    name: str = "hcpcs_code_lookup"
    description: str = (
        "Use this to look up HCPCS or CPT billing codes. "
        "Input a specific code like E0601 or K0001. "
        "Returns description, coverage category, CMN requirement, and billing notes."
    )
    args_schema: type = HCPCSLookupInput

    HCPCS_DB: dict = {
        "E0601": {
            "description": "Continuous Positive Airway Pressure (CPAP) Device",
            "category": "Respiratory Equipment",
            "medicare_covered": True,
            "cmn_required": True,
            "prior_auth": False,
            "rental_purchase": "Rental — 13-month capped, then patient owns",
            "notes": "Requires compliance data after month 3 (KX modifier)",
        },
        "E0470": {
            "description": "Respiratory Assist Device, BiPAP without backup rate",
            "category": "Respiratory Equipment",
            "medicare_covered": True,
            "cmn_required": True,
            "prior_auth": True,
            "rental_purchase": "Rental",
            "notes": "Covered for COPD, OHS, CSA. Prior auth required by most payers.",
        },
        "E0471": {
            "description": "Respiratory Assist Device, BiPAP with backup rate",
            "category": "Respiratory Equipment",
            "medicare_covered": True,
            "cmn_required": True,
            "prior_auth": True,
            "rental_purchase": "Rental",
            "notes": "Higher complexity documentation required",
        },
        "K0001": {
            "description": "Standard Manual Wheelchair",
            "category": "Mobility Equipment",
            "medicare_covered": True,
            "cmn_required": False,
            "prior_auth": False,
            "rental_purchase": "Purchase",
            "notes": "Written order required. Face-to-face exam needed.",
        },
        "K0005": {
            "description": "Ultralightweight Manual Wheelchair (< 17 lbs)",
            "category": "Mobility Equipment",
            "medicare_covered": True,
            "cmn_required": True,
            "prior_auth": True,
            "rental_purchase": "Purchase",
            "notes": "Requires detailed functional assessment and mobility evaluation",
        },
        "E0143": {
            "description": "Walker, folding, wheeled, adjustable or fixed height",
            "category": "Mobility Aids",
            "medicare_covered": True,
            "cmn_required": False,
            "prior_auth": False,
            "rental_purchase": "Purchase",
            "notes": "Written order from treating physician required",
        },
        "E0105": {
            "description": "Cane, adjustable or fixed, with tips",
            "category": "Mobility Aids",
            "medicare_covered": True,
            "cmn_required": False,
            "prior_auth": False,
            "rental_purchase": "Purchase",
            "notes": "Standard coverage — keep prescription on file",
        },
        "A4253": {
            "description": "Blood Glucose Test Strips, per 50",
            "category": "Diabetic Supplies",
            "medicare_covered": True,
            "cmn_required": False,
            "prior_auth": False,
            "rental_purchase": "Purchase",
            "notes": "Quantity limits apply — check LCD for covered amounts",
        },
        "E1390": {
            "description": "Oxygen Concentrator, Single Delivery Port",
            "category": "Respiratory Equipment",
            "medicare_covered": True,
            "cmn_required": True,
            "prior_auth": False,
            "rental_purchase": "Rental — 36-month capped",
            "notes": "Requires qualifying PO2 or SpO2 test results",
        },
        "L3900": {
            "description": "Orthosis, wrist-hand-finger, dynamic",
            "category": "Orthotics",
            "medicare_covered": True,
            "cmn_required": False,
            "prior_auth": False,
            "rental_purchase": "Purchase",
            "notes": "Custom vs prefab distinction affects coding",
        },
    }

    def _run(self, code: str) -> str:
        code = code.upper().strip().replace(" ", "")
        if code in self.HCPCS_DB:
            info = self.HCPCS_DB[code]
            return (
                f"HCPCS Code: {code}\n"
                f"Description: {info['description']}\n"
                f"Category: {info['category']}\n"
                f"Medicare Covered: {'Yes' if info['medicare_covered'] else 'No'}\n"
                f"CMN Required: {'Yes' if info['cmn_required'] else 'No'}\n"
                f"Prior Authorization: {'Required' if info['prior_auth'] else 'Not Required'}\n"
                f"Rental / Purchase: {info['rental_purchase']}\n"
                f"Notes: {info['notes']}"
            )
        return (
            f"Code {code} not found in local database.\n"
            "Tip: Search the CMS HCPCS database at https://www.cms.gov/medicare/coding-billing/hcpcs-codes"
        )

    async def _arun(self, code: str) -> str:
        return self._run(code)


# ── Tool 2: Claim Validator ──────────────────────────────────

class ClaimValidatorInput(BaseModel):
    claim_data: str = Field(
        description=(
            "JSON string with claim fields: hcpcs_code, patient_id, "
            "date_of_service, diagnosis_codes (list), provider_npi"
        )
    )


class ClaimValidatorTool(BaseTool):
    """Validate a DME claim for common errors before submission."""

    name: str = "validate_dme_claim"
    description: str = (
        "Validate a DME claim before submitting it. "
        "Input is a JSON string with claim fields. "
        "Returns a list of errors or a success message."
    )
    args_schema: type = ClaimValidatorInput

    # Codes that require prior authorization (expand as needed)
    PRIOR_AUTH_REQUIRED = {"E0470", "E0471", "K0005", "K0856", "E1390"}

    def _run(self, claim_data: str) -> str:
        try:
            claim = json.loads(claim_data)
        except json.JSONDecodeError:
            return "Error: Input is not valid JSON. Please reformat and try again."

        errors, warnings = [], []

        # Required fields
        for field in ["hcpcs_code", "date_of_service", "diagnosis_codes", "provider_npi"]:
            if not claim.get(field):
                errors.append(f"Missing required field: {field}")

        # HCPCS format (5 alphanumeric characters)
        hcpcs = claim.get("hcpcs_code", "").upper()
        if hcpcs and not re.match(r"^[A-Z0-9]{5}$", hcpcs):
            errors.append(f"Invalid HCPCS format: '{hcpcs}' — must be 5 alphanumeric chars")

        # NPI format (exactly 10 digits)
        npi = str(claim.get("provider_npi", ""))
        if npi and not re.match(r"^\d{10}$", npi):
            errors.append(f"Invalid NPI: '{npi}' — must be exactly 10 digits")

        # ICD-10 diagnosis code format
        dx_codes = claim.get("diagnosis_codes", [])
        if isinstance(dx_codes, str):
            dx_codes = [dx_codes]
        for dx in dx_codes:
            if not re.match(r"^[A-Z]\d{2}(\.\w+)?$", dx.upper()):
                warnings.append(f"Unusual ICD-10 format: '{dx}' — expected format like G47.33")

        # Prior authorization check
        if hcpcs in self.PRIOR_AUTH_REQUIRED:
            if not claim.get("prior_auth_number"):
                warnings.append(
                    f"{hcpcs} typically requires prior authorization. "
                    "Add 'prior_auth_number' field to confirm it was obtained."
                )

        # CMN check for certain codes
        cmn_required_codes = {"E0601", "E0470", "E0471", "E1390", "K0005"}
        if hcpcs in cmn_required_codes and not claim.get("cmn_on_file"):
            warnings.append(f"{hcpcs} requires a Certificate of Medical Necessity on file")

        if errors:
            return "❌ VALIDATION FAILED\nErrors:\n" + "\n".join(f"  • {e}" for e in errors)
        if warnings:
            return "⚠️ PASSED WITH WARNINGS\nWarnings:\n" + "\n".join(f"  • {w}" for w in warnings)
        return "✅ CLAIM VALIDATION PASSED — All checks successful"

    async def _arun(self, claim_data: str) -> str:
        return self._run(claim_data)


# ── Tool 3: Denial Analyzer ──────────────────────────────────

class DenialAnalyzerInput(BaseModel):
    denial_code: str = Field(
        description="Denial reason code or description from EOB/remittance, e.g. CO-4 or CO-197"
    )


class DenialAnalyzerTool(BaseTool):
    """Analyze a claim denial and return remediation steps."""

    name: str = "analyze_denial"
    description: str = (
        "Analyze why a DME claim was denied and get specific remediation steps. "
        "Input should be the denial code (e.g., CO-4) or denial reason text."
    )
    args_schema: type = DenialAnalyzerInput

    DENIAL_GUIDE: dict = {
        "CO-4": {
            "description": "Service/equipment not authorized by payer",
            "action": "Obtain prior authorization. Contact payer to request retro-auth if urgent.",
            "appeal": "Submit retro-authorization request with clinical documentation",
            "timeframe": "120 days from denial date",
        },
        "CO-22": {
            "description": "Coordination of Benefits — another payer is primary",
            "action": "Bill primary insurance first, then submit secondary with primary EOB attached",
            "appeal": "Submit COB information and primary EOB to secondary payer",
            "timeframe": "File primary within 1 year of DOS; secondary within plan limits",
        },
        "CO-50": {
            "description": "Non-covered service — not deemed medically necessary",
            "action": "Gather detailed medical records, physician notes, and CMN supporting necessity",
            "appeal": "Write medical necessity appeal letter with supporting clinical documentation",
            "timeframe": "120 days from denial",
        },
        "CO-97": {
            "description": "Payment is included in the allowance for another service/procedure",
            "action": "Review if bundling rules apply. Check if correct modifiers were used.",
            "appeal": "Attach modifier 59 if procedures are distinct. Request itemized denial.",
            "timeframe": "120 days",
        },
        "CO-167": {
            "description": "Diagnosis is not covered / does not match equipment",
            "action": "Verify ICD-10 code is on the covered diagnosis list in the applicable LCD",
            "appeal": "Submit with amended diagnosis code or additional supporting diagnosis",
            "timeframe": "120 days",
        },
        "CO-197": {
            "description": "Precertification / prior authorization absent",
            "action": "Request retro-authorization immediately. Document any emergency circumstances.",
            "appeal": "Emergency exception or retro-auth request — act within 72 hours",
            "timeframe": "Varies by payer — act immediately",
        },
        "CO-B7": {
            "description": "Provider not accredited for this procedure",
            "action": "Verify supplier accreditation status. Check CBA eligibility if applicable.",
            "appeal": "Submit proof of accreditation or corrected billing",
            "timeframe": "90 days",
        },
        "PR-96": {
            "description": "Non-covered charge — patient responsibility",
            "action": "Check if ABN was signed. If yes: bill patient. If no: write off the charge.",
            "appeal": "If ABN issue exists, file a grievance with Medicare",
            "timeframe": "Administrative — varies",
        },
        "CO-16": {
            "description": "Claim requires information not on file",
            "action": "Review the remittance remarks codes for the specific missing item",
            "appeal": "Resubmit with complete documentation (CMN, physician order, medical notes)",
            "timeframe": "120 days",
        },
        "CO-55": {
            "description": "Procedure inconsistent with patient age",
            "action": "Verify patient date of birth on claim is correct",
            "appeal": "Correct patient demographics and resubmit as corrected claim",
            "timeframe": "Correct and resubmit within 120 days",
        },
    }

    def _run(self, denial_code: str) -> str:
        code = denial_code.upper().strip()

        if code in self.DENIAL_GUIDE:
            info = self.DENIAL_GUIDE[code]
            return (
                f"Denial Code: {code}\n"
                f"Description: {info['description']}\n\n"
                f"Recommended Action:\n{info['action']}\n\n"
                f"Appeal Strategy:\n{info['appeal']}\n\n"
                f"Timeframe: {info['timeframe']}"
            )

        # Partial / fuzzy match
        for known, info in self.DENIAL_GUIDE.items():
            if known in code or code.replace("-", "") in known.replace("-", ""):
                return f"Closest match for '{denial_code}':\n\n" + self._run(known)

        return (
            f"Denial code '{denial_code}' not found in local database.\n\n"
            "General DME Denial Response Steps:\n"
            "1. Review the full EOB/remittance for detailed remarks codes\n"
            "2. Check the payer portal for denial reason detail\n"
            "3. Verify all documentation is complete (CMN, physician order, medical records)\n"
            "4. Confirm prior authorization was obtained if required\n"
            "5. Contact payer provider relations for clarification\n"
            "6. File appeal within the deadline (typically 120 days)\n\n"
            "Reference: https://x12.org/codes/claim-adjustment-reason-codes"
        )

    async def _arun(self, denial_code: str) -> str:
        return self._run(denial_code)
