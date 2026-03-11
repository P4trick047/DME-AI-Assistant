# ============================================================
# tests/test_tools.py
# Week 4 verification — test the billing tools
# Run: python tests/test_tools.py
# ============================================================

import sys
sys.path.insert(0, ".")

from src.billing_tools import HCPCSLookupTool, ClaimValidatorTool, DenialAnalyzerTool


def test_hcpcs_lookup():
    print("=" * 55)
    print("🧪 Test 1: HCPCS Lookup Tool")
    print("=" * 55)
    tool = HCPCSLookupTool()

    result = tool._run("E0601")
    print(result)
    assert "CPAP" in result
    assert "CMN Required: Yes" in result

    result2 = tool._run("K0001")
    assert "Wheelchair" in result2
    print("✅ HCPCS Lookup — PASS\n")


def test_claim_validator():
    print("=" * 55)
    print("🧪 Test 2: Claim Validator Tool")
    print("=" * 55)
    tool = ClaimValidatorTool()
    import json

    # Valid claim
    valid = json.dumps({
        "hcpcs_code": "E0601",
        "date_of_service": "2024-01-15",
        "diagnosis_codes": ["G47.33"],
        "provider_npi": "1234567890",
    })
    result = tool._run(valid)
    print(f"Valid claim result: {result}")

    # Invalid claim — bad NPI
    invalid = json.dumps({
        "hcpcs_code": "E0601",
        "date_of_service": "2024-01-15",
        "diagnosis_codes": ["G47.33"],
        "provider_npi": "123",  # too short
    })
    result2 = tool._run(invalid)
    assert "FAILED" in result2 or "WARNINGS" in result2
    print(f"Invalid claim result: {result2}")
    print("✅ Claim Validator — PASS\n")


def test_denial_analyzer():
    print("=" * 55)
    print("🧪 Test 3: Denial Analyzer Tool")
    print("=" * 55)
    tool = DenialAnalyzerTool()

    result = tool._run("CO-50")
    print(result)
    assert "medical necessity" in result.lower() or "Necessary" in result

    result2 = tool._run("CO-197")
    assert "authorization" in result2.lower() or "auth" in result2.lower()
    print("✅ Denial Analyzer — PASS\n")


if __name__ == "__main__":
    print("🚀 Running Billing Tools Tests\n")
    test_hcpcs_lookup()
    test_claim_validator()
    test_denial_analyzer()
    print("🎉 All billing tools tests passed!")
