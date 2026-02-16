"""
LLM interface module
Provides abstraction for LLM calls with mock implementation
"""
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os

try:
    import streamlit as st
except Exception:
    st = None

def get_secret(key: str) -> Optional[str]:
    # Priority: environment variable -> Streamlit secrets
    return os.getenv(key) or (st.secrets.get(key) if st else None)


class LLMInterface(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """
        Generate completion from prompt

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        pass

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None,
                     temperature: float = 0.3) -> Dict[str, Any]:
        """
        Generate JSON response from prompt

        Args:
            prompt: User prompt (should request JSON output)
            system_prompt: System prompt
            temperature: Lower temperature for more deterministic JSON

        Returns:
            Parsed JSON dictionary
        """
        response = self.generate(prompt, system_prompt, temperature=temperature)

        # Try to extract JSON from response
        try:
            # Try parsing directly
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON in markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # Try to find JSON object in text
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))

            raise ValueError(f"Could not parse JSON from response: {response[:200]}")


class MockLLM(LLMInterface):
    """Mock LLM that returns deterministic outputs for testing"""

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """
        Generate mock response based on prompt type

        This returns realistic but deterministic outputs so the app runs without API keys
        """
        prompt_lower = prompt.lower()

        # Detect prompt type and return appropriate mock response
        if "change" in prompt_lower and "summary" in prompt_lower:
            return self._mock_change_summary()
        elif "impact" in prompt_lower:
            return self._mock_impact_analysis()
        elif "test" in prompt_lower and "scenario" in prompt_lower:
            return self._mock_test_scenarios()
        elif "executive summary" in prompt_lower:
            return self._mock_executive_summary()
        else:
            return "Mock LLM response for: " + prompt[:100]

    def _mock_change_summary(self) -> str:
        """Mock change summary response"""
        return json.dumps({
            "what_changed": [
                "Coverage limit increased from $500,000 to $1,000,000",
                "Deductible calculation method changed from per-incident to annual aggregate",
                "New exclusion added for cyber-related claims"
            ],
            "why_it_matters": [
                "Higher coverage limits may affect premium calculations and underwriting risk assessment",
                "Annual aggregate deductible changes claims processing workflow and customer expectations"
            ],
            "evidence_quotes_a": [
                "coverage shall not exceed $500,000 per occurrence",
                "deductible applies to each separate claim incident"
            ],
            "evidence_quotes_b": [
                "maximum coverage limit of $1,000,000 per occurrence",
                "annual aggregate deductible of $10,000"
            ],
            "severity": 4
        })

    def _mock_impact_analysis(self) -> str:
        """Mock impact analysis response"""
        return json.dumps({
            "claims": [
                {
                    "description": "Claims processing must verify new annual aggregate deductible logic instead of per-incident",
                    "evidence": "annual aggregate deductible of $10,000"
                },
                {
                    "description": "Cyber-related claims must be flagged and excluded under new policy terms",
                    "evidence": "excluding all losses arising from cyber incidents"
                }
            ],
            "underwriting": [
                {
                    "description": "Risk models must be updated to reflect new $1M coverage limit",
                    "evidence": "maximum coverage limit of $1,000,000"
                },
                {
                    "description": "Premium calculation formulas require adjustment for higher limits",
                    "evidence": "coverage limit increased to one million dollars"
                }
            ],
            "customer_support": [
                {
                    "description": "Customer FAQs must be updated to explain new annual aggregate deductible structure",
                    "evidence": "deductible calculated on annual aggregate basis"
                },
                {
                    "description": "Support scripts need updates for cyber exclusion questions",
                    "evidence": "cyber incidents are not covered under this policy"
                }
            ]
        })

    def _mock_test_scenarios(self) -> str:
        """Mock test scenarios response"""
        return json.dumps({
            "scenarios": [
                {
                    "scenario_id": "test_001",
                    "scenario_text": "Customer files a claim for $750,000 property damage after multiple incidents totaling $12,000 in deductibles throughout the year",
                    "expected_handling": "Claim should be processed under new annual aggregate deductible; customer pays $10,000 max instead of $12,000",
                    "section_refs": ["policy_b_2"],
                    "evidence": ["annual aggregate deductible of $10,000"],
                    "is_edge_case": False
                },
                {
                    "scenario_id": "test_002",
                    "scenario_text": "Customer experiences ransomware attack causing $200,000 in business interruption losses",
                    "expected_handling": "Claim must be denied due to new cyber exclusion clause in policy version B",
                    "section_refs": ["policy_b_4"],
                    "evidence": ["excluding all losses arising from cyber incidents"],
                    "is_edge_case": True
                },
                {
                    "scenario_id": "test_003",
                    "scenario_text": "High-value claim at exactly $1,000,000 submitted under new policy",
                    "expected_handling": "Should be approved at limit; triggers underwriting review for new high-limit exposure",
                    "section_refs": ["policy_b_1"],
                    "evidence": ["maximum coverage limit of $1,000,000"],
                    "is_edge_case": True
                }
            ]
        })

    def _mock_executive_summary(self) -> str:
        """Mock executive summary"""
        return """Policy Version Comparison: Executive Summary

The updated policy introduces 3 major changes and 2 minor modifications affecting claims processing, underwriting risk models, and customer communications.

**Critical Changes:**
1. Coverage limit increased from $500K to $1M (Severity: 4/5)
2. Deductible structure changed from per-incident to annual aggregate (Severity: 4/5)
3. New cyber incident exclusion added (Severity: 5/5)

**Impact Areas:**
- Claims: Requires workflow updates for deductible calculations and cyber exclusion checks
- Underwriting: Risk models and premium formulas need recalibration
- Customer Support: FAQs and scripts require updates for new terms

**Recommended Actions:**
- Update claims processing systems within 30 days
- Retrain underwriting staff on new limits
- Prepare customer communication materials
- Generate test cases for all changed sections
"""


class OpenAILLM(LLMInterface):
    """OpenAI API implementation (requires openai package and API key)"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize OpenAI LLM

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name to use
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")

        self.api_key = api_key or get_secret("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided.\n"
                "Set OPENAI_API_KEY environment variable or add to .streamlit/secrets.toml\n"
                "Or use provider='mock' to run without API keys"
            )

        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate completion using OpenAI API"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content


class AnthropicLLM(LLMInterface):
    """Anthropic Claude API implementation (requires anthropic package and API key)"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        """
        Initialize Anthropic LLM

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Model name to use
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")

        self.api_key = api_key or get_secret("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided.\n"
                "Set ANTHROPIC_API_KEY environment variable or add to .streamlit/secrets.toml\n"
                "Or use provider='mock' to run without API keys"
            )

        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate completion using Anthropic API"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text


def get_llm(provider: str = "mock", **kwargs) -> LLMInterface:
    """
    Factory function to get LLM instance

    Args:
        provider: LLM provider ("mock", "openai", "anthropic")
        **kwargs: Additional arguments for LLM initialization

    Returns:
        LLM instance
    """
    if provider == "mock":
        return MockLLM()
    elif provider == "openai":
        return OpenAILLM(**kwargs)
    elif provider == "anthropic":
        return AnthropicLLM(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
