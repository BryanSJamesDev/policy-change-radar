"""
Impact analysis module
Analyzes impact of policy changes on downstream workflows
"""
from typing import List, Dict
from src.schemas import (
    ChangeSummary, Section, ImpactAnalysis, ImpactItem,
    Evidence, LLMImpactResponse
)
from src.llm import LLMInterface


IMPACT_ANALYSIS_SYSTEM_PROMPT = """You are an insurance policy analyst. Your job is to analyze policy changes and determine their impact on three key areas:
1. Claims processing
2. Underwriting risk assessment
3. Customer support

You must provide evidence-based analysis. Every impact item must cite specific text from the policy documents.
Output ONLY valid JSON with no additional text."""


def build_impact_prompt(
    changes: List[ChangeSummary],
    sections_a_dict: Dict[str, Section],
    sections_b_dict: Dict[str, Section]
) -> str:
    """
    Build prompt for LLM impact analysis

    Args:
        changes: List of detected changes
        sections_a_dict: Dictionary of sections from policy A
        sections_b_dict: Dictionary of sections from policy B

    Returns:
        Formatted prompt string
    """
    prompt_parts = [
        "Analyze the following policy changes and determine their impact on Claims, Underwriting, and Customer Support.",
        "\n## Policy Changes:\n"
    ]

    # Include only high-severity changes or modifications
    significant_changes = [
        c for c in changes
        if c.severity_score >= 3 and c.change_type.value != "unchanged"
    ]

    for i, change in enumerate(significant_changes[:10], 1):  # Limit to top 10
        prompt_parts.append(f"\n### Change {i}:")
        prompt_parts.append(f"Type: {change.change_type.value}")
        prompt_parts.append(f"Severity: {change.severity_score}/5")

        if change.section_a_id and change.section_a_id in sections_a_dict:
            section_a = sections_a_dict[change.section_a_id]
            prompt_parts.append(f"\nVersion A - {section_a.title}:")
            prompt_parts.append(section_a.text[:500])

        if change.section_b_id and change.section_b_id in sections_b_dict:
            section_b = sections_b_dict[change.section_b_id]
            prompt_parts.append(f"\nVersion B - {section_b.title}:")
            prompt_parts.append(section_b.text[:500])

    prompt_parts.append("\n\n## Instructions:")
    prompt_parts.append("""
For each impact area (Claims, Underwriting, Customer Support), provide 1-3 specific impacts.

For each impact, include:
- description: What needs to change in this workflow
- evidence: Exact quote from the policy (max 25 words)

Output format (JSON):
{
  "claims": [
    {"description": "...", "evidence": "quote from policy"},
    ...
  ],
  "underwriting": [
    {"description": "...", "evidence": "quote from policy"},
    ...
  ],
  "customer_support": [
    {"description": "...", "evidence": "quote from policy"},
    ...
  ]
}

Output ONLY the JSON object, no other text.
""")

    return "\n".join(prompt_parts)


def parse_impact_response(
    response_json: Dict,
    sections_a_dict: Dict[str, Section],
    sections_b_dict: Dict[str, Section]
) -> ImpactAnalysis:
    """
    Parse LLM response into ImpactAnalysis object

    Args:
        response_json: Parsed JSON from LLM
        sections_a_dict: Sections from policy A
        sections_b_dict: Sections from policy B

    Returns:
        ImpactAnalysis object
    """
    impact_analysis = ImpactAnalysis()

    # Process Claims impacts
    for item in response_json.get("claims", []):
        evidence_list = []
        evidence_text = item.get("evidence", "")

        # Try to find which section this evidence comes from
        section_id = find_section_for_evidence(
            evidence_text,
            {**sections_a_dict, **sections_b_dict}
        )

        if evidence_text:
            evidence = Evidence(
                source="B",  # Assume version B for simplicity
                section_id=section_id or "unknown",
                quote=evidence_text[:100]  # Limit to 100 chars
            )
            evidence_list.append(evidence)

        impact_item = ImpactItem(
            area="Claims",
            description=item.get("description", ""),
            evidence=evidence_list
        )
        impact_analysis.claims.append(impact_item)

    # Process Underwriting impacts
    for item in response_json.get("underwriting", []):
        evidence_list = []
        evidence_text = item.get("evidence", "")

        section_id = find_section_for_evidence(
            evidence_text,
            {**sections_a_dict, **sections_b_dict}
        )

        if evidence_text:
            evidence = Evidence(
                source="B",
                section_id=section_id or "unknown",
                quote=evidence_text[:100]
            )
            evidence_list.append(evidence)

        impact_item = ImpactItem(
            area="Underwriting",
            description=item.get("description", ""),
            evidence=evidence_list
        )
        impact_analysis.underwriting.append(impact_item)

    # Process Customer Support impacts
    for item in response_json.get("customer_support", []):
        evidence_list = []
        evidence_text = item.get("evidence", "")

        section_id = find_section_for_evidence(
            evidence_text,
            {**sections_a_dict, **sections_b_dict}
        )

        if evidence_text:
            evidence = Evidence(
                source="B",
                section_id=section_id or "unknown",
                quote=evidence_text[:100]
            )
            evidence_list.append(evidence)

        impact_item = ImpactItem(
            area="Customer Support",
            description=item.get("description", ""),
            evidence=evidence_list
        )
        impact_analysis.customer_support.append(impact_item)

    return impact_analysis


def find_section_for_evidence(evidence: str, sections_dict: Dict[str, Section]) -> str:
    """
    Find which section contains the given evidence text

    Args:
        evidence: Evidence text to search for
        sections_dict: Dictionary of all sections

    Returns:
        Section ID or None
    """
    evidence_lower = evidence.lower()

    for section_id, section in sections_dict.items():
        if evidence_lower in section.text.lower():
            return section_id

    return None


def analyze_impact(
    changes: List[ChangeSummary],
    sections_a: List[Section],
    sections_b: List[Section],
    llm: LLMInterface
) -> ImpactAnalysis:
    """
    Analyze impact of changes on downstream workflows

    Args:
        changes: List of detected changes
        sections_a: Sections from policy A
        sections_b: Sections from policy B
        llm: LLM interface

    Returns:
        ImpactAnalysis object
    """
    # Create section dictionaries
    sections_a_dict = {s.section_id: s for s in sections_a}
    sections_b_dict = {s.section_id: s for s in sections_b}

    # Build prompt
    prompt = build_impact_prompt(changes, sections_a_dict, sections_b_dict)

    # Get LLM response
    try:
        response_json = llm.generate_json(
            prompt=prompt,
            system_prompt=IMPACT_ANALYSIS_SYSTEM_PROMPT,
            temperature=0.3
        )

        # Parse response
        impact_analysis = parse_impact_response(
            response_json,
            sections_a_dict,
            sections_b_dict
        )

        return impact_analysis

    except Exception as e:
        print(f"Error in impact analysis: {e}")
        # Return empty impact analysis on error
        return ImpactAnalysis()


def get_impact_summary(impact_analysis: ImpactAnalysis) -> str:
    """
    Generate human-readable summary of impact analysis

    Args:
        impact_analysis: ImpactAnalysis object

    Returns:
        Summary string
    """
    lines = ["Impact Analysis Summary\n"]

    lines.append(f"Claims Impacts: {len(impact_analysis.claims)}")
    for item in impact_analysis.claims:
        lines.append(f"  - {item.description}")

    lines.append(f"\nUnderwriting Impacts: {len(impact_analysis.underwriting)}")
    for item in impact_analysis.underwriting:
        lines.append(f"  - {item.description}")

    lines.append(f"\nCustomer Support Impacts: {len(impact_analysis.customer_support)}")
    for item in impact_analysis.customer_support:
        lines.append(f"  - {item.description}")

    return "\n".join(lines)
