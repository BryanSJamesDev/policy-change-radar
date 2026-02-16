"""
Test case generation module
Generates synthetic test scenarios for validating policy changes
"""
from typing import List, Dict
from src.schemas import (
    ChangeSummary, Section, TestScenario,
    Evidence, LLMTestScenarioResponse
)
from src.llm import LLMInterface


TEST_GEN_SYSTEM_PROMPT = """You are an insurance test case designer. Your job is to generate realistic test scenarios that validate policy changes.

Each scenario should:
- Describe a realistic customer/claim situation
- Specify what the expected handling should be under the new policy
- Reference specific policy sections
- Include direct quotes as evidence
- At least 2 scenarios must be edge cases

Output ONLY valid JSON with no additional text."""


def build_test_generation_prompt(
    changes: List[ChangeSummary],
    sections_a_dict: Dict[str, Section],
    sections_b_dict: Dict[str, Section],
    num_scenarios: int = 8
) -> str:
    """
    Build prompt for test case generation

    Args:
        changes: List of detected changes
        sections_a_dict: Sections from policy A
        sections_b_dict: Sections from policy B
        num_scenarios: Number of test scenarios to generate

    Returns:
        Formatted prompt string
    """
    prompt_parts = [
        f"Generate {num_scenarios} test scenarios to validate the following policy changes.",
        "At least 2 scenarios must be edge cases (is_edge_case: true).",
        "\n## Policy Changes:\n"
    ]

    # Focus on significant changes
    significant_changes = [
        c for c in changes
        if c.severity_score >= 3 and c.change_type.value in ["modified_major", "modified_minor", "added", "removed"]
    ]

    for i, change in enumerate(significant_changes[:8], 1):  # Limit to top 8
        prompt_parts.append(f"\n### Change {i}:")
        prompt_parts.append(f"Type: {change.change_type.value}")
        prompt_parts.append(f"Severity: {change.severity_score}/5")

        if change.section_a_id and change.section_a_id in sections_a_dict:
            section_a = sections_a_dict[change.section_a_id]
            prompt_parts.append(f"\nOLD - {section_a.title}:")
            prompt_parts.append(section_a.text[:400])

        if change.section_b_id and change.section_b_id in sections_b_dict:
            section_b = sections_b_dict[change.section_b_id]
            prompt_parts.append(f"\nNEW - {section_b.title}:")
            prompt_parts.append(section_b.text[:400])

        if change.what_changed:
            prompt_parts.append("\nWhat changed: " + "; ".join(change.what_changed))

    prompt_parts.append("\n\n## Instructions:")
    prompt_parts.append(f"""
Generate {num_scenarios} test scenarios. Each scenario should test a specific change.
At least 2 must be edge cases (boundary conditions, unusual situations, corner cases).

Output format (JSON):
{{
  "scenarios": [
    {{
      "scenario_id": "test_001",
      "scenario_text": "Detailed customer/claim scenario description...",
      "expected_handling": "What should happen under the new policy...",
      "section_refs": ["section_id_from_policy_b"],
      "evidence": ["exact quote from new policy supporting this handling"],
      "is_edge_case": true or false
    }},
    ...
  ]
}}

Output ONLY the JSON object, no other text.
""")

    return "\n".join(prompt_parts)


def parse_test_scenarios_response(
    response_json: Dict,
    sections_b_dict: Dict[str, Section]
) -> List[TestScenario]:
    """
    Parse LLM response into TestScenario objects

    Args:
        response_json: Parsed JSON from LLM
        sections_b_dict: Sections from policy B

    Returns:
        List of TestScenario objects
    """
    scenarios = []

    for scenario_data in response_json.get("scenarios", []):
        evidence_list = []

        # Parse evidence quotes
        for evidence_text in scenario_data.get("evidence", []):
            # Try to find section for this evidence
            section_id = None
            for sid, section in sections_b_dict.items():
                if evidence_text.lower() in section.text.lower():
                    section_id = sid
                    break

            if evidence_text:
                evidence = Evidence(
                    source="B",
                    section_id=section_id or "unknown",
                    quote=evidence_text[:100]  # Limit quote length
                )
                evidence_list.append(evidence)

        scenario = TestScenario(
            scenario_id=scenario_data.get("scenario_id", "unknown"),
            scenario_text=scenario_data.get("scenario_text", ""),
            expected_handling=scenario_data.get("expected_handling", ""),
            section_refs=scenario_data.get("section_refs", []),
            evidence=evidence_list,
            is_edge_case=scenario_data.get("is_edge_case", False)
        )
        scenarios.append(scenario)

    return scenarios


def generate_test_scenarios(
    changes: List[ChangeSummary],
    sections_a: List[Section],
    sections_b: List[Section],
    llm: LLMInterface,
    num_scenarios: int = 8
) -> List[TestScenario]:
    """
    Generate test scenarios for policy changes

    Args:
        changes: List of detected changes
        sections_a: Sections from policy A
        sections_b: Sections from policy B
        llm: LLM interface
        num_scenarios: Number of scenarios to generate

    Returns:
        List of TestScenario objects
    """
    # Create section dictionaries
    sections_a_dict = {s.section_id: s for s in sections_a}
    sections_b_dict = {s.section_id: s for s in sections_b}

    # Build prompt
    prompt = build_test_generation_prompt(
        changes,
        sections_a_dict,
        sections_b_dict,
        num_scenarios
    )

    # Get LLM response
    try:
        response_json = llm.generate_json(
            prompt=prompt,
            system_prompt=TEST_GEN_SYSTEM_PROMPT,
            temperature=0.7  # Higher temperature for more creative scenarios
        )

        # Parse response
        scenarios = parse_test_scenarios_response(
            response_json,
            sections_b_dict
        )

        return scenarios

    except Exception as e:
        print(f"Error generating test scenarios: {e}")
        # Return empty list on error
        return []


def validate_test_coverage(
    scenarios: List[TestScenario],
    changes: List[ChangeSummary]
) -> Dict[str, any]:
    """
    Validate that test scenarios provide adequate coverage of changes

    Args:
        scenarios: Generated test scenarios
        changes: List of changes

    Returns:
        Coverage metrics
    """
    # Get all section IDs from significant changes
    significant_sections = set()
    for change in changes:
        if change.severity_score >= 3:
            if change.section_b_id:
                significant_sections.add(change.section_b_id)

    # Get sections covered by test scenarios
    covered_sections = set()
    for scenario in scenarios:
        covered_sections.update(scenario.section_refs)

    # Calculate coverage
    if significant_sections:
        coverage = len(covered_sections & significant_sections) / len(significant_sections)
    else:
        coverage = 1.0

    # Count edge cases
    edge_case_count = sum(1 for s in scenarios if s.is_edge_case)

    return {
        "total_scenarios": len(scenarios),
        "edge_case_count": edge_case_count,
        "sections_needing_tests": len(significant_sections),
        "sections_covered": len(covered_sections & significant_sections),
        "coverage_ratio": coverage,
        "has_minimum_edge_cases": edge_case_count >= 2
    }


def get_test_scenario_summary(scenarios: List[TestScenario]) -> str:
    """
    Generate human-readable summary of test scenarios

    Args:
        scenarios: List of test scenarios

    Returns:
        Summary string
    """
    lines = [f"Generated {len(scenarios)} Test Scenarios\n"]

    edge_cases = [s for s in scenarios if s.is_edge_case]
    regular_cases = [s for s in scenarios if not s.is_edge_case]

    lines.append(f"Regular scenarios: {len(regular_cases)}")
    lines.append(f"Edge cases: {len(edge_cases)}\n")

    for scenario in scenarios:
        marker = "[EDGE CASE]" if scenario.is_edge_case else "[TEST]"
        lines.append(f"{marker} {scenario.scenario_id}")
        lines.append(f"  Scenario: {scenario.scenario_text[:100]}...")
        lines.append(f"  Expected: {scenario.expected_handling[:100]}...")
        lines.append("")

    return "\n".join(lines)
