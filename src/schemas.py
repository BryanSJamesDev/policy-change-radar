"""
Pydantic schemas for Policy Change Radar
All data structures for sections, changes, impacts, and test cases
"""
from typing import List, Optional, Dict, Literal, Any
from pydantic import BaseModel, Field
from enum import Enum


class ChangeType(str, Enum):
    """Classification of section changes"""
    UNCHANGED = "unchanged"
    MODIFIED_MINOR = "modified_minor"
    MODIFIED_MAJOR = "modified_major"
    REMOVED = "removed"
    ADDED = "added"


class Section(BaseModel):
    """A single section from a policy document"""
    section_id: str
    title: str
    text: str
    start_char: int
    end_char: int

    class Config:
        frozen = False


class Evidence(BaseModel):
    """Evidence snippet from policy text"""
    source: Literal["A", "B"]  # Which policy version
    section_id: str
    quote: str = Field(..., description="Max 25 words")

    class Config:
        frozen = False


class SectionAlignment(BaseModel):
    """Alignment between sections from version A and B"""
    section_a_id: Optional[str] = None
    section_b_id: Optional[str] = None
    similarity_score: float
    is_matched: bool

    class Config:
        frozen = False


class ChangeSummary(BaseModel):
    """Summary of a single section change"""
    section_a_id: Optional[str] = None
    section_b_id: Optional[str] = None
    change_type: ChangeType
    severity_score: int = Field(..., ge=1, le=5, description="1=minor, 5=critical")
    what_changed: List[str] = Field(default_factory=list, description="1-3 bullets")
    why_it_matters: List[str] = Field(default_factory=list, description="1-2 bullets")
    evidence: List[Evidence] = Field(default_factory=list, description="1-3 quotes")

    class Config:
        frozen = False


class ImpactItem(BaseModel):
    """Single impact on a downstream workflow"""
    area: Literal["Claims", "Underwriting", "Customer Support"]
    description: str
    evidence: List[Evidence] = Field(default_factory=list)

    class Config:
        frozen = False


class ImpactAnalysis(BaseModel):
    """Structured impact analysis"""
    claims: List[ImpactItem] = Field(default_factory=list)
    underwriting: List[ImpactItem] = Field(default_factory=list)
    customer_support: List[ImpactItem] = Field(default_factory=list)

    class Config:
        frozen = False


class TestScenario(BaseModel):
    """Synthetic test case for validating policy changes"""
    scenario_id: str
    scenario_text: str = Field(..., description="Customer/claim narrative")
    expected_handling: str = Field(..., description="What should happen")
    section_refs: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    is_edge_case: bool = False

    class Config:
        frozen = False


class EvaluationMetrics(BaseModel):
    """Metrics from evaluation harness"""
    citation_coverage: float = Field(..., ge=0.0, le=1.0, description="% of bullets with evidence")
    avg_retrieval_relevance: float = Field(..., ge=0.0, le=1.0)
    total_latency_seconds: float
    num_sections_analyzed: int

    class Config:
        frozen = False


class AnalysisReport(BaseModel):
    """Complete analysis output"""
    policy_a_name: str
    policy_b_name: str
    executive_summary: str
    sections_a: List[Section]
    sections_b: List[Section]
    alignments: List[SectionAlignment]
    changes: List[ChangeSummary]
    impact_analysis: ImpactAnalysis
    test_scenarios: List[TestScenario]
    evaluation_metrics: Optional[EvaluationMetrics] = None

    class Config:
        frozen = False


# LLM Response Schemas
class LLMChangeSummaryResponse(BaseModel):
    """Expected JSON response from LLM for change summarization"""
    what_changed: List[str] = Field(..., max_items=3)
    why_it_matters: List[str] = Field(..., max_items=2)
    evidence_quotes_a: List[str] = Field(default_factory=list, max_items=3)
    evidence_quotes_b: List[str] = Field(default_factory=list, max_items=3)
    severity: int = Field(..., ge=1, le=5)


class LLMImpactResponse(BaseModel):
    """Expected JSON response from LLM for impact analysis"""
    claims: List[Dict[str, str]] = Field(default_factory=list)
    underwriting: List[Dict[str, str]] = Field(default_factory=list)
    customer_support: List[Dict[str, str]] = Field(default_factory=list)


class LLMTestScenarioResponse(BaseModel):
    """Expected JSON response from LLM for test generation"""
    scenarios: List[Dict[str, Any]] = Field(default_factory=list)
