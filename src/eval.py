"""
Evaluation harness module
Measures citation coverage, retrieval relevance, and latency
"""
import time
from typing import List, Tuple, Dict
from src.schemas import (
    ChangeSummary, ImpactAnalysis, TestScenario,
    EvaluationMetrics, AnalysisReport
)
from src.rag import RAGSystem


def calculate_citation_coverage(changes: List[ChangeSummary]) -> float:
    """
    Calculate percentage of change summaries that include evidence citations

    Args:
        changes: List of ChangeSummary objects

    Returns:
        Citation coverage ratio (0.0 to 1.0)
    """
    if not changes:
        return 0.0

    # Only consider non-unchanged sections
    relevant_changes = [
        c for c in changes
        if c.change_type.value != "unchanged"
    ]

    if not relevant_changes:
        return 1.0

    # Count changes with at least one evidence citation
    with_citations = sum(
        1 for c in relevant_changes
        if len(c.evidence) > 0
    )

    return with_citations / len(relevant_changes)


def calculate_impact_citation_coverage(impact_analysis: ImpactAnalysis) -> float:
    """
    Calculate percentage of impact items that include evidence

    Args:
        impact_analysis: ImpactAnalysis object

    Returns:
        Coverage ratio (0.0 to 1.0)
    """
    all_impacts = (
        impact_analysis.claims +
        impact_analysis.underwriting +
        impact_analysis.customer_support
    )

    if not all_impacts:
        return 0.0

    with_evidence = sum(
        1 for item in all_impacts
        if len(item.evidence) > 0
    )

    return with_evidence / len(all_impacts)


def calculate_test_scenario_coverage(scenarios: List[TestScenario]) -> float:
    """
    Calculate percentage of test scenarios that include evidence

    Args:
        scenarios: List of TestScenario objects

    Returns:
        Coverage ratio (0.0 to 1.0)
    """
    if not scenarios:
        return 0.0

    with_evidence = sum(
        1 for s in scenarios
        if len(s.evidence) > 0
    )

    return with_evidence / len(scenarios)


def calculate_overall_citation_coverage(
    changes: List[ChangeSummary],
    impact_analysis: ImpactAnalysis,
    scenarios: List[TestScenario]
) -> float:
    """
    Calculate overall citation coverage across all outputs

    Args:
        changes: Change summaries
        impact_analysis: Impact analysis
        scenarios: Test scenarios

    Returns:
        Average coverage ratio
    """
    coverages = [
        calculate_citation_coverage(changes),
        calculate_impact_citation_coverage(impact_analysis),
        calculate_test_scenario_coverage(scenarios)
    ]

    return sum(coverages) / len(coverages) if coverages else 0.0


def evaluate_retrieval_relevance(
    rag_system: RAGSystem,
    test_queries: List[str]
) -> Tuple[float, List[float]]:
    """
    Evaluate retrieval relevance using test queries

    Args:
        rag_system: RAG system to evaluate
        test_queries: List of test queries

    Returns:
        Tuple of (average_score, individual_scores)
    """
    if not test_queries:
        return 0.0, []

    scores = []

    for query in test_queries:
        # Retrieve top 3 results
        results = rag_system.retrieve(query, k=3)

        if results:
            # Average the similarity scores
            avg_score = sum(score for _, score in results) / len(results)
            scores.append(avg_score)
        else:
            scores.append(0.0)

    avg_relevance = sum(scores) / len(scores) if scores else 0.0

    return avg_relevance, scores


def generate_golden_test_set() -> List[Dict[str, str]]:
    """
    Generate a small golden test set for evaluation

    Returns:
        List of test questions with expected attributes
    """
    return [
        {
            "query": "What are the coverage limits?",
            "expected_topic": "coverage"
        },
        {
            "query": "How is the deductible calculated?",
            "expected_topic": "deductible"
        },
        {
            "query": "What exclusions apply to this policy?",
            "expected_topic": "exclusion"
        },
        {
            "query": "What are the claim filing requirements?",
            "expected_topic": "claim"
        },
        {
            "query": "What notification timeframes are required?",
            "expected_topic": "notification"
        }
    ]


def evaluate_golden_set(
    rag_system: RAGSystem,
    golden_set: List[Dict[str, str]]
) -> Dict[str, any]:
    """
    Evaluate retrieval against golden test set

    Args:
        rag_system: RAG system
        golden_set: List of test queries with expected topics

    Returns:
        Evaluation results
    """
    results = []

    for test_case in golden_set:
        query = test_case["query"]
        expected_topic = test_case["expected_topic"]

        # Retrieve results
        retrieved = rag_system.retrieve(query, k=3)

        # Check if retrieved chunks contain expected topic
        relevant = False
        if retrieved:
            for chunk, score in retrieved:
                if expected_topic.lower() in chunk.text.lower():
                    relevant = True
                    break

        results.append({
            "query": query,
            "expected_topic": expected_topic,
            "retrieved_chunks": len(retrieved),
            "is_relevant": relevant,
            "top_score": retrieved[0][1] if retrieved else 0.0
        })

    # Calculate accuracy
    accuracy = sum(1 for r in results if r["is_relevant"]) / len(results) if results else 0.0

    return {
        "accuracy": accuracy,
        "total_queries": len(results),
        "relevant_count": sum(1 for r in results if r["is_relevant"]),
        "details": results
    }


class PerformanceTimer:
    """Simple timer for measuring latency"""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer"""
        self.start_time = time.time()

    def stop(self):
        """Stop the timer"""
        self.end_time = time.time()

    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


def evaluate_analysis(
    report: AnalysisReport,
    rag_system: RAGSystem,
    latency_seconds: float
) -> EvaluationMetrics:
    """
    Complete evaluation of analysis report

    Args:
        report: AnalysisReport object
        rag_system: RAG system used for analysis
        latency_seconds: Total analysis latency

    Returns:
        EvaluationMetrics object
    """
    # Calculate citation coverage
    citation_coverage = calculate_overall_citation_coverage(
        report.changes,
        report.impact_analysis,
        report.test_scenarios
    )

    # Evaluate retrieval relevance using golden set
    golden_set = generate_golden_test_set()
    golden_results = evaluate_golden_set(rag_system, golden_set)
    avg_retrieval_relevance = golden_results["accuracy"]

    # Count analyzed sections
    num_sections = len(report.sections_a) + len(report.sections_b)

    metrics = EvaluationMetrics(
        citation_coverage=citation_coverage,
        avg_retrieval_relevance=avg_retrieval_relevance,
        total_latency_seconds=latency_seconds,
        num_sections_analyzed=num_sections
    )

    return metrics


def format_evaluation_report(metrics: EvaluationMetrics) -> str:
    """
    Format evaluation metrics as human-readable report

    Args:
        metrics: EvaluationMetrics object

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "EVALUATION REPORT",
        "=" * 60,
        "",
        "Citation Coverage:",
        f"  - Overall: {metrics.citation_coverage * 100:.1f}%",
        f"  - Target: ≥ 80%",
        f"  - Status: {'✓ PASS' if metrics.citation_coverage >= 0.8 else '✗ NEEDS IMPROVEMENT'}",
        "",
        "Retrieval Relevance:",
        f"  - Accuracy: {metrics.avg_retrieval_relevance * 100:.1f}%",
        f"  - Target: ≥ 70%",
        f"  - Status: {'✓ PASS' if metrics.avg_retrieval_relevance >= 0.7 else '✗ NEEDS IMPROVEMENT'}",
        "",
        "Performance:",
        f"  - Total latency: {metrics.total_latency_seconds:.2f} seconds",
        f"  - Sections analyzed: {metrics.num_sections_analyzed}",
        f"  - Avg time per section: {metrics.total_latency_seconds / max(1, metrics.num_sections_analyzed):.2f}s",
        "",
        "=" * 60
    ]

    return "\n".join(lines)
