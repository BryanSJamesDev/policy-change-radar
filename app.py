"""
Policy Change Radar - Streamlit Application
Main entry point for the web UI
"""
import streamlit as st
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.extract import extract_text, clean_text
from src.segment import segment_document
from src.rag import RAGSystem
from src.align import align_sections, get_alignment_stats
from src.diff import detect_changes, get_change_statistics
from src.llm import get_llm
from src.impact import analyze_impact, get_impact_summary
from src.tests_gen import generate_test_scenarios, validate_test_coverage, get_test_scenario_summary
from src.eval import evaluate_analysis, format_evaluation_report, PerformanceTimer
from src.schemas import AnalysisReport, ChangeType


# Page config
st.set_page_config(
    page_title="Policy Change Radar - Insurance Policy Analysis",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main title styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }

    /* Feature cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    /* Info boxes */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .warning-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Improve spacing */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def load_sample_policy(version: str) -> str:
    """Load sample policy from data directory"""
    sample_path = Path(__file__).parent / f"data/sample_policies/policy_{version}.txt"
    if sample_path.exists():
        with open(sample_path, 'r') as f:
            return f.read()
    return ""


def run_analysis(text_a: str, text_b: str, policy_a_name: str, policy_b_name: str, llm_provider: str):
    """
    Run complete policy change analysis

    Args:
        text_a: Policy version A text
        text_b: Policy version B text
        policy_a_name: Name of policy A
        policy_b_name: Name of policy B
        llm_provider: LLM provider to use

    Returns:
        AnalysisReport object
    """
    timer = PerformanceTimer()
    timer.start()

    # Initialize components
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Clean text
    status_text.text("Cleaning text...")
    text_a = clean_text(text_a)
    text_b = clean_text(text_b)
    progress_bar.progress(10)

    # Step 2: Segment documents
    status_text.text("Segmenting documents into sections...")
    sections_a = segment_document(text_a, policy_a_name)
    sections_b = segment_document(text_b, policy_b_name)
    progress_bar.progress(20)

    # Step 3: Initialize RAG system
    status_text.text("Initializing RAG system and embeddings...")
    rag_system = RAGSystem()
    rag_system.index_sections(sections_a + sections_b)
    progress_bar.progress(35)

    # Step 4: Align sections
    status_text.text("Aligning sections between versions...")
    alignments = align_sections(sections_a, sections_b, rag_system)
    progress_bar.progress(50)

    # Step 5: Detect changes
    status_text.text("Detecting and classifying changes...")
    changes = detect_changes(sections_a, sections_b, alignments)
    progress_bar.progress(60)

    # Step 6: Initialize LLM
    status_text.text("Initializing LLM...")
    llm = get_llm(provider=llm_provider)
    progress_bar.progress(65)

    # Step 7: Generate impact analysis
    status_text.text("Analyzing impact on downstream workflows...")
    impact_analysis = analyze_impact(changes, sections_a, sections_b, llm)
    progress_bar.progress(80)

    # Step 8: Generate test scenarios
    status_text.text("Generating test scenarios...")
    test_scenarios = generate_test_scenarios(changes, sections_a, sections_b, llm, num_scenarios=8)
    progress_bar.progress(90)

    # Step 9: Generate executive summary
    status_text.text("Creating executive summary...")
    exec_summary = generate_executive_summary(changes, impact_analysis, llm)
    progress_bar.progress(95)

    # Step 10: Evaluate
    timer.stop()
    status_text.text("Running evaluation...")
    report = AnalysisReport(
        policy_a_name=policy_a_name,
        policy_b_name=policy_b_name,
        executive_summary=exec_summary,
        sections_a=sections_a,
        sections_b=sections_b,
        alignments=alignments,
        changes=changes,
        impact_analysis=impact_analysis,
        test_scenarios=test_scenarios
    )

    metrics = evaluate_analysis(report, rag_system, timer.elapsed())
    report.evaluation_metrics = metrics

    progress_bar.progress(100)
    status_text.text("Analysis complete!")

    return report


def generate_executive_summary(changes, impact_analysis, llm):
    """Generate executive summary"""
    stats = get_change_statistics(changes)

    prompt = f"""Generate a concise executive summary for a policy change analysis.

Changes detected:
- Total changes: {stats['total_changes']}
- Modified (major): {stats['modified_major']}
- Modified (minor): {stats['modified_minor']}
- Added sections: {stats['added']}
- Removed sections: {stats['removed']}
- High severity changes: {stats['high_severity_count']}

Impact areas:
- Claims: {len(impact_analysis.claims)} impacts
- Underwriting: {len(impact_analysis.underwriting)} impacts
- Customer Support: {len(impact_analysis.customer_support)} impacts

Generate a 3-4 paragraph executive summary highlighting the most critical changes and their business impact.
"""

    try:
        summary = llm.generate(prompt, temperature=0.5, max_tokens=500)

        # If the model returns JSON (mock often does), format it nicely
        import json
        try:
            obj = json.loads(summary)

            bullets = "\n".join([f"- {x}" for x in obj.get("what_changed", [])])
            why = "\n".join([f"- {x}" for x in obj.get("why_it_matters", [])])

            out = "### Key Changes\n"
            out += (bullets + "\n\n") if bullets else "- (No key changes returned)\n\n"
            out += "### Why it Matters\n"
            out += why if why else "- (No rationale returned)"

            return out

        except Exception:
            # Normal case: plain text from the model
            return summary

    except Exception:
        # Fallback summary if LLM fails
        return f"""Policy Change Analysis Summary

This analysis detected {stats['total_changes']} changes between policy versions, with {stats['high_severity_count']} high-severity modifications requiring immediate attention.

Key changes include {stats['modified_major']} major modifications, {stats['added']} new sections, and {stats['removed']} removed sections. These changes impact multiple business areas including claims processing, underwriting risk assessment, and customer support operations.

Impact analysis identified {len(impact_analysis.claims)} claims workflow impacts, {len(impact_analysis.underwriting)} underwriting process changes, and {len(impact_analysis.customer_support)} customer support updates required.

Immediate action is recommended to update systems, retrain staff, and communicate changes to stakeholders."""



def render_executive_summary(report: AnalysisReport):
    """Render executive summary tab"""
    st.markdown("## 📊 Executive Summary")

    # Summary in a styled box
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 2rem; border-radius: 10px; margin: 1rem 0;">
        {report.executive_summary}
    </div>
    """, unsafe_allow_html=True)

    # Stats cards
    st.markdown("### 📈 Key Metrics")

    stats = get_change_statistics(report.changes)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📄 Total Sections (A)", len(report.sections_a))
        st.metric("📄 Total Sections (B)", len(report.sections_b))

    with col2:
        st.metric("🔄 Total Changes", stats['total_changes'], delta=f"{stats['total_changes']} detected")
        st.metric("🔴 High Severity", stats['high_severity_count'], delta="Requires attention" if stats['high_severity_count'] > 0 else "None")

    with col3:
        st.metric("⚠️ Major Changes", stats['modified_major'])
        st.metric("📝 Minor Changes", stats['modified_minor'])

    with col4:
        st.metric("➕ Added Sections", stats['added'])
        st.metric("➖ Removed Sections", stats['removed'])


def render_changes_tab(report: AnalysisReport):
    """Render changes by section tab"""
    st.markdown("## 🔍 Detailed Changes")

    # Create section lookup
    sections_a_dict = {s.section_id: s for s in report.sections_a}
    sections_b_dict = {s.section_id: s for s in report.sections_b}

    # Filter controls in columns
    st.markdown("### Filters")
    col1, col2 = st.columns(2)

    with col1:
        severity_filter = st.multiselect(
            "🎯 Severity Level",
            options=[1, 2, 3, 4, 5],
            default=[3, 4, 5],
            help="Filter changes by severity (1=minor, 5=critical)"
        )

    with col2:
        change_type_filter = st.multiselect(
            "📋 Change Type",
            options=[ct.value for ct in ChangeType],
            default=["modified_major", "modified_minor", "added", "removed"],
            help="Filter by type of change detected"
        )

    # Filter changes
    filtered_changes = [
        c for c in report.changes
        if c.severity_score in severity_filter
        and c.change_type.value in change_type_filter
    ]

    st.write(f"Showing {len(filtered_changes)} of {len(report.changes)} changes")

    # Display changes
    for change in filtered_changes:
        severity_color = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "🔴"}
        severity_emoji = severity_color.get(change.severity_score, "⚪")

        with st.expander(
            f"{severity_emoji} [{change.change_type.value.upper()}] Severity {change.severity_score}/5",
            expanded=change.severity_score >= 4
        ):
            # Side-by-side comparison
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Version A (Old)")
                if change.section_a_id and change.section_a_id in sections_a_dict:
                    section = sections_a_dict[change.section_a_id]
                    st.markdown(f"**{section.title}**")
                    st.text_area("", section.text, height=200, key=f"a_{change.section_a_id}")
                else:
                    st.info("Section not present in Version A")

            with col2:
                st.subheader("Version B (New)")
                if change.section_b_id and change.section_b_id in sections_b_dict:
                    section = sections_b_dict[change.section_b_id]
                    st.markdown(f"**{section.title}**")
                    st.text_area("", section.text, height=200, key=f"b_{change.section_b_id}")
                else:
                    st.info("Section not present in Version B")

            # Evidence section
            if change.evidence:
                st.subheader("Evidence")
                for i, ev in enumerate(change.evidence):
                    st.markdown(f"**[Version {ev.source}]** _{ev.section_id}_")
                    st.info(f"'{ev.quote}'")


def render_impact_tab(report: AnalysisReport):
    """Render impact analyzer tab"""
    st.markdown("## 💼 Impact Analysis")

    st.markdown("""
    <div class="info-box">
        <strong>How policy changes affect downstream workflows:</strong><br>
        • <strong>Claims</strong> - Claim processing, triage, and handling<br>
        • <strong>Underwriting</strong> - Risk assessment and premium calculation<br>
        • <strong>Customer Support</strong> - FAQs, scripts, and customer communications
    </div>
    """, unsafe_allow_html=True)

    # Claims impacts
    st.subheader(f"Claims Processing ({len(report.impact_analysis.claims)} impacts)")
    for i, impact in enumerate(report.impact_analysis.claims, 1):
        with st.container():
            st.markdown(f"**{i}. {impact.description}**")
            if impact.evidence:
                for ev in impact.evidence:
                    st.caption(f"📝 Evidence: _{ev.quote}_")
            st.divider()

    # Underwriting impacts
    st.subheader(f"Underwriting ({len(report.impact_analysis.underwriting)} impacts)")
    for i, impact in enumerate(report.impact_analysis.underwriting, 1):
        with st.container():
            st.markdown(f"**{i}. {impact.description}**")
            if impact.evidence:
                for ev in impact.evidence:
                    st.caption(f"📝 Evidence: _{ev.quote}_")
            st.divider()

    # Customer Support impacts
    st.subheader(f"Customer Support ({len(report.impact_analysis.customer_support)} impacts)")
    for i, impact in enumerate(report.impact_analysis.customer_support, 1):
        with st.container():
            st.markdown(f"**{i}. {impact.description}**")
            if impact.evidence:
                for ev in impact.evidence:
                    st.caption(f"📝 Evidence: _{ev.quote}_")
            st.divider()


def render_tests_tab(report: AnalysisReport):
    """Render test cases tab"""
    st.markdown("## ✅ Generated Test Cases")

    # Coverage stats
    coverage = validate_test_coverage(report.test_scenarios, report.changes)

    st.markdown("### 📊 Test Coverage Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Total Scenarios", coverage['total_scenarios'])
    with col2:
        st.metric("🔺 Edge Cases", coverage['edge_case_count'], delta="Critical test scenarios")
    with col3:
        st.metric("📈 Section Coverage", f"{coverage['coverage_ratio']*100:.0f}%", delta="Target: 80%+")

    # Separate edge cases and regular cases
    edge_cases = [s for s in report.test_scenarios if s.is_edge_case]
    regular_cases = [s for s in report.test_scenarios if not s.is_edge_case]

    # Display edge cases first
    if edge_cases:
        st.subheader("Edge Cases")
        for scenario in edge_cases:
            with st.expander(f"🔺 {scenario.scenario_id}", expanded=True):
                st.markdown("**Scenario:**")
                st.write(scenario.scenario_text)

                st.markdown("**Expected Handling:**")
                st.success(scenario.expected_handling)

                if scenario.evidence:
                    st.markdown("**Evidence:**")
                    for ev in scenario.evidence:
                        st.caption(f"📝 _{ev.quote}_")

    # Display regular test cases
    if regular_cases:
        st.subheader("Regular Test Cases")
        for scenario in regular_cases:
            with st.expander(f"✓ {scenario.scenario_id}"):
                st.markdown("**Scenario:**")
                st.write(scenario.scenario_text)

                st.markdown("**Expected Handling:**")
                st.info(scenario.expected_handling)

                if scenario.evidence:
                    st.markdown("**Evidence:**")
                    for ev in scenario.evidence:
                        st.caption(f"📝 _{ev.quote}_")


def render_evaluation_tab(report: AnalysisReport):
    """Render evaluation tab"""
    st.markdown("## 📈 Quality Metrics")

    if not report.evaluation_metrics:
        st.warning("No evaluation metrics available")
        return

    metrics = report.evaluation_metrics

    # Metrics display
    st.markdown("### 🎯 Accuracy Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📝 Citation Coverage")
        st.metric(
            "Percentage of outputs with evidence citations",
            f"{metrics.citation_coverage * 100:.1f}%",
            delta="Target: ≥80%"
        )
        if metrics.citation_coverage >= 0.8:
            st.success("✓ Meets target")
        else:
            st.warning("⚠ Below target")

    with col2:
        st.markdown("#### 🎯 Retrieval Relevance")
        st.metric(
            "Average retrieval accuracy",
            f"{metrics.avg_retrieval_relevance * 100:.1f}%",
            delta="Target: ≥70%"
        )
        if metrics.avg_retrieval_relevance >= 0.7:
            st.success("✓ Meets target")
        else:
            st.warning("⚠ Below target")

    # Performance metrics
    st.subheader("Performance")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Latency", f"{metrics.total_latency_seconds:.2f}s")

    with col2:
        st.metric("Sections Analyzed", metrics.num_sections_analyzed)

    with col3:
        avg_per_section = metrics.total_latency_seconds / max(1, metrics.num_sections_analyzed)
        st.metric("Avg Time/Section", f"{avg_per_section:.2f}s")

    # Detailed report
    st.subheader("Detailed Evaluation Report")
    st.code(format_evaluation_report(metrics))


def main():
    """Main Streamlit app"""

    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1>📋 Policy Change Radar</h1>
        <p>Evidence-based insurance policy analysis powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # LLM provider selection
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["mock", "openai", "anthropic"],
            help="Select 'mock' to run without API keys"
        )

        st.divider()

        # Sample policy loader
        st.markdown("### 🚀 Quick Start")
        st.markdown("Try with **example insurance policies** to see the platform in action")
        if st.button("📄 Load Sample Policies", use_container_width=True):
            st.session_state['policy_a_text'] = load_sample_policy('v1')
            st.session_state['policy_b_text'] = load_sample_policy('v2')
            st.rerun()

        st.divider()

        # About section
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Policy Change Radar** automatically detects, analyzes, and summarizes changes between policy versions.

        ✓ Evidence-based analysis
        ✓ Impact assessments
        ✓ Automated test cases
        ✓ Measurable accuracy
        """)

        st.divider()
        st.caption("v1.0 | Built with Streamlit")
        st.caption("© 2024 Policy Change Radar")

    # Value proposition section
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3 style="margin:0; color:#667eea;">🔍 Smart Detection</h3>
            <p style="margin:0.5rem 0 0 0; color:#666;">Automatically identifies all changes between policy versions</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3 style="margin:0; color:#667eea;">📊 Impact Analysis</h3>
            <p style="margin:0.5rem 0 0 0; color:#666;">Analyzes effects on Claims, Underwriting, and Support</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3 style="margin:0; color:#667eea;">✅ Test Generation</h3>
            <p style="margin:0.5rem 0 0 0; color:#666;">Creates targeted test scenarios including edge cases</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="feature-card">
            <h3 style="margin:0; color:#667eea;">📈 Measured Quality</h3>
            <p style="margin:0.5rem 0 0 0; color:#666;">Built-in evaluation with citation coverage metrics</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Upload section with better styling
    st.markdown("## 📤 Upload Policy Documents")
    st.markdown("Upload or paste two versions of your insurance policy to begin analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📄 Policy Version A (Baseline)")
        st.markdown("Upload the original or older version of your policy")
        policy_a_file = st.file_uploader(
            "Upload PDF or TXT file",
            type=['pdf', 'txt'],
            key='policy_a',
            help="Supported formats: PDF, TXT"
        )

        policy_a_text = st.text_area(
            "Or paste policy text directly",
            height=200,
            key='policy_a_text',
            placeholder="Paste your policy text here..."
        )


    with col2:
        st.markdown("### 📄 Policy Version B (Updated)")
        st.markdown("Upload the new or revised version of your policy")
        policy_b_file = st.file_uploader(
            "Upload PDF or TXT file",
            type=['pdf', 'txt'],
            key='policy_b',
            help="Supported formats: PDF, TXT"
        )

        policy_b_text = st.text_area(
            "Or paste policy text directly",
            height=200,
            key='policy_b_text',
            placeholder="Paste your policy text here..."
        )


    # Extract text from files if uploaded
    if policy_a_file:
        file_bytes = policy_a_file.read()
        file_type = 'pdf' if policy_a_file.name.endswith('.pdf') else 'txt'
        policy_a_text = extract_text(file_bytes, file_type)

    if policy_b_file:
        file_bytes = policy_b_file.read()
        file_type = 'pdf' if policy_b_file.name.endswith('.pdf') else 'txt'
        policy_b_text = extract_text(file_bytes, file_type)

    # Info banner
    if not policy_a_text and not policy_b_text:
        st.markdown("""
        <div class="info-box">
            <strong>💡 First time here?</strong><br>
            Click <strong>"Load Sample Policies"</strong> in the sidebar to see the platform in action with example commercial liability policies (v1.0 → v2.0)
        </div>
        """, unsafe_allow_html=True)

    # Run analysis button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Run Analysis", type="primary", disabled=not (policy_a_text and policy_b_text), use_container_width=True):
            with st.spinner("🔄 Running comprehensive analysis..."):
                try:
                    report = run_analysis(
                        policy_a_text,
                        policy_b_text,
                        "Policy_A",
                        "Policy_B",
                        llm_provider
                    )
                    st.session_state['report'] = report
                    st.markdown("""
                    <div class="success-box">
                        <strong>✅ Analysis complete!</strong> View the results in the tabs below.
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"""
                    <div class="warning-box">
                        <strong>⚠️ Error during analysis:</strong><br>{str(e)}
                    </div>
                    """, unsafe_allow_html=True)
                    import traceback
                    with st.expander("View error details"):
                        st.code(traceback.format_exc())

    # Display results
    if 'report' in st.session_state:
        report = st.session_state['report']

        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        st.markdown("Navigate through the tabs below to explore different aspects of the analysis")

        # Tabs for different views with icons
        tabs = st.tabs([
            "📊 Executive Summary",
            "🔍 Detailed Changes",
            "💼 Impact Analysis",
            "✅ Test Scenarios",
            "📈 Quality Metrics"
        ])

        with tabs[0]:
            render_executive_summary(report)

        with tabs[1]:
            render_changes_tab(report)

        with tabs[2]:
            render_impact_tab(report)

        with tabs[3]:
            render_tests_tab(report)

        with tabs[4]:
            render_evaluation_tab(report)


if __name__ == "__main__":
    main()
