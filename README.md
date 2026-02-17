# Policy Change Radar

**Compare insurance policy versions**

An evidence-based policy change analysis system that detects, categorizes, and summarizes changes between insurance policy versions, then generates impact assessments and test scenarios.

## Features

- **Section Segmentation**: Automatically detects headings and segments policies into structured sections
- **Intelligent Alignment**: Uses embedding similarity and bipartite matching to align sections across versions
- **Change Detection**: Classifies changes as unchanged/minor/major/added/removed with severity scoring
- **Evidence-Based Summaries**: Every change summary cites exact snippets from both policy versions
- **Impact Analysis**: Analyzes effects on Claims, Underwriting, and Customer Support workflows
- **Test Case Generation**: Creates synthetic test scenarios including edge cases
- **Evaluation Harness**: Measures citation coverage, retrieval relevance, and latency

## Architecture

This is NOT a generic "chat with PDF" app. It's a specialized pipeline:

1. **Extract** → PDF/text extraction (PyMuPDF)
2. **Segment** → Heading detection using heuristics (ALL CAPS, numbered sections)
3. **Embed** → Local embeddings via sentence-transformers (all-MiniLM-L6-v2)
4. **Align** → Greedy bipartite matching with similarity threshold
5. **Diff** → Change classification with keyword-based severity scoring
6. **Analyze** → LLM-powered impact analysis with evidence extraction
7. **Generate** → Test scenario generation with edge case requirements
8. **Evaluate** → Citation coverage + retrieval metrics + latency tracking

## Tech Stack

- **Python 3.11+**
- **Streamlit** - Web UI
- **sentence-transformers** - Local embeddings
- **FAISS** - Vector similarity search
- **PyMuPDF** - PDF text extraction
- **Pydantic** - Schema validation
- **pytest** - Unit testing

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup

```bash
# Clone the repository
cd policy-change-radar

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

The app will download the sentence-transformers model on first run (~90MB). This happens automatically.

## Usage

### Quick Start with Sample Policies

```bash
# Run the Streamlit app
streamlit run app.py
```

1. Click "Load Sample Policies" in the sidebar
2. Select LLM provider: "mock" (no API key required)
3. Click "Run Analysis"
4. Explore the tabs: Executive Summary, Changes, Impact, Test Cases, Evaluation

### Using Your Own Policies

**Option 1: Upload Files**
- Upload PDF or TXT files for both Policy Version A and Policy Version B

**Option 2: Paste Text**
- Copy/paste policy text directly into the text areas

**Option 3: Use API (Advanced)**

```python
from src.extract import extract_text, clean_text
from src.segment import segment_document
from src.rag import RAGSystem
from src.align import align_sections
from src.diff import detect_changes
from src.llm import get_llm
from src.impact import analyze_impact
from src.tests_gen import generate_test_scenarios

# Extract and clean
text_a = clean_text(extract_text("policy_v1.pdf"))
text_b = clean_text(extract_text("policy_v2.pdf"))

# Segment
sections_a = segment_document(text_a, "policy_a")
sections_b = segment_document(text_b, "policy_b")

# Initialize RAG
rag = RAGSystem()
rag.index_sections(sections_a + sections_b)

# Align and detect changes
alignments = align_sections(sections_a, sections_b, rag)
changes = detect_changes(sections_a, sections_b, alignments)

# Analyze with LLM
llm = get_llm(provider="mock")  # or "openai" or "anthropic"
impact = analyze_impact(changes, sections_a, sections_b, llm)
test_scenarios = generate_test_scenarios(changes, sections_a, sections_b, llm)
```

## LLM Providers

### Mock Provider (Default)

No API key required. Returns deterministic outputs for testing.

```python
llm = get_llm(provider="mock")
```

### OpenAI Provider

Requires OpenAI API key.

```bash
# Set environment variable
export OPENAI_API_KEY="your-key-here"
```

```python
llm = get_llm(provider="openai", api_key="your-key")
```

### Anthropic Provider

Requires Anthropic API key.

```bash
# Set environment variable
export ANTHROPIC_API_KEY="your-key-here"
```

```python
llm = get_llm(provider="anthropic", api_key="your-key")
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_segment.py -v
```

## Project Structure

```
policy-change-radar/
├── app.py                      # Streamlit UI entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── pytest.ini                  # Pytest configuration
├── src/
│   ├── __init__.py
│   ├── schemas.py              # Pydantic data models
│   ├── extract.py              # PDF/text extraction
│   ├── segment.py              # Section segmentation with heuristics
│   ├── rag.py                  # Embeddings + FAISS vector store
│   ├── align.py                # Bipartite matching alignment
│   ├── diff.py                 # Change detection + classification
│   ├── llm.py                  # LLM interface + MockLLM
│   ├── impact.py               # Impact analysis with evidence
│   ├── tests_gen.py            # Test scenario generation
│   └── eval.py                 # Evaluation harness
├── data/
│   └── sample_policies/
│       ├── policy_v1.txt       # Sample policy version A
│       └── policy_v2.txt       # Sample policy version B
└── tests/
    ├── __init__.py
    ├── test_segment.py         # Segmentation tests
    └── test_align.py           # Alignment tests
```

## Key Differentiators

1. **Evidence-First Design**: Every output must cite specific text snippets
2. **No Hallucination**: Uses retrieval + structured prompts + validation
3. **Measurable Quality**: Built-in evaluation harness with metrics
4. **Production-Ready**: Runs without API keys via MockLLM
5. **Not a Chatbot**: Specialized pipeline for policy diff analysis

## Evaluation Metrics

The system measures itself on three dimensions:

### 1. Citation Coverage
Percentage of analysis outputs that include evidence citations.
- **Target**: ≥80%
- **Measurement**: Count outputs with at least one evidence quote

### 2. Retrieval Relevance
Accuracy of semantic search for policy-related queries.
- **Target**: ≥70%
- **Measurement**: Golden test set with 5 standard queries

### 3. Latency
Total processing time for complete analysis.
- **Measurement**: End-to-end pipeline execution time

## Sample Output

```
Executive Summary:
The updated policy introduces 3 major changes affecting claims processing,
underwriting risk models, and customer communications.

Critical Changes:
1. Coverage limit increased from $500K to $1M (Severity: 4/5)
   Evidence A: "coverage shall not exceed $500,000 per occurrence"
   Evidence B: "maximum coverage limit of $1,000,000 per occurrence"

2. Deductible structure changed from per-incident to annual aggregate (Severity: 4/5)
   Evidence A: "deductible applies to each separate claim incident"
   Evidence B: "annual aggregate deductible of $10,000"

3. New cyber incident exclusion added (Severity: 5/5)
   Evidence B: "excluding all losses arising from cyber incidents"

Impact Analysis:
- Claims: Workflow updates required for deductible calculations
- Underwriting: Risk models need recalibration for new limits
- Customer Support: FAQs require updates for new terms

Test Scenarios: 8 generated (2 edge cases)
Evaluation: 95% citation coverage, 82% retrieval relevance, 8.3s latency
```

## Customization

### Adding New LLM Providers

Extend the `LLMInterface` class in `src/llm.py`:

```python
class CustomLLM(LLMInterface):
    def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=2000):
        # Your implementation
        return response_text
```

### Adjusting Segmentation Rules

Modify heuristics in `src/segment.py`:

```python
def is_custom_heading(line: str) -> bool:
    # Add your custom heading detection logic
    return custom_condition
```

### Changing Alignment Threshold

In `src/align.py`:

```python
alignments = align_sections(
    sections_a,
    sections_b,
    rag_system,
    threshold=0.6  # Adjust threshold (0.0-1.0)
)
```

## Limitations

1. **Heading Detection**: Relies on heuristics (ALL CAPS, numbered sections) - may miss unconventional formats
2. **Language**: English only
3. **LLM Dependency**: Impact analysis and test generation require LLM (mock mode has limited intelligence)
4. **PDF Tables**: Complex table structures may not extract perfectly
5. **Context Window**: Very long policies (>50 pages) may need chunking strategies

## Troubleshooting

**Issue**: "sentence-transformers not found"
```bash
pip install sentence-transformers
```

**Issue**: "FAISS not installed"
```bash
pip install faiss-cpu
```

**Issue**: "PyMuPDF import error"
```bash
pip install PyMuPDF
```

**Issue**: Streamlit page won't load
```bash
# Check Python version
python --version  # Should be 3.11+

# Reinstall streamlit
pip install --upgrade streamlit
```

**Issue**: LLM provider error
- For OpenAI: Verify `OPENAI_API_KEY` is set
- For Anthropic: Verify `ANTHROPIC_API_KEY` is set
- Or use `provider="mock"` to run without API keys

## Contributing

This is a demonstration project. For production use, consider:
- Adding support for more document formats (DOCX, HTML)
- Implementing async processing for large batches
- Adding database persistence for analysis history
- Expanding evaluation golden sets
- Supporting multilingual policies

## License

MIT License - Free for commercial and personal use.

## Contact

For questions or issues, please open a GitHub issue.

## Citation

If you use this system in your research or product, please cite:

```
Policy Change Radar: Evidence-Based Insurance Policy Analysis
GitHub: https://github.com/BryanSJamesDev/policy-change-radar
Year: 2024
```

