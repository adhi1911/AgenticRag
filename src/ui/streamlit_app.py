import streamlit as st
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.orchestrator.agentic_orchestrator import AgenticOrchestrator
from config.settings import settings, PDF_DIR, VIDEO_DIR, TRANSCRIPTS_DIR

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Suppress HuggingFace/transformers warnings about __path__ access and other verbose logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.utils.generic").setLevel(logging.ERROR)
logging.getLogger("transformers.models").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

@st.cache_resource
def get_orchestrator():
    """Cached function to initialize the orchestrator only once"""
    try:
        with st.spinner("Initializing Agentic RAG system..."):
            orchestrator = AgenticOrchestrator()
        st.success("✓ Agentic RAG system initialized successfully!")
        logger.info("Agentic RAG UI initialized")
        return orchestrator
    except Exception as e:
        st.error(f"Failed to initialize Agentic RAG system: {str(e)}")
        logger.error(f"Failed to initialize orchestrator: {e}")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #f0fff0;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff8e1;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .citation-inline {
        background-color: #e3f2fd;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        font-size: 0.9em;
    }
    .citation-footnote {
        border-left: 2px solid #1f77b4;
        padding-left: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9em;
        color: #555;
    }
    .source-card {
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


class AgenticRAGApp:
    """Streamlit application for Agentic RAG system"""

    def __init__(self):
        self.orchestrator = get_orchestrator()

    def render_header(self):
        """Render the application header"""
        st.markdown('<div class="main-header">🤖 Agentic RAG Assistant</div>', unsafe_allow_html=True)
        st.markdown("""
        **Intelligent Knowledge Base**

        Ask questions about any topic and get AI-powered answers with source citations.
        """)

    def render_sidebar(self):
        """Render the sidebar with controls and information"""
        with st.sidebar:
            st.header("🎛️ Controls")

            # Initialize session state for controls if not exists
            if 'response_format' not in st.session_state:
                st.session_state.response_format = "concise"
            if 'citation_style' not in st.session_state:
                st.session_state.citation_style = "inline"
            if 'temperature' not in st.session_state:
                st.session_state.temperature = 0.3

            # Response format selector
            st.session_state.response_format = st.selectbox(
                "Response Format",
                ["narrative", "structured", "concise", "research"],
                index=["narrative", "structured", "concise", "research"].index(st.session_state.response_format),
                help="Choose how detailed and structured the answer should be"
            )

            # Citation style selector
            st.session_state.citation_style = st.selectbox(
                "Citation Style",
                ["inline", "footnote", "numbered"],
                index=["inline", "footnote", "numbered"].index(st.session_state.citation_style),
                help="How sources should be cited in the answer"
            )

            # Temperature slider
            st.session_state.temperature = st.slider(
                "Creativity",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.1,
                help="Higher values make answers more creative, lower values more focused"
            )

            st.divider()

            # File upload section
            st.header("📄 Document Upload")
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=["pdf", "txt", "md", "json"],
                accept_multiple_files=True,
                help="Upload documents to add to the knowledge base"
            )

            if uploaded_files and st.button("🚀 Ingest Documents", type="primary"):
                self.handle_file_upload(uploaded_files)

            st.divider()

            # System info
            st.header("ℹ️ System Info")
            st.markdown("""
            **Features:**
            - Hybrid retrieval (dense + sparse)
            - Cross-encoder reranking
            - Query transformation
            - Flexible citations
            - Multi-document support

            **Supported Formats:**
            - PDF documents
            - Text files (.txt, .md)
            - JSON data
            - Log files
            """)

            # Help section
            with st.expander("❓ Help & Examples"):
                st.markdown("""
                **Example Questions:**
                - What are the key concepts in [your domain]?
                - How does [topic] work?
                - Best practices for [subject]
                - Technical details about [topic]

                **Commands:**
                - Type questions normally
                - Upload files to expand knowledge base
                - Use controls to customize responses
                """)

    def handle_file_upload(self, uploaded_files: List) -> None:
        """Handle file upload and ingestion"""
        if not uploaded_files:
            return

        with st.spinner(f"Ingesting {len(uploaded_files)} document(s)..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            processed = 0
            errors = []
            skipped = []

            for i, file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing: {file.name}")

                    # Check if file already exists in PDFs directory (deduplication)
                    target_path = PDF_DIR / file.name
                    if target_path.exists():
                        skipped.append(f"{file.name}: Already ingested")
                        logger.info(f"Skipped duplicate: {file.name}")
                    else:
                        # Save uploaded file to persistent location (not temp)
                        file_path = PDF_DIR / file.name
                        PDF_DIR.mkdir(parents=True, exist_ok=True)
                        with open(file_path, 'wb') as f:
                            f.write(file.getvalue())

                        # Ingest the file from persistent location
                        result = self.orchestrator.route_request("ingest documents", files=[Path(file_path)])

                        if result["success"]:
                            processed += 1
                            st.success(f"✓ Ingested: {file.name}")
                        else:
                            errors.append(f"{file.name}: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    errors.append(f"{file.name}: {str(e)}")
                    logger.error(f"Error processing {file.name}: {e}")

                progress_bar.progress((i + 1) / len(uploaded_files))

            progress_bar.empty()
            status_text.empty()

            if processed > 0:
                st.success(f"✅ Successfully ingested {processed} document(s)")
            if skipped:
                st.info(f"⊘ Skipped {len(skipped)} duplicate file(s): {', '.join([s.split(':')[0] for s in skipped])}")
            if errors:
                st.warning(f"⚠️ {len(errors)} file(s) had errors: {', '.join(errors)}")

    def render_query_interface(self):
        """Render the main query interface"""
        st.markdown('<div class="sub-header">💬 Ask Questions</div>', unsafe_allow_html=True)

        # Query input
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What are the key concepts in artificial intelligence?",
            help="Ask any question and get AI-powered answers with source citations"
        )

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            ask_button = st.button("🚀 Ask Question", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        with col3:
            help_button = st.button("❓ Help", use_container_width=True)

        if clear_button:
            st.rerun()

        if help_button:
            self.show_help()

        if ask_button and query.strip():
            self.handle_query(query.strip())

    def handle_query(self, query: str):
        """Handle user query and display results"""
        with st.spinner("🔍 Searching knowledge base..."):
            start_time = time.time()

            try:
                # Call orchestrator with user preferences
                result = self.orchestrator.route_request(
                    query,
                    response_format=st.session_state.response_format,
                    temperature=st.session_state.temperature,
                    citation_style=st.session_state.citation_style
                )

                processing_time = time.time() - start_time

                if result["success"]:
                    self.display_answer(result, processing_time)
                else:
                    st.error(f"❌ Error: {result.get('response', 'Unknown error occurred')}")

            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                logger.error(f"Query error: {e}")

    def display_answer(self, result: Dict[str, Any], processing_time: float):
        """Display the answer and sources"""
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        metadata = result.get("metadata", {})

        # Show mode indicator (AGENTIC vs STANDARD RAG)
        mode = metadata.get("mode", "standard")
        if mode == "agentic":
            st.info("🤖 **Agentic Mode**: Using intelligent reasoning loop (THINK → ACT → OBSERVE → DECIDE)")
        
        # Answer section
        st.markdown("### 📝 Answer")
        st.markdown(answer)

        # AGENTIC: Show reasoning trace if agentic mode was used
        if metadata.get("mode") == "agentic":
            st.markdown("### 🤖 Agent Reasoning Trace")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Iterations", metadata.get("iterations", 0))
            with col2:
                reformations = metadata.get("reformulations", 0)
                st.metric("Reformulations", reformations)
            with col3:
                confidence_pct = metadata.get("confidence", 0) * 100
                st.metric("Confidence", f"{confidence_pct:.0f}%")
            
            # Show thought process - expanded if there were reformulations
            reasoning_steps = metadata.get("agent_reasoning", [])
            has_reformulations = metadata.get("reformulations", 0) > 0
            
            if has_reformulations:
                st.markdown("**🧠 Agent Thought Process** (Query was reformulated):")
                for step in reasoning_steps:
                    step_num = step.get("step", "?")
                    action = step.get("action", "").upper()
                    reasoning = step.get("reasoning", "")
                    result_text = step.get("result", "")
                    
                    st.write(f"**Step {step_num}: {action}**")
                    st.write(f"*{reasoning}*")
                    st.write(f"`→ {result_text}`")
                    st.divider()
            else:
                with st.expander("🧠 Show Agent Thought Process (First iteration - no reformulation)"):
                    for step in reasoning_steps:
                        step_num = step.get("step", "?")
                        action = step.get("action", "").upper()
                        reasoning = step.get("reasoning", "")
                        result_text = step.get("result", "")
                        
                        st.write(f"**Step {step_num}: {action}**")
                        st.write(f"*{reasoning}*")
                        st.write(f"`→ {result_text}`")
                        st.divider()

        # Original metadata display
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Processing Time", f"{processing_time:.2f}s")
        with col2:
            st.metric("Sources Found", len(sources))
        with col3:
            # Show actual temperature used, not just static confidence
            temp = metadata.get("temperature", st.session_state.temperature)
            st.metric("Temperature", f"{temp:.2f}")
        with col4:
            model = metadata.get("model", "unknown").replace("llama-3.1-8b-instant", "Llama 3.1")
            st.metric("Model", model[:12])

        # Sources section
        if sources:
            st.markdown("### 📚 Sources")
            for i, source in enumerate(sources, 1):
                with st.expander(f"Source {i}: {source.get('source', 'Unknown')} (Page {source.get('page', 'N/A')})"):
                    st.markdown(f"**Content Preview:** {source.get('preview', 'N/A')}")
                    st.markdown(f"**Relevance Score:** {source.get('rerank_score', 'N/A')}")
                    st.markdown(f"**Chunk ID:** {source.get('chunk_id', 'N/A')}")

        # Response format info
        with st.expander("ℹ️ Response Details"):
            st.json({
                "query": result.get("query", ""),
                "response_format": metadata.get("response_format", "unknown"),
                "citation_style": metadata.get("citation_style", "unknown"),
                "temperature": metadata.get("temperature", 0.3),
                "num_sources": metadata.get("num_sources", 0)
            })

    def show_help(self):
        """Show help information"""
        st.markdown("""
        ### ❓ Agentic RAG Assistant Help

        **What is this?**
        An intelligent AI assistant that combines document search with AI generation to provide accurate, sourced answers.
        It uses advanced RAG (Retrieval-Augmented Generation) technology for comprehensive knowledge retrieval.

        **How to use:**
        1. **Ask Questions**: Type your question in the text area
        2. **Customize Response**: Use sidebar controls for format, citations, and creativity
        3. **Upload Documents**: Add new documents to expand the knowledge base
        4. **Review Sources**: Check the sources section for answer verification

        **Response Formats:**
        - **Narrative**: Comprehensive, story-like explanations
        - **Structured**: Organized with sections and bullet points
        - **Concise**: Brief, direct answers
        - **Research**: In-depth analysis with multiple perspectives

        **Citation Styles:**
        - **Inline**: Citations within the text [1], [2]
        - **Footnote**: References at the bottom
        - **Numbered**: Sources listed at the end

        **Tips:**
        - Be specific in your questions for better results
        - Upload relevant documents to improve answers
        - Use the temperature slider for creative vs. focused responses
        - Check sources to verify information accuracy
        """)

    def run(self):
        """Run the Streamlit application"""
        self.render_header()
        self.render_sidebar()
        self.render_query_interface()


def main():
    """Main application entry point"""
    app = AgenticRAGApp()
    app.run()


if __name__ == "__main__":
    main()