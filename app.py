import streamlit as st
import os
from dotenv import load_dotenv
from Backend.embeddings_manager import EmbeddingsManager
from Backend.gemini_api import GeminiAPI
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Set page title and favicon
st.set_page_config(
    page_title="Grain AI",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .chat-message.user {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .chat-message.bot {
        background-color: #f0f4c3;
        border-left: 5px solid #8bc34a;
    }
    .chat-message .avatar {
        width: 40px;
    }
    .chat-message .avatar img {
        max-width: 40px;
        max-height: 40px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: calc(100% - 60px);
        padding-left: 1rem;
    }
    .stTextInput>div>div>input {
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "use_session_memory" not in st.session_state:
    st.session_state.use_session_memory = True
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
if "embeddings_loaded" not in st.session_state:
    st.session_state.embeddings_loaded = False
if "embeddings_manager" not in st.session_state:
    st.session_state.embeddings_manager = None
if "gemini_api" not in st.session_state:
    st.session_state.gemini_api = None
if "data_source" not in st.session_state:
    st.session_state.data_source = "all"

# Title
st.title("🎓 GRAIN AI")
st.markdown("#### Built for students. Not Sci-fi")
st.markdown("#### Course Stuff. Campus Stuff. Wierdly specific forms. We've got you covered.")

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    # API key is now loaded from .env file
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        st.error("No API key found in environment variables. Please check your .env file.")
    else:
        st.success("API key loaded")
    
    # Session memory toggle
    use_memory = st.toggle(
        "Enable Session Memory", 
        value=st.session_state.use_session_memory,
        help="When enabled, the chatbot will remember previous queries in this session"
    )
    
    # Display notification when session memory setting is changed
    if use_memory != st.session_state.use_session_memory:
        st.session_state.use_session_memory = use_memory
        if use_memory:
            st.success("Session memory enabled! The chatbot will remember your previous questions.")
        else:
            st.warning("Session memory disabled. The chatbot will not use previous questions for context.")
    
    # Debug mode toggle
    st.session_state.debug_mode = st.toggle(
        "Debug Mode",
        value=st.session_state.debug_mode,
        help="Show additional diagnostic information about chunk retrieval"
    )
    
    # Update environment variable when debug mode changes
    os.environ["DEBUG_MODE"] = str(st.session_state.debug_mode).lower()
    
    # Load embeddings
    if st.button("Load Knowledge Base"):
        with st.spinner("Loading university knowledge base..."):
            try:
                # Initialize Embeddings Manager
                embeddings_manager = EmbeddingsManager()
                
                # Try to load existing embeddings
                if embeddings_manager.load_embeddings():
                    st.session_state.embeddings_manager = embeddings_manager
                    st.session_state.embeddings_loaded = True
                    st.success("Knowledge base loaded successfully!")
                else:
                    st.error("No existing knowledge base found. Please run process_pdfs.py first.")
            except Exception as e:
                st.error(f"Error loading knowledge base: {str(e)}")
    
    # Initialize Gemini API
    if st.session_state.embeddings_loaded:
        try:
            st.session_state.gemini_api = GeminiAPI()
            st.success("Gemini API initialized.")
        except Exception as e:
            st.error(f"Error initializing Gemini API: {str(e)}")
    
    # Data source selector
    if st.session_state.embeddings_loaded:
        st.subheader("Data Source")
        data_source = st.radio(
            "Choose which data sources to use:",
            ["All Sources", "PDF Documents Only", "Website Content Only"],
            index=0
        )
        
        if data_source == "All Sources":
            st.session_state.data_source = "all"
        elif data_source == "PDF Documents Only":
            st.session_state.data_source = "pdf"
        else:
            st.session_state.data_source = "web"
            
        # Advanced settings section (shown in debug mode or with a toggle)
        if st.session_state.debug_mode:
            st.subheader("Advanced Settings")
            
            # Relevance threshold slider (only appears in debug mode)
            if "relevance_threshold" not in st.session_state:
                st.session_state.relevance_threshold = 0.65
            
            new_threshold = st.slider(
                "Relevance Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.relevance_threshold,
                step=0.05,
                help="Higher values require more relevant chunks (may reduce hallucinations but could limit responses)"
            )
            
            # Update the threshold if changed
            if new_threshold != st.session_state.relevance_threshold:
                st.session_state.relevance_threshold = new_threshold
                if st.session_state.embeddings_manager:
                    st.session_state.embeddings_manager.relevance_threshold = new_threshold
                st.success(f"Relevance threshold updated to {new_threshold}")
    
    st.divider()
    st.markdown("### About")
    st.markdown("""
    This chatbot uses:
    - bge-base-en-v1.5 for embeddings
    - FAISS for similarity search
    - Google's Gemini 2.5 Flash Lite model for highly accurate responses
    - Combined knowledge from PDFs and university website
    """)
    
    # Display session history in the sidebar
    if st.session_state.query_history:
        st.divider()
        st.markdown("### Session History")
        st.markdown("Type `/history` in the chat to view your previous questions.")
        with st.expander("View previous queries", expanded=False):
            for i, query in enumerate(st.session_state.query_history):
                st.markdown(f"{i+1}. {query}")
        
        # Add option to clear history
        if st.button("Clear Query History"):
            st.session_state.query_history = []
            st.success("Query history cleared!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about admissions, fees, courses, or university life..."):
    # Check for special commands
    if prompt.strip().lower() in ["/history", "/show history", "show my history"]:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Show history as response
        with st.chat_message("assistant"):
            if not st.session_state.query_history or len(st.session_state.query_history) == 0:
                response = "You haven't asked any questions yet."
            else:
                response = "**Your conversation history:**\n\n"
                for i, q in enumerate(st.session_state.query_history):
                    response += f"{i+1}. {q}\n"
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        
    else:
        # Regular question processing
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Add to query history if session memory is enabled
        if st.session_state.use_session_memory:
            st.session_state.query_history.append(prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if embeddings and API are loaded
        if not st.session_state.embeddings_loaded:
            with st.chat_message("assistant"):
                st.error("Please load the knowledge base first using the button in the sidebar.")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "⚠️ Please load the knowledge base first using the button in the sidebar."
                })
        elif not st.session_state.gemini_api:
            with st.chat_message("assistant"):
                st.error("Gemini API not initialized. Please check your .env file for a valid API key.")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "⚠️ Gemini API not initialized. Please check your .env file for a valid API key."
                })
        else:
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get chunks based on selected data source - increase number of initial chunks to search
                    if st.session_state.data_source == "all":
                        # Search for relevant chunks
                        relevant_chunks = st.session_state.embeddings_manager.search_similar_chunks(prompt, k=20)
                    elif st.session_state.data_source == "pdf":
                        # Search in all chunks first, then filter by type
                        all_chunks = st.session_state.embeddings_manager.search_similar_chunks(prompt, k=30)
                        relevant_chunks = [chunk for chunk in all_chunks if chunk["metadata"].get("type") == "pdf"][:15]
                    else:  # web only
                        # Search in all chunks first, then filter by type
                        all_chunks = st.session_state.embeddings_manager.search_similar_chunks(prompt, k=30)
                        relevant_chunks = [chunk for chunk in all_chunks if chunk["metadata"].get("type") == "web"][:15]
                    
                    # Log information about the chunks if in debug mode
                    if st.session_state.debug_mode:
                        with st.expander("Debug Information", expanded=True):
                            st.write(f"Found {len(relevant_chunks)} relevant chunks")
                            if relevant_chunks:
                                st.write(f"Top relevance score: {relevant_chunks[0]['metadata'].get('relevance_score', 0):.2f}")
                                # Show the top 3 chunks and their relevance scores
                                for i, chunk in enumerate(relevant_chunks[:3]):
                                    st.markdown(f"**Chunk {i+1}** (Score: {chunk['metadata'].get('relevance_score', 0):.2f})")
                                    st.text(chunk['text'][:200] + "...")
                    
                    # Generate response
                    response = st.session_state.gemini_api.generate_response(
                        prompt, 
                        relevant_chunks,
                        st.session_state.query_history[:-1] if st.session_state.use_session_memory else None  # Exclude current query if session memory enabled
                    )
                    
                    # Add a special case for meta-questions about previous queries (debug info)
                    if st.session_state.debug_mode and st.session_state.use_session_memory and any(phrase in prompt.lower() for phrase in ["what did i ask", "previous question", "what have i asked"]):
                        with st.expander("Session Memory Debug", expanded=True):
                            st.write("Query History:")
                            for i, q in enumerate(st.session_state.query_history[:-1]):
                                st.write(f"{i+1}. {q}")
                    
                    # Check if response indicates no information and provide feedback
                    if "don't have enough information" in response.lower():
                        st.warning("I couldn't find specific information about that in my knowledge base. Please try rephrasing your question or ask about a different university topic.")
                        
                    # Display response
                    st.markdown(response)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

# Run instructions
if not st.session_state.messages:
    st.info("""
    ### How to use this chatbot:
    
    1. Click "Load Knowledge Base" in the sidebar
    2. Select which data sources to use (PDF documents, website content, or both)
    3. Ask questions about your university in the chat input below
    
    **Special Commands:**
    - Type `/history` to view your conversation history
    
    **Tips:**
    - Make sure "Enable Session Memory" is toggled on in the sidebar if you want the chatbot to remember your previous questions
    - You can ask follow-up questions and the bot will understand the context
    
    """) 

# capture the prompts and log in logs\prompt_logs.txt file:
if prompt:
    log_entry = f"{datetime.now().isoformat()} - {prompt}\n"
    with open("logs/prompt_logs.txt", "a", encoding="utf-8") as log_file:
        log_file.write(log_entry)