import streamlit as st
import requests
import json
import time
from datetime import datetime, timedelta
from typing import List, Tuple
import uuid
from streamlit_local_storage import LocalStorage

# --- CONFIGURATION ---
API_BASE_URL = "http://localhost:8000" 
API_ENDPOINT = f"{API_BASE_URL}/v1/chat/completions"

st.set_page_config(
    page_title="DocuChat AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOCAL STORAGE & STATE ---
localS = LocalStorage()

# Initialize Session State
if "user_id" not in st.session_state:
    stored_user_id = localS.getItem("rag_user_id")
    if stored_user_id and stored_user_id.strip():
        st.session_state.user_id = stored_user_id
    else:
        new_user_id = "user-" + str(uuid.uuid4())
        st.session_state.user_id = new_user_id
        localS.setItem("rag_user_id", new_user_id)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

if "user_sessions" not in st.session_state:
    st.session_state.user_sessions = []

# --- HELPER FUNCTIONS ---

def get_smart_time_format(dt):
    """Display relative time for recent updates"""
    if isinstance(dt, str): return dt 
    now = datetime.now()
    diff = now - dt
    seconds = diff.total_seconds()
    if seconds < 60: return "just now"
    elif seconds < 3600: return f"{int(seconds / 60)}m ago"
    elif seconds < 86400: return f"{int(seconds / 3600)}h ago"
    elif seconds < 518400: return f"{int(seconds / 86400)}d ago"
    else: return dt.strftime('%b %d %Y, %I:%M %p')

def create_new_session():
    """Create a new chat session"""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    # Add DB creation logic here if needed

def load_session(session_id: str):
    """Load a specific session"""
    st.session_state.session_id = session_id
    st.session_state.messages = [] # Placeholder for DB load
    # st.session_state.messages = run_async(db.load_session_messages(session_id))

def send_message(message: str, stream: bool = True) -> Tuple[str, float]:
    """Send message to the RAG API and get response"""
    start_time = time.perf_counter()

    api_messages = []
    for msg in st.session_state.messages:
        api_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    payload = {
        "model": "rag-model", 
        "messages": api_messages,
        "stream": stream,
        "session_id": st.session_state.session_id
    }

    headers = {
        "Content-Type": "application/json",
        "x-openwebui-chat-id": st.session_state.session_id,
        "X-OpenWebUI-User-Id": st.session_state.user_id,
    }

    try:
        if stream:
            response = requests.post(
                API_ENDPOINT,
                json=payload,
                headers=headers,
                stream=True,
                timeout=120
            )
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]': break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    full_response += content
                            elif 'error' in chunk:
                                full_response += f"\nError: {chunk['error']}"
                        except json.JSONDecodeError:
                            pass

            elapsed = time.perf_counter() - start_time
            return full_response if full_response else "No response received.", elapsed
        else:
            response = requests.post(API_ENDPOINT, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()
            elapsed = time.perf_counter() - start_time
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'], elapsed
            return "No response received.", elapsed

    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - start_time
        return f"Error: Request timeout. The backend took too long to respond.", elapsed
    except requests.exceptions.ConnectionError:
        elapsed = time.perf_counter() - start_time
        return f"Error: Unable to connect to RAG API. Make sure the backend is running on {API_BASE_URL}", elapsed
    except requests.exceptions.RequestException as e:
        elapsed = time.perf_counter() - start_time
        return f"Error: {str(e)}", elapsed

def process_user_message(message: str):
    """Process a user message and get response"""
    if st.session_state.processing: return
    
    st.session_state.processing = True
    
    try:
        st.session_state.messages.append({"role": "user", "content": message})
        
        # Status update using Streamlit native elements for cleaner look with complex CSS
        status_placeholder = st.empty()
        with status_placeholder.container():
            st.info("🔍 Retrieving context and generating answer...")
            
        response, elapsed_time = send_message(message, stream=True)
        status_placeholder.empty()

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "elapsed_time": elapsed_time
        })

    finally:
        st.session_state.processing = False

# --- ORIGINAL CSS STYLING ---
st.markdown("""
    <style>
    /* ===== GLOBAL STYLES (applies to all devices) ===== */
    .main {
        background-color: var(--background-color);
    }
    
    /* Prevent any horizontal overflow */
    body, html, .main, [data-testid="stAppViewContainer"] {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }
    
    /* Text input - adapts to theme */
    .stTextInput > div > div > input {
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 0.2rem rgba(38, 132, 255, 0.25) !important;
    }
    
    /* Session list styling */
    .session-item {
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 0.5rem;
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .session-item:hover {
        background-color: var(--primary-color);
        color: white;
        transform: translateX(5px);
    }
    
    .session-item.active {
        background-color: var(--primary-color);
        color: white;
        border-color: var(--primary-color);
    }
    
    /* Active chat button styling - Orange color */
    [data-testid="stSidebar"] button[disabled] {
    background-color: transparent !important;
    color: var(--text-color) !important;
    border: 2px solid #ff9800 !important;
    border-radius: 0.5rem !important;
    opacity: 1 !important;
    font-weight: 500 !important;
    box-shadow: 0 0 0 2px rgba(255, 152, 0, 0.1) !important;
    }
    
    .session-title {
        font-weight: 500;
        margin-bottom: 0.25rem;
    }
    
    .session-meta {
        font-size: 0.75rem;
        opacity: 0.7;
    }
    
    /* Hide duplicate form inputs */
    [data-testid="stForm"]:first-of-type {
        display: block !important;
    }
    
    [data-testid="stForm"]:not(:first-of-type) {
        display: none !important;
    }
    
    /* ===== COMPACT SIDEBAR CHAT LIST ===== */
    /* Reduce spacing between chat items in sidebar */
    [data-testid="stSidebar"] [data-testid="column"] {
        padding: 0 !important;
        margin-bottom: 0.25rem !important;
    }
    
    /* Reduce button spacing in sidebar */
    [data-testid="stSidebar"] button {
        margin-bottom: 0.25rem !important;
    }
    
    /* Reduce caption spacing */
    [data-testid="stSidebar"] .stCaption {
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }
    
    /* Reduce horizontal rule spacing in sidebar */
    [data-testid="stSidebar"] hr {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Compact the column layout for session items */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        gap: 0.25rem !important;
    }
    
    /* ===== DESKTOP/LAPTOP STYLES (default) ===== */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        overflow-wrap: break-word;
        word-break: break-word;
    }
    
    .chat-message.user {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    
    .chat-message.assistant {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    
    .chat-message .message-content {
        margin-top: 0.5rem;
        color: #333;
        line-height: 1.6;
    }
    
    .chat-message .message-role {
        font-weight: bold;
        font-size: 0.9rem;
        color: #666;
    }
    
    .sidebar .sidebar-content {
        background-color: #fafafa;
    }
    
    .timing {
        font-size: 0.85rem;
        color: #999;
        margin-top: 0.5rem;
        font-style: italic;
    }
    
    /* Desktop spacing for content */
    .chat-message .message-content p {
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
    }
    
    .chat-message .message-content p:first-child {
        margin-top: 0;
    }
    
    .chat-message .message-content p:last-child {
        margin-bottom: 0;
    }
    
    .chat-message .message-content p + p {
        margin-top: 0.75rem;
    }
    
    /* Desktop table styling */
    .chat-message table {
        margin-top: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #ddd;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        width: 100%;
        border-collapse: collapse;
        table-layout: auto;
    }
    
    .chat-message table td,
    .chat-message table th {
        padding: 0.5rem 0.75rem;
        border: 1px solid #ddd;
        vertical-align: top;
        word-wrap: break-word;
    }
    
    /* Desktop list styling */
    .chat-message ul, .chat-message ol {
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
        padding-left: 2rem;
    }
    
    .chat-message li {
        margin-bottom: 0.5rem;
    }
    
    /* Desktop heading styling */
    .chat-message h1, .chat-message h2, .chat-message h3,
    .chat-message h4, .chat-message h5, .chat-message h6 {
        margin-top: 1.25rem;
        margin-bottom: 0.75rem;
        line-height: 1.3;
    }
    
    .chat-message h1:first-child, .chat-message h2:first-child, 
    .chat-message h3:first-child, .chat-message h4:first-child {
        margin-top: 0;
    }
    
   /* ===== MOBILE/TABLET STYLES (768px and below) ===== */
    @media (max-width: 768px) {
        /* Container fixes */
        .main .block-container {
            padding: 0.5rem !important;
            max-width: 100% !important;
        }
        
        /* Chat message adjustments for mobile */
        .chat-message {
            padding: 1rem 0.5rem !important;
            font-size: 0.9rem !important;
            margin-bottom: 0.75rem !important;
        }
        
        .chat-message .message-content {
            white-space: normal !important;
            line-height: 1.5 !important;
            max-width: 100% !important;
            overflow-x: auto !important;
        }
        
        .chat-message .message-role {
            font-size: 0.85rem !important;
        }
        
        /* Mobile spacing - more compact */
        .chat-message .message-content p {
            margin-top: 0.4rem !important;
            margin-bottom: 0.4rem !important;
        }
        
        .chat-message .message-content p + p {
            margin-top: 0.4rem !important;
        }
        
        /* Mobile table styling - improved scrolling */
        .chat-message table {
            display: block !important;
            overflow-x: auto !important;
            white-space: normal !important;
            font-size: 0.8rem !important;
            width: 100% !important;
            -webkit-overflow-scrolling: touch !important;
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
            border: 1px solid #ccc !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        
        .chat-message table thead {
            display: table-header-group !important;
        }
        
        .chat-message table tbody {
            display: table-row-group !important;
        }
        
        .chat-message table tr {
            display: table-row !important;
        }
        
        .chat-message table td,
        .chat-message table th {
            display: table-cell !important;
            padding: 0.4rem 0.5rem !important;
            font-size: 0.8rem !important;
            border: 1px solid #ddd !important;
            vertical-align: top !important;
            min-width: 80px !important;
            max-width: none !important;
            word-wrap: normal !important;
            white-space: nowrap !important;
        }
        
        /* Header cells */
        .chat-message table th {
            background-color: #f5f5f5 !important;
            font-weight: 600 !important;
            position: sticky !important;
            top: 0 !important;
            z-index: 10 !important;
        }
        
        /* Make Devanagari text properly sized */
        .chat-message table td {
            line-height: 1.6 !important;
        }
        
        /* First column (Sanskrit) - wider, single line for each element */
        .chat-message table td:first-child,
        .chat-message table th:first-child {
            min-width: 150px !important;
            font-size: 0.85rem !important;
        }
        
        /* Keep clickable code elements in first column together */
        .chat-message table td:first-child code {
            white-space: nowrap !important;
            word-break: keep-all !important;
            display: inline-block !important;
        }
        
        /* Code elements in tables should not wrap */
        .chat-message table code {
            white-space: nowrap !important;
            word-break: keep-all !important;
            display: inline !important;
        }
        
        /* Middle column (transliteration) */
        .chat-message table td:nth-child(2),
        .chat-message table th:nth-child(2) {
            min-width: 120px !important;
        }
        
        /* Last column (Hindi/translation) */
        .chat-message table td:last-child,
        .chat-message table th:last-child {
            min-width: 120px !important;
            font-size: 0.85rem !important;
        }
        
        /* Scroll indicator - improved positioning */
        .chat-message .message-content:has(table)::after {
            content: "← Swipe left/right to see more →";
            display: block;
            text-align: center;
            font-size: 0.7rem;
            color: #666;
            margin-top: 0.25rem;
            padding: 0.25rem;
            background-color: #f9f9f9;
            border-radius: 4px;
            font-style: italic;
        }
        
        /* Mobile list styling */
        .chat-message ul, .chat-message ol {
            margin-top: 0.4rem !important;
            margin-bottom: 0.4rem !important;
            padding-left: 1.5rem !important;
        }
        
        .chat-message li {
            margin-bottom: 0.25rem !important;
        }
        
        /* Mobile heading styling */
        .chat-message h1, .chat-message h2, .chat-message h3,
        .chat-message h4, .chat-message h5, .chat-message h6 {
            margin-top: 0.6rem !important;
            margin-bottom: 0.4rem !important;
            line-height: 1.3 !important;
        }
        
        /* Sidebar adjustments for mobile */
        [data-testid="stSidebar"] {
            max-width: 85vw !important;
        }
        
        /* Input area - prevent iOS zoom and improve touch targets */
        .stTextInput input {
            font-size: 16px !important;
            min-height: 44px !important;
        }
        
        /* Send button - better for mobile */
        .stButton button {
            min-height: 44px !important;
            font-size: 16px !important;
        }
        
        /* Follow-up buttons - more touchable */
        div[data-testid="column"] button {
            font-size: 0.8rem !important;
            padding: 0.5rem !important;
            white-space: normal !important;
            height: auto !important;
            min-height: 44px !important;
        }
        
        /* Timing text smaller on mobile */
        .timing {
            font-size: 0.75rem !important;
            margin-top: 0.3rem !important;
        }
    }
        
        /* ===== TABLET LANDSCAPE (769px - 1024px) ===== */
        @media (min-width: 769px) and (max-width: 1024px) {
            .chat-message {
                padding: 1.25rem;
            }
            
            .chat-message table {
                font-size: 0.9rem;
            }
            
            .chat-message table td,
            .chat-message table th {
                padding: 0.4rem 0.6rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("favicon.png", width=80) # Ensure you have this image or comment it out
    st.markdown("### DocuChat AI")

    st.subheader("Chat Sessions")
    
    if st.button("New Chat", use_container_width=True, type="primary"):
        create_new_session()
        st.rerun()
    
    st.markdown("---")

    if st.session_state.user_sessions:
        st.markdown("**Recent Chats:**")
        for session in st.session_state.user_sessions[:10]:
             # Mock logic for display - replaced with actual keys if DB was active
            title = "Session " + session.get("session_id", "")[:5]
            if st.button(title, use_container_width=True):
                load_session(session.get("session_id"))
                st.rerun()
    else:
        st.info("No previous chats")

    st.markdown("---")
    
    # --- NEW: FILE UPLOAD SECTION ---
    st.subheader("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        if st.button("Process Files", use_container_width=True):
            with st.spinner("Ingesting documents..."):
                # 1. Prepare files for API
                files_payload = [
                    ('files', (file.name, file.getvalue(), file.type)) 
                    for file in uploaded_files
                ]
                
                # 2. Send to your Backend API
                try:
                    upload_response = requests.post(
                        f"{API_BASE_URL}/v1/ingest", 
                        files=files_payload,
                        data={"session_id": st.session_state.session_id}
                    )
                    upload_response.raise_for_status()
                    
                    st.success(f"Successfully processed {len(uploaded_files)} documents!")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")
    
    st.markdown("---")

    # Maintenance / Utility
    with st.expander("Settings"):
        if st.button("Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        st.caption(f"Session: {st.session_state.session_id[:8]}...")

# --- MAIN CONTENT ---
st.markdown("""
<div class="main-header">
    <h2>DocuChat AI</h2>
    <p>Your Advanced RAG Assistant</p>
</div>
""", unsafe_allow_html=True)

chat_container = st.container()

with chat_container:
    # Display Messages using the HTML structure that matches the complex CSS
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        # The CSS specific classes: 'chat-message', 'user'/'assistant', 'message-role', 'message-content'
        if role == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div class="message-role">User</div>
                <div class="message-content">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant">
                <div class="message-role">DocuChat AI</div>
                <div class="message-content">{content}</div>
            </div>
            """, unsafe_allow_html=True)
            
            elapsed_time = message.get("elapsed_time")
            if elapsed_time:
                st.markdown(f"<div class='timing'>Response time: {elapsed_time:.2f}s</div>", unsafe_allow_html=True)

# --- INPUT AREA ---
st.markdown("---")

with st.form(key="message_form", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message here...",
            key="user_input",
            label_visibility="collapsed",
            placeholder="Ask a question about your documents..."
        )
    
    with col2:
        send_clicked = st.form_submit_button("Send ➤", use_container_width=True, type="primary", disabled=st.session_state.processing)

    if send_clicked and user_input and user_input.strip():
        process_user_message(user_input.strip())
        st.rerun()

# --- EMPTY STATE ---
if len(st.session_state.messages) == 0:
    st.info("""
    **Welcome to DocuChat AI!**
    
    I am ready to help you analyze your documents. 
    
    * Ask questions about uploaded files
    * Get summaries and key insights
    * Retrieve specific details contextually
    """)

st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 20px;'>"
    "DocuChat AI v1.0"
    "</div>",
    unsafe_allow_html=True
)