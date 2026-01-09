"""
Professional News Explorer GUI
Modern, clean interface for news exploration and analysis
"""

import os
import re
import json
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from news_fetcher import NewsFetcher
from db import NewsVectorDB

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="News Explorer | AI-Powered News Analysis",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS STYLING ==========
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
    
    /* Minimalistic Professional Theme - Neutral with Emerald Accents */
    :root {
        --primary-dark: #0a0a0a;
        --secondary-dark: #171717;
        --tertiary-dark: #262626;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --text-primary: #fafafa;
        --text-secondary: #a3a3a3;
        --text-dark: #171717;
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.06);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.15);
        --shadow-xl: 0 16px 48px rgba(0, 0, 0, 0.2);
        --shadow-glass: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    }
    
    /* Global font */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
    }
    
    /* Main background - Subtle gradient */
    .main {
        background: linear-gradient(180deg, #fafafa 0%, #f5f5f5 100%);
    }
    
    /* Header styling - Glassmorphic design */
    .main-header {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        padding: 3.5rem 2rem;
        border-radius: 24px;
        margin-bottom: 2.5rem;
        box-shadow: var(--shadow-lg), inset 0 1px 0 rgba(255, 255, 255, 0.5);
        text-align: center;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 30% 50%, rgba(16, 185, 129, 0.05) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .main-header h1 {
        color: var(--primary-dark);
        font-size: 3.75rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.03em;
        position: relative;
        z-index: 1;
    }
    
    .main-header h1::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 4px;
        background: linear-gradient(90deg, var(--accent-emerald), var(--accent-amber));
        border-radius: 2px;
    }
    
    .main-header p {
        color: #525252;
        font-size: 1.15rem;
        margin-top: 1.25rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
        letter-spacing: 0.01em;
    }
    
    /* Sidebar styling - Dark glassmorphic theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--primary-dark) 0%, var(--secondary-dark) 100%);
        box-shadow: 4px 0 32px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: var(--text-primary) !important;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-secondary);
    }
    
    /* Sidebar section headers */
    [data-testid="stSidebar"] h2 {
        color: var(--text-primary) !important;
        font-size: 1.4rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] h3 {
        color: var(--accent-emerald) !important;
        font-size: 1rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.85rem;
    }
    
    /* Stats card in sidebar - Glass effect */
    .stats-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.25rem;
        border-radius: 16px;
        box-shadow: var(--shadow-sm);
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    /* Article cards - Modern glassmorphic design */
    .article-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: var(--shadow-md);
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-left: 5px solid;
        border-image: var(--gradient-premium) 1;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .article-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.03) 0%, rgba(139, 92, 246, 0.03) 100%);
        border-radius: 20px;
        opacity: 0;
        transition: opacity 0.4s ease;
        pointer-events: none;
    }
    
    .article-card:hover {
        transform: translateY(-8px);
        box-shadow: var(--shadow-lg);
    }
    
    .article-card:hover::before {
        opacity: 1;
    }
    
    .article-card h4 {
        color: var(--primary-dark);
        font-size: 1.25rem;
        margin-bottom: 0.75rem;
        line-height: 1.4;
    }
    
    .article-card p {
        color: #475569;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* Button styling - Modern gradient buttons */
    .stButton > button {
        background: var(--gradient-premium);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.02em;
        text-transform: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Chat messages - Glassmorphic design */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px) saturate(180%);
        -webkit-backdrop-filter: blur(12px) saturate(180%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm), inset 0 1px 0 rgba(255, 255, 255, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: all 0.3s ease;
    }
    
    .stChatMessage:hover {
        box-shadow: var(--shadow-md), inset 0 1px 0 rgba(255, 255, 255, 0.3);
        transform: translateX(4px);
    }
    
    /* User message - minimalist styling */
    .stChatMessage[data-testid*="user"] {
        background: rgba(245, 245, 245, 0.8);
        border-left: 3px solid var(--text-dark);
    }
    
    /* Assistant message - subtle accent */
    .stChatMessage[data-testid*="assistant"] {
        background: rgba(236, 253, 245, 0.6);
        border-left: 3px solid var(--accent-emerald);
    }
    
    /* Expander styling - Glass effect */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        font-weight: 600;
        color: var(--text-dark);
        padding: 1rem 1.5rem;
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255,Glassmorphic */
    .stSuccess {
        background: rgba(236, 253, 245, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-left: 3px solid var(--accent-emerald);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        box-shadow: var(--shadow-sm);
        color: #065f46;
        font-weight: 500;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .stInfo {
        background: rgba(240, 249, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-left: 3px solid #737373;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        box-shadow: var(--shadow-sm);
        color: #404040;
        font-weight: 500;
        border: 1px solid rgba(115, 115, 115, 0.2);
    }
    
    .stWarning {
        background: rgba(255, 251, 235, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-left: 3px solid var(--accent-amber);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        box-shadow: var(--shadow-sm);
        color: #92400e;
        font-weight: 500;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    .stError {
        background: rgba(254, 242, 242, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-left: 3px solid #dc2626;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        box-shadow: var(--shadow-sm);
        color: #991b1b;
        font-weight: 500;
        border: 1px solid rgba(220, 38, 38, 0.2);
    }
    
    /* Metrics - Minimalist gradient text */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-emerald), var(--accent-amber));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--tGlassmorphic styling */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--accent-emerald);
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
        outline: none;
        background: rgba(255, 255, 255, 0.9);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, var(--accent-emerald), var(--accent-amber));
    }
    
    /* Date input */
    .stDateInput > div > div > input {
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        padding: 0.75rem 1rem;
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    
    /* Checkbox */
    .stCheckbox > label {
        font-weight: 500;
        color: var(--text-primary);
    }
    
    /* Tabs - Glassmorphic design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #525252;
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.7);
        border-color: var(--accent-emerald);
        color: var(--accent-emerald);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-dark);
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f8fafc;
        border-color: var(--accent-emerald);
        color: var(--accent-emerald);
    }
    
    /* Minimal gradient */
    hr {
        margin: 3rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 0, 0, 0.1), transparent);
        border-radius: 1px;
    }
    
    /* Spinner - Accent color */
    .stSpinner > div {
        border-top-color: var(--accent-emerald) !important;
    }
    
    /* Feature highlights - Glass effect */
    .feature-box {
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        border: 1px solid rgba(0, 0, 0, 0.05);
        box-shadow: var(--shadow-sm);
    }
    
    /* Footer - Minimal and elegant */
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        color: #737373;
        font-size: 0.9rem;
        margin-top: 4rem;
        background: rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-top: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .footer strong {
        background: linear-gradient(90deg, var(--accent-emerald), var(--accent-amber));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    
    /* Chat input - Glass effect */
    .stChatInputContainer {
        border-top: 1px solid rgba(0, 0, 0, 0.05);
        padding-top: 1.5rem;
        margin-top: 2rem;
        background: rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    
    /* Scrollbar styling - Minimal */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.02);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-emerald), var(--accent-amber));
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #059669, #d97706);
    }
    
    /* Loading skeleton effect */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    /* Frosted glass utility class */
    .glass {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px) saturate(180%);
        -webkit-backdrop-filter: blur(12px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
</style>
""", unsafe_allow_html=True)

# ========== CONFIGURATION ==========
load_dotenv()
GOOGLE_API_KEY = "your gemini api key"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
NEWS_API_KEY = "your newsapi key"
NEWSDATA_API_KEY = "your newsdata api key"

news_fetcher = NewsFetcher(newsapi_key=NEWS_API_KEY, newdata_key=NEWSDATA_API_KEY)
vector_db = NewsVectorDB()
CHAT_HISTORY_FILE = "chat_history.json"

SYSTEM_INSTRUCTION = """You are an expert news analyst. Provide comprehensive, detailed answers based on news articles.
Always provide detailed responses. Synthesize from multiple sources. Include dates, locations, names, events.
For "full article" requests, provide extensive responses. Be confident and professional."""

# ========== UTILITY FUNCTIONS ==========
def get_chat_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        model_kwargs={"system_instruction": SYSTEM_INSTRUCTION}
    )

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        return json.load(open(CHAT_HISTORY_FILE))
    return []

def save_chat_history(history):
    json.dump(history, open(CHAT_HISTORY_FILE, 'w'), indent=2)

def _format_date(date_val):
    if isinstance(date_val, datetime) or hasattr(date_val, 'strftime'):
        return date_val.strftime("%Y-%m-%dT%H:%M:%S")
    return str(date_val) if date_val else None

def _clean_query(query):
    cleaned = re.sub(r'\b(any|news|about|the|a|an|what|when|where|who|how)\s+', '', query, flags=re.IGNORECASE).strip()
    return cleaned if cleaned and len(cleaned) >= 2 else query

def _fetch_agent(agent_func, name, *args, **kwargs):
    try:
        articles = agent_func(*args, **kwargs)
        return articles
    except:
        return []

def fetch_news_multi_agent(query="technology", country=None, category=None, 
                          from_date=None, to_date=None, days_back=2):
    """Fetch news using multiple agents"""
    from_dt, to_dt = _format_date(from_date), _format_date(to_date)
    all_articles = []
    
    all_articles.extend(_fetch_agent(
        news_fetcher.fetch_from_newsapi, "NewsAPI",
        q=query, from_days=days_back, page_size=20, from_date=from_dt, to_date=to_dt
    ))
    all_articles.extend(_fetch_agent(
        news_fetcher.fetch_top_from_newsapi, "Top Headlines",
        country=(country or "us").lower(), category=category, page_size=20
    ))
    search_query = _clean_query(query)
    all_articles.extend(_fetch_agent(
        news_fetcher.fetch_from_newdata, "NewsData.io",
        q=search_query, days=days_back, limit=20, country=country.lower() if country else None,
        from_date=from_dt, to_date=to_dt
    ))
    
    seen, unique_articles = set(), []
    for article in all_articles:
        uid = article.get("url") or article.get("id")
        if uid and uid not in seen:
            seen.add(uid)
            unique_articles.append(article)
    
    return unique_articles[:10]

def should_fetch_new_data(user_query, current_topic, chat_history):
    """Detect if query needs new data"""
    follow_up_patterns = ['do you have', 'any info', 'any information', 'tell me about', 'what about', 'any news']
    if any(pattern in user_query.lower() for pattern in follow_up_patterns):
        keywords = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b\w{4,}\b', user_query)
        return True, ' '.join(keywords[:3]) if keywords else user_query
    
    if not current_topic:
        keywords = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b\w{4,}\b', user_query)
        return True, ' '.join(keywords[:3]) if keywords else user_query
    
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', user_query)
    if entities and len(entities) > 0:
        topic_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', current_topic)
        if not any(e.lower() in ' '.join(topic_entities).lower() for e in entities):
            return True, ' '.join(entities[:2]) if entities else user_query
    
    query_words = set(re.findall(r'\b\w{4,}\b', user_query.lower()))
    topic_words = set(re.findall(r'\b\w{4,}\b', current_topic.lower()))
    overlap = len(query_words & topic_words) / max(len(query_words), 1)
    
    if overlap < 0.4:
        keywords = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b\w{4,}\b', user_query)
        return True, ' '.join(keywords[:3]) if keywords else user_query
    return False, None

def build_retriever_from_db(k=20):
    return vector_db.as_retriever(k=k)

def generate_comprehensive_article(query, retrieved_docs, llm):
    context = "\n\n---\n\n".join([
        f"Source: {doc.metadata.get('source', 'Unknown')}\nTitle: {doc.metadata.get('title', 'N/A')}\n"
        f"Published: {doc.metadata.get('published_at', 'N/A')}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    ])
    prompt = f"""Write a comprehensive article about: "{query}"
Include what/when/where/who/how. Synthesize from sources. Be confident.
News Articles:\n{context}\nWrite article:"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else str(response)

def create_qa_chain(k=20):
    retriever = build_retriever_from_db(k)
    llm = get_chat_llm()
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)

def enhance_question(question, is_detailed=False):
    base = f"Provide detailed answer with dates, locations, names, events. Synthesize from sources.\nQuestion: {question}"
    if is_detailed:
        return f"Provide comprehensive answer. Include what/when/where/who/how/consequences. Never say 'I don't know'.\n{base}"
    return base

def _get_answer_from_docs(query, docs, llm):
    context = "\n\n---\n\n".join([
        f"Title: {doc.metadata.get('title', 'N/A')}\nSource: {doc.metadata.get('source', 'Unknown')}\n"
        f"Content: {doc.page_content}" for doc in docs[:10]
    ])
    prompt = f"Based on these articles, provide comprehensive answer to: \"{query}\"\n\nArticles:\n{context}\n\nAnswer:"
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else str(response)

# ========== MAIN APPLICATION ==========

# Header
st.markdown("""
    <div class="main-header">
        <h1>📰 News Explorer</h1>
        <p>AI-Powered News Analysis & Intelligent Information Retrieval</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">Powered by Multi-Source Aggregation | Real-time Analysis | Smart Insights</p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
for key in ["chat_history", "qa_chain", "current_articles", "current_topic"]:
    if key not in st.session_state:
        st.session_state[key] = load_chat_history() if key == "chat_history" else None if key != "current_articles" else []

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("## 🔍 News Configuration")
    
    st.markdown("### Search Parameters")
    query_topic = st.text_input("📌 Topic to Explore", "technology", help="Enter your topic of interest")
    
    col1, col2 = st.columns(2)
    with col1:
        country = st.text_input("🌍 Country", "", help="e.g., us, gb, in")
    with col2:
        category = st.selectbox("📁 Category", 
                               [None, "business", "entertainment", "general", "health", "science", "sports", "technology"],
                               help="Filter by news category")
    
    st.markdown("### 📅 Time Range")
    use_date_range = st.checkbox("📆 Custom Date Range", help="Enable to set specific dates")
    from_date = to_date = None
    
    if use_date_range:
        col1, col2 = st.columns(2)
        with col1:
            from_date = st.date_input("From", value=datetime.now() - timedelta(days=30))
        with col2:
            to_date = st.date_input("To", value=datetime.now())
    else:
        days_back = st.slider("📆 Days Back", 1, 30, 2, help="How far back to search")
    
    st.markdown("---")
    
    # Fetch button
    if st.button("🚀 Fetch News", use_container_width=True):
        with st.spinner("🔄 Fetching news from multiple sources..."):
            articles = fetch_news_multi_agent(
                query=query_topic, 
                country=country if country else None, 
                category=category,
                from_date=from_date if use_date_range else None, 
                to_date=to_date if use_date_range else None,
                days_back=days_back if not use_date_range else 2
            )
            if articles:
                st.session_state.current_articles = articles
                st.session_state.current_topic = query_topic
                vector_db.add_articles(articles, replace=False)
                st.session_state.qa_chain = create_qa_chain(20)
                st.success(f"✅ Successfully loaded {len(articles)} articles!")
            else:
                st.error("❌ No articles found. Try different parameters.")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        save_chat_history([])
        st.success("✨ Chat history cleared!")
    
    # Stats
    if st.session_state.current_articles:
        st.markdown("### 📊 Statistics")
        col1, col2 = st.columns(2)
        col1.metric("Articles", len(st.session_state.current_articles))
        col2.metric("Chats", len(st.session_state.chat_history))

# ========== MAIN CONTENT ==========

# Display articles
if st.session_state.current_articles:
    st.markdown("## 📋 Latest News Articles")
    st.markdown(f"**Current Topic:** `{st.session_state.current_topic}`")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📰 Article List", "📊 Quick View"])
    
    with tab1:
        for idx, article in enumerate(st.session_state.current_articles, 1):
            with st.expander(f"**{idx}.** {article.get('title', 'Untitled')}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**🏢 Source:** {article.get('source', 'Unknown')}")
                    if article.get('published_at'):
                        st.markdown(f"**📅 Published:** {article.get('published_at')}")
                
                with col2:
                    if article.get('url'):
                        st.markdown(f"[🔗 Read Full Article]({article.get('url')})")
                
                st.markdown("---")
                content = article.get('content', article.get('description', 'No content available'))
                st.markdown(f"**Content:** {content}")
    
    with tab2:
        cols = st.columns(2)
        for idx, article in enumerate(st.session_state.current_articles, 1):
            with cols[(idx-1) % 2]:
                st.markdown(f"""
                    <div class="article-card">
                        <h4>{idx}. {article.get('title', 'Untitled')}</h4>
                        <p><strong>🏢 Source:</strong> {article.get('source', 'Unknown')}</p>
                        <p><strong>📅 Published:</strong> {article.get('published_at', 'N/A')[:10] if article.get('published_at') else 'N/A'}</p>
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")

# Chat Interface
st.markdown("## 💬 AI News Assistant")
st.markdown("Ask questions about the news articles, request summaries, or explore specific topics.")

# Display chat history
for user, bot in st.session_state.chat_history:
    with st.chat_message("user", avatar="👤"):
        st.markdown(user)
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(bot)

# Chat input
user_msg = st.chat_input("💭 Ask me anything about the news...")

if user_msg:
    # Display user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_msg)
    
    # Process query
    needs_fetch, fetch_topic = should_fetch_new_data(user_msg, st.session_state.current_topic, st.session_state.chat_history)
    detailed_keywords = ['full article', 'detailed article', 'comprehensive', 'complete article', 'full description',
                        'detailed description', 'everything about', 'all details', 'complete information',
                        'how did it happen', 'who was', 'what happened', 'when did', 'where did', 'how', 'when', 'what']
    is_detailed = any(kw in user_msg.lower() for kw in detailed_keywords)
    
    if needs_fetch and fetch_topic:
        with st.spinner(f"🔍 Fetching news about '{fetch_topic}'..."):
            search_query = user_msg if is_detailed else fetch_topic
            new_articles = fetch_news_multi_agent(query=search_query, days_back=90 if is_detailed else 30)
            if len(new_articles) < 5 and search_query != fetch_topic:
                additional = fetch_news_multi_agent(query=fetch_topic, days_back=90 if is_detailed else 30)
                seen_urls = {a.get('url') or a.get('id') for a in new_articles}
                new_articles.extend([a for a in additional if (a.get('url') or a.get('id')) not in seen_urls])
            if new_articles:
                vector_db.add_articles(new_articles, replace=False)
                st.session_state.qa_chain = create_qa_chain(25 if is_detailed else 20)
                st.info(f"📰 Retrieved {len(new_articles)} relevant articles")
    
    if st.session_state.qa_chain:
        with st.spinner("🤔 Analyzing articles and generating response..."):
            enhanced_question = enhance_question(user_msg, is_detailed)
            result = st.session_state.qa_chain.invoke({"question": enhanced_question, "chat_history": st.session_state.chat_history})
            answer = result["answer"]
            
            # Check for incomplete answers
            incomplete_indicators = ["don't know", "not available", "cannot provide", "does not provide",
                                    "does not give", "does not offer", "does not include", "no information",
                                    "i'm sorry", "i cannot", "unable to", "limited information", "does not include any"]
            answer_lower = answer.lower()
            is_incomplete = any(ind in answer_lower for ind in incomplete_indicators)
            
            if is_incomplete:
                keywords = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b\w{4,}\b', user_msg)
                fetch_query = ' '.join(keywords[:3]) if keywords else user_msg
                
                with st.spinner("🔄 Fetching more specific information..."):
                    specific_articles = fetch_news_multi_agent(query=fetch_query, days_back=90)
                    if specific_articles:
                        vector_db.add_articles(specific_articles, replace=False)
                        st.session_state.qa_chain = create_qa_chain(40)
                        retriever = build_retriever_from_db(40)
                        docs = retriever.invoke(enhanced_question)
                        if docs:
                            answer = generate_comprehensive_article(user_msg, docs, get_chat_llm())
                        else:
                            docs2 = build_retriever_from_db(30).invoke(fetch_query)
                            answer = _get_answer_from_docs(user_msg, docs2, get_chat_llm())
            
            st.session_state.chat_history.append((user_msg, answer))
            save_chat_history(st.session_state.chat_history)
            
            # Display assistant response
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(answer)
    else:
        answer = "Please fetch some news first using the sidebar, or I'll try to fetch relevant news automatically."
        with st.spinner("🔍 Fetching relevant news..."):
            new_articles = fetch_news_multi_agent(query=user_msg, days_back=30)
            if new_articles:
                vector_db.add_articles(new_articles, replace=False)
                st.session_state.qa_chain = create_qa_chain(20)
                answer = "✅ News fetched successfully! Please ask your question again."
        
        st.session_state.chat_history.append((user_msg, answer))
        save_chat_history(st.session_state.chat_history)
        
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(answer)

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">🚀 <strong>News Explorer</strong></p>
        <p style="margin: 0.25rem 0;">Powered by Streamlit • LangChain • Google Gemini AI</p>
        <p style="margin: 0.25rem 0; font-size: 0.85rem; color: #94a3b8;">Multi-source news aggregation | Real-time analysis | Intelligent retrieval</p>
        <p style="margin-top: 1rem; font-size: 0.8rem; color: #94a3b8;">© 2025 News Explorer. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
