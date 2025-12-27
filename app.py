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

# Load API keys
load_dotenv()
GOOGLE_API_KEY = "your gemini api key"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
NEWS_API_KEY = "your newsapi key"
NEWSDATA_API_KEY = "your newsdata api key"

# Initialize components
news_fetcher = NewsFetcher(newsapi_key=NEWS_API_KEY, newdata_key=NEWSDATA_API_KEY)
vector_db = NewsVectorDB()
CHAT_HISTORY_FILE = "chat_history.json"

SYSTEM_INSTRUCTION = """You are an expert news analyst. Provide comprehensive, detailed answers based on news articles.
Always provide detailed responses. Synthesize from multiple sources. Include dates, locations, names, events.
For "full article" requests, provide extensive responses. Be confident and professional."""

# Use single stable model to avoid quota issues
def get_chat_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # change the model to the model you want to use
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
        st.info(f"{name}: Found {len(articles)} articles")
        return articles
    except:
        return []

def fetch_news_multi_agent(query="technology", country=None, category=None, 
                          from_date=None, to_date=None, days_back=2):
    """Fetch news using multiple agents"""
    from_dt, to_dt = _format_date(from_date), _format_date(to_date)
    all_articles = []
    
    # Fetch from all agents
    all_articles.extend(_fetch_agent(
        news_fetcher.fetch_from_newsapi, "Agent 1 (NewsAPI)",
        q=query, from_days=days_back, page_size=20, from_date=from_dt, to_date=to_dt
    ))
    all_articles.extend(_fetch_agent(
        news_fetcher.fetch_top_from_newsapi, "Agent 2 (Top Headlines)",
        country=(country or "us").lower(), category=category, page_size=20
    ))
    search_query = _clean_query(query)
    all_articles.extend(_fetch_agent(
        news_fetcher.fetch_from_newdata, "Agent 3 (NewsData.io)",
        q=search_query, days=days_back, limit=20, country=country.lower() if country else None,
        from_date=from_dt, to_date=to_dt
    ))
    
    # Deduplicate and return top 10
    seen, unique_articles = set(), []
    for article in all_articles:
        uid = article.get("url") or article.get("id")
        if uid and uid not in seen:
            seen.add(uid)
            unique_articles.append(article)
    
    return unique_articles[:10]

def should_fetch_new_data(user_query, current_topic, chat_history):
    """Detect if query needs new data - check for specific entities and follow-up questions"""
    # Check for follow-up questions
    follow_up_patterns = ['do you have', 'any info', 'any information', 'tell me about', 'what about', 'any news']
    if any(pattern in user_query.lower() for pattern in follow_up_patterns):
        # Extract entity from previous context or current query
        keywords = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b\w{4,}\b', user_query)
        return True, ' '.join(keywords[:3]) if keywords else user_query
    
    if not current_topic:
        keywords = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b\w{4,}\b', user_query)
        return True, ' '.join(keywords[:3]) if keywords else user_query
    
    # Check for specific entities (proper nouns, sports teams, countries, etc.)
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', user_query)
    if entities and len(entities) > 0:
        # If query has specific entities not in current topic, fetch
        topic_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', current_topic)
        if not any(e.lower() in ' '.join(topic_entities).lower() for e in entities):
            return True, ' '.join(entities[:2]) if entities else user_query
    
    # Check word overlap
    query_words = set(re.findall(r'\b\w{4,}\b', user_query.lower()))
    topic_words = set(re.findall(r'\b\w{4,}\b', current_topic.lower()))
    overlap = len(query_words & topic_words) / max(len(query_words), 1)
    
    if overlap < 0.4:  # Less than 40% overlap = different topic
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

# Streamlit app
st.set_page_config(page_title="News Explorer", layout="wide")
st.title("📰 News Explorer – Your Personal News Analyst")

# Initialize session state
for key in ["chat_history", "qa_chain", "current_articles", "current_topic"]:
    if key not in st.session_state:
        st.session_state[key] = load_chat_history() if key == "chat_history" else None if key != "current_articles" else []

# Sidebar
with st.sidebar:
    st.header("News Setup")
    query_topic = st.text_input("Topic to explore", "technology")
    country = st.text_input("Country (e.g., in, us, gb) - Optional", "")
    category = st.selectbox("Category - Optional", 
                           [None, "business", "entertainment", "general", "health", "science", "sports", "technology"])
    st.subheader("📅 Date Range (for past news)")
    use_date_range = st.checkbox("Use date range")
    from_date = to_date = None
    if use_date_range:
        col1, col2 = st.columns(2)
        from_date = col1.date_input("From Date", value=datetime.now() - timedelta(days=30))
        to_date = col2.date_input("To Date", value=datetime.now())
    days_back = st.slider("Days back (if not using date range)", 1, 30, 2)
    
    if st.button("Fetch News"):
        with st.spinner("Fetching news from multiple sources..."):
            articles = fetch_news_multi_agent(
                query=query_topic, country=country if country else None, category=category,
                from_date=from_date if use_date_range else None, to_date=to_date if use_date_range else None,
                days_back=days_back
            )
            if articles:
                st.session_state.current_articles = articles
                st.session_state.current_topic = query_topic
                vector_db.add_articles(articles, replace=False)
                st.success(f"✅ Stored {len(articles)} articles")
                st.session_state.qa_chain = create_qa_chain(20)
                st.success(f"✅ Loaded {len(articles)} news bulletins for '{query_topic}'.")
            else:
                st.error("No news found or API error.")
    
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        save_chat_history([])
        st.success("Chat history cleared!")

# Display articles
if st.session_state.current_articles:
    st.header("📋 Fetched News Articles")
    for idx, article in enumerate(st.session_state.current_articles, 1):
        with st.expander(f"{idx}. {article.get('title', 'No title')}", expanded=False):
            st.write(f"**Source:** {article.get('source', 'Unknown')}")
            if article.get('published_at'):
                st.write(f"**Published:** {article.get('published_at')}")
            st.write(f"**Content:** {article.get('content', article.get('description', 'No content available'))}")
            if article.get('url'):
                st.markdown(f"[Read full article]({article.get('url')})")
    st.divider()

# Chat UI
st.header("💬 Ask Questions About the News")
user_msg = st.chat_input("Ask about the news...")
if user_msg:
    needs_fetch, fetch_topic = should_fetch_new_data(user_msg, st.session_state.current_topic, st.session_state.chat_history)
    detailed_keywords = ['full article', 'detailed article', 'comprehensive', 'complete article', 'full description',
                        'detailed description', 'everything about', 'all details', 'complete information',
                        'how did it happen', 'who was', 'what happened', 'when did', 'where did', 'how', 'when', 'what']
    is_detailed = any(kw in user_msg.lower() for kw in detailed_keywords)
    
    if needs_fetch and fetch_topic:
        with st.spinner(f"Fetching news about '{fetch_topic}'..."):
            search_query = user_msg if is_detailed else fetch_topic
            new_articles = fetch_news_multi_agent(query=search_query, days_back=90 if is_detailed else 30)
            if len(new_articles) < 5 and search_query != fetch_topic:
                additional = fetch_news_multi_agent(query=fetch_topic, days_back=90 if is_detailed else 30)
                seen_urls = {a.get('url') or a.get('id') for a in new_articles}
                new_articles.extend([a for a in additional if (a.get('url') or a.get('id')) not in seen_urls])
            if new_articles:
                vector_db.add_articles(new_articles, replace=False)
                st.session_state.qa_chain = create_qa_chain(25 if is_detailed else 20)
                st.info(f"📰 Fetched {len(new_articles)} new articles")
    
    if st.session_state.qa_chain:
        with st.spinner("Gathering information..." if is_detailed else "Thinking..."):
            enhanced_question = enhance_question(user_msg, is_detailed)
            result = st.session_state.qa_chain.invoke({"question": enhanced_question, "chat_history": st.session_state.chat_history})
            answer = result["answer"]
            
            # Check if answer indicates no information - fetch automatically
            incomplete_indicators = ["don't know", "not available", "cannot provide", "does not provide",
                                    "does not give", "does not offer", "does not include", "no information",
                                    "i'm sorry", "i cannot", "unable to", "limited information", "does not include any"]
            answer_lower = answer.lower()
            is_incomplete = any(ind in answer_lower for ind in incomplete_indicators)
            
            if is_incomplete:
                # Extract keywords from query to fetch
                keywords = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b\w{4,}\b', user_msg)
                fetch_query = ' '.join(keywords[:3]) if keywords else user_msg
                
                with st.spinner("Fetching relevant information..."):
                    specific_articles = fetch_news_multi_agent(query=fetch_query, days_back=90)
                    if specific_articles:
                        vector_db.add_articles(specific_articles, replace=False)
                        st.session_state.qa_chain = create_qa_chain(40)
                        retriever = build_retriever_from_db(40)
                        docs = retriever.invoke(enhanced_question)
                        if docs:
                            answer = generate_comprehensive_article(user_msg, docs, get_chat_llm())
                        else:
                            # Try direct answer from new articles
                            docs2 = build_retriever_from_db(30).invoke(fetch_query)
                            answer = _get_answer_from_docs(user_msg, docs2, get_chat_llm())
            
            st.session_state.chat_history.append((user_msg, answer))
            save_chat_history(st.session_state.chat_history)
    else:
        answer = "Please fetch some news first, or I'll try to fetch relevant news."
        st.session_state.chat_history.append((user_msg, answer))
        save_chat_history(st.session_state.chat_history)
        with st.spinner("Fetching relevant news..."):
            new_articles = fetch_news_multi_agent(query=user_msg, days_back=30)
            if new_articles:
                vector_db.add_articles(new_articles, replace=False)
                st.session_state.qa_chain = create_qa_chain(20)
                st.success("News fetched! Please ask your question again.")

# Render chat
for user, bot in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(user)
    with st.chat_message("assistant"):
        st.write(bot)
