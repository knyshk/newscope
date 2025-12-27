# Newscope

Newscope is a Python based news exploration and summarization system that combines live news ingestion, semantic indexing, and language model driven reasoning into a single interactive application.

The project focuses on **how unstructured news data can be collected, organized semantically, and queried contextually**, rather than just displaying headlines or summaries.

It is designed as a practical implementation of modern NLP system design concepts such as embeddings, vector search, and context aware text generation.

---

## What This Project Does

At a high level, Newscope allows a user to:

- Fetch current news articles from external news APIs
- Convert raw news text into semantic representations
- Store and retrieve articles based on meaning rather than keywords
- Ask natural language questions about the news
- Receive concise, contextual summaries and answers

Instead of treating news as static text, the system treats it as **searchable semantic knowledge**.

---

## How It Works (End to End Flow)

### 1. News Ingestion

- The system fetches news articles using third party news APIs
- Article metadata such as title, source, and content is extracted
- Raw HTML or noisy text is cleaned and normalized

This step converts live news into plain structured text suitable for processing.

---

### 2. Text Chunking and Preparation

- Articles are split into smaller text chunks
- Chunking prevents loss of context during embedding
- Each chunk represents a semantically meaningful portion of an article

This improves retrieval accuracy and relevance.

---

### 3. Semantic Embedding

- Each text chunk is converted into a numerical embedding
- Embeddings capture the semantic meaning of the text
- Similar meanings result in closer vectors in embedding space

This step enables meaning based retrieval rather than keyword matching.

---

### 4. Vector Storage and Indexing

- Embeddings are stored in a local vector database using ChromaDB
- Each vector is stored alongside its source text and metadata
- The database supports similarity search at query time

This allows fast retrieval of relevant news context.

---

### 5. Query Processing and Retrieval

- User queries are converted into embeddings
- The vector database is searched for the most relevant news chunks
- Retrieved chunks form the contextual knowledge for the response

This ensures responses are grounded in real news content.

---

### 6. Summarization and Response Generation

- Retrieved news context is passed to a language model
- The model generates concise summaries or answers
- Responses are grounded in retrieved articles, not free generation

This reduces hallucinations and improves factual relevance.

---

### 7. User Interface

- Streamlit provides an interactive web interface
- Users can enter queries and follow up questions
- Results are displayed in real time

The UI is intentionally simple to keep focus on system design.

---

## Key Features

- Live news ingestion from external APIs
- Semantic indexing using embeddings
- Vector based similarity search
- Context aware summarization
- Interactive Streamlit interface
- Modular and extensible codebase

---

## Project Structure

```text
news_explorer/
├── app.py              # Main Streamlit application and UI logic
├── news_fetcher.py     # News API integration and data ingestion
├── summarizer.py       # Language model interaction and summarization
├── db.py               # Vector database setup and retrieval logic
├── test_app.py         # Basic test scaffolding
└── requirements.txt    # Python dependencies
