# PDF Query Answer System using RAG  

![Project Preview](assets/app-preview.png)


A **sophisticated** document query system leveraging `Llama 2`, **RAG**, `Faiss`, `LangChain`, and `Streamlit` to extract insights from PDFs and provide accurate, context-aware answers.

---

## Table of Contents  
  * [Introduction](#introduction)
  * [System Overview](#system-overview)
  * [Technical Stack](#technical-stack)
  * [Getting Started](#getting-started)
  * [Usage](#usage)
 
---

## Introduction
This project introduces a PDF query answer system that merges cutting-edge technologies to facilitate efficient information extraction from documents. By leveraging **RAG** (Retrieval-Augmented Generation) with `Llama 2` and a robust retrieval system powered by Faiss, the solution enables users to pose questions about PDF content and receive precise, relevant answers. The system is designed for seamless interaction through a Streamlit interface, making complex document analysis accessible and user-friendly.

---

## System Overview
The architecture integrates several state-of-the-art components to deliver high accuracy and responsiveness:
* **Llama 2**: Handles natural language understanding and response generation.
* **Faiss**: Accelerates similarity searches for rapid document retrieval.
* **LangChain**: Manages language processing workflows and data preparation.
* **Streamlit**: Provides an intuitive interface for users to interact with the system.

Key features:
  * **Document Embeddings**: Semantic embeddings generated via `SentenceTransformers` for content understanding.
  * **Query Processing**: Adapts to factual questions and complex analytical queries.
  * **Response Generation**: Synthesizes answers using extracted contextual information from PDFs.

---

## Technical Stack
  * **Machine Learning**:  
    * SentenceTransformers (Embeddings)
    * Llama 2 (Language Model)
    * Faiss (Vector Indexing)
  * **Libraries**:  
    * PyPDFLoader (PDF Parsing)
    * LangChain (Workflows)
    * Torch, Transformers (Backend Processing)
  * **UI Framework**:  
    * Streamlit (Interactive Frontend)

---

## Getting Started
### Prerequisites
Install Python 3.8+, then clone the repository:
```bash
git clone https://github.com/shaikssam65/chat_with_pdf.git
cd chat_with_pdf
pip install -r requirements.txt
streamlit run app.py
