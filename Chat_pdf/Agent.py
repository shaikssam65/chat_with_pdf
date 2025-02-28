from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import shutil

# Constants
DATA_PATH = 'data/'
DB_FAISS_PATH = 'VectorStore/'
RELEVANCE_THRESHOLD = 0.2

# Remove the directory if it exists
if os.path.exists(DB_FAISS_PATH):
    shutil.rmtree(DB_FAISS_PATH)

# Recreate the directory
os.makedirs(DB_FAISS_PATH)

# Functions for vector database creation
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

# Functions for loading LLM
def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.01
    )
    return llm

# Functions for retrieval QA chain
def retrieval_qa_chain(llm, prompt, db, query):
    retriever = db.as_retriever(search_kwargs={'k': 1})  # Reduce retrieved documents
    retrieved_docs = retriever.get_relevant_documents(query)
    retrieved_documents_str = "\n".join([doc.page_content for doc in retrieved_docs])

    # Truncate to 512 tokens if too long
    if len(retrieved_documents_str.split()) > 450:
        retrieved_documents_str = " ".join(retrieved_documents_str.split()[:450])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Functions for query embedding generation
def generate_query_embedding(query, model_name):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode(query, convert_to_tensor=False)
    faiss.normalize_L2(query_embedding.reshape(1, -1))  # Normalize for cosine similarity
    return query_embedding

# Functions for score conversion
def convert_distance_to_score(distance):
    score = 1 / (1 + distance)
    return max(0, min(score, 1))

# Functions for setting custom prompt
def set_custom_prompt():
    custom_prompt_template = """
    **Context:**
    {context}

    **Question:**
    {question}

    **Instructions:**
    - Carefully read the provided context and question.
    - If the context contains relevant information to answer the question, provide a detailed and comprehensive response.
    - If the context does not contain relevant information, clearly state that the answer is not available in the provided documents.
    - Suggest rephrasing the question or searching for broader information related to the topic if the context is insufficient.
    - Ensure the response is coherent, concise, and directly addresses the question.
    - Avoid making assumptions or providing information not supported by the context.

    **Answer:**
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])


# Main function to get the final result
def final_result(query, file_path=None):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    query_embedding = generate_query_embedding(query, model_name)

    # Ensure the FAISS index exists
    if not os.path.exists(f'{DB_FAISS_PATH}/index.faiss'):
        create_vector_db()

    index = faiss.read_index(f'{DB_FAISS_PATH}/index.faiss')
    faiss.normalize_L2(np.ascontiguousarray(query_embedding.reshape(1, -1)))
    D, I = index.search(np.ascontiguousarray(query_embedding).reshape(1, -1), 10)
    scores = [1 - (d / 2) for d in D[0]]
    highest_score = max(scores)

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()

    # Retrieve documents
    retriever = db.as_retriever(search_kwargs={'k': 2})
    retrieved_docs = retriever.get_relevant_documents(query)
    retrieved_documents_str = "\n".join([doc.page_content for doc in retrieved_docs])

    qa = retrieval_qa_chain(llm, qa_prompt, db, query)
    print(qa.input_keys)
    qa_result = qa({
        'query': query,
        'context': retrieved_documents_str,
    })

    return qa_result['result'], highest_score


