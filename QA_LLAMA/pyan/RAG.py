from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import shutil
from transformers import AutoTokenizer
import nltk

# nltk.download('punkt')

DATA_PATH = 'data/'
DB_FAISS_PATH = 'VectorStore/'

# Adjust the text splitter to create smaller chunks
def create_vector_db(uploaded_file):
    # loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    # documents = loader.load()
    # # Adjust chunk size and overlap to ensure smaller chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    # texts = text_splitter.split_documents(documents)

    # embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    # db = FAISS.from_documents(texts, embeddings)
    # db.save_local(DB_FAISS_PATH)
    # Temporarily save the uploaded file to process it
    temp_path = os.path.join(DATA_PATH, uploaded_file.name)
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Now, assume you need a file path to load the document
    loader = PyPDFLoader(file_path=temp_path)
    documents = loader.load() 

    # Continue with the processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

# Load and prepare the language model
def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,  # Ensure this matches the model's limit
        temperature=0.5
    )
    return llm

# Adjust QA retrieval to handle token limit
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',  # Adjusted to a valid chain type
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Define the prompt template
def set_custom_prompt():
    custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt


def qa_bot(query):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    # Set allow_dangerous_deserialization to True when loading the database
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    response = qa({'query': query})
    return response

# Use the adjusted QA bot
res = qa_bot("what are requirements for F1 Visa if i am a citizen of india")
print(res['result'])


# def qa_bot_2(query):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
#     db = FAISS.load_local(DB_FAISS_PATH, embeddings)
#     llm = load_llm()

#     qa_prompt = set_custom_prompt()
#     qa = retrieval_qa_chain(llm, qa_prompt, db)

#     context = "Your retrieved context goes here. It could be one or multiple documents."
#     combined_input = context + query  # This is a simplification

#     max_length = 512  # Adjust based on your model's token limit
#     if len(combined_input.split()) > max_length:
#         # Truncate the context while preserving the full query
#         preserved_query_length = len(query.split())
#         max_context_length = max_length - preserved_query_length
#         truncated_context = ' '.join(context.split()[:max_context_length])
#         combined_input = truncated_context + query

#     # Dummy response to illustrate flow
#     response = {"result": "Processed input without exceeding token limit."}
#     return response