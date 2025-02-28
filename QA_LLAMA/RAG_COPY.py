from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.chains import RetrievalQA
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# nltk.download('punkt')
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

DATA_PATH = 'data/'
DB_FAISS_PATH = 'VectorStore/'

# # Create vector database
# def create_vector_db():
#     loader = DirectoryLoader(DATA_PATH,
#                              glob='*.pdf',
#                              loader_cls=PyPDFLoader)

#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
#                                                    chunk_overlap=50)
#     texts = text_splitter.split_documents(documents)

#     embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
#                                        model_kwargs={'device': 'cpu'})

#     db = FAISS.from_documents(texts, embeddings)
#     db.save_local(DB_FAISS_PATH)

# if __name__ == "__main__":
#     create_vector_db()
    
    
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
db = FAISS.load_local(DB_FAISS_PATH, embeddings)
llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )

def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',  # Note: 'stuff' should be replaced with the actual chain type if needed
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain
#QA Model Function

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

qa_prompt = set_custom_prompt()
qa = retrieval_qa_chain(llm, qa_prompt, db)
#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

res=final_result("what are requirements for F1 Visa if i am a citizen of india")
res['result']