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

# Constants
DATA_PATH = 'data/'
DB_FAISS_PATH = 'VectorStore/'
RELEVANCE_THRESHOLD = 0.2

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
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
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
    Given the context and the most relevant documents retrieved, provide a comprehensive answer to the user's question.

    If the retrieved documents don't contain relevant information, clearly state that the answer is likely outside the scope of the available documents.

    Context: {context}
    Question: {question}

    Answer based on the retrieved documents:
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
# Function to extract keywords (replace with your preferred method)
def extract_keywords(text):
    # Sample implementation using frequency-based approach
    word_counts = {}
    for word in text.lower().split():
        word_counts[word] = word_counts.get(word, 0) + 1
    keywords = sorted(word_counts, key=word_counts.get, reverse=True)[:5]  # Top 5 words
    return keywords


# Function to generate prompts
def generate_prompts(documents):
    prompts = []
    for document in documents:
        keywords = extract_keywords(document)
        for keyword in keywords:
            prompts.extend([
                f"Ask a question about {keyword} based on this information: {document[:500]}...",
                f"What can you learn about {keyword} from this text?",
                f"What key points does this document make about {keyword}?",
                f"What are some interesting insights related to {keyword} in this document?",
            ])
    prompts = list(set(prompts))  # Remove duplicates
    return prompts


# Function to load the LLM model
def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.01
    )
    return llm



# Main function to get the final result
def final_result(query):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Generate query embedding
    query_embedding = generate_query_embedding(query, model_name)

    # Load FAISS index and normalize embedding
    index = faiss.read_index(f"{DB_FAISS_PATH}/index.faiss")
    faiss.normalize_L2(np.ascontiguousarray(query_embedding.reshape(1, -1)))

    # Search for top K documents and calculate scores
    D, I = index.search(np.ascontiguousarray(query_embedding).reshape(1, -1), 10)
    scores = [convert_distance_to_score(d) for d in D[0]]
    highest_score = max(scores)

    # Check for out-of-bound queries based on score threshold
    if highest_score < RELEVANCE_THRESHOLD:
        return "I couldn't find relevant information to answer your question. It might be outside the scope of the available documents.", highest_score

    # Retrieve the most relevant documents
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    most_relevant_documents = db.retrieve(query_embedding, k=3)  # Adjust k as needed

    # Generate prompts based on retrieved documents
    prompts = generate_prompts(most_relevant_documents)

    # Present prompts to the user
    print("Here are some questions you might be interested in, based on the available documents:")
    for i, prompt in enumerate(prompts):
        print(f"{i+1}. {prompt}")

    # Get user's choice or suggest a prompt
    user_choice = input("Choose a prompt by number (or type your own): ")
    if not user_choice:
        user_choice = prompts[0]  # Suggest the first prompt
    elif user_choice.isdigit():
        user_choice = prompts[int(user_choice) - 1]  # Convert choice to index

    # Use the chosen prompt for retrieval QA
    qa_prompt = PromptTemplate(template=user_choice, input_variables=['context', 'question'])
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    qa_result = qa({'query': query})

    return qa_result['result'], highest_score

# Execute and print the final result
res,highest_score = final_result("WHo is the president of United states")
print(f"Response: {res}, Relevant Score: {highest_score}")