import streamlit as st
import hashlib
from scoring_latest import final_result, create_vector_db

# Set page title and favicon
st.set_page_config(page_title="QA Bot", page_icon=":robot_face:")

# Define CSS styles
css = """
    .stApp {
        background-color: #1E1E1E; /* Dark background color */
        color: #CCCCCC; /* Text color */
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
        background-color: #2E2E2E; /* Darker input background color */
        color: #CCCCCC; /* Text color */
    }
    .stButton>button {
        background-color: #4CAF50;
        color: #FFFFFF; /* Brighter text color */
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 12px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .subheader-text {
        color: #FFD700; /* Bright yellow color */
    }
    .radio-header {
        font-size: 20px; /* Increase font size */
        color: #FFD700; /* Bright yellow color */
    }
    .qa-bot {
        color: #FFA500; /* Orange color */
        font-size: 36px; /* Increase font size */
        font-weight: bold; /* Bold font */
    }
"""

# Apply CSS styles
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Main content
st.markdown("<h1 class='qa-bot'>QA Bot</h1>", unsafe_allow_html=True)

st.markdown("<h3 class='subheader-text' style='font-size: 18px;'>Upload a file:</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("")
    
def file_hash(uploaded_file):
    # Create a hash of the uploaded file
    file_content = uploaded_file.getvalue()
    return hashlib.sha256(file_content).hexdigest()

def process_file(uploaded_file):
    # Process the file only if it hasn't been processed before or if it's a new file
    if 'file_hash' not in st.session_state or st.session_state['file_hash'] != file_hash(uploaded_file):
        create_vector_db(uploaded_file)
        # Update the session state with the new file hash
        st.session_state['file_hash'] = file_hash(uploaded_file)

# In your Streamlit UI logic
if uploaded_file is not None:
    process_file(uploaded_file)


# Initialize chat history in the session state if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def handle_question():
    user_query = st.session_state.user_query
    if user_query:
        # Integrate your function to get the answer from the QA bot
        result = final_result(user_query)

        # Update chat history with the new interaction at the beginning
        st.session_state.chat_history.insert(0, {"question": user_query, "answer": result})

        # Clear the input box after the question is asked
        st.session_state.user_query = ""

# User input
st.markdown("<h3 class='subheader-text' style='font-size: 18px;'>Enter a question:</h3>", unsafe_allow_html=True)
user_query = st.text_area("", key="user_query")

if st.button('Ask', on_click=handle_question):
    pass  # The logic is now handled in the handle_question function

# Display chat history in reversed order (latest first)
for interaction in st.session_state.chat_history:
    # Using markdown to differentiate questions and answers visually
    st.markdown(f"**Query:** {interaction['question']}")
    st.markdown(f"**Answer:** {interaction['answer']}")

