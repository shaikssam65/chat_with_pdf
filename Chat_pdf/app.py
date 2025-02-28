import streamlit as st
from Agent import final_result
import os
import shutil

# Custom CSS
st.markdown(
    """
<style>
    .file-drop-container { 
        background-color: #f8f9fa;
        padding: 20px;
        border: 2px dashed #0000FF;
        border-radius: 10px;
        text-align: center;
        max-width: 400px;
        margin: 0 auto;
    }

    .query-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }

    .response-card {
        border-left: 4px solid #00FF00;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    .score-explanation {
        font-size: 12px;
        color: #6c757d;
        margin-top: -10px;
        margin-bottom: 15px;
    }

    .streamlit-expander {
        background-color: #ffffff !important;
        border-left: 4px solid #007bff;
        border-radius: 8px;
    }

    .streamlit-expanderHeader {
        padding: 10px 15px !important;
        font-weight: bold;
        color: #007bff;
    }

    .streamlit-expanderContent {
        padding: 15px !important;
        border-bottom-left-radius: 8px;
        border-bottom-right-radius: 8px;
    }

    .upload-status {
        font-size: 14px;
        color: #28a745;
        margin-bottom: 20px;
    }

    .big-font {
        font-size: 20px;
        margin-bottom: 20px;
        color: #17a2b8;
    }
</style>
    """, unsafe_allow_html=True
)

st.markdown(
    '<div class="big-font">**Document-Based AI Assistant**</div>', 
    unsafe_allow_html=True
)

st.write("---")

file_uploaded = False

with st.container():
    st.markdown(
        '<div class="file-drop-container">'
        '<h5>Drag your PDF here or click to browse:</h5>'
        '</div>', 
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("", type="pdf", key="pdf_uploader", label_visibility="collapsed")
    
    if uploaded_file:
        file_uploaded = True
        st.markdown(
            f'<div class="upload-status">Uploaded: {uploaded_file.name}</div>', 
            unsafe_allow_html=True
        )

with st.container():
    if file_uploaded:
        st.markdown(
            '<div class="query-container">'
            'Ask me anything about the document:'
            '</div>', 
            unsafe_allow_html=True
        )
        user_query = st.text_area("", "", height=100)
        process_query = st.button("Generate Response")
    else:
        st.warning("Please upload a PDF file to get started.")
DATA_PATH = "data/"  # Ensure this matches your vector DB path
# Remove the directory if it exists
if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)

# Recreate the directory
os.makedirs(DATA_PATH)


if file_uploaded and user_query and process_query:
    file_path = os.path.join(DATA_PATH, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    with st.spinner('**Processing Query...**'):
        response, highest_score = final_result(user_query, file_path)
        
    st.success('**Response:**')
    st.markdown(
        f'<div class="response-card">{response}</div>', 
        unsafe_allow_html=True
    )
    
    st.markdown(
        '<div class="score-explanation">'
        f'Relevance Score: {highest_score:.2f} '
        '(Higher indicates better match to uploaded document)'
        '</div>', 
        unsafe_allow_html=True
    )
    
    with st.expander(".CREATED WITH ❤️ AND LANGCHAIN"):
        st.write("This app is built using LangChain RAG + Llama2-7B-Chat-GGML 🦙各行各")