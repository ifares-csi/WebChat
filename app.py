import streamlit as st
import validators
import requests
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from langchain_community.llms import Ollama

# Initialize environment (Ollama doesn't need API keys)
def initialize_env():
    load_dotenv()
    
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

initialize_env()

def check_url_accessibility(url):
    try:
        response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def extract_text_from_url(url):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs if docs else "Error: No content extracted."
    except Exception as e:
        return f"Error: {str(e)}"

def summarize_webpage(url, user_prompt):
    try:
        docs = extract_text_from_url(url)
        if isinstance(docs, str) and docs.startswith("Error"):
            return docs

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.split_documents(docs)

        #  Embedding & Storing >> Vector Embeddings
        embeddings = (OllamaEmbeddings(model="deepseek-r1:1.5b"))
        
        vectorstoredb = FAISS.from_documents(documents, embeddings)
        
        # Initialize Ollama
        llm = Ollama(model='deepseek-r1:1.5b')
        
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the following question based only on the provided context:
            <context>
            {context}
            </context>
            
            Question: {input}
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Input--->Retriever--->vectorstoredb
        retriever = vectorstoredb.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        
        
        ## Get the response form the LLM
        response = retrieval_chain.invoke({"input": user_prompt})
        return response if response else "No relevant summary found."
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI Configuration
st.set_page_config(page_title="Chat With WebPage", layout="wide")

# Centered title with custom styling
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1; font-family: Arial, sans-serif;'>
        WebPage Chat (Deepseek R1)
    </h1>
    """,
    unsafe_allow_html=True
)

# Divider
st.markdown("---")

# Input Section
col1, col2 = st.columns([3, 2])
with col1:
    url = st.text_input("🌐 Enter Webpage URL", placeholder="https://example.com")
with col2:
    prompt = st.text_area(
        "💬 Your Question", 
        "Summarize this webpage in a concise manner.",
        height=100
    )

# Validation and Processing
if url and not validators.url(url):
    st.error("❌ Invalid URL format. Please enter a valid web address.")

if st.button("🚀 Submit", use_container_width=True):
    if url and validators.url(url):
        if not check_url_accessibility(url):
            st.error("🔴 Unable to access the webpage. Please check the URL and try again.")
        else:
            with st.spinner("🔍 Analyzing content with Deepseek R1..."):
                summary = summarize_webpage(url, prompt)
                
            st.markdown("---")
             # Think Section with Custom Background
            with st.expander("🤔 Reasoning Process", expanded=False):
                st.markdown(
                    f"""
                    <div style='background-color: #E8F8F5; padding: 20px; border-radius: 10px;'>
                        <h4>Think</h4>
                        {summary['answer'].split("</think>")[0].replace("<think>", "")}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Final Answer Section with Custom Background
            st.markdown(
                f"""
                <div style='background-color: #F4F6F6; padding: 20px; border-radius: 10px;'>
                    <h4>Final Answer</h4>
                    {summary['answer'].split("</think>")[1]}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("---")
    else:
        st.error("⚠️ Please enter a valid URL before submitting.")

# Footer
st.markdown(
    """
    <div style='text-align: center; margin-top: 50px; color: #707B7C;'>
        Copyright &copy; 2025 El-Fares. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)