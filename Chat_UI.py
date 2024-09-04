import streamlit as st
import openai
import os
import io
import sys
import ollama
import json
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from typing import List
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# App title
st.set_page_config(page_title="Document Query Interface")

load_dotenv()

__import__('pysqlite3') 
import sys 
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def validate_openai_api_key(api_key: str) -> bool:
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.APIConnectionError:
        return False
    else:
        return True

class CapturePrints:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.captured_output = io.StringIO()

    def __enter__(self):
        sys.stdout = self.captured_output
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = sys.__stdout__
        if self.log_callback:
            self.log_callback(self.captured_output.getvalue())

if 'log' not in st.session_state:
    st.session_state.log = ""

def update_log(message):
    st.session_state.log += message

def extract_pages(source_file: str):
    print("="*30)
    print(f">>>Extracting from: {source_file}")

    try:
        with open(source_file, 'r') as file:
            data = json.load(file)
    except json.JSONDecodeError:
        print("Invalid JSON file.")
        return []
    except FileNotFoundError:
        print("File not found.")
        return []

    documents = []
    for key, value in data.items():
        doc = Document(page_content=value, metadata={"source": key})
        documents.append(doc)

    print(f">>>Extracted {len(documents)} pages.")
    print("="*30)
    return documents

def process_batch(batch: List[Document], chunk_size: int, chunk_overlap: int):
    print(f"Processing batch of {len(batch)} pages...")

    # Create and store embeddings
    vectorstore_path = os.environ.get('VECTORSTORE_PATH', 'Vec_Store')
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator=False,
    )
    docs = text_splitter.split_documents(batch)
    vectorstore = Chroma.from_documents(docs, embedding=embd, persist_directory=vectorstore_path)

    print("Batch processing complete.")
    return vectorstore

def process_all_documents(documents: List[Document], chunk_size: int, chunk_overlap: int):
    batch_size = 6
    total_pages = len(documents)
    vectorstore = None
    
    for i in range(0, total_pages, batch_size):
        batch = documents[i:i+batch_size]
        batch_vectorstore = process_batch(batch, chunk_size, chunk_overlap)
        
        if vectorstore is None:
            vectorstore = batch_vectorstore
        else:
            vectorstore.add_documents(batch)
        
        pages_processed = min(i + batch_size, total_pages)
        update_log(f"{pages_processed} pages processed\n")
    
    return vectorstore

# Sidebar
with st.sidebar:
    st.title('Document Query Interface')

    llm_type = st.sidebar.radio("Choose LLM Type", ['API', 'Local'])

    if llm_type == 'API':
        api_key = st.sidebar.text_input('Enter API key:', type='password', key='api_key')

        if api_key:
            api_key_valid = validate_openai_api_key(api_key)
        
            if api_key_valid:
                st.sidebar.success('API key validated!', icon='✅')
                st.session_state.api_key_final = api_key
                os.environ['OPENAI_API_KEY'] = api_key
                embd = OpenAIEmbeddings(openai_api_key=api_key)
                model = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()],temperature=0, model="gpt-4o-mini")
            else:
                st.sidebar.error('Invalid API key!', icon='⚠️')
                st.stop()
        else:
            st.warning("Please enter your OpenAI API key to continue.")
            st.stop()
            
    elif llm_type == 'Local':
        list_response = ollama.list()
        pulled_models = [model['name'] for model in list_response['models']]
        selected_model = st.sidebar.selectbox('Select a Model', list(reversed(pulled_models)))

    # Document Update and Processing Section
    st.markdown('---')
    st.subheader('Document Processing')
    update_enabled = st.checkbox("Update Documents", key='update_documents')

    if update_enabled:
        source_directory = st.text_input("Path to JSON file:")
        if st.button("Load Document"):
            if not os.path.exists(source_directory):
                st.error("Invalid file path!")
            else:
                with CapturePrints(log_callback=update_log):
                    st.session_state.total_pages = extract_pages(source_directory)
                st.success("Documents loaded successfully.")
        
        col1, col2 = st.columns([1,1])
        with col1:
            chunk_size = st.number_input("Chunk Size", min_value=100, max_value=1000, value=500, help='Size of text chunk for processing.')
        with col2:
            chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=100, value=50, help='Overlap size between chunks.')
        
        if st.button("Process Documents"):
            if 'total_pages' not in st.session_state:
                st.error("Please load the document first.")
            else:
                with CapturePrints(log_callback=update_log):
                    st.session_state.vectorstore = process_all_documents(
                        st.session_state.total_pages, chunk_size, chunk_overlap
                    )
                st.success("Document processing complete.")
                st.success('VectorStore is Loaded')

    else:
        with CapturePrints(log_callback=update_log):
            if 'vectorstore' not in st.session_state: 
                st.session_state.vectorstore = Chroma(persist_directory="Pre_stored_Vec_Store", embedding_function=embd)
        st.success('Pre-stored VectorStore is Loaded')

    st.markdown('---')
    st.text_area("Log", st.session_state.log, height=300)

# Chat Interface
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

if 'sessions' not in st.session_state:
    st.session_state.sessions = []

def save_session():
    if st.session_state.messages:
        st.session_state.sessions.append(st.session_state.messages.copy())

def new_session():
    save_session()
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.success("New session created and old session saved", icon='✅')

with st.sidebar:
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button('Clear Log'):
            st.session_state.log = ""
    with col2:
        st.button('New Session', on_click=new_session)

col1, col2 = st.columns([1,1])
with col1:
    if st.session_state.sessions:
        session_index = st.selectbox('Select Previous Session', range(len(st.session_state.sessions)), format_func=lambda x: f"Session {x+1}")
    if st.button('Load Selected Session'):
        st.session_state.messages = st.session_state.sessions[session_index].copy()
        st.success("Loaded selected session")
with col2:
    st.session_state.topk = st.number_input("Top K Retrieval", min_value=1, max_value=10, value=3, help='Number of chunks Retrieved')

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

disable_chat = not (llm_type == 'API' and api_key_valid or llm_type == 'Local' and selected_model)

if not disable_chat:
    if prompt := st.chat_input("Enter your question:", disabled=disable_chat):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if llm_type == 'Local':
        st.session_state.rag_chain = RetrievalQA.from_chain_type(
            llm=Ollama(model=selected_model),
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": st.session_state.topk})
        )
    else:
        st.session_state.rag_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": st.session_state.topk})
        )

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": st.session_state.topk})
                docs = retriever.get_relevant_documents(prompt)
                with st.expander("See Context"):
                    for doc in docs:
                        st.write(doc.page_content)
                        file_path = doc.metadata.get('source', 'Raptor Cluster Summary File')
                        st.markdown(f"**Source:** `{file_path}`")
                response = st.session_state.rag_chain.run(prompt)
                st.write(response)
            
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

