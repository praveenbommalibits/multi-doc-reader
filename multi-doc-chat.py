import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv


####
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from huggingface_hub import notebook_login
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import sys
from PyPDF2 import PdfReader
from docx import Document
from htmlTemplates import css, bot_template, user_template


embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME')
llama_7b_chat_model_name = os.environ.get('LLAMA_7B_CHAT_MODEL_NAME')
hftoken = st.secrets['HUGGINGFACEHUB_API_TOKEN']



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_document_chunks(documents):
    document_splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)
    document_chunks = document_splitter.split_documents(documents)
    # text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    # chunks = text_splitter.split_text(text)
    return document_chunks


def get_vector_store(document_chunks):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectordb = Chroma.from_documents(document_chunks, embedding=embeddings, persist_directory='./data')
    #embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    #vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectordb.persist()
    return vectordb

# Function to read PDF content
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to read DOC content
def read_docx(file):
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def prepare_documents(generic_docs):
    if generic_docs is not None:
        documents = []

        for doc_file in generic_docs:
            if doc_file.type == 'application/pdf':
                # Read PDF content
                pdf_text = read_pdf(doc_file)
                documents.append(pdf_text)
            elif doc_file.type == 'application/msword':
                # Read DOC content
                doc_text = read_docx(doc_file)
                documents.append(doc_text)
        return documents


def prepare_documents1():
    document = []
    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            pdf_path = "./docs/" + file
            loader = PyPDFLoader(pdf_path)
            document.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = "./docs/" + file
            loader = Docx2txtLoader(doc_path)
            document.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = "./docs/" + file
            loader = TextLoader(text_path)
            document.extend(loader.load())
    return document


def get_conversation_chain(vectorstore):
    #cache_dir = "./model_cache"
    tokenizer = AutoTokenizer.from_pretrained(
        llama_7b_chat_model_name,
        #cache_dir=cache_dir,
        trust_remote_code=True,
        #token=hftoken,
        use_auth_token = hftoken
    )

    model = AutoModelForCausalLM.from_pretrained(llama_7b_chat_model_name,
                                                 device_map='auto',
                                                 torch_dtype=torch.float16,
                                                 #use_auth_token=True,
                                                 # load_in_8bit=True,
                                                 load_in_4bit=True,
                                                 use_auth_token=hftoken
                                                 )
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hftoken
    )
    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})
    #llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                             retriever=vectorstore.as_retriever(search_kwargs={'k':6}),
                                             verbose=False, memory=memory)
    return conversation_chain


def handle_user_input(user_question):
    response=st.session_state.conversation({'question':user_question})
    st.write(response)
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2==0:
            #st.write(message)
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            #st.write(message)
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)




def main():
    load_dotenv()
    st.set_page_config("Chat with Multiple PDFs", page_icon=":books:")
    st.header("Chat with Multiple PDFs :mobiles:")
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None
    user_question = st.text_input("Ask a question from your documents")
    if user_question:
        handle_user_input(user_question)
    with st.sidebar:
        st.header("Chat with Multi Docs")
        st.title("Multimodal Chatapp using LangChain, Llama2")
        st.subheader("Your Documents")
        #generic_docs = st.file_uploader("Upload the Files here and Click on Process", accept_multiple_files=True,
         #                               type=['pdf', 'doc'])
        #documents = prepare_documents()
        documents = prepare_documents1()
        print("Total no of documents :", len(documents))
        st.markdown('''
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [Model](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) meta-llama/Llama-2-7b-chat-hf
        ''')
        st.write('App developed using Streamlit cloud and different models using Huggingface')

        if st.button('Process'):
            with st.spinner("Processing"):
                # Extract Text from PDF
                raw_text = documents  # get_pdf_text(documents)
                # Split the Text into Chunks
                document_chunks = get_document_chunks(raw_text)
                # Create Vector Store
                vectorstore = get_vector_store(document_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Done!")


if __name__ == "__main__":
    main()
