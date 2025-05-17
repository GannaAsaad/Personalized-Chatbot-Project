# --- START OF FILE rag_module.py ---

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os
from functools import lru_cache # For caching resource-intensive loads

# --- Configuration ---
GEMINI_MODEL_ID = "gemini-1.5-flash-latest"
DEFAULT_K_CONTEXT_CHUNKS = 4
DEFAULT_TEMPERATURE = 0.5
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Core LangChain and Helper Functions ---

@lru_cache(maxsize=1)
def load_embeddings_model():
    print(f"INFO (RAG): Loading embeddings model '{EMBEDDINGS_MODEL_NAME}'...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
        print("INFO (RAG): Embeddings model loaded successfully.")
        return embeddings
    except Exception as e:
        print(f"ERROR (RAG): Failed to load embeddings model: {e}")
        raise

def extract_text_from_pdf(pdf_file_path: str) -> str | None:
    print(f"INFO (RAG): Extracting text from '{os.path.basename(pdf_file_path)}' for RAG...")
    try:
        doc = fitz.open(pdf_file_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text() + "\n"
        doc.close()
        if not text.strip():
            print("WARNING (RAG): No text found in the PDF for RAG.")
            return None
        print("INFO (RAG): Text extraction for RAG successful.")
        return text
    except FileNotFoundError:
        print(f"ERROR (RAG): PDF file not found at '{pdf_file_path}'.")
        return None
    except Exception as e:
        print(f"ERROR (RAG): Could not read or process PDF '{pdf_file_path}' for RAG: {e}")
        return None

def create_vector_store(pdf_text: str, embeddings_model) -> FAISS | None:
    if not pdf_text:
        print("WARNING (RAG): Cannot create vector store from empty text.")
        return None
    print("INFO (RAG): Creating vector store...")
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_text(pdf_text)

        if not chunks:
            print("WARNING (RAG): No text chunks were generated from the PDF content for RAG.")
            return None

        print(f"INFO (RAG): Generated {len(chunks)} text chunks for RAG.")
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings_model)
        print("INFO (RAG): Vector store created successfully for RAG.")
        return vector_store
    except Exception as e:
        print(f"ERROR (RAG): Failed to create vector store for RAG: {e}")
        return None

def conversation_answering(
    vector_store: FAISS,
    question: str,
    api_key: str, # Google Gemini API Key
    k: int = DEFAULT_K_CONTEXT_CHUNKS,
    temperature: float = DEFAULT_TEMPERATURE,
    conversation_obj: dict | None = None
):
    if not api_key:
        return {"answer": "ERROR (RAG): API Key is missing for RAG. Please provide it.", "source_documents": []}, conversation_obj
    if not vector_store:
        return {"answer": "ERROR (RAG): Vector store (PDF index) is not available for RAG.", "source_documents": []}, conversation_obj

    try:
        if conversation_obj is None:
            print("INFO (RAG): Initializing new conversation chain...")
            qa_template = """You are an expert AI research assistant. Your goal is to answer questions based on the provided context from a research paper.
Be precise and technical. If specific details or data are present in the context, include them.

Context from the paper:
---
{context}
---

Question: {question}

Precise Answer based *only* on the context:"""
            QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])

            llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_ID,
                google_api_key=api_key,
                temperature=temperature,
                convert_system_message_to_human=True
            )
            retriever = vector_store.as_retriever(search_kwargs={'k': k})
            memory = ConversationBufferMemory(
                memory_key='chat_history',
                output_key='answer',
                return_messages=True
            )
            _template_condense = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
If the follow up question is already a standalone question or if there is no chat history, just return the question as is.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
            CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template_condense)

            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                chain_type="stuff",
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": QA_PROMPT},
                output_key='answer'
            )
            conversation_obj = {"chain": chain, "memory": memory}
            print("INFO (RAG): Conversation chain initialized.")
        
        result = conversation_obj["chain"].invoke({'question': question})
        # Ensure source documents are serializable
        if 'source_documents' in result and result['source_documents'] is not None:
            for doc in result['source_documents']:
                if 'metadata' in doc and isinstance(doc.metadata, dict):
                    # Convert any non-serializable items in metadata if necessary
                    # For now, assume metadata is simple enough or FAISS handles it.
                    pass 
        return result, conversation_obj

    except Exception as e:
        print(f"ERROR (RAG): An error occurred in conversation_answering: {str(e)}")
        # Attempt to get more detailed error if it's from Google API
        if hasattr(e, 'args') and e.args:
            error_detail = str(e.args[0])
            if "API key not valid" in error_detail:
                 return {"answer": f"Error (RAG): Google API Key is not valid. Please check your key. Details: {error_detail}", "source_documents": []}, conversation_obj
        return {"answer": f"Error during LangChain processing (RAG): {str(e)}", "source_documents": []}, conversation_obj

# --- END OF FILE rag_module.py ---
