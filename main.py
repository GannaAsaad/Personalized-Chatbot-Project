import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException # Removed Form as API key is not a form field now
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from dotenv import load_dotenv # Import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import functionalities from our modules
import explanation_module
import rag_module

app = FastAPI(title="Paper Processor and Q&A API")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State and Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Load API key from environment

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    

app_state = {
    "pdf_path": None,
    "summaries": None,
    "vector_store": None,
    "conversation_obj": None,
    # "gemini_api_key": None, # No longer needed here, using global GEMINI_API_KEY
    "embeddings_model": None,
    "is_initialized": False
}
TEMP_PDF_DIR = "temp_pdf_storage"
os.makedirs(TEMP_PDF_DIR, exist_ok=True)

# --- Orchestrator Functions ---

def _reset_and_prepare_state():
    """Cleans up old PDF path and resets the application state for a new PDF."""
    global app_state
    logger.info("Resetting application state for new PDF processing.")
    
    if app_state.get("pdf_path") and os.path.exists(app_state["pdf_path"]):
        try:
            os.remove(app_state["pdf_path"])
            logger.info(f"Removed previous temporary PDF: {app_state['pdf_path']}")
        except OSError as e:
            logger.warning(f"Could not remove old temp PDF {app_state['pdf_path']}: {e}")

    app_state = {
        "pdf_path": None, "summaries": None, "vector_store": None,
        "conversation_obj": None,
        "embeddings_model": None, "is_initialized": False
    }

def _process_and_initialize_pdf(file_content_bytes: bytes, original_filename: str):
    """
    Handles the entire PDF processing and RAG initialization workflow.
    Modifies global app_state. Uses the globally loaded GEMINI_API_KEY.
    """
    global app_state
    if not GEMINI_API_KEY:
        logger.error("Gemini API key is not configured. Cannot process PDF.")
        raise HTTPException(status_code=500, detail="Server configuration error: Gemini API key not found.")

    _reset_and_prepare_state()

    logger.info("Configuring Gemini for summarization module...")
    explanation_module.configure_gemini_explicitly(GEMINI_API_KEY)
    if not explanation_module.test_gemini():
        raise HTTPException(status_code=500, detail="Gemini self-test failed for summarization module. Check API Key.")
    logger.info("Gemini for summarization module configured and tested successfully.")

    temp_pdf_path = os.path.join(TEMP_PDF_DIR, original_filename)
    try:
        with open(temp_pdf_path, "wb") as f:
            f.write(file_content_bytes)
        app_state["pdf_path"] = temp_pdf_path
        logger.info(f"Uploaded PDF saved temporarily to: {temp_pdf_path}")
    except IOError as e:
        logger.error(f"Failed to save temporary PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save temporary PDF: {str(e)}")

    logger.info("Processing PDF for section summaries...")
    summaries_result = explanation_module.process_pdf(file_content_bytes)
    if not summaries_result or "error" in summaries_result:
        error_detail = summaries_result.get("error", "Unknown error during summarization or no summaries produced.")
        logger.error(f"Error processing PDF for summaries: {error_detail}")
        if os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)
        app_state["pdf_path"] = None
        raise HTTPException(status_code=400, detail=f"Error processing PDF for summaries: {error_detail}")
    app_state["summaries"] = summaries_result
    logger.info("PDF summaries generated successfully.")

    logger.info("Initializing RAG system...")
    app_state["embeddings_model"] = rag_module.load_embeddings_model()
    
    pdf_text_for_rag = rag_module.extract_text_from_pdf(temp_pdf_path)
    if not pdf_text_for_rag:
        logger.error("Could not extract text from PDF for RAG system.")
        raise HTTPException(status_code=400, detail="Could not extract text from PDF for RAG system.")
    
    app_state["vector_store"] = rag_module.create_vector_store(pdf_text_for_rag, app_state["embeddings_model"])
    if not app_state["vector_store"]:
        logger.error("Failed to create vector store for RAG system.")
        raise HTTPException(status_code=500, detail="Failed to create vector store for RAG system.")
    
    app_state["is_initialized"] = True
    logger.info(f"PDF '{original_filename}' processed and RAG initialized successfully.")
    return app_state["summaries"]


def _get_answer_from_rag(query: str):
    """Handles querying the RAG system and managing conversation state."""
    global app_state
    if not GEMINI_API_KEY:
        logger.error("Gemini API key is not configured. Cannot answer questions.")
        raise HTTPException(status_code=500, detail="Server configuration error: Gemini API key not found.")
        
    if not app_state.get("is_initialized") or \
       not app_state.get("vector_store"):
        logger.warning("Attempted to ask question before PDF processing or RAG initialization.")
        raise HTTPException(status_code=400, detail="System not initialized. Please upload and process a PDF first.")

    rag_result, updated_conv_obj = rag_module.conversation_answering(
        vector_store=app_state["vector_store"],
        question=query,
        api_key=GEMINI_API_KEY, # Use the globally loaded API key
        conversation_obj=app_state.get("conversation_obj")
    )
    app_state["conversation_obj"] = updated_conv_obj

    answer = rag_result.get("answer", "No answer found or an error occurred.")
    sources = rag_result.get("source_documents", [])
    
    serializable_sources = []
    if sources:
        for doc in sources:
            serializable_sources.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata if isinstance(doc.metadata, dict) else str(doc.metadata)
            })
    
    logger.info(f"Answer for '{query}': '{answer}'")
    return answer, serializable_sources

# --- API Endpoints ---

@app.get("/")
async def home():
    if not GEMINI_API_KEY:
        return {"warning": "Server is running, but GEMINI_API_KEY is not configured. Functionality will be limited."}
    return {"message": "Welcome to the Paper Explanation and Q&A API. Use /docs for API documentation."}

@app.post("/upload_pdf/")
async def upload_pdf_endpoint(
    file: UploadFile = File(..., description="The PDF file to process")
    # gemini_api_key: str = Form(...) # Removed: API key is now from .env
):
    logger.info(f"Received API call to /upload_pdf/ for file: {file.filename}")
    if not GEMINI_API_KEY:
        logger.error("Cannot upload PDF: Gemini API key is not configured on the server.")
        raise HTTPException(status_code=503, detail="Service unavailable: API key not configured on server.")
    try:
        file_content_bytes = await file.read()
        if not file_content_bytes:
            logger.error("Uploaded PDF file is empty.")
            raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")

        # _process_and_initialize_pdf now uses the global GEMINI_API_KEY
        summaries = _process_and_initialize_pdf(file_content_bytes, file.filename)
        
        return {
            "filename": file.filename,
            "message": "PDF processed successfully. Ready for Q&A.",
            "summaries": summaries
        }
    except HTTPException as http_exc:
        logger.error(f"HTTPException during PDF upload for {file.filename}: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.exception(f"Unexpected error during PDF upload for {file.filename}: {str(e)}")
        if app_state.get("pdf_path") and os.path.exists(app_state["pdf_path"]):
             try:
                os.remove(app_state["pdf_path"])
                app_state["pdf_path"] = None
             except OSError as err_os:
                 logger.warning(f"Could not remove temp PDF {app_state.get('pdf_path')} during error cleanup: {err_os}")
        app_state["is_initialized"] = False
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")
    finally:
        if file:
            await file.close()
            logger.info(f"Closed file: {file.filename}")


class QuestionRequest(BaseModel):
    query: str

@app.post("/ask/")
async def ask_question_endpoint(payload: QuestionRequest):
    logger.info(f"Received API call to /ask/ with query: '{payload.query}'")
    if not GEMINI_API_KEY:
        logger.error("Cannot ask question: Gemini API key is not configured on the server.")
        raise HTTPException(status_code=503, detail="Service unavailable: API key not configured on server.")
    try:
        answer, sources = _get_answer_from_rag(payload.query)
        return {
            "question": payload.query,
            "answer": answer,
            "source_documents": sources
        }
    except HTTPException as http_exc:
        logger.error(f"HTTPException during /ask/ for query '{payload.query}': {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.exception(f"Unexpected error during /ask/ for query '{payload.query}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")


@app.on_event("shutdown")
def shutdown_event():
    global app_state
    logger.info("Application shutting down. Cleaning up temporary PDF if it exists...")
    pdf_path_to_remove = app_state.get("pdf_path")
    if pdf_path_to_remove and os.path.exists(pdf_path_to_remove):
        try:
            os.remove(pdf_path_to_remove)
            logger.info(f"Successfully removed temporary PDF: {pdf_path_to_remove}")
        except OSError as e:
            logger.error(f"Could not remove temporary PDF {pdf_path_to_remove} on shutdown: {e}")

if __name__ == "__main__":
    import uvicorn
    if not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY is not set in the environment. The application might not function correctly.")
        print("Please create a .env file with GEMINI_API_KEY= or set it as an environment variable.")
    uvicorn.run(app) # Added host and port for clarity

# --- END OF FILE main2.py ---
