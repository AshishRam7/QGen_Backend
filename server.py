import os
import time
import re
import base64
import requests # For Jina API and Gemini
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple
import sys
import shutil
import urllib.parse
import json

# Text Processing & Embeddings
from PIL import Image
import nltk
import torch # For converting list to tensor for sbert_util

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    # nltk.data.find('tokenizers/punkt_tab') # punkt_tab is not a standard resource name, punkt covers sentence tokenization.
except LookupError:
    print("NLTK data not found or incomplete. Downloading essential resources...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    print("NLTK data downloaded.")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import util as sbert_util # sbert_util is still used for cosine similarity

# Qdrant
from qdrant_client import models, QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, ScoredPoint
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

# Moondream for Image Description
import moondream as md_moondream

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Configuration ---
from dotenv import load_dotenv
load_dotenv()

# --- Load Environment Variables ---
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DATALAB_API_KEY = os.environ.get("DATALAB_API_KEY")
DATALAB_MARKER_URL = os.environ.get("DATALAB_MARKER_URL")
MOONDREAM_API_KEY = os.environ.get("MOONDREAM_API_KEY")

# Jina API Configuration
JINA_API_KEY = os.environ.get("JINA_API_KEY")
JINA_API_EMBEDDING_URL = os.environ.get("JINA_API_EMBEDDING_URL", "https://api.jina.ai/v1/embeddings")
JINA_API_EMBEDDING_MODEL_NAME = os.environ.get("JINA_API_EMBEDDING_MODEL_NAME", "jina-embeddings-v2-base-en") # Defaulting to a common one

# --- Initialize Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s][%(lineno)d] %(message)s')
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('nltk').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING) # Jina API calls use requests
logger = logging.getLogger(__name__)

# --- Check Essential Variables ---
essential_vars = {
    "QDRANT_URL": QDRANT_URL, "GEMINI_API_KEY": GEMINI_API_KEY,
    "DATALAB_API_KEY": DATALAB_API_KEY, "DATALAB_MARKER_URL": DATALAB_MARKER_URL,
    "MOONDREAM_API_KEY": MOONDREAM_API_KEY,
    "JINA_API_KEY": JINA_API_KEY
}
missing_vars = [var_name for var_name, var_value in essential_vars.items() if not var_value]
if missing_vars:
    logger.critical(f"FATAL ERROR: Missing essential environment variables: {', '.join(missing_vars)}")
    sys.exit(f"Missing essential environment variables: {', '.join(missing_vars)}")

# --- Directories & Paths ---
SERVER_BASE_DIR = Path(__file__).parent
PERSISTENT_DATA_BASE_DIR = SERVER_BASE_DIR / "server_data_question_gen"
TEMP_UPLOAD_DIR = PERSISTENT_DATA_BASE_DIR / "temp_uploads"
JOB_DATA_DIR = PERSISTENT_DATA_BASE_DIR / "job_data"
PROMPT_DIR = SERVER_BASE_DIR / "content_prompts"

# --- Constants ---
DATALAB_POST_TIMEOUT = 180
DATALAB_POLL_TIMEOUT = 90
DATALAB_MAX_POLLS = 300
DATALAB_POLL_INTERVAL = 5
GEMINI_TIMEOUT = 300
MAX_GEMINI_RETRIES = 3
GEMINI_RETRY_DELAY = 60
JINA_API_TIMEOUT = 120 # Timeout for Jina API requests
QDRANT_COLLECTION_NAME = "markdown_docs_v3_semantic_qg"
EMBEDDING_MODEL_NAME = JINA_API_EMBEDDING_MODEL_NAME # Use the Jina API model name
VECTOR_SIZE = 768  # Dimension for jina-embeddings-v2-base-en. ADJUST IF USING A DIFFERENT JINA MODEL!
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:{action}?key={api_key}"
QSTS_THRESHOLD = 0.5
QUALITATIVE_METRICS = ["Understandable", "TopicRelated", "Grammatical", "Clear", "Central"]
MAX_INTERACTIVE_REGENERATION_ATTEMPTS = 15
MAX_HISTORY_TURNS = 10
ANSWER_RETRIEVAL_LIMIT = 5
MIN_CHUNK_SIZE_WORDS = 30
MAX_CHUNK_SIZE_WORDS = 300

# Prompt File Paths
FINAL_USER_PROMPT_PATH = PROMPT_DIR / "final_user_prompt.txt"
HYPOTHETICAL_PROMPT_PATH = PROMPT_DIR / "hypothetical_prompt.txt"
QUALITATIVE_EVAL_PROMPT_PATH = PROMPT_DIR / "qualitative_eval_prompt.txt"
ENHANCED_ANSWERABILITY_PROMPT_PATH = PROMPT_DIR / "enhanced_answerability_prompt.txt"

def ensure_server_dirs_and_prompts():
    for dir_path in [PERSISTENT_DATA_BASE_DIR, TEMP_UPLOAD_DIR, JOB_DATA_DIR, PROMPT_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")

    mandatory_prompts_list = [FINAL_USER_PROMPT_PATH, HYPOTHETICAL_PROMPT_PATH, QUALITATIVE_EVAL_PROMPT_PATH, ENHANCED_ANSWERABILITY_PROMPT_PATH]
    missing_prompts_actual = [p.name for p in mandatory_prompts_list if not p.exists()]
    if missing_prompts_actual:
        logger.critical(f"FATAL ERROR: Missing required prompt template files in '{PROMPT_DIR}': {', '.join(missing_prompts_actual)}.")
        sys.exit(f"Missing prompt files: {', '.join(missing_prompts_actual)}")

    if not ENHANCED_ANSWERABILITY_PROMPT_PATH.exists():
        logger.warning(f"'{ENHANCED_ANSWERABILITY_PROMPT_PATH.name}' not found in '{PROMPT_DIR}'. Creating a default template.")
        default_answerability_content = """
You are an expert evaluator assessing if a question is appropriately answerable for a specific student profile, given retrieved context snippets from a document they are expected to have studied.

**Student Profile:**
*   **Academic Level:** {academic_level}
*   **Major/Field:** {major}
*   **Course Name:** {course_name}
*   **Question Weight:** {marks_for_question} marks

**Question Details:**
*   **Generated Question:**
    ```
    {question}
    ```
*   **Target Bloom's Taxonomy Level:** {taxonomy_level}

**Instructions:**

1.  **Review the 'Context Snippets for Answering' below.** These were retrieved from the document based on the 'Generated Question'.
2.  **Consider the Student Profile, Bloom's Level, and Question Weight.** Assume the student has read the source material and can apply cognitive skills appropriate for the specified Bloom's level and the expected depth for the question's marks.
3.  **Judge Sufficiency:** Determine if the provided 'Context Snippets for Answering' contain *sufficient* information for this student to *derive* a complete and accurate answer.
    *   The context does **NOT** need to contain the answer verbatim.
    *   The context **MUST** provide the necessary building blocks, facts, concepts, or data points from which an answer can be constructed through reasoning, synthesis, or application of knowledge.
    *   Focus on whether the *foundational information* is present.
4.  **Output Format:** Respond ONLY with a valid JSON object containing two keys:
    *   `"is_answerable"`: `true` if the context provides the core concepts or data points necessary for the student to formulate a reasonable answer, even if it requires some synthesis or inference. `false` if critical information is clearly missing or the context is wholly irrelevant.
    *   `"reasoning"`: A concise string (2-3 sentences) explaining your judgment, specifically highlighting what information is present or absent.

**Context Snippets for Answering:**
(Top {answer_retrieval_limit} snippets retrieved based on the question itself)

{answer_context}

---
Respond now with the JSON object.
"""
        try:
            ENHANCED_ANSWERABILITY_PROMPT_PATH.write_text(default_answerability_content.strip(), encoding='utf-8')
            logger.info(f"Created default enhanced answerability prompt at: {ENHANCED_ANSWERABILITY_PROMPT_PATH}")
        except Exception as e:
            logger.critical(f"FATAL ERROR: Could not create default enhanced answerability prompt: {e}")
            sys.exit("Failed to create default prompt file.")
ensure_server_dirs_and_prompts()

# --- Global Model Initializations ---
qdrant_client: Optional[QdrantClient] = None
model_moondream: Optional[Any] = None
stop_words_nltk: Optional[set] = None
# No Jina client to initialize globally

def initialize_models():
    global qdrant_client, model_moondream, stop_words_nltk
    try:
        logger.info(f"Using Jina Embeddings API for model '{JINA_API_EMBEDDING_MODEL_NAME}' via URL '{JINA_API_EMBEDDING_URL}'.")
        # Test Jina API connectivity (optional but good practice)
        try:
            test_payload = {"input": ["test"], "model": JINA_API_EMBEDDING_MODEL_NAME}
            headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
            response = requests.post(JINA_API_EMBEDDING_URL, json=test_payload, headers=headers, timeout=10)
            response.raise_for_status()
            response_data = response.json()
            if not response_data.get("data") or not response_data["data"][0].get("embedding"):
                raise ValueError("Jina API test call returned unexpected data format.")
            test_embedding_dim = len(response_data["data"][0]["embedding"])
            if test_embedding_dim != VECTOR_SIZE:
                logger.critical(f"FATAL: Jina API test returned embedding dimension {test_embedding_dim}, but VECTOR_SIZE is {VECTOR_SIZE}. Please align these values.")
                sys.exit(f"Jina API embedding dimension mismatch. Expected {VECTOR_SIZE}, got {test_embedding_dim}.")
            logger.info("Jina Embeddings API connectivity test successful.")
        except requests.exceptions.RequestException as e:
            logger.critical(f"FATAL: Failed to connect or authenticate with Jina Embeddings API: {e}", exc_info=True)
            sys.exit("Jina API connectivity test failed.")
        except ValueError as e:
            logger.critical(f"FATAL: Jina API test call failed: {e}", exc_info=True)
            sys.exit("Jina API test call failed.")


        logger.info(f"Initializing Moondream model...")
        model_moondream = md_moondream.vl(api_key=MOONDREAM_API_KEY)
        logger.info("Moondream model initialized successfully.")

        logger.info(f"Initializing Qdrant client for URL: {QDRANT_URL}...")
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=180)
        try:
            collection_info = qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
            logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' found.")
            current_vector_size = None
            if isinstance(collection_info.config.params.vectors, dict): 
                default_vector_name = next(iter(collection_info.config.params.vectors)) # Assuming default unnamed or first named
                current_vector_size = collection_info.config.params.vectors[default_vector_name].size
            elif hasattr(collection_info.config.params.vectors, 'size'): 
                current_vector_size = collection_info.config.params.vectors.size

            if current_vector_size is not None and current_vector_size != VECTOR_SIZE:
                logger.warning(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' exists but has vector size {current_vector_size} instead of configured {VECTOR_SIZE}. Recreating collection.")
                qdrant_client.recreate_collection(
                    collection_name=QDRANT_COLLECTION_NAME,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
                )
                logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' recreated with new vector size {VECTOR_SIZE}.")
        except Exception as e: 
            err_str = str(e).lower()
            is_not_found_or_mismatch = any(keyword in err_str for keyword in ["not found", "status_code=404", "collection not found", "vector size mismatch", "vektör boyutu uyuşmazlığı", "not exist"])

            if is_not_found_or_mismatch or (hasattr(e, 'status_code') and e.status_code == 404):
                if "vector size mismatch" in err_str : 
                    logger.warning(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' vector size mismatch. Recreating...")
                else:
                    logger.warning(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' not found. Creating...")
                qdrant_client.recreate_collection(
                    collection_name=QDRANT_COLLECTION_NAME,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
                )
                logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created/recreated with vector size {VECTOR_SIZE}.")
            else:
                logger.error(f"Qdrant connection/access error for collection '{QDRANT_COLLECTION_NAME}': {e}", exc_info=True)
                raise Exception(f"Qdrant connection/access error: {e}")


        logger.info("Loading NLTK stopwords...")
        stop_words_nltk = set(stopwords.words('english'))
        logger.info("NLTK stopwords loaded.")

    except SystemExit: 
        raise
    except Exception as e:
        logger.critical(f"Fatal error during global model initialization: {e}", exc_info=True)
        sys.exit("Model initialization failed.")

# --- FastAPI App Setup ---
app = FastAPI(title="Interactive Question Generation API")
initialize_models()

origins = ["http://localhost:3000", "http://localhost:3001", "https://q-gen-frontend.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
job_status_storage: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models ---
class QuestionGenerationRequest(BaseModel):
    academic_level: str = "Undergraduate"
    major: str = "Computer Science"
    course_name: str = "Data Structures and Algorithms"
    taxonomy_level: str = Field("Evaluate", pattern="^(Remember|Understand|Apply|Analyze|Evaluate|Create)$")
    marks_for_question: str = Field("10", pattern="^(5|10|15|20)$")
    topics_list: str = "Breadth First Search, Shortest path"
    retrieval_limit_generation: int = Field(15, gt=0)
    similarity_threshold_generation: float = Field(0.4, ge=0.0, le=1.0)
    generate_diagrams: bool = False

class JobCreationResponse(BaseModel):
    job_id: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None
    error_details: Optional[str] = None
    job_params: Optional[QuestionGenerationRequest] = None
    original_filename: Optional[str] = None
    current_question: Optional[str] = None
    current_evaluations: Optional[Dict[str, Any]] = None
    regeneration_attempts_made: Optional[int] = None
    max_regeneration_attempts: Optional[int] = None
    final_result: Optional[Dict[str, Any]] = None

class RegenerationRequest(BaseModel):
    user_feedback: str

class FinalizeRequest(BaseModel):
    final_question: str

# --- Helper Functions ---

def get_bloom_guidance(level: str, job_id_for_log: str) -> str:
    logger.debug(f"[{job_id_for_log}] Getting Bloom's guidance for level: {level}")
    guidance = {
        "Remember": "Focus on recalling facts and basic concepts. Use verbs like: Define, List, Name, Recall, Repeat, State.",
        "Understand": "Focus on explaining ideas or concepts. Use verbs like: Classify, Describe, Discuss, Explain, Identify, Report, Select, Translate.",
        "Apply": "Focus on using information in new situations. Use verbs like: Apply, Choose, Demonstrate, Employ, Illustrate, Interpret, Solve, Use.",
        "Analyze": "Focus on drawing connections among ideas. Use verbs like: Analyze, Compare, Contrast, Differentiate, Examine, Organize, Relate, Test.",
        "Evaluate": "Focus on justifying a stand or decision. Use verbs like: Appraise, Argue, Defend, Judge, Justify, Critique, Support, Value.",
        "Create": "Focus on producing new or original work. Use verbs like: Assemble, Construct, Create, Design, Develop, Formulate, Generate, Invent."
    }
    return guidance.get(level, "No specific guidance available for this Bloom's level. Generate a thoughtful question appropriate for a university student.")

def fill_template_string(template_path: Path, placeholders: Dict[str, Any], job_id_for_log: str) -> str:
    log_placeholder_snippet = {
        k: (str(v)[:70] + "..." if len(str(v)) > 70 else str(v))
        for k, v in placeholders.items()
        if k in ["academic_level", "major", "course_name", "taxonomy_level", "marks_for_question", "topics_list", "question", "context", "retrieved_context", "answer_context"]
    }
    logger.debug(f"[{job_id_for_log}] Filling template: {template_path.name}. Placeholder snippet: {log_placeholder_snippet}")
    
    if not template_path.exists():
        logger.error(f"[{job_id_for_log}] Prompt template file not found: {template_path}")
        raise FileNotFoundError(f"Prompt template file not found: {template_path}")
    try:
        template_content = template_path.read_text(encoding="utf-8")
        for key, value in placeholders.items():
            template_content = template_content.replace(f"{{{key}}}", str(value))
        if "{" in template_content and "}" in template_content:
             unfilled_placeholders = re.findall(r"\{([\w_]+)\}", template_content)
             if unfilled_placeholders:
                logger.warning(f"[{job_id_for_log}] Template {template_path.name} may still have unfilled placeholders after processing: {unfilled_placeholders}. Original keys provided: {list(placeholders.keys())}")
        return template_content
    except Exception as e:
        logger.error(f"[{job_id_for_log}] Error filling template {template_path}: {e}", exc_info=True)
        raise

def get_gemini_response(
    job_id_for_log: str, system_prompt: Optional[str], user_prompt: str,
    conversation_history: List[Dict[str, Any]], temperature: float = 0.6,
    top_p: float = 0.9, top_k: int = 32, max_output_tokens: int = 8192,
) -> str:
    logger.info(f"[{job_id_for_log}] Calling Gemini API. History length: {len(conversation_history)}. User prompt (first 100 chars): {user_prompt[:100]}")
    if not GEMINI_API_KEY:
        logger.error(f"[{job_id_for_log}] GEMINI_API_KEY is not set.")
        return "Error: GEMINI_API_KEY not configured."

    api_url = GEMINI_API_URL_TEMPLATE.format(model_name=GEMINI_MODEL_NAME, action="generateContent", api_key=GEMINI_API_KEY)
    processed_history = conversation_history[-(MAX_HISTORY_TURNS*2):]
    contents = list(processed_history)
    contents.append({"role": "user", "parts": [{"text": user_prompt}]})

    payload = {
        "contents": contents,
        "generationConfig": {"temperature": temperature, "topP": top_p, "topK": top_k, "maxOutputTokens": max_output_tokens,},
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
    }
    if system_prompt and "1.5" in GEMINI_MODEL_NAME:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    elif system_prompt:
        logger.warning(f"[{job_id_for_log}] System prompt provided for non-1.5 model '{GEMINI_MODEL_NAME}'; ensure it's handled in conversation history or user prompt.")


    for attempt in range(MAX_GEMINI_RETRIES):
        try:
            response = requests.post(api_url, json=payload, timeout=GEMINI_TIMEOUT)
            response.raise_for_status()
            response_data = response.json()

            if "candidates" in response_data and response_data["candidates"]:
                candidate = response_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"] and candidate["content"]["parts"]:
                    model_response_text = candidate["content"]["parts"][0]["text"]
                    logger.info(f"[{job_id_for_log}] Gemini API call successful. Response (first 100 chars): {model_response_text[:100]}")
                    return model_response_text.strip()
                elif "finishReason" in candidate and candidate["finishReason"] != "STOP":
                    reason = candidate["finishReason"]
                    safety_ratings = candidate.get("safetyRatings", [])
                    logger.warning(f"[{job_id_for_log}] Gemini generation finished with reason: {reason}. Safety: {safety_ratings}")
                    if response_data.get("promptFeedback", {}).get("blockReason"):
                        block_reason = response_data["promptFeedback"]["blockReason"]
                        return f"Error: Gemini content blocked. Reason: {block_reason}. Details: {response_data['promptFeedback'].get('safetyRatings', [])}"
                    return f"Error: Gemini generation stopped. Reason: {reason}."

            logger.error(f"[{job_id_for_log}] Gemini API response malformed: {response_data}")
            return "Error: Malformed response from Gemini API."
        except requests.exceptions.Timeout:
            logger.warning(f"[{job_id_for_log}] Gemini API call timed out (attempt {attempt+1}/{MAX_GEMINI_RETRIES}).")
        except requests.exceptions.RequestException as e:
            logger.warning(f"[{job_id_for_log}] Gemini API request failed (attempt {attempt+1}/{MAX_GEMINI_RETRIES}): {e}")
            if e.response is not None:
                logger.warning(f"[{job_id_for_log}] Gemini API error response content: {e.response.text}")

        if attempt < MAX_GEMINI_RETRIES - 1:
            logger.info(f"[{job_id_for_log}] Retrying Gemini API call in {GEMINI_RETRY_DELAY} seconds...")
            time.sleep(GEMINI_RETRY_DELAY)
        else:
            logger.error(f"[{job_id_for_log}] Gemini API call failed after {MAX_GEMINI_RETRIES} retries.")
            return "Error: Gemini API call failed after multiple retries."
    return "Error: Gemini API call failed (exhausted retries - code path should ideally not be reached)."

def clean_text_for_embedding(text: str, job_id_for_log: str) -> str:
    global stop_words_nltk
    logger.debug(f"[{job_id_for_log}] Cleaning text for embedding (first 50 chars): {text[:50]}")
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text) # Remove HTML tags
    text = re.sub(r"[^\w\s\.\-\']", "", text) # Keep alphanumeric, whitespace, period, hyphen, apostrophe
    text = re.sub(r"\s+", " ", text).strip() # Normalize whitespace
    if stop_words_nltk:
        try:
            tokens = word_tokenize(text)
        except LookupError as e:
            logger.error(f"[{job_id_for_log}] NLTK LookupError during word_tokenize in clean_text_for_embedding: {e}. Ensure NLTK 'punkt' is downloaded.")
            if 'punkt' in str(e).lower():
                logger.warning(f"[{job_id_for_log}] NLTK 'punkt' resource missing. Skipping stopword removal.")
                return text # Return text as is if punkt is missing, to avoid complete failure
            raise # Re-raise if it's another LookupError
        tokens = [word for word in tokens if word not in stop_words_nltk and len(word) > 1]
        text = " ".join(tokens)
    return text

def get_embeddings_with_jina_api(texts: List[str], job_id_for_log: str) -> List[List[float]]:
    if not JINA_API_KEY:
        logger.error(f"[{job_id_for_log}] JINA_API_KEY is not configured.")
        raise ValueError("Jina API Key not configured.")
    if not texts:
        logger.warning(f"[{job_id_for_log}] No texts provided to get_embeddings_with_jina_api.")
        return []

    logger.info(f"[{job_id_for_log}] Requesting Jina API embeddings for {len(texts)} texts using model '{JINA_API_EMBEDDING_MODEL_NAME}' at '{JINA_API_EMBEDDING_URL}'.")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
    }
    payload = {
        "input": texts,
        "model": JINA_API_EMBEDDING_MODEL_NAME,
    }

    try:
        response = requests.post(JINA_API_EMBEDDING_URL, headers=headers, json=payload, timeout=JINA_API_TIMEOUT)
        response.raise_for_status() 
        
        response_data = response.json()
        embedding_data = response_data.get("data")

        if not embedding_data or not isinstance(embedding_data, list):
            logger.error(f"[{job_id_for_log}] Invalid or missing 'data' in response from Jina API: {response_data}")
            raise ValueError("Invalid embedding data format from Jina API.")
        
        embeddings = [item.get("embedding") for item in embedding_data]
        
        if not all(isinstance(emb, list) for emb in embeddings): # Ensure all items are lists
            logger.error(f"[{job_id_for_log}] One or more embeddings in Jina API response are not lists: {embeddings}")
            raise ValueError("Invalid embedding format within data from Jina API.")

        if len(embeddings) != len(texts):
            logger.error(f"[{job_id_for_log}] Mismatch in number of texts sent ({len(texts)}) and embeddings received ({len(embeddings)}).")
            raise ValueError("Mismatch in text/embedding count from Jina API.")

        if embeddings and embeddings[0] and len(embeddings[0]) != VECTOR_SIZE:
            dim_received = len(embeddings[0])
            logger.error(f"[{job_id_for_log}] Jina API returned embedding dimension {dim_received}, expected {VECTOR_SIZE} based on config. Model: '{JINA_API_EMBEDDING_MODEL_NAME}'.")
            raise ValueError(f"Jina API embedding dimension mismatch. Expected {VECTOR_SIZE}, got {dim_received}. Check JINA_API_EMBEDDING_MODEL_NAME and VECTOR_SIZE alignment.")

        logger.info(f"[{job_id_for_log}] Successfully received {len(embeddings)} embeddings from Jina API.")
        return embeddings
    except requests.exceptions.Timeout:
        logger.error(f"[{job_id_for_log}] Timeout connecting to Jina API at {JINA_API_EMBEDDING_URL}.")
        raise TimeoutError(f"Jina API request timed out for job {job_id_for_log}.")
    except requests.exceptions.RequestException as e:
        logger.error(f"[{job_id_for_log}] Error connecting to Jina API at {JINA_API_EMBEDDING_URL}: {e}", exc_info=True)
        if e.response is not None:
            logger.error(f"[{job_id_for_log}] Jina API error response content: {e.response.text}")
        raise 
    except ValueError as ve: 
        logger.error(f"[{job_id_for_log}] Data processing error with Jina API response: {ve}", exc_info=True)
        raise


def hierarchical_chunk_markdown(markdown_text: str, source_filename: str, job_id_for_log: str,
                                min_words: int = MIN_CHUNK_SIZE_WORDS, max_words: int = MAX_CHUNK_SIZE_WORDS) -> List[Dict]:
    logger.info(f"[{job_id_for_log}] Starting hierarchical chunking for {source_filename}.")
    chunks = []
    header_pattern = re.compile(r"^(#{1,4})\s+(.*)", re.MULTILINE) # Matches H1 to H4
    current_chunk_text = []
    current_header_trail = [] # Stores titles of current H1, H2, H3, H4 path
    chunk_index_counter = 0 # To give unique IDs to initially split chunks

    lines = markdown_text.splitlines()

    for i, line in enumerate(lines):
        header_match = header_pattern.match(line)
        if header_match:
            # New header found
            level = len(header_match.group(1)) # Level 1 for #, 2 for ##, etc.
            title = header_match.group(2).strip()

            # If there's content in current_chunk_text, finalize it as a chunk
            if current_chunk_text:
                full_text = "\n".join(current_chunk_text).strip()
                if len(full_text.split()) >= min_words: # Only add if it meets min word count
                    chunk_data = {
                        "text": full_text,
                        "metadata": {
                            "source_file": source_filename,
                            "header_trail": list(current_header_trail), # Copy of current trail
                            "chunk_index_original_split": chunk_index_counter,
                            "estimated_char_length": len(full_text),
                            "estimated_word_count": len(full_text.split())
                        }
                    }
                    chunks.append(chunk_data)
                    chunk_index_counter +=1
                current_chunk_text = [] # Reset for the new section

            # Update header trail for the new header
            current_header_trail = current_header_trail[:level-1] # Keep headers above current level
            current_header_trail.append(title)
            current_chunk_text.append(line) # Start new chunk with this header line
        else:
            # Regular line, add to current chunk
            current_chunk_text.append(line)

    # Add any remaining text as the last chunk
    if current_chunk_text:
        full_text = "\n".join(current_chunk_text).strip()
        if len(full_text.split()) >= min_words:
            chunks.append({
                "text": full_text,
                "metadata": {
                    "source_file": source_filename,
                    "header_trail": list(current_header_trail),
                    "chunk_index_original_split": chunk_index_counter,
                    "estimated_char_length": len(full_text),
                    "estimated_word_count": len(full_text.split())
                }
            })

    # Secondary split for chunks that are too large
    final_chunks = []
    for chunk_idx, chunk in enumerate(chunks):
        if chunk['metadata']['estimated_word_count'] > max_words:
            logger.debug(f"[{job_id_for_log}] Chunk from '{chunk['metadata']['header_trail']}' (orig_idx {chunk_idx}) is too large ({chunk['metadata']['estimated_word_count']} words), splitting by paragraphs.")
            paragraphs = chunk['text'].split('\n\n') # Simple paragraph split
            temp_sub_chunk_text = ""
            sub_chunk_id_counter_local = 0
            for para_idx, para in enumerate(paragraphs):
                para_words = len(para.split())
                current_sub_chunk_words = len(temp_sub_chunk_text.split())

                if temp_sub_chunk_text and (current_sub_chunk_words + para_words > max_words):
                    # Current sub-chunk + new paragraph would exceed max_words
                    if current_sub_chunk_words >= min_words: # Finalize current sub-chunk if valid
                        final_chunks.append({
                            "text": temp_sub_chunk_text.strip(),
                            "metadata": {**chunk['metadata'], "sub_chunk_id": sub_chunk_id_counter_local, "estimated_word_count": current_sub_chunk_words}
                        })
                        sub_chunk_id_counter_local += 1
                    temp_sub_chunk_text = para # Start new sub-chunk with current paragraph
                else:
                    # Add current paragraph to temp_sub_chunk_text
                    temp_sub_chunk_text = (temp_sub_chunk_text + "\n\n" + para).strip() if temp_sub_chunk_text else para
            
            # Add any remaining part of the sub-chunk
            if temp_sub_chunk_text.strip() and len(temp_sub_chunk_text.strip().split()) >= min_words :
                final_chunks.append({
                    "text": temp_sub_chunk_text.strip(),
                    "metadata": {**chunk['metadata'], "sub_chunk_id": sub_chunk_id_counter_local, "estimated_word_count": len(temp_sub_chunk_text.strip().split())}
                })
        elif chunk['metadata']['estimated_word_count'] >= min_words: # Chunk is within limits
            final_chunks.append(chunk)

    # Add a final_chunk_index to all resulting chunks for easier reference if needed
    for i, chk in enumerate(final_chunks):
        chk['metadata']['final_chunk_index'] = i

    logger.info(f"[{job_id_for_log}] Chunking for {source_filename} resulted in {len(final_chunks)} final chunks.")
    if not final_chunks and markdown_text.strip(): # Handle case where no chunks are made but text exists
        logger.warning(f"[{job_id_for_log}] No chunks generated for {source_filename}, but text exists. Creating a single chunk for the whole document.")
        return [{"text": markdown_text, "metadata": {"source_file": source_filename, "header_trail": ["Full Document"], "final_chunk_index": 0, "estimated_char_length": len(markdown_text), "estimated_word_count": len(markdown_text.split())}}]
    return final_chunks

def embed_chunks(chunks_data: List[Dict], job_id_for_log: str) -> List[List[float]]:
    logger.info(f"[{job_id_for_log}] Embedding {len(chunks_data)} chunks via Jina API.")
    
    texts_to_embed = [chunk['text'] for chunk in chunks_data]
    if not texts_to_embed: 
        logger.warning(f"[{job_id_for_log}] No texts to embed in embed_chunks.")
        return []
    
    cleaned_texts_to_embed = [clean_text_for_embedding(text, job_id_for_log) for text in texts_to_embed]

    try:
        embeddings = get_embeddings_with_jina_api(cleaned_texts_to_embed, job_id_for_log)
    except (ValueError, requests.exceptions.RequestException, TimeoutError, RuntimeError) as e:
        logger.error(f"[{job_id_for_log}] Failed to get embeddings from Jina API: {e}", exc_info=True)
        raise 
    
    logger.info(f"[{job_id_for_log}] Finished embedding {len(chunks_data)} chunks via Jina API.")
    return embeddings

def upsert_to_qdrant(job_id_for_log: str, collection_name: str, embeddings: List[List[float]],
                     chunks_data: List[Dict], batch_size: int = 100) -> int:
    global qdrant_client
    logger.info(f"[{job_id_for_log}] Upserting {len(chunks_data)} points to Qdrant collection '{collection_name}'.")
    if not qdrant_client:
        logger.error(f"[{job_id_for_log}] Qdrant client not initialized.")
        raise ValueError("Qdrant client not available.")
    if len(embeddings) != len(chunks_data):
        logger.error(f"[{job_id_for_log}] Mismatch between embeddings ({len(embeddings)}) and chunks_data ({len(chunks_data)}).")
        raise ValueError("Embeddings and chunks_data count mismatch.")

    points_to_upsert = []
    for i, chunk in enumerate(chunks_data):
        point_id = str(uuid.uuid4())
        # Ensure metadata is clean for Qdrant (e.g., convert Path objects to strings if any)
        payload = {"text": chunk['text'], "metadata": {k: str(v) if isinstance(v, Path) else v for k, v in chunk['metadata'].items()}}
        
        points_to_upsert.append(PointStruct(id=point_id, vector=embeddings[i], payload=payload))

    upserted_count = 0
    for i in range(0, len(points_to_upsert), batch_size):
        batch = points_to_upsert[i:i + batch_size]
        try:
            qdrant_client.upsert(collection_name=collection_name, points=batch, wait=True)
            upserted_count += len(batch)
            logger.debug(f"[{job_id_for_log}] Upserted batch of {len(batch)} points to Qdrant.")
        except Exception as e:
            logger.error(f"[{job_id_for_log}] Error upserting batch to Qdrant: {e}", exc_info=True)
            if "vector size mismatch" in str(e).lower() or "params.vectors.size" in str(e).lower():
                logger.critical(f"[{job_id_for_log}] QDRANT VECTOR SIZE MISMATCH! Collection expected different size than {VECTOR_SIZE}. This might require manual Qdrant collection deletion/recreation if server restart didn't fix it.")
            raise
    logger.info(f"[{job_id_for_log}] Successfully upserted {upserted_count} points to Qdrant collection '{collection_name}'.")
    return upserted_count

def find_topics_and_generate_hypothetical_text(job_id_for_log: str, academic_level: str, major: str,
                                             course_name: str, taxonomy_level: str, topics: str, marks_for_question: str) -> str:
    logger.info(f"[{job_id_for_log}] Generating hypothetical text for user-provided topics: '{topics}'")
    placeholders = {
        "academic_level": academic_level, "major": major, "course_name": course_name,
        "taxonomy_level": taxonomy_level, "topics": topics,
        "marks_for_question": marks_for_question,
        "bloom_guidance": get_bloom_guidance(taxonomy_level, job_id_for_log)
    }
    try:
        hypothetical_user_prompt = fill_template_string(HYPOTHETICAL_PROMPT_PATH, placeholders, job_id_for_log)
        hypothetical_system_prompt = "You are an AI assistant helping to generate a hypothetical search query based on student profile and topics."
        response_text = get_gemini_response(job_id_for_log, hypothetical_system_prompt, hypothetical_user_prompt, [])
        if response_text.startswith("Error:"):
            logger.error(f"[{job_id_for_log}] Failed to generate hypothetical text from Gemini: {response_text}")
            return f"Could not generate hypothetical text. Error: {response_text}"
        logger.info(f"[{job_id_for_log}] Successfully generated hypothetical text (first 100 chars): {response_text[:100]}")
        return response_text
    except FileNotFoundError:
        logger.error(f"[{job_id_for_log}] Hypothetical prompt file not found. Cannot generate text.")
        return "Error: Hypothetical prompt template file missing."
    except Exception as e:
        logger.error(f"[{job_id_for_log}] Error in find_topics_and_generate_hypothetical_text: {e}", exc_info=True)
        return f"Error generating hypothetical text: {str(e)}"

def search_qdrant(job_id_for_log: str, collection_name: str, embedded_vector: List[float],
                  query_text_for_log: str, limit: int, score_threshold: Optional[float] = None,
                  document_ids_filter: Optional[List[str]] = None,
                  session_id_filter: Optional[str] = None) -> List[ScoredPoint]:
    global qdrant_client
    logger.info(f"[{job_id_for_log}] Searching Qdrant collection '{collection_name}' for query (log): {query_text_for_log[:100]}. Limit: {limit}, Threshold: {score_threshold}, DocIDs: {document_ids_filter}, SessionID: {session_id_filter}")
    if not qdrant_client:
        logger.error(f"[{job_id_for_log}] Qdrant client not initialized.")
        raise ValueError("Qdrant client not available.")
    
    q_filter_conditions = []
    if document_ids_filter:
        q_filter_conditions.append(FieldCondition(key="metadata.document_id", match=MatchAny(any=document_ids_filter)))
    if session_id_filter:
        q_filter_conditions.append(FieldCondition(key="metadata.session_id", match=MatchValue(value=session_id_filter)))
    
    final_filter = models.Filter(must=q_filter_conditions) if q_filter_conditions else None
    
    logger.debug(f"[{job_id_for_log}] Qdrant search final_filter: {final_filter.model_dump_json(indent=2) if final_filter else 'None'}")
    try:
        search_results = qdrant_client.search(
            collection_name=collection_name, 
            query_vector=embedded_vector, 
            query_filter=final_filter,
            limit=limit, 
            score_threshold=score_threshold
        )
        logger.info(f"[{job_id_for_log}] Qdrant search returned {len(search_results)} results.")
        # Log details of first few results for debugging
        for i, res in enumerate(search_results[:3]):
            logger.debug(f"[{job_id_for_log}] Search Result {i+1}: Score={res.score:.4f}, ID={res.id}, Metadata={res.payload.get('metadata') if res.payload else None}")
        return search_results
    except Exception as e:
        logger.error(f"[{job_id_for_log}] Error searching Qdrant: {e}", exc_info=True)
        raise

def parse_questions_from_llm_response(job_id_for_log: str, question_block: str, num_expected: int = 1) -> List[str]:
    logger.debug(f"[{job_id_for_log}] Parsing questions from LLM response (first 100 chars): {question_block[:100]}")
    lines = [line.strip() for line in question_block.splitlines() if line.strip()]
    question_prefixes = ["Question:", "Q:", "Generated Question:"] # Common prefixes
    cleaned_questions = []

    for line in lines:
        original_line = line # Keep original for later if no prefix match
        for prefix in question_prefixes:
            if line.lower().startswith(prefix.lower()): # Case-insensitive prefix match
                line = line[len(prefix):].strip()
                break
        # Remove common list markers like "1. ", "a) "
        line = re.sub(r"^\s*\d+\.\s*", "", line)
        line = re.sub(r"^\s*[a-zA-Z]\)\s*", "", line)
        line = line.strip() # Final strip
        if line: 
            cleaned_questions.append(line)
        elif original_line and not any(original_line.lower().startswith(p.lower()) for p in question_prefixes) and num_expected == 1:
            # If no prefix matched, and we expect only one question, consider the original non-empty line
            # after basic list marker removal as a potential candidate if it wasn't just a prefix itself.
            candidate_line = re.sub(r"^\s*\d+\.\s*", "", original_line).strip()
            candidate_line = re.sub(r"^\s*[a-zA-Z]\)\s*", "", candidate_line).strip()
            if candidate_line and not any(p.strip().lower() == candidate_line.lower() for p in question_prefixes):
                 cleaned_questions.append(candidate_line)


    if not cleaned_questions and lines and num_expected == 1:
        # Fallback: If no questions were extracted using prefixes, and we expect one,
        # take the first non-empty line after cleaning list markers, if it seems substantial.
        logger.warning(f"[{job_id_for_log}] No question prefixes matched. Using first cleaned non-empty line as question if available. Original lines: {lines}")
        first_line_cleaned = re.sub(r"^\s*\d+\.\s*", "", lines[0]).strip()
        first_line_cleaned = re.sub(r"^\s*[a-zA-Z]\)\s*", "", first_line_cleaned).strip()
        if first_line_cleaned: 
            cleaned_questions = [first_line_cleaned]

    final_questions = cleaned_questions[:num_expected] if cleaned_questions else []
    
    if not final_questions and question_block.strip(): # If still no questions, and original block wasn't empty
        logger.warning(f"[{job_id_for_log}] Could not parse distinct questions using prefixes or simple fallbacks. Raw block: {question_block}")
        if num_expected == 1: # If only one question is expected, use the whole block as a last resort
            final_questions = [question_block.strip()]

    logger.info(f"[{job_id_for_log}] Parsed {len(final_questions)} questions. First: {final_questions[0][:100] if final_questions else 'None'}")
    return final_questions

def evaluate_question_qsts(job_id_for_log: str, question: str, context: str) -> float:
    logger.info(f"[{job_id_for_log}] Evaluating QSTS for question (first 100 chars): {question[:100]}")
    if not question or not context: 
        logger.warning(f"[{job_id_for_log}] Empty question or context for QSTS. Returning 0.0")
        return 0.0
    
    try:
        cleaned_question = clean_text_for_embedding(question, job_id_for_log)
        cleaned_context = clean_text_for_embedding(context, job_id_for_log)
        
        if not cleaned_question or not cleaned_context:
            logger.warning(f"[{job_id_for_log}] QSTS: Question or context became empty after cleaning. Q: '{question[:50]}', Ctx: '{context[:50]}'. Returning 0.0")
            return 0.0
            
        embeddings_list_of_lists = get_embeddings_with_jina_api([cleaned_question, cleaned_context], job_id_for_log)
        
        if len(embeddings_list_of_lists) != 2 or not embeddings_list_of_lists[0] or not embeddings_list_of_lists[1]:
            logger.error(f"[{job_id_for_log}] QSTS: Expected 2 valid embeddings from Jina API, got {len(embeddings_list_of_lists)} (or empty embeddings).")
            return 0.0

        q_embed_list = embeddings_list_of_lists[0]
        c_embed_list = embeddings_list_of_lists[1]

        q_embed_tensor = torch.tensor(q_embed_list, dtype=torch.float32).unsqueeze(0) 
        c_embed_tensor = torch.tensor(c_embed_list, dtype=torch.float32).unsqueeze(0) 
        
        score = sbert_util.pytorch_cos_sim(q_embed_tensor, c_embed_tensor).item()
        logger.info(f"[{job_id_for_log}] QSTS score: {score:.4f}")
        return score
    except (ValueError, requests.exceptions.RequestException, TimeoutError, RuntimeError) as e_embed:
        logger.error(f"[{job_id_for_log}] QSTS: Failed to get embeddings via Jina API: {e_embed}", exc_info=True)
        return 0.0
    except Exception as e: 
        logger.error(f"[{job_id_for_log}] Error during QSTS calculation: {e}", exc_info=True)
        return 0.0

def evaluate_question_qualitative_llm(job_id_for_log: str, question: str, context_for_eval: str,
                                   academic_level: str, major: str, course_name: str, taxonomy_level: str, marks_for_question: str) -> Dict[str, Any]:
    logger.info(f"[{job_id_for_log}] Performing qualitative LLM evaluation for question (first 100 chars): {question[:100]}")
    placeholders = {
        "question": question, "context": context_for_eval, "academic_level": academic_level,
        "major": major, "course_name": course_name, "taxonomy_level": taxonomy_level,
        "marks_for_question": marks_for_question,
        "bloom_guidance": get_bloom_guidance(taxonomy_level, job_id_for_log)
    }
    default_error_return = {**{metric: False for metric in QUALITATIVE_METRICS}} # Default to False for all metrics on error
    try:
        eval_user_prompt = fill_template_string(QUALITATIVE_EVAL_PROMPT_PATH, placeholders, job_id_for_log)
        eval_system_prompt = "You are an expert AI assistant evaluating the quality of a generated question based on provided criteria and context."
        response_text = get_gemini_response(job_id_for_log, eval_system_prompt, eval_user_prompt, [])

        if response_text.startswith("Error:"):
            logger.error(f"[{job_id_for_log}] Gemini call failed for qualitative eval: {response_text}")
            return {**default_error_return, "error_message": response_text, "llm_raw_response": response_text}
        
        logger.debug(f"[{job_id_for_log}] Qualitative eval - Raw LLM response:\n{response_text}")
        try:
            # Try to find JSON block, leniently
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```|({[\s\S]*})", response_text, re.DOTALL | re.MULTILINE)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                eval_results = json.loads(json_str)
                # Ensure all defined qualitative metrics are present, defaulting to False if missing
                final_results = {metric: eval_results.get(metric, False) for metric in QUALITATIVE_METRICS}
                final_results["reasoning"] = eval_results.get("reasoning", "No reasoning provided by LLM.") # Capture reasoning if available
                logger.info(f"[{job_id_for_log}] Qualitative LLM evaluation results: {final_results}")
                return final_results
            else:
                logger.warning(f"[{job_id_for_log}] No JSON block found in qualitative eval response: {response_text}")
                return {**default_error_return, "error_message": "No JSON in LLM response for qualitative eval.", "llm_raw_response": response_text}
        except json.JSONDecodeError as jde:
            logger.error(f"[{job_id_for_log}] Failed to decode JSON from qualitative eval. Error: {jde}. Response: {response_text}", exc_info=True)
            return {**default_error_return, "error_message": f"JSON decode error in qualitative eval: {str(jde)}", "llm_raw_response": response_text}
    except Exception as e:
        logger.error(f"[{job_id_for_log}] Error in evaluate_question_qualitative_llm: {e}", exc_info=True)
        return {**default_error_return, "error_message": str(e)}

def evaluate_question_answerability_llm(job_id_for_log: str, question: str, academic_level: str, major: str,
                                     course_name: str, taxonomy_level: str, marks_for_question: str,
                                     document_ids_filter: List[str], session_id_filter: str
                                     ) -> Tuple[bool, str, List[Dict]]:
    logger.info(f"[{job_id_for_log}] Performing enhanced answerability LLM evaluation for question (first 100 chars): '{question[:100]}'")
    logger.debug(f"[{job_id_for_log}] Answerability - Student Profile: Level='{academic_level}', Major='{major}', Course='{course_name}', Marks='{marks_for_question}', Bloom='{taxonomy_level}'")
    ans_context_metadata = []
    try:
        cleaned_question_for_embedding = clean_text_for_embedding(question, job_id_for_log)
        if not cleaned_question_for_embedding:
            logger.warning(f"[{job_id_for_log}] Answerability: Question became empty after cleaning: '{question[:50]}'. Returning not answerable.")
            return False, "Question became empty after cleaning, cannot embed.", []

        embedding_response_list_of_lists = get_embeddings_with_jina_api([cleaned_question_for_embedding], job_id_for_log)
        
        if not embedding_response_list_of_lists or len(embedding_response_list_of_lists) != 1 or not embedding_response_list_of_lists[0]:
            logger.error(f"[{job_id_for_log}] Answerability: Failed to get valid embedding for the question via Jina API.")
            return False, "Error: Could not embed question for answerability check (Jina API).", []

        question_embedding_vector = embedding_response_list_of_lists[0]
        
        answer_search_results = search_qdrant(
            job_id_for_log=job_id_for_log, collection_name=QDRANT_COLLECTION_NAME,
            embedded_vector=question_embedding_vector, query_text_for_log=f"Answerability search: {question[:50]}",
            limit=ANSWER_RETRIEVAL_LIMIT, document_ids_filter=document_ids_filter, session_id_filter=session_id_filter
        )
        if not answer_search_results: 
            logger.warning(f"[{job_id_for_log}] Answerability: No relevant context found in document for question '{question[:100]}'. Filters: doc_ids={document_ids_filter}, session_id={session_id_filter}. Returning not answerable.")
            return False, "No relevant context found in document to answer this question.", []
        
        ans_context_parts = [f"Snippet {idx+1} (Score: {res.score:.4f}):\n{res.payload.get('text', '')}" for idx, res in enumerate(answer_search_results)]
        ans_context_for_llm = "\n\n---\n\n".join(filter(None, ans_context_parts))
        ans_context_metadata = [res.payload for res in answer_search_results] # Store full payload for potential UI display
        
        logger.debug(f"[{job_id_for_log}] Answerability - Context for LLM (first 300 chars of total {len(ans_context_for_llm)} chars):\n{ans_context_for_llm[:300]}")
        
        placeholders = {
            "academic_level": academic_level, "major": major, "course_name": course_name,
            "question": question, "taxonomy_level": taxonomy_level, "marks_for_question": marks_for_question,
            "answer_retrieval_limit": ANSWER_RETRIEVAL_LIMIT, "answer_context": ans_context_for_llm
        }
        user_prompt = fill_template_string(ENHANCED_ANSWERABILITY_PROMPT_PATH, placeholders, job_id_for_log)
        system_prompt = "You are an expert AI evaluating question answerability from context, adhering strictly to JSON output format." # Slightly more directive system prompt
        
        logger.debug(f"[{job_id_for_log}] Answerability - User prompt for LLM (first 300 chars):\n{user_prompt[:300]}")
        response_text = get_gemini_response(job_id_for_log, system_prompt, user_prompt, [])
        logger.debug(f"[{job_id_for_log}] Answerability - Raw LLM response:\n{response_text}")


        if response_text.startswith("Error:"):
            logger.error(f"[{job_id_for_log}] Answerability - LLM call failed: {response_text}")
            return False, f"LLM call failed for answerability: {response_text}", ans_context_metadata
        try:
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```|({[\s\S]*})", response_text, re.DOTALL | re.MULTILINE)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                eval_data = json.loads(json_str)
                is_answerable = eval_data.get("is_answerable", False) # Default to False if key missing
                reasoning = eval_data.get("reasoning", "No reasoning provided by LLM.")
                logger.info(f"[{job_id_for_log}] Answerability Eval - Parsed is_answerable: {is_answerable}, Reasoning (first 100): {reasoning[:100]}")
                return is_answerable, reasoning, ans_context_metadata
            else:
                logger.warning(f"[{job_id_for_log}] Answerability - LLM response did not contain expected JSON. Response: {response_text}")
                return False, "LLM response for answerability not in expected JSON format.", ans_context_metadata
        except json.JSONDecodeError as jde:
            logger.error(f"[{job_id_for_log}] Answerability - Failed to parse LLM's JSON. Error: {jde}. Response: {response_text}", exc_info=True)
            return False, "Failed to parse LLM's JSON response for answerability.", ans_context_metadata
    except (ValueError, requests.exceptions.RequestException, TimeoutError, RuntimeError) as e_embed: # Catch Jina client errors
        logger.error(f"[{job_id_for_log}] Answerability: Failed to get Jina API embeddings for question: {e_embed}", exc_info=True)
        return False, f"Error getting question embedding for answerability (Jina API): {str(e_embed)}", []
    except Exception as e:
        logger.error(f"[{job_id_for_log}] Unexpected error in answerability_eval: {e}", exc_info=True)
        return False, f"Unexpected error in answerability_eval: {str(e)}", ans_context_metadata

# --- Datalab, Moondream, File Processing ---
def call_datalab_marker(file_path: Path, job_id_for_log: str) -> Dict[str, Any]:
    logger.info(f"[{job_id_for_log}] Attempting to call Datalab Marker API for {file_path.name} at path {file_path}")
    if not DATALAB_API_KEY or not DATALAB_MARKER_URL:
        raise ValueError("Datalab API Key or URL not configured.")
    if not file_path.exists():
        raise FileNotFoundError(f"Datalab input file not found: {file_path}")
    if file_path.stat().st_size == 0:
        raise ValueError(f"Input file for Datalab is empty: {file_path.name}")

    try:
        with open(file_path, "rb") as f_read: file_content_bytes = f_read.read()
        if not file_content_bytes: raise ValueError(f"Failed to read content from Datalab input file: {file_path.name}")

        files_payload = {"file": (file_path.name, file_content_bytes, "application/pdf")} # Ensure correct MIME type
        form_data = {"output_format": (None, "markdown"), "disable_image_extraction": (None, "false")}
        headers = {"X-Api-Key": DATALAB_API_KEY}
        
        logger.info(f"[{job_id_for_log}] Posting to Datalab Marker URL: {DATALAB_MARKER_URL}")
        response = requests.post(DATALAB_MARKER_URL, files=files_payload, data=form_data, headers=headers, timeout=DATALAB_POST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        logger.info(f"[{job_id_for_log}] Datalab initial response: {data}")
    except requests.exceptions.RequestException as e:
        error_text = e.response.text if e.response else "No response text"
        logger.error(f"Datalab API request failed for {file_path.name}: {e} - Response: {error_text}", exc_info=True)
        raise Exception(f"Datalab API request failed: {e} - {error_text}") from e


    if not data.get("success"): raise Exception(f"Datalab API error: {data.get('error', 'Unknown error')}")
    check_url = data["request_check_url"]
    logger.info(f"[{job_id_for_log}] Polling Datalab check URL: {check_url}")
    for poll_num in range(DATALAB_MAX_POLLS):
        time.sleep(DATALAB_POLL_INTERVAL)
        try:
            poll_resp = requests.get(check_url, headers=headers, timeout=DATALAB_POLL_TIMEOUT)
            poll_resp.raise_for_status(); poll_data = poll_resp.json()
            logger.info(f"[{job_id_for_log}] Datalab poll {poll_num+1}/{DATALAB_MAX_POLLS}: status {poll_data.get('status')}, progress {poll_data.get('progress_percent')}%")
            if poll_data.get("status") == "complete": 
                logger.info(f"[{job_id_for_log}] Datalab processing complete.")
                return {"markdown": poll_data.get("markdown", ""), "images": poll_data.get("images", {})}
            if poll_data.get("status") == "error": 
                raise Exception(f"Datalab processing failed: {poll_data.get('error', 'Unknown error')}")
        except requests.exceptions.RequestException as e_poll: 
            logger.warning(f"[{job_id_for_log}] Polling Datalab error (attempt {poll_num+1}): {e_poll}. Retrying...")
    raise TimeoutError(f"Polling timed out for Datalab processing of {file_path.name}.")

def generate_moondream_image_description(image_path: Path, caption_text: str = "") -> str:
    global model_moondream
    if not model_moondream: 
        logger.error(f"Moondream model not initialized. Cannot describe image {image_path.name}.")
        return "Error: Moondream model not available."
    logger.info(f"Generating Moondream description for image: {image_path.name}, caption hint: '{caption_text[:50]}'")
    try:
        image = Image.open(image_path)
        if image.mode != "RGB": image = image.convert("RGB") # Moondream might prefer RGB
        
        prompt_parts = ["Describe key technical findings, data, or important visual elements in this figure/visualization."]
        if caption_text:
            prompt_parts.append(f"The provided caption/context for this image is: '{caption_text}'. Use this to refine your description if relevant.")
        prompt_parts.append("Focus on information that would be useful for understanding its content in an academic context.")
        prompt = " ".join(prompt_parts)
        
        encoded_image = model_moondream.encode_image(image)
        # Assuming query method takes encoded_image and prompt directly.
        # The moondream library's API might vary; adjust if 'query' has different signature.
        response_dict_or_str = model_moondream.query(encoded_image, prompt) # Or `answer_question` if that's the method
        
        desc = ""
        if isinstance(response_dict_or_str, dict):
            desc = response_dict_or_str.get("answer", response_dict_or_str.get("text", "Error: No 'answer' or 'text' key in Moondream response."))
        elif isinstance(response_dict_or_str, str):
            desc = response_dict_or_str
        else:
            desc = f"Error: Unexpected response type from Moondream: {type(response_dict_or_str)}"

        desc = desc.replace('\n', ' ').strip()
        if not desc or desc.startswith("Error:"): 
            logger.warning(f"Moondream [{image_path.name}]: No valid description generated or error returned. Response: {desc}")
            return "No valid description generated." if not desc.startswith("Error:") else desc
        logger.info(f"Moondream [{image_path.name}] desc (first 100 chars): {desc[:100]}")
        return desc
    except Exception as e: 
        logger.error(f"Moondream [{image_path.name}]: Error during description generation: {e}", exc_info=True)
        return f"Error generating description: {str(e)}"

def save_extracted_images_api(images_dict: Dict[str,str], images_folder: Path, job_id_for_log: str) -> Dict[str, str]:
    images_folder.mkdir(parents=True, exist_ok=True)
    saved_files_map = {} # Maps original name from MD to new path on disk
    for name_in_md, b64_data in images_dict.items():
        try:
            # Sanitize name for filesystem and ensure unique
            original_path_obj = Path(name_in_md)
            base = original_path_obj.stem
            suffix = original_path_obj.suffix if original_path_obj.suffix else ".png" # Default to .png if no suffix
            
            safe_base = "".join(c for c in base if c.isalnum() or c in ('-', '_')).strip()[:50] # Limit length
            if not safe_base: safe_base = f"img_{uuid.uuid4().hex[:6]}" # Ensure base is not empty

            counter = 0
            disk_name = f"{safe_base}{suffix}"
            disk_path = images_folder / disk_name
            while disk_path.exists(): # Avoid overwriting
                counter += 1
                disk_name = f"{safe_base}_{counter}{suffix}"
                disk_path = images_folder / disk_name
            
            with open(disk_path, "wb") as img_file:
                img_file.write(base64.b64decode(b64_data))
            saved_files_map[name_in_md] = str(disk_path) # Store original name as key
            logger.info(f"[{job_id_for_log}] Saved image '{name_in_md}' to '{disk_path}'")
        except Exception as e:
            logger.warning(f"[{job_id_for_log}] Could not save image '{name_in_md}': {e}")
    return saved_files_map

def process_markdown_api(markdown_text: str, saved_images_map: Dict[str, str], job_id_for_log: str) -> str:
    logger.info(f"[{job_id_for_log}] Processing markdown for image descriptions... Found {len(saved_images_map)} saved images to map.")
    lines = markdown_text.splitlines()
    processed_lines = []
    img_count = 0 # For untitled figures

    # Regex for typical Markdown image syntax: ![alt text](path "title")
    # We care about the path and potentially alt text or a nearby caption.
    fig_pattern = re.compile(r"^\s*!\[(?P<alt>.*?)\]\((?P<path>[^)]+?)\s*(?:\"(?P<title>.*?)\")?\)\s*$")
    # Regex for captions like "Figure 1: Description" or "Table 2. Details"
    # This is a simple heuristic; complex caption finding is hard.
    cap_pattern = re.compile(r"^\s*(Figure|Table|Chart|Fig\.?|Tbl\.?)\s?([\w\d\.\-]+[:.]?)\s?(.*)", re.IGNORECASE)

    i = 0
    while i < len(lines):
        line = lines[i]
        match = fig_pattern.match(line.strip()) # Check current line for image pattern

        if match:
            img_count += 1
            alt_text = match.group("alt")
            image_path_in_md_encoded = match.group("path")
            image_path_in_md_decoded = urllib.parse.unquote(image_path_in_md_encoded)
            
            logger.debug(f"[{job_id_for_log}] Found image ref: Alt='{alt_text}', Path Enc='{image_path_in_md_encoded}', Path Dec='{image_path_in_md_decoded}'")

            # Attempt to find a caption on the next non-empty line(s)
            caption_text_for_moondream = alt_text # Default to alt text
            potential_caption_line_index = -1

            # Look ahead a couple of lines for a caption pattern
            for k_lookahead in range(1, 3): # Check next 2 lines
                next_line_idx = i + k_lookahead
                if next_line_idx < len(lines):
                    next_line_content = lines[next_line_idx].strip()
                    if not next_line_content: # Skip empty lines
                        continue 
                    if cap_pattern.match(next_line_content):
                        logger.debug(f"[{job_id_for_log}] Found potential caption for '{image_path_in_md_decoded}': '{next_line_content}'")
                        caption_text_for_moondream = next_line_content # Use this as the caption
                        potential_caption_line_index = next_line_idx
                        break # Found a caption, stop looking ahead
                    else:
                        # If the next line is not empty and not a caption, assume no immediate caption
                        break 
                else:
                    break # End of document

            # Get the disk path from our map
            actual_disk_path_str = saved_images_map.get(image_path_in_md_decoded) or saved_images_map.get(image_path_in_md_encoded)
            moondream_description = ""

            if actual_disk_path_str:
                disk_path_obj = Path(actual_disk_path_str)
                if disk_path_obj.exists():
                    moondream_description = generate_moondream_image_description(disk_path_obj, caption_text_for_moondream)
                else:
                    logger.warning(f"[{job_id_for_log}] Image file '{disk_path_obj.name}' (from '{image_path_in_md_decoded}') not found on disk at '{actual_disk_path_str}'.")
                    moondream_description = f"*Error: Image file '{disk_path_obj.name}' not found on disk.*"
            else:
                logger.warning(f"[{job_id_for_log}] Image path '{image_path_in_md_decoded}' (or encoded '{image_path_in_md_encoded}') not found in saved_images_map.")
                moondream_description = f"*Error: Image '{image_path_in_md_decoded}' not found in pre-processed image map.*"
            
            # Construct the replacement block
            # Use identified caption or a generic title if no good caption
            figure_title_text = caption_text_for_moondream if cap_pattern.match(caption_text_for_moondream) else f"Figure {img_count}"

            image_block = f"\n---\n### {figure_title_text}\n\n"
            image_block += f"**Original Markdown Reference:** `![{alt_text}]({image_path_in_md_encoded})`\n"
            if caption_text_for_moondream != alt_text and not cap_pattern.match(caption_text_for_moondream): # If alt_text was used but wasn't a formal caption
                 image_block += f"**Context/Alt Text:** {alt_text}\n"
            image_block += f"\n**Moondream AI Description:**\n{moondream_description}\n---\n"
            
            processed_lines.append(image_block)

            if potential_caption_line_index > i : # If we used a subsequent line as caption, skip it
                i = potential_caption_line_index
            # else, we just processed line 'i' (the image line itself)
        else:
            # Not an image line, just append it
            processed_lines.append(line)
        
        i += 1 # Move to next line

    logger.info(f"[{job_id_for_log}] Finished processing markdown. Processed {img_count} image references.")
    return "\n".join(processed_lines)

# --- Main Background Task Logic ---
def process_document_and_generate_first_question(
    job_id: str, pdf_path_on_disk: Path, original_filename: str,
    params: QuestionGenerationRequest, job_specific_temp_dir: Path
):
    global job_status_storage
    job_status_storage[job_id]["status"] = "processing_setup"
    job_status_storage[job_id]["message"] = "Preparing document..."
    
    current_job_data_dir = JOB_DATA_DIR / job_id
    current_job_data_dir.mkdir(parents=True, exist_ok=True)
    job_images_dir = job_specific_temp_dir / "images_from_datalab" # More specific name
    final_md_path = current_job_data_dir / f"{job_id}_{Path(original_filename).stem}_processed.md"

    try:
        if not pdf_path_on_disk.exists() or pdf_path_on_disk.stat().st_size == 0:
            raise FileNotFoundError(f"Input PDF {pdf_path_on_disk.name} is missing or empty before Datalab call.")
        logger.info(f"[{job_id}] Starting Datalab processing for {pdf_path_on_disk.name}")
        job_status_storage[job_id]["message"] = "Extracting content (Datalab)... This may take several minutes."
        marker_result = call_datalab_marker(pdf_path_on_disk, job_id)
        raw_md = marker_result.get("markdown", "")
        img_dict = marker_result.get("images", {}) # This is {filename_in_md: base64_data}
        
        if not raw_md.strip(): 
            logger.error(f"[{job_id}] Datalab returned empty markdown content for {original_filename}.")
            raise ValueError("Markdown content is empty after Datalab processing.")
        logger.info(f"[{job_id}] Datalab completed. Markdown length: {len(raw_md)}, Images found: {len(img_dict)}")

        job_status_storage[job_id]["message"] = "Processing images with Moondream & enriching markdown..."
        job_images_dir.mkdir(parents=True, exist_ok=True)
        # saved_imgs_map is {filename_in_md: path_on_disk_where_saved}
        saved_imgs_map = save_extracted_images_api(img_dict, job_images_dir, job_id)
        processed_md = process_markdown_api(raw_md, saved_imgs_map, job_id)
        final_md_path.write_text(processed_md, encoding="utf-8")
        logger.info(f"[{job_id}] Processed markdown saved to {final_md_path}")
        
        job_status_storage[job_id].update({
            "processed_markdown_path_relative": str(final_md_path.relative_to(PERSISTENT_DATA_BASE_DIR)),
            "processed_markdown_filename_on_server": final_md_path.name
        })
        if job_images_dir.exists(): # Clean up temp images dir after processing MD
            shutil.rmtree(job_images_dir) 
            logger.info(f"[{job_id}] Cleaned up temporary Datalab images directory: {job_images_dir}")

        job_status_storage[job_id]["message"] = "Chunking document, generating embeddings (Jina API), and upserting to Qdrant..."
        # Unique document ID for Qdrant, incorporating job_id for session scoping
        doc_id_qdrant = f"doc_{job_id}_{Path(original_filename).stem.replace('.', '_').replace(' ', '_')}"
        job_status_storage[job_id]["document_id_for_qdrant"] = doc_id_qdrant
        
        chunks = hierarchical_chunk_markdown(processed_md, original_filename, job_id)
        if not chunks: 
            logger.error(f"[{job_id}] No chunks were generated from the processed markdown of {original_filename}.")
            raise ValueError("No chunks generated from document. Cannot proceed with embedding.")
        
        # Add document_id and session_id to each chunk's metadata for Qdrant filtering
        for chunk_data in chunks:
            chunk_data.setdefault('metadata', {}) # Ensure metadata dict exists
            chunk_data['metadata']['document_id'] = doc_id_qdrant
            chunk_data['metadata']['session_id'] = job_id # Use job_id as a session_id
        
        embeddings = embed_chunks(chunks, job_id) 
        if not embeddings: 
            logger.error(f"[{job_id}] Embedding process returned no embeddings for {original_filename}.")
            raise ValueError("Embedding failed or returned no embeddings (Jina API).")
        
        upsert_count = upsert_to_qdrant(job_id, QDRANT_COLLECTION_NAME, embeddings, chunks)
        if upsert_count == 0:
            logger.error(f"[{job_id}] No points were upserted to Qdrant for {original_filename}.")
            raise ValueError("No points upserted to Qdrant. Check chunking or Qdrant setup.")
        logger.info(f"[{job_id}] Successfully upserted {upsert_count} chunks to Qdrant for document {doc_id_qdrant}.")

        job_status_storage[job_id]["status"] = "generating_initial_question"
        job_status_storage[job_id]["message"] = "Generating hypothetical query for context retrieval..."
        hypo_text = find_topics_and_generate_hypothetical_text(
            job_id, params.academic_level, params.major, params.course_name,
            params.taxonomy_level, params.topics_list, params.marks_for_question
        )
        if hypo_text.startswith("Error:") or not hypo_text.strip(): 
            logger.error(f"[{job_id}] Hypothetical text generation failed: {hypo_text}")
            raise ValueError(f"Hypothetical text generation failed: {hypo_text}")
        
        cleaned_hypo_text = clean_text_for_embedding(hypo_text, job_id)
        if not cleaned_hypo_text:
            raise ValueError(f"Hypothetical text became empty after cleaning: '{hypo_text[:100]}'")

        hypo_embedding_response_list_of_lists = get_embeddings_with_jina_api([cleaned_hypo_text], job_id)
        if not hypo_embedding_response_list_of_lists or len(hypo_embedding_response_list_of_lists) != 1 or not hypo_embedding_response_list_of_lists[0]:
            logger.error(f"[{job_id}] Failed to get a valid Jina API embedding for the hypothetical text.")
            raise ValueError("Failed to get valid Jina API embedding for hypothetical text.")
        query_embed = hypo_embedding_response_list_of_lists[0]


        job_status_storage[job_id]["message"] = "Retrieving initial context for question generation (Qdrant)..."
        # Search Qdrant using the document_id and session_id (job_id) as filters
        gen_results = search_qdrant(
            job_id_for_log=job_id, 
            collection_name=QDRANT_COLLECTION_NAME, 
            embedded_vector=query_embed, 
            query_text_for_log=hypo_text,
            limit=params.retrieval_limit_generation, 
            score_threshold=params.similarity_threshold_generation,
            document_ids_filter=[doc_id_qdrant], # Filter by the specific document ID
            session_id_filter=job_id             # Filter by the current job_id (session)
        )
        if not gen_results: 
            logger.warning(f"[{job_id}] No context retrieved from Qdrant for generation (query: {hypo_text[:100]}...). Check Qdrant data, filters, or thresholds for doc '{doc_id_qdrant}'.")
            # Proceeding with empty context might be possible if the LLM can generate from topics alone,
            # but it's better to flag this. For now, let's raise an error if crucial.
            raise ValueError(f"No context retrieved from Qdrant for question generation. Document: '{doc_id_qdrant}'. This could be due to restrictive filters, high similarity threshold, or issues with the content/query.")
        
        gen_ctx_text = "\n\n---\n\n".join(filter(None, [r.payload.get('text', '') for r in gen_results]))
        gen_ctx_meta = [r.payload for r in gen_results] # Storing full payloads can be large, consider storing only essential metadata
        job_status_storage[job_id].update({
            "generation_context_text_for_llm": gen_ctx_text,
            # Be cautious about storing large metadata directly in job_status_storage if it becomes too big.
            # For now, this is fine for debugging/information.
            "generation_context_snippets_metadata": [
                {"source_file": m.get("metadata", {}).get("source_file"), "header_trail": m.get("metadata", {}).get("header_trail"), "final_chunk_index": m.get("metadata", {}).get("final_chunk_index")} 
                for m in gen_ctx_meta if m
            ]
        })

        job_status_storage[job_id]["message"] = "Generating initial question using LLM..."
        q_text, q_evals, convo_hist, ans_ctx_meta = generate_and_evaluate_question_once(
            job_id, params, gen_ctx_text, [], "", [doc_id_qdrant], job_id, job_specific_temp_dir, 1
        )
        job_status_storage[job_id].update({
            "status": "awaiting_feedback",
            "message": "Initial question generated. Please review.",
            "current_question": q_text, "current_evaluations": q_evals,
            "answerability_context_snippets_metadata": [ # Store relevant metadata for answerability context
                 {"source_file": m.get("metadata", {}).get("source_file"), "header_trail": m.get("metadata", {}).get("header_trail"), "final_chunk_index": m.get("metadata", {}).get("final_chunk_index")} 
                for m in ans_ctx_meta if m
            ],
            "conversation_history_for_qgen": convo_hist,
            "regeneration_attempts_made": 1,
            "max_regeneration_attempts": MAX_INTERACTIVE_REGENERATION_ATTEMPTS,
        })

    except (ValueError, FileNotFoundError, requests.exceptions.RequestException, TimeoutError, RuntimeError) as e: 
        logger.error(f"[{job_id}] Critical error during initial processing or first question generation: {e}", exc_info=True)
        job_status_storage[job_id].update({
            "status": "error", "message": f"Job setup or initial question generation failed: {str(e)}", "error_details": str(e)
        })
        if job_specific_temp_dir.exists(): shutil.rmtree(job_specific_temp_dir, ignore_errors=True)
    except Exception as e: 
        logger.error(f"[{job_id}] Unexpected critical error during initial processing or first question generation: {e}", exc_info=True)
        job_status_storage[job_id].update({
            "status": "error", "message": f"Unexpected job setup or initial generation error: {str(e)}", "error_details": str(e)
        })
        if job_specific_temp_dir.exists(): shutil.rmtree(job_specific_temp_dir, ignore_errors=True)


def generate_and_evaluate_question_once(
    job_id: str, params: QuestionGenerationRequest, gen_ctx_text: str,
    convo_hist: List[Dict[str, Any]], user_feedback: str, doc_ids_filter: List[str],
    session_id_filter: str, job_specific_temp_dir: Path, attempt_num: int
) -> Tuple[str, Dict[str, Any], List[Dict[str,Any]], List[Dict]]:
    
    logger.info(f"[{job_id}] Generating/Evaluating Question Attempt {attempt_num} with params: AcademicLevel='{params.academic_level}', Major='{params.major}', Course='{params.course_name}', Taxonomy='{params.taxonomy_level}', Marks='{params.marks_for_question}', Topics='{params.topics_list}'")
    placeholders = {
        "academic_level": params.academic_level, "major": params.major, "course_name": params.course_name,
        "taxonomy_level": params.taxonomy_level, "taxonomy": params.taxonomy_level, # "taxonomy" for broader compatibility if prompts use it
        "marks_for_question": params.marks_for_question, "topics_list": params.topics_list,
        "bloom_guidance": get_bloom_guidance(params.taxonomy_level, job_id),
        "blooms_taxonomy_descriptions": get_bloom_guidance(params.taxonomy_level, job_id), # Alias for clarity in prompts
        "retrieved_context": gen_ctx_text if gen_ctx_text else "No specific context was retrieved. Please generate a question based on the general topics and student profile.", 
        "feedback_on_previous_attempt": user_feedback if user_feedback else "This is the first attempt, or no specific feedback was provided for the previous one.", 
        "num_questions": "1" # We always generate one question at a time in this interactive flow
    }
    
    # Save the prompt for debugging
    prompt_filename = f"qgen_prompt_job_{job_id}_attempt_{attempt_num}.txt"
    prompt_path_on_disk = job_specific_temp_dir / prompt_filename 
    
    user_prompt_qgen = fill_template_string(FINAL_USER_PROMPT_PATH, placeholders, job_id)
    try: # Save prompt for inspection
        prompt_path_on_disk.write_text(user_prompt_qgen, encoding='utf-8')
        logger.info(f"[{job_id}] Question generation prompt for attempt {attempt_num} saved to {prompt_path_on_disk}")
    except Exception as e_io:
        logger.warning(f"[{job_id}] Could not save QGEN prompt to disk: {e_io}")


    sys_prompt_qgen = "You are an expert AI specializing in educational content, skilled in crafting diverse and challenging questions based on provided material and adhering to specific pedagogical guidelines like Bloom's Taxonomy. Ensure your output contains only the generated question text directly, without any preamble like 'Question:' unless it's naturally part of the question itself."
    llm_resp_block = get_gemini_response(job_id, sys_prompt_qgen, user_prompt_qgen, convo_hist)
    
    updated_hist = list(convo_hist) # Make a copy
    updated_hist.append({"role": "user", "parts": [{"text": user_prompt_qgen}]})

    if llm_resp_block.startswith("Error:"):
        logger.error(f"[{job_id}] LLM failed to generate question on attempt {attempt_num}: {llm_resp_block}")
        return ("Error: LLM failed to generate question.",
                {"error_message": llm_resp_block, "generation_status_message": "LLM API error during question generation."},
                updated_hist, [])

    updated_hist.append({"role": "model", "parts": [{"text": llm_resp_block}]})
    parsed_q_list = parse_questions_from_llm_response(job_id, llm_resp_block, 1)
    
    if not parsed_q_list:
        logger.error(f"[{job_id}] Failed to parse question from LLM response on attempt {attempt_num}. Response: {llm_resp_block[:200]}")
        return ("Error: Failed to parse question from LLM.",
                {"error_message": "LLM response was unparsable or did not yield a question.", "generation_status_message": "Could not parse question from LLM output.", "llm_raw_response_for_parsing": llm_resp_block},
                updated_hist, [])
    
    curr_q_text = parsed_q_list[0]
    logger.info(f"[{job_id}] Attempt {attempt_num} - Generated question (first 100 chars): {curr_q_text[:100]}")

    qsts = evaluate_question_qsts(job_id, curr_q_text, gen_ctx_text) 
    qual_evals = evaluate_question_qualitative_llm(
        job_id, curr_q_text, gen_ctx_text, params.academic_level, params.major,
        params.course_name, params.taxonomy_level, params.marks_for_question
    )
    is_ans, ans_reason, ans_ctx_meta_payloads = evaluate_question_answerability_llm( 
        job_id, curr_q_text, params.academic_level, params.major, params.course_name,
        params.taxonomy_level, params.marks_for_question, doc_ids_filter, session_id_filter
    )
    # Prompt file is now saved under job_specific_temp_dir, so no need to unlink here
    # if prompt_path_on_disk.exists(): prompt_path_on_disk.unlink(missing_ok=True) 

    eval_metrics = {
        "qsts_score": qsts, 
        "qualitative_metrics": qual_evals, # This will include its own "reasoning" if provided by LLM
        "llm_answerability": {"is_answerable": is_ans, "reasoning": ans_reason},
        "generation_status_message": f"Question evaluation for attempt {attempt_num} complete."
    }
    return curr_q_text, eval_metrics, updated_hist, ans_ctx_meta_payloads # Return full payloads for answerability context

class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path): return str(obj)
        return json.JSONEncoder.default(self, obj)

# --- API Endpoints ---
@app.post("/generate-questions", response_model=JobCreationResponse)
async def start_question_generation_endpoint(
    background_tasks: BackgroundTasks, file: UploadFile = File(...),
    academic_level: str = Form("Undergraduate"), major: str = Form("Computer Science"),
    course_name: str = Form("Data Structures and Algorithms"), taxonomy_level: str = Form("Evaluate"),
    marks_for_question: str = Form("10"), topics_list: str = Form("Breadth First Search, Shortest path"),
    retrieval_limit_generation: int = Form(15), similarity_threshold_generation: float = Form(0.4),
    generate_diagrams: bool = Form(False) # This param is not used yet in the backend logic flow
):
    job_id = str(uuid.uuid4())
    logger.info(f"[{job_id}] Received new request for question generation. File: {file.filename}, Taxonomy: {taxonomy_level}, Marks: {marks_for_question}, Topics: {topics_list}")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.error(f"[{job_id}] Invalid file type uploaded: {file.filename}")
        raise HTTPException(status_code=400, detail="Invalid file. Only PDF allowed.")

    # Create a temporary directory specific to this job ID under TEMP_UPLOAD_DIR
    job_temp_dir = TEMP_UPLOAD_DIR / job_id 
    job_temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{job_id}] Created temporary directory for job: {job_temp_dir}")
    
    # Sanitize filename before saving
    original_file_name_path = Path(file.filename)
    safe_stem = "".join(c for c in original_file_name_path.stem if c.isalnum() or c in ('-', '_')).strip()[:100] or "upload" # Limit stem length
    safe_suffix = original_file_name_path.suffix if original_file_name_path.suffix.lower() == ".pdf" else ".pdf" # Ensure .pdf
    safe_fname_for_disk = f"input_{safe_stem}{safe_suffix}" # Prepend "input_"
    temp_pdf_path = job_temp_dir / safe_fname_for_disk


    try:
        content = await file.read()
        if not content: 
            logger.error(f"[{job_id}] Uploaded file '{file.filename}' is empty.")
            if job_temp_dir.exists(): shutil.rmtree(job_temp_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        with temp_pdf_path.open("wb") as buf: buf.write(content)
        logger.info(f"[{job_id}] File '{file.filename}' ({len(content)}B) saved to '{temp_pdf_path}'")
    except Exception as e:
        if job_temp_dir.exists(): shutil.rmtree(job_temp_dir, ignore_errors=True)
        logger.error(f"[{job_id}] Could not save uploaded file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}") from e
    finally: await file.close()

    req_params_dict = {
        "academic_level": academic_level, "major": major, "course_name": course_name,
        "taxonomy_level": taxonomy_level, "marks_for_question": marks_for_question,
        "topics_list": topics_list, "retrieval_limit_generation": retrieval_limit_generation,
        "similarity_threshold_generation": similarity_threshold_generation,
        "generate_diagrams": generate_diagrams
    }
    try:
        req_params = QuestionGenerationRequest(**req_params_dict)
        logger.info(f"[{job_id}] Parsed request parameters: {req_params.model_dump_json(indent=2)}")
    except Exception as pydantic_error: 
        if job_temp_dir.exists(): shutil.rmtree(job_temp_dir, ignore_errors=True)
        logger.error(f"[{job_id}] Invalid parameters for question generation: {pydantic_error}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Invalid parameters: {pydantic_error}")

    job_status_storage[job_id] = {
        "status": "queued", "message": "Job queued for initial processing.",
        "job_params": req_params.model_dump(), # Store as dict for easier JSON serialization if needed later
        "original_filename": file.filename, # Store original filename
        "regeneration_attempts_made": 0, 
        "max_regeneration_attempts": MAX_INTERACTIVE_REGENERATION_ATTEMPTS,
        "job_temp_dir_path_str": str(job_temp_dir) # Store path to job's temp dir for cleanup or later use
    }
    # Pass the Pydantic model `req_params` and the `job_specific_temp_dir` (which is job_temp_dir here)
    background_tasks.add_task(process_document_and_generate_first_question,
                              job_id, temp_pdf_path, file.filename, req_params, job_temp_dir)
    return JobCreationResponse(job_id=job_id, message="Job successfully queued. Check status endpoint.")

def _get_job_status_response_dict(job_id: str, job_data: Dict) -> Dict:
    # Prepare a dictionary that matches the JobStatusResponse model fields
    response_dict = {k: job_data.get(k) for k in JobStatusResponse.model_fields.keys() if k != 'job_id'}
    
    # Ensure job_params is correctly formatted if it exists
    job_params_data = job_data.get("job_params")
    if isinstance(job_params_data, dict):
        try:
            # Validate and convert dict to Pydantic model instance for the response
            response_dict['job_params'] = QuestionGenerationRequest(**job_params_data)
        except Exception: # If validation fails, set to None or keep as dict if preferred
            response_dict['job_params'] = None 
            logger.warning(f"[{job_id}] Could not parse job_params dict into QuestionGenerationRequest model for status response.")
    elif not isinstance(job_params_data, QuestionGenerationRequest): # If it's neither dict nor Pydantic model
        response_dict['job_params'] = None
    
    # Ensure current_evaluations is a dict or None
    if 'current_evaluations' in response_dict and not isinstance(response_dict['current_evaluations'], (dict, type(None))):
        logger.warning(f"[{job_id}] current_evaluations is not a dict, converting to str for response: {type(response_dict['current_evaluations'])}")
        response_dict['current_evaluations'] = {"raw_eval_data": str(response_dict['current_evaluations'])}

    # Ensure final_result is a dict or None
    if 'final_result' in response_dict and not isinstance(response_dict['final_result'], (dict, type(None))):
        logger.warning(f"[{job_id}] final_result is not a dict, converting to str for response: {type(response_dict['final_result'])}")
        response_dict['final_result'] = {"raw_final_data": str(response_dict['final_result'])}
        
    return response_dict

@app.post("/regenerate-question/{job_id}", response_model=JobStatusResponse)
async def regenerate_question_endpoint(job_id: str, regen_request: RegenerationRequest):
    logger.info(f"[{job_id}] Received regeneration request. Feedback: '{regen_request.user_feedback[:100]}'")
    if job_id not in job_status_storage: 
        logger.warning(f"[{job_id}] Regeneration attempt for non-existent job.")
        raise HTTPException(status_code=404, detail="Job not found.")
    job_data = job_status_storage[job_id]
    
    status = job_data.get("status")
    attempts = job_data.get("regeneration_attempts_made", 0)
    max_attempts = job_data.get("max_regeneration_attempts", MAX_INTERACTIVE_REGENERATION_ATTEMPTS)

    if status not in ["awaiting_feedback", "max_attempts_reached"]:
        logger.warning(f"[{job_id}] Regeneration attempt from invalid status: {status}")
        raise HTTPException(status_code=400, detail=f"Cannot regenerate from current job status: {status}")
    
    if attempts >= max_attempts:
        job_data["status"] = "max_attempts_reached" 
        job_data["message"] = "Max regeneration attempts already reached. Further regeneration will not change the outcome unless limits are adjusted or job is reset."
        logger.info(f"[{job_id}] Max regeneration attempts ({max_attempts}) reached.")
        return JobStatusResponse(job_id=job_id, **_get_job_status_response_dict(job_id, job_data))

    job_data["status"], job_data["message"] = "regenerating_question", "Regenerating question based on feedback..."
    
    # Retrieve job-specific temp directory path from storage
    job_temp_dir_path_str = job_data.get("job_temp_dir_path_str")
    if not job_temp_dir_path_str:
        logger.error(f"[{job_id}] job_temp_dir_path_str not found in job_data. Cannot proceed with regeneration.")
        job_data.update({"status": "error", "message": "Critical error: Job temporary directory path missing.", "error_details": "Job temp dir path missing."})
        raise HTTPException(status_code=500, detail="Critical error: Job temporary directory path missing.")
    job_temp_dir = Path(job_temp_dir_path_str)
    job_temp_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists

    try:
        params = QuestionGenerationRequest(**job_data["job_params"]) 
        
        gen_ctx_text = job_data.get("generation_context_text_for_llm")
        convo_hist = job_data.get("conversation_history_for_qgen", [])
        doc_id_qdrant = job_data.get("document_id_for_qdrant")

        if not all([gen_ctx_text is not None, doc_id_qdrant is not None]): # gen_ctx_text can be empty string if no context was found initially
            missing_data_msg = "Missing essential data for regeneration (e.g., document ID, or generation context was never set)."
            logger.error(f"[{job_id}] {missing_data_msg}. GenCtxPresent={gen_ctx_text is not None}, DocIdPresent={doc_id_qdrant is not None}")
            job_data.update({"status": "error", "message": missing_data_msg, "error_details": missing_data_msg})
            # Don't raise HTTPException here if we want to return JobStatusResponse
            # Fall through to return the error status via JobStatusResponse
            return JobStatusResponse(job_id=job_id, **_get_job_status_response_dict(job_id, job_data))


        q_text, q_evals, updated_convo_hist, ans_meta_payloads = generate_and_evaluate_question_once(
            job_id, params, gen_ctx_text if gen_ctx_text else "", # Pass empty string if None
            convo_hist, regen_request.user_feedback,
            [doc_id_qdrant], job_id, job_temp_dir, attempts + 1
        )
        job_data.update({
            "current_question": q_text, "current_evaluations": q_evals,
            "conversation_history_for_qgen": updated_convo_hist,
            "answerability_context_snippets_metadata": [ # Store relevant metadata for answerability context
                 {"source_file": m.get("metadata", {}).get("source_file"), "header_trail": m.get("metadata", {}).get("header_trail"), "final_chunk_index": m.get("metadata", {}).get("final_chunk_index")} 
                for m in ans_meta_payloads if m # ans_meta_payloads is list of dicts (payloads)
            ], 
            "regeneration_attempts_made": attempts + 1, "error_details": None # Clear previous error details if successful
        })
        if job_data["regeneration_attempts_made"] >= max_attempts:
            job_data["status"], job_data["message"] = "max_attempts_reached", "Max regeneration attempts reached."
            logger.info(f"[{job_id}] Max regeneration attempts ({max_attempts}) reached after current regeneration.")
        else:
            job_data["status"], job_data["message"] = "awaiting_feedback", "Question regenerated successfully. Please review."
    except Exception as e:
        logger.error(f"[{job_id}] Error during regeneration process: {e}", exc_info=True)
        # Revert to awaiting_feedback but with error message, so user can see previous state if possible
        job_data.update({ 
            "status": "awaiting_feedback", # Or "error_during_regeneration"
            "message": f"An error occurred during regeneration: {str(e)}. Previous question state retained if possible.",
            "error_details": str(e) # Store the new error
        })
        # Optionally, append error to current_evaluations if it exists
        if isinstance(job_data.get("current_evaluations"), dict):
            job_data["current_evaluations"]["error_message_regeneration"] = str(e)
        else: # If no previous evaluations, create one with the error
            job_data["current_evaluations"] = {"error_message_regeneration": str(e)}
        
    return JobStatusResponse(job_id=job_id, **_get_job_status_response_dict(job_id, job_data))

@app.post("/finalize-question/{job_id}", response_model=JobStatusResponse)
async def finalize_question_endpoint(job_id: str, finalize_request: FinalizeRequest):
    logger.info(f"[{job_id}] Received finalize request for question: '{finalize_request.final_question[:100]}'")
    if job_id not in job_status_storage: 
        logger.warning(f"[{job_id}] Finalize attempt for non-existent job.")
        raise HTTPException(status_code=404, detail="Job not found.")
    job_data = job_status_storage[job_id]
    
    current_status = job_data.get("status")
    if current_status not in ["awaiting_feedback", "max_attempts_reached"]:
        logger.warning(f"[{job_id}] Finalize attempt from invalid status: {current_status}")
        raise HTTPException(status_code=400, detail=f"Cannot finalize from current job status: {current_status}")
    
    current_q_on_server = job_data.get("current_question")
    if not current_q_on_server or current_q_on_server.startswith("Error:"):
         logger.error(f"[{job_id}] Attempt to finalize with an errored or missing question. Server has: '{current_q_on_server}'")
         raise HTTPException(status_code=400, detail="Cannot finalize with an errored or missing question on the server. Current question is invalid.")
    
    finalized_question_text = finalize_request.final_question
    if finalized_question_text != current_q_on_server:
        logger.warning(f"[{job_id}] Finalizing question mismatch. Client sent: '{finalized_question_text[:70]}...', Server had: '{current_q_on_server[:70]}...'. Using question sent by client for finalization: '{finalized_question_text[:70]}'")
        # If you want to strictly use server's question:
        # finalized_question_text = current_q_on_server
        # logger.warning(f"[{job_id}] Using server's current question '{current_q_on_server[:70]}' for finalization despite client sending a different one.")


    job_data["status"], job_data["message"] = "finalizing", "Finalizing job and saving results..."
    
    job_params_dict = job_data.get("job_params", {})
    try:
        job_params_model = QuestionGenerationRequest(**job_params_dict) if job_params_dict else None
    except Exception: # Should not happen if job_params was validated on creation
        job_params_model = None 
        logger.error(f"[{job_id}] Could not parse job_params into model during finalization. Storing as dict.")

    # Prepare the final payload to be saved
    final_payload = {
        "job_id": job_id, 
        "original_filename": job_data.get("original_filename"),
        "parameters": job_params_model.model_dump() if job_params_model else job_params_dict, 
        "generated_question": finalized_question_text, # Using the question from the request
        "evaluation_metrics": job_data.get("current_evaluations"), # Last known evaluations
        "generation_context_snippets_metadata": job_data.get("generation_context_snippets_metadata"), 
        "answerability_context_snippets_metadata": job_data.get("answerability_context_snippets_metadata"), 
        "processed_markdown_path_relative": job_data.get("processed_markdown_path_relative"),
        "processed_markdown_filename_on_server": job_data.get("processed_markdown_filename_on_server"),
        "total_regeneration_attempts_made": job_data.get("regeneration_attempts_made"),
        "finalized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    # Define path for the final result JSON file
    # Results are saved under JOB_DATA_DIR / job_id / <filename>
    job_final_data_dir = JOB_DATA_DIR / job_id
    job_final_data_dir.mkdir(parents=True, exist_ok=True) 
    result_file_path = job_final_data_dir / f"{job_id}_final_interactive_result.json"
    
    try: 
        result_file_path.write_text(json.dumps(final_payload, indent=2, cls=PathEncoder), encoding="utf-8")
        logger.info(f"[{job_id}] Final result successfully saved to {result_file_path}")
        job_data.update({"final_result": final_payload, "status": "completed", "message": "Job finalized successfully by user. Results saved."})
    except Exception as e: 
        logger.error(f"[{job_id}] Failed to save final result JSON to {result_file_path}: {e}", exc_info=True)
        # Update status to reflect error but still provide the payload if possible
        job_data.update({
            "status": "error_saving_final", 
            "message": f"Job finalized, but an error occurred while saving the result file: {str(e)}", 
            "final_result": final_payload, # Include the data that was supposed to be saved
            "error_details": f"Error saving final result JSON: {str(e)}"
        })

    # Clean up the job-specific temporary upload directory
    job_temp_dir_path_str = job_data.get("job_temp_dir_path_str")
    if job_temp_dir_path_str:
        job_temp_dir_to_clean = Path(job_temp_dir_path_str)
        if job_temp_dir_to_clean.exists(): 
            try:
                shutil.rmtree(job_temp_dir_to_clean)
                logger.info(f"[{job_id}] Cleaned up temporary job directory: {job_temp_dir_to_clean}")
            except Exception as e_clean:
                logger.warning(f"[{job_id}] Failed to clean up temporary job directory {job_temp_dir_to_clean}: {e_clean}")
        else:
            logger.info(f"[{job_id}] Temporary job directory {job_temp_dir_to_clean} not found for cleanup (might have been cleaned already or failed before creation).")
    else:
        logger.warning(f"[{job_id}] No job_temp_dir_path_str found in job_data for cleanup during finalization.")
    
    return JobStatusResponse(job_id=job_id, **_get_job_status_response_dict(job_id, job_data))


@app.get("/job-status/{job_id}", response_model=JobStatusResponse)
async def get_job_status_endpoint(job_id: str):
    job_info = job_status_storage.get(job_id)
    if not job_info: 
        logger.warning(f"Status request for non-existent job ID: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found.")
    
    # logger.debug(f"[{job_id}] Current raw job_info for status request: {job_info}")
    # Use the helper to structure the response, ensuring Pydantic models are handled
    response_payload = _get_job_status_response_dict(job_id, job_info)
    # logger.debug(f"[{job_id}] Prepared JobStatusResponse payload: {response_payload}")
    
    return JobStatusResponse(job_id=job_id, **response_payload)

@app.get("/health")
async def health_check():
    jina_api_status = "unknown"
    jina_test_error_details = None
    if JINA_API_KEY and JINA_API_EMBEDDING_URL:
        try:
            test_texts = ["health check test string"]
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {JINA_API_KEY}",
            }
            payload = {
                "input": test_texts,
                "model": JINA_API_EMBEDDING_MODEL_NAME,
            }
            response = requests.post(JINA_API_EMBEDDING_URL, headers=headers, json=payload, timeout=10) 
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("data") and isinstance(response_data["data"], list) and \
                   len(response_data["data"]) > 0 and response_data["data"][0].get("embedding"):
                    jina_api_status = "ok"
                else:
                    jina_api_status = "error_malformed_response"
                    jina_test_error_details = f"Malformed response: {str(response_data)[:200]}"
            else:
                jina_api_status = f"error_status_{response.status_code}"
                jina_test_error_details = f"Status {response.status_code}, Body: {response.text[:200]}"
        except requests.exceptions.Timeout:
            jina_api_status = "error_timeout"
            jina_test_error_details = "Request to Jina API timed out."
        except requests.exceptions.RequestException as e:
            jina_api_status = "error_unreachable_or_request_failed"
            jina_test_error_details = f"RequestException: {str(e)}"
        except Exception as e:
            logger.warning(f"Health check: Jina API test failed unexpectedly: {e}", exc_info=True)
            jina_api_status = "error_exception"
            jina_test_error_details = f"Unexpected Exception: {str(e)}"
    else:
        jina_api_status = "not_configured (API Key or URL missing)"
    
    health_response = {
        "status": "ok", 
        "message": "Interactive QG API is running.",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dependencies": {
            "jina_embedding_api": {
                "status": jina_api_status,
                "model_configured": JINA_API_EMBEDDING_MODEL_NAME,
                "url_configured": JINA_API_EMBEDDING_URL,
                "error_details": jina_test_error_details if jina_test_error_details else "N/A"
            },
            "qdrant": {
                "url_configured": bool(QDRANT_URL),
                "api_key_configured": bool(QDRANT_API_KEY) # Don't log the key itself
            },
            "gemini_api": {"api_key_configured": bool(GEMINI_API_KEY)},
            "datalab_api": {
                "api_key_configured": bool(DATALAB_API_KEY),
                "url_configured": bool(DATALAB_MARKER_URL)
            },
            "moondream_api": {"api_key_configured": bool(MOONDREAM_API_KEY)}
        },
        "expected_vector_size": VECTOR_SIZE,
    }
    return health_response

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Interactive Question Generation API with Uvicorn...")
    logger.info(f"CONFIG: Jina API URL='{JINA_API_EMBEDDING_URL}', Jina Model='{JINA_API_EMBEDDING_MODEL_NAME}', Vector Size='{VECTOR_SIZE}'")
    logger.info(f"CONFIG: Qdrant URL='{QDRANT_URL}'")
    logger.info(f"CONFIG: Gemini Model='{GEMINI_MODEL_NAME}'")
    logger.info(f"CONFIG: Datalab Marker URL='{DATALAB_MARKER_URL}'")
    logger.info(f"Server data base directory: {PERSISTENT_DATA_BASE_DIR}")
    logger.info(f"Prompt directory: {PROMPT_DIR}")
    
    # Note for user: Ensure JINA_API_KEY, QDRANT_URL, GEMINI_API_KEY, etc.
    # are set in your .env file. VECTOR_SIZE must match the output of JINA_API_EMBEDDING_MODEL_NAME.
    # Pip install: requests torch sentence-transformers Pillow nltk python-dotenv fastapi uvicorn[standard] qdrant-client moondream
    uvicorn.run(app, host="0.0.0.0", port=8002)