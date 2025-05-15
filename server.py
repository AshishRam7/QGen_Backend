# -*- coding: utf-8 -*-
import os
import time
import re
import base64
import requests
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

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("NLTK data not found or incomplete. Downloading essential resources...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    print("NLTK data downloaded.")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sbert_util

# Qdrant
from qdrant_client import models, QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, ScoredPoint,FilterSelector
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny, MatchText

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

# --- Initialize Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s][%(lineno)d] %(message)s')
logging.getLogger('PIL').setLevel(logging.WARNING) # Reduce PIL logging noise
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('nltk').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Check Essential Variables ---
essential_vars = {
    "QDRANT_URL": QDRANT_URL, "GEMINI_API_KEY": GEMINI_API_KEY,
    "DATALAB_API_KEY": DATALAB_API_KEY, "DATALAB_MARKER_URL": DATALAB_MARKER_URL,
    "MOONDREAM_API_KEY": MOONDREAM_API_KEY
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
QDRANT_COLLECTION_NAME = "markdown_docs_v3_semantic_qg"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:{action}?key={api_key}"
QSTS_THRESHOLD = 0.5 # Default threshold for Question-to-Source-Text Similarity
QUALITATIVE_METRICS = ["Understandable", "TopicRelated", "Grammatical", "Clear", "Central"]
MAX_INTERACTIVE_REGENERATION_ATTEMPTS = 15
MAX_HISTORY_TURNS = 10 # For LLM conversation history
ANSWER_RETRIEVAL_LIMIT = 5 # For answerability check context
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
    *   The context **MUST** provide the necessary building blocks.
4.  **Output Format:** Respond ONLY with a valid JSON object containing two keys:
    *   `"is_answerable"`: `true` if the question is sufficiently answerable, `false` otherwise.
    *   `"reasoning"`: A concise string explaining your judgment.

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
model_st: Optional[SentenceTransformer] = None
qdrant_client: Optional[QdrantClient] = None
model_moondream: Optional[Any] = None
stop_words_nltk: Optional[set] = None

def initialize_models():
    global model_st, qdrant_client, model_moondream, stop_words_nltk
    try:
        logger.info(f"Initializing Sentence Transformer model '{EMBEDDING_MODEL_NAME}'...")
        model_st = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Sentence Transformer model loaded.")

        logger.info(f"Initializing Moondream model...")
        model_moondream = md_moondream.vl(api_key=MOONDREAM_API_KEY)
        logger.info("Moondream model initialized successfully.")

        logger.info(f"Initializing Qdrant client for URL: {QDRANT_URL}...")
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        try:
            qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
            logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' found.")
        except Exception as e:
            err_str = str(e).lower()
            is_not_found = ("not found" in err_str or "status_code=404" in err_str or "collection not found" in err_str or " কারণেই" in err_str)
            if is_not_found or (hasattr(e, 'status_code') and e.status_code == 404):
                logger.warning(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' not found. Creating...")
                qdrant_client.recreate_collection(
                    collection_name=QDRANT_COLLECTION_NAME,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
                )
                logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created.")
            else:
                logger.error(f"Qdrant connection/access error for collection '{QDRANT_COLLECTION_NAME}': {e}", exc_info=True)
                raise Exception(f"Qdrant connection/access error: {e}")

        logger.info("Loading NLTK stopwords...")
        stop_words_nltk = set(stopwords.words('english'))
        logger.info("NLTK stopwords loaded.")

    except Exception as e:
        logger.critical(f"Fatal error during global model initialization: {e}", exc_info=True)
        sys.exit("Model initialization failed.")

# --- FastAPI App Setup ---
app = FastAPI(title="Interactive Question Generation API")
initialize_models()

origins = ["https://qgen-frontend-app.vercel.app/","http://localhost:3000", "http://localhost:3001", "https://q-gen-frontend.vercel.app"] # Adjust to your frontend URL
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
    error_details: Optional[str] = None # For backend errors during job processing
    job_params: Optional[QuestionGenerationRequest] = None
    original_filename: Optional[str] = None
    current_question: Optional[str] = None
    current_evaluations: Optional[Dict[str, Any]] = None # Stores eval metrics, including potential error messages from LLM calls
    regeneration_attempts_made: Optional[int] = None
    max_regeneration_attempts: Optional[int] = None
    final_result: Optional[Dict[str, Any]] = None # Populated when job is 'completed'

class RegenerationRequest(BaseModel):
    user_feedback: str

class FinalizeRequest(BaseModel):
    final_question: str

# --- Helper Functions ---
def get_bloom_guidance(level: str, job_id_for_log: str) -> str:
    logger.info(f"[{job_id_for_log}] Getting Bloom's guidance for level: {level}")
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
    logger.debug(f"[{job_id_for_log}] Filling template string from path: {template_path} with placeholders: {list(placeholders.keys())}")
    if not template_path.exists():
        logger.error(f"[{job_id_for_log}] Prompt template file not found: {template_path}")
        raise FileNotFoundError(f"Prompt template file not found: {template_path}")
    try:
        template_content = template_path.read_text(encoding="utf-8")
        for key, value in placeholders.items():
            template_content = template_content.replace(f"{{{key}}}", str(value))
        # Check for unfilled placeholders as a warning
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
    processed_history = conversation_history[-(MAX_HISTORY_TURNS*2):] # Keep last N turns
    contents = list(processed_history) # Make a copy
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
        logger.warning(f"[{job_id_for_log}] System prompt provided for non-1.5 model; ensure it's handled in conversation history or user prompt.")

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
                    return f"Error: Gemini generation stopped. Reason: {reason}." # e.g. SAFETY, RECITATION etc.

            logger.error(f"[{job_id_for_log}] Gemini API response malformed: {response_data}")
            return "Error: Malformed response from Gemini API." # Should be caught by caller
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
            return "Error: Gemini API call failed after multiple retries." # Caller must handle this
    return "Error: Gemini API call failed (exhausted retries - code path should ideally not be reached)."


def clean_text_for_embedding(text: str, job_id_for_log: str) -> str:
    global stop_words_nltk
    logger.debug(f"[{job_id_for_log}] Cleaning text for embedding (first 50 chars): {text[:50]}")
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text) # Remove HTML tags
    text = re.sub(r"[^\w\s\.\-\']", "", text) # Remove special characters except word chars, whitespace, '.', '-'
    text = re.sub(r"\s+", " ", text).strip() # Normalize whitespace
    if stop_words_nltk:
        try:
            tokens = word_tokenize(text)
        except LookupError as e:
            logger.error(f"[{job_id_for_log}] NLTK LookupError during word_tokenize in clean_text_for_embedding: {e}. This may indicate missing 'punkt' or 'punkt_tab'. Ensure NLTK resources are downloaded.")
            # Depending on severity, you might re-raise or return text without stopword removal
            # For now, proceed without stopword removal if punkt is missing.
            if 'punkt' in str(e).lower():
                logger.warning(f"[{job_id_for_log}] NLTK 'punkt' resource missing. Skipping stopword removal.")
                return text # Return text as is, if punkt is critical and missing.
            raise # Re-raise other lookup errors
        tokens = [word for word in tokens if word not in stop_words_nltk and len(word) > 1] # Remove stopwords and short tokens
        text = " ".join(tokens)
    return text

def hierarchical_chunk_markdown(markdown_text: str, source_filename: str, job_id_for_log: str,
                                min_words: int = MIN_CHUNK_SIZE_WORDS, max_words: int = MAX_CHUNK_SIZE_WORDS) -> List[Dict]:
    logger.info(f"[{job_id_for_log}] Starting hierarchical chunking for {source_filename}.")
    chunks = []
    header_pattern = re.compile(r"^(#{1,4})\s+(.*)", re.MULTILINE)
    current_chunk_text = []
    current_header_trail = []
    chunk_index_counter = 0

    lines = markdown_text.splitlines()

    for i, line in enumerate(lines):
        header_match = header_pattern.match(line)
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            if current_chunk_text:
                full_text = "\n".join(current_chunk_text).strip()
                if len(full_text.split()) >= min_words:
                    chunk_data = {
                        "text": full_text,
                        "metadata": {
                            "source_file": source_filename,
                            "header_trail": list(current_header_trail),
                            "chunk_index_original_split": chunk_index_counter,
                            "estimated_char_length": len(full_text),
                            "estimated_word_count": len(full_text.split())
                        }
                    }
                    chunks.append(chunk_data)
                    chunk_index_counter +=1
                current_chunk_text = []
            current_header_trail = current_header_trail[:level-1]
            current_header_trail.append(title)
            current_chunk_text.append(line)
        else:
            current_chunk_text.append(line)

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

    final_chunks = []
    for chunk_idx, chunk in enumerate(chunks):
        if chunk['metadata']['estimated_word_count'] > max_words:
            logger.debug(f"[{job_id_for_log}] Chunk from '{chunk['metadata']['header_trail']}' (orig_idx {chunk_idx}) is too large ({chunk['metadata']['estimated_word_count']} words), splitting by paragraphs.")
            paragraphs = chunk['text'].split('\n\n')
            temp_sub_chunk_text = ""
            sub_chunk_id_counter_local = 0
            for para_idx, para in enumerate(paragraphs):
                para_words = len(para.split())
                current_sub_chunk_words = len(temp_sub_chunk_text.split())
                if temp_sub_chunk_text and (current_sub_chunk_words + para_words > max_words):
                    if current_sub_chunk_words >= min_words:
                        final_chunks.append({
                            "text": temp_sub_chunk_text.strip(),
                            "metadata": {**chunk['metadata'], "sub_chunk_id": sub_chunk_id_counter_local, "estimated_word_count": current_sub_chunk_words}
                        })
                        sub_chunk_id_counter_local += 1
                    temp_sub_chunk_text = para
                else:
                    temp_sub_chunk_text = (temp_sub_chunk_text + "\n\n" + para).strip() if temp_sub_chunk_text else para
            
            if temp_sub_chunk_text.strip() and len(temp_sub_chunk_text.strip().split()) >= min_words :
                final_chunks.append({
                    "text": temp_sub_chunk_text.strip(),
                    "metadata": {**chunk['metadata'], "sub_chunk_id": sub_chunk_id_counter_local, "estimated_word_count": len(temp_sub_chunk_text.strip().split())}
                })
        elif chunk['metadata']['estimated_word_count'] >= min_words:
            final_chunks.append(chunk) # No sub_chunk_id needed if not split further

    for i, chk in enumerate(final_chunks):
        chk['metadata']['final_chunk_index'] = i # Universal final index

    logger.info(f"[{job_id_for_log}] Chunking for {source_filename} resulted in {len(final_chunks)} final chunks.")
    if not final_chunks and markdown_text.strip():
        logger.warning(f"[{job_id_for_log}] No chunks generated for {source_filename}, but text exists. Creating a single chunk for the whole document.")
        return [{"text": markdown_text, "metadata": {"source_file": source_filename, "header_trail": ["Full Document"], "final_chunk_index": 0, "estimated_char_length": len(markdown_text), "estimated_word_count": len(markdown_text.split())}}]
    return final_chunks

def embed_chunks(chunks_data: List[Dict], job_id_for_log: str) -> List[List[float]]:
    global model_st
    logger.info(f"[{job_id_for_log}] Embedding {len(chunks_data)} chunks.")
    if not model_st:
        logger.error(f"[{job_id_for_log}] SentenceTransformer model not initialized.")
        raise ValueError("SentenceTransformer model not available for embedding.")
    texts_to_embed = [chunk['text'] for chunk in chunks_data]
    if not texts_to_embed: return []
    embeddings = model_st.encode(texts_to_embed, show_progress_bar=False) # Set show_progress_bar=False for server logs
    logger.info(f"[{job_id_for_log}] Finished embedding {len(chunks_data)} chunks.")
    return embeddings.tolist()

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
        payload = {"text": chunk['text'], "metadata": chunk['metadata']}
        for k, v in payload["metadata"].items():
            if isinstance(v, Path): payload["metadata"][k] = str(v) # Qdrant needs serializable types

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
            return f"Could not generate hypothetical text. Error: {response_text}" # Return error string
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
    logger.info(f"[{job_id_for_log}] Searching Qdrant collection '{collection_name}' for query (log): {query_text_for_log[:100]}. Limit: {limit}, Threshold: {score_threshold}")
    if not qdrant_client:
        logger.error(f"[{job_id_for_log}] Qdrant client not initialized.")
        raise ValueError("Qdrant client not available.")
    q_filter = models.Filter(must=[])
    if document_ids_filter:
        q_filter.must.append(FieldCondition(key="metadata.document_id", match=MatchAny(any=document_ids_filter)))
    if session_id_filter:
        q_filter.must.append(FieldCondition(key="metadata.session_id", match=MatchValue(value=session_id_filter)))
    final_filter = q_filter if q_filter.must else None
    logger.debug(f"[{job_id_for_log}] Qdrant search final_filter: {final_filter}")
    try:
        search_results = qdrant_client.search(
            collection_name=collection_name, query_vector=embedded_vector, query_filter=final_filter,
            limit=limit, score_threshold=score_threshold
        )
        logger.info(f"[{job_id_for_log}] Qdrant search returned {len(search_results)} results.")
        return search_results
    except Exception as e:
        logger.error(f"[{job_id_for_log}] Error searching Qdrant: {e}", exc_info=True)
        raise

def parse_questions_from_llm_response(job_id_for_log: str, question_block: str, num_expected: int = 1) -> List[str]:
    logger.debug(f"[{job_id_for_log}] Parsing questions from LLM response: {question_block[:100]}")
    lines = [line.strip() for line in question_block.splitlines() if line.strip()]
    question_prefixes = ["Question:", "Q:", "Generated Question:"]
    cleaned_questions = []
    for line in lines:
        for prefix in question_prefixes:
            if line.startswith(prefix): line = line[len(prefix):].strip(); break
        line = re.sub(r"^\s*\d+\.\s*", "", line) # Remove "1. ", " 2. " etc.
        line = re.sub(r"^\s*[a-zA-Z]\)\s*", "", line) # Remove "a) ", "b) " etc.
        if line: cleaned_questions.append(line)

    if not cleaned_questions and lines and num_expected == 1: # Fallback for single question if no prefixes matched
        logger.warning(f"[{job_id_for_log}] No question prefixes matched. Using first cleaned line as question. Lines: {lines}")
        first_line_cleaned = re.sub(r"^\s*\d+\.\s*", "", lines[0]).strip()
        first_line_cleaned = re.sub(r"^\s*[a-zA-Z]\)\s*", "", first_line_cleaned).strip()
        if first_line_cleaned: cleaned_questions = [first_line_cleaned]

    final_questions = cleaned_questions[:num_expected] if cleaned_questions else []
    if not final_questions and question_block.strip():
        logger.warning(f"[{job_id_for_log}] Could not parse distinct questions. Raw block: {question_block}")
        if num_expected == 1: final_questions = [question_block.strip()] # Last resort for single Q
    logger.info(f"[{job_id_for_log}] Parsed {len(final_questions)} questions. First: {final_questions[0][:100] if final_questions else 'None'}")
    return final_questions

def evaluate_question_qsts(job_id_for_log: str, question: str, context: str) -> float:
    global model_st
    logger.info(f"[{job_id_for_log}] Evaluating QSTS for question: {question[:100]}")
    if not model_st: logger.error(f"[{job_id_for_log}] SentenceTransformer model not initialized for QSTS."); return 0.0
    if not question or not context: logger.warning(f"[{job_id_for_log}] Empty question or context for QSTS."); return 0.0
    try:
        q_embed = model_st.encode(clean_text_for_embedding(question, job_id_for_log))
        c_embed = model_st.encode(clean_text_for_embedding(context, job_id_for_log))
        score = sbert_util.pytorch_cos_sim(q_embed, c_embed).item()
        logger.info(f"[{job_id_for_log}] QSTS score: {score:.4f}")
        return score
    except Exception as e: logger.error(f"[{job_id_for_log}] Error during QSTS: {e}", exc_info=True); return 0.0

def evaluate_question_qualitative_llm(job_id_for_log: str, question: str, context_for_eval: str,
                                   academic_level: str, major: str, course_name: str, taxonomy_level: str, marks_for_question: str) -> Dict[str, Any]:
    logger.info(f"[{job_id_for_log}] Performing qualitative LLM evaluation for question: {question[:100]}")
    placeholders = {
        "question": question, "context": context_for_eval, "academic_level": academic_level,
        "major": major, "course_name": course_name, "taxonomy_level": taxonomy_level,
        "marks_for_question": marks_for_question,
        "bloom_guidance": get_bloom_guidance(taxonomy_level, job_id_for_log)
    }
    default_error_return = {**{metric: False for metric in QUALITATIVE_METRICS}} # Base for error returns
    try:
        eval_user_prompt = fill_template_string(QUALITATIVE_EVAL_PROMPT_PATH, placeholders, job_id_for_log)
        eval_system_prompt = "You are an expert AI assistant evaluating the quality of a generated question."
        response_text = get_gemini_response(job_id_for_log, eval_system_prompt, eval_user_prompt, [])

        if response_text.startswith("Error:"):
            logger.error(f"[{job_id_for_log}] Gemini call failed for qualitative eval: {response_text}")
            return {**default_error_return, "error_message": response_text}
        try:
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```|({[\s\S]*})", response_text, re.DOTALL | re.MULTILINE)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                eval_results = json.loads(json_str)
                final_results = {metric: eval_results.get(metric, False) for metric in QUALITATIVE_METRICS}
                logger.info(f"[{job_id_for_log}] Qualitative LLM evaluation results: {final_results}")
                return final_results
            else:
                logger.warning(f"[{job_id_for_log}] No JSON block in qualitative eval response: {response_text}")
                return {**default_error_return, "error_message": "No JSON in LLM response for qualitative eval."}
        except json.JSONDecodeError as jde:
            logger.error(f"[{job_id_for_log}] Failed to decode JSON from qualitative eval: {response_text}", exc_info=True)
            return {**default_error_return, "error_message": f"JSON decode error in qualitative eval: {str(jde)}"}
    except Exception as e:
        logger.error(f"[{job_id_for_log}] Error in evaluate_question_qualitative_llm: {e}", exc_info=True)
        return {**default_error_return, "error_message": str(e)}

def evaluate_question_answerability_llm(job_id_for_log: str, question: str, academic_level: str, major: str,
                                     course_name: str, taxonomy_level: str, marks_for_question: str,
                                     document_ids_filter: List[str], session_id_filter: str
                                     ) -> Tuple[bool, str, List[Dict]]:
    global model_st
    logger.info(f"[{job_id_for_log}] Performing enhanced answerability LLM evaluation for: {question[:100]}")
    if not model_st: return False, "Error: Embedding model not available.", []
    ans_context_metadata = []
    try:
        question_embedding = model_st.encode(clean_text_for_embedding(question, job_id_for_log))
        answer_search_results = search_qdrant(
            job_id_for_log=job_id_for_log, collection_name=QDRANT_COLLECTION_NAME,
            embedded_vector=question_embedding.tolist(), query_text_for_log=f"Answerability search: {question[:50]}",
            limit=ANSWER_RETRIEVAL_LIMIT, document_ids_filter=document_ids_filter, session_id_filter=session_id_filter
        )
        if not answer_search_results: return False, "No relevant context found in document to answer this question.", []
        
        ans_context_parts = [res.payload.get('text', '') for res in answer_search_results]
        ans_context_for_llm = "\n\n---\n\n".join(filter(None, ans_context_parts))
        ans_context_metadata = [res.payload for res in answer_search_results]
        
        placeholders = {
            "academic_level": academic_level, "major": major, "course_name": course_name,
            "question": question, "taxonomy_level": taxonomy_level, "marks_for_question": marks_for_question,
            "answer_retrieval_limit": ANSWER_RETRIEVAL_LIMIT, "answer_context": ans_context_for_llm
        }
        user_prompt = fill_template_string(ENHANCED_ANSWERABILITY_PROMPT_PATH, placeholders, job_id_for_log)
        system_prompt = "You are an expert AI evaluating question answerability from context."
        response_text = get_gemini_response(job_id_for_log, system_prompt, user_prompt, [])

        if response_text.startswith("Error:"):
            return False, f"LLM call failed for answerability: {response_text}", ans_context_metadata
        try:
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```|({[\s\S]*})", response_text, re.DOTALL | re.MULTILINE)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                eval_data = json.loads(json_str)
                is_answerable = eval_data.get("is_answerable", False)
                reasoning = eval_data.get("reasoning", "No reasoning from LLM.")
                logger.info(f"[{job_id_for_log}] Answerability: {is_answerable}, Reasoning: {reasoning[:100]}")
                return is_answerable, reasoning, ans_context_metadata
            else:
                return False, "LLM response for answerability not in expected JSON format.", ans_context_metadata
        except json.JSONDecodeError:
            return False, "Failed to parse LLM's JSON response for answerability.", ans_context_metadata
    except Exception as e:
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

        files_payload = {"file": (file_path.name, file_content_bytes, "application/pdf")}
        form_data = {"output_format": (None, "markdown"), "disable_image_extraction": (None, "false")}
        headers = {"X-Api-Key": DATALAB_API_KEY}
        
        response = requests.post(DATALAB_MARKER_URL, files=files_payload, data=form_data, headers=headers, timeout=DATALAB_POST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Datalab API request failed: {e} - {e.response.text if e.response else 'No response'}") from e

    if not data.get("success"): raise Exception(f"Datalab API error: {data.get('error', 'Unknown error')}")
    check_url = data["request_check_url"]
    for _ in range(DATALAB_MAX_POLLS): # Use _ if i is not needed
        time.sleep(DATALAB_POLL_INTERVAL)
        try:
            poll_resp = requests.get(check_url, headers=headers, timeout=DATALAB_POLL_TIMEOUT)
            poll_resp.raise_for_status(); poll_data = poll_resp.json()
            if poll_data.get("status") == "complete": return {"markdown": poll_data.get("markdown", ""), "images": poll_data.get("images", {})}
            if poll_data.get("status") == "error": raise Exception(f"Datalab processing failed: {poll_data.get('error', 'Unknown error')}")
            logger.info(f"[{job_id_for_log}] Polling Datalab: status {poll_data.get('status')}")
        except requests.exceptions.RequestException as e_poll: logger.warning(f"[{job_id_for_log}] Polling Datalab error: {e_poll}. Retrying...")
    raise TimeoutError(f"Polling timed out for Datalab processing of {file_path.name}.")

def generate_moondream_image_description(image_path: Path, caption_text: str = "") -> str:
    global model_moondream
    if not model_moondream: return "Error: Moondream model not available."
    try:
        image = Image.open(image_path)
        if image.mode != "RGB": image = image.convert("RGB")
        prompt = (f"Describe key technical findings in this figure/visualization {('captioned: ' + caption_text) if caption_text else ''}..." ) # Keep prompt as before
        encoded_image = model_moondream.encode_image(image)
        response_dict = model_moondream.query(encoded_image, prompt)
        desc = response_dict.get("answer", "Error: No answer key.") if isinstance(response_dict, dict) else str(response_dict)
        desc = desc.replace('\n', ' ').strip()
        if not desc or desc.startswith("Error:"): return "No valid description generated." if not desc.startswith("Error:") else desc
        return desc
    except Exception as e: logger.error(f"Moondream [{image_path.name}]: Error: {e}", exc_info=True); return f"Error generating description: {str(e)}"

def save_extracted_images_api(images_dict: Dict[str,str], images_folder: Path, job_id_for_log: str) -> Dict[str, str]:
    images_folder.mkdir(parents=True, exist_ok=True); saved_files_map = {}
    for name_in_md, b64_data in images_dict.items():
        try:
            base, suffix = Path(name_in_md).stem, Path(name_in_md).suffix or ".png"
            safe_base = "".join(c for c in base if c.isalnum() or c in ('-', '_')).strip()[:50] or f"img_{uuid.uuid4().hex[:6]}"
            counter = 0; disk_name = f"{safe_base}{suffix}"; disk_path = images_folder / disk_name
            while disk_path.exists(): counter += 1; disk_name = f"{safe_base}_{counter}{suffix}"; disk_path = images_folder / disk_name
            with open(disk_path, "wb") as img_file: img_file.write(base64.b64decode(b64_data))
            saved_files_map[name_in_md] = str(disk_path)
        except Exception as e: logger.warning(f"[{job_id_for_log}] Could not save image '{name_in_md}': {e}")
    return saved_files_map

def process_markdown_api(markdown_text: str, saved_images_map: Dict[str, str], job_id_for_log: str) -> str:
    logger.info(f"[{job_id_for_log}] Processing markdown for image descriptions...")
    lines = markdown_text.splitlines(); processed_lines = []; img_count = 0
    fig_pattern = re.compile(r"^\s*!\[(?P<alt>.*?)\]\((?P<path>[^)]+)\)\s*$")
    cap_pattern = re.compile(r"^\s*(Figure|Table|Chart|Fig\.?|Tbl\.?)\s?([\w\d\.\-]+[:.]?)\s?(.*)", re.IGNORECASE)
    i = 0
    while i < len(lines):
        line = lines[i]; match = fig_pattern.match(line.strip())
        if match:
            img_count += 1; alt, path_enc = match.group("alt"), match.group("path")
            path_dec = urllib.parse.unquote(path_enc)
            caption, cap_idx = alt, -1
            # Simple caption lookahead
            for k_ahead in range(2): # Look 2 lines ahead
                next_idx = i + 1 + k_ahead
                if next_idx < len(lines):
                    next_line = lines[next_idx].strip()
                    if not next_line: 
                        if k_ahead == 0:
                            k_ahead -=1 
                            continue # Skip first empty line, effectively looking one more line down
                    if cap_pattern.match(next_line): caption, cap_idx = next_line, next_idx; break
                    break # Non-empty, non-caption line: stop lookahead
                else: break
            
            disk_path_str = saved_images_map.get(path_dec) or saved_images_map.get(path_enc)
            desc = ""
            if disk_path_str:
                if Path(disk_path_str).exists(): desc = generate_moondream_image_description(Path(disk_path_str), caption)
                else: desc = f"*Error: Image file '{Path(disk_path_str).name}' not found on disk.*"
            else: desc = f"*Error: Image '{path_dec}' not in saved images map.*"
            
            title = caption if caption and len(caption) > 5 else f"Figure {img_count}"
            block = f"\n---\n### {title}\n\n**Original Ref:** `{path_dec}` (Alt: `{alt}`)\n\n**Moondream Desc:**\n{desc}\n---\n"
            processed_lines.append(block)
            if cap_idx != -1: i = cap_idx # Skip processed caption line
        else: processed_lines.append(line)
        i += 1
    logger.info(f"[{job_id_for_log}] Processed markdown with {img_count} image references.")
    return "\n".join(processed_lines)

# --- Main Background Task Logic ---
def process_document_and_generate_first_question(
    job_id: str, pdf_path_on_disk: Path, original_filename: str,
    params: QuestionGenerationRequest, job_specific_temp_dir: Path
):
    global model_st, job_status_storage
    job_status_storage[job_id]["status"] = "processing_setup"
    job_status_storage[job_id]["message"] = "Preparing document..."
    
    current_job_data_dir = JOB_DATA_DIR / job_id
    current_job_data_dir.mkdir(parents=True, exist_ok=True)
    job_images_dir = job_specific_temp_dir / "images"
    final_md_path = current_job_data_dir / f"{job_id}_{Path(original_filename).stem}_processed.md"

    try:
        if not pdf_path_on_disk.exists() or pdf_path_on_disk.stat().st_size == 0:
            raise FileNotFoundError(f"Input PDF {pdf_path_on_disk.name} is missing or empty before Datalab.")

        job_status_storage[job_id]["message"] = "Extracting content (Datalab)..."
        marker_result = call_datalab_marker(pdf_path_on_disk, job_id)
        raw_md = marker_result.get("markdown", "")
        img_dict = marker_result.get("images", {})
        if not raw_md.strip(): raise ValueError("Markdown empty after Datalab.")

        job_status_storage[job_id]["message"] = "Processing images & markdown..."
        job_images_dir.mkdir(parents=True, exist_ok=True)
        saved_imgs_map = save_extracted_images_api(img_dict, job_images_dir, job_id)
        processed_md = process_markdown_api(raw_md, saved_imgs_map, job_id)
        final_md_path.write_text(processed_md, encoding="utf-8")
        job_status_storage[job_id].update({
            "processed_markdown_path_relative": str(final_md_path.relative_to(PERSISTENT_DATA_BASE_DIR)),
            "processed_markdown_filename_on_server": final_md_path.name
        })
        if job_images_dir.exists(): shutil.rmtree(job_images_dir)

        job_status_storage[job_id]["message"] = "Chunking, embedding, upserting (Qdrant)..."
        doc_id_qdrant = f"doc_{job_id}_{Path(original_filename).stem.replace('.', '_')}"
        job_status_storage[job_id]["document_id_for_qdrant"] = doc_id_qdrant
        chunks = hierarchical_chunk_markdown(processed_md, original_filename, job_id)
        if not chunks: raise ValueError("No chunks generated.")
        for chunk_data in chunks:
            chunk_data.setdefault('metadata', {})['document_id'] = doc_id_qdrant
            chunk_data['metadata']['session_id'] = job_id
        embeddings = embed_chunks(chunks, job_id)
        if not embeddings: raise ValueError("Embedding failed.")
        if upsert_to_qdrant(job_id, QDRANT_COLLECTION_NAME, embeddings, chunks) == 0:
            raise ValueError("No points upserted to Qdrant.")

        job_status_storage[job_id]["status"] = "generating_initial_question"
        job_status_storage[job_id]["message"] = "Generating hypothetical query..."
        hypo_text = find_topics_and_generate_hypothetical_text(
            job_id, params.academic_level, params.major, params.course_name,
            params.taxonomy_level, params.topics_list, params.marks_for_question
        )
        if hypo_text.startswith("Error:") or not hypo_text.strip(): raise ValueError(f"Hypothetical text gen failed: {hypo_text}")
        if not model_st: raise ValueError("Embedding model unavailable for hypothetical text.")
        query_embed = model_st.encode(clean_text_for_embedding(hypo_text, job_id)).tolist()

        job_status_storage[job_id]["message"] = "Retrieving generation context (Qdrant)..."
        gen_results = search_qdrant(
            job_id, QDRANT_COLLECTION_NAME, query_embed, hypo_text,
            params.retrieval_limit_generation, params.similarity_threshold_generation,
            [doc_id_qdrant], job_id
        )
        if not gen_results: raise ValueError(f"No context from Qdrant for generation (query: {hypo_text[:100]}...).")
        
        gen_ctx_text = "\n\n---\n\n".join(filter(None, [r.payload.get('text', '') for r in gen_results]))
        gen_ctx_meta = [r.payload for r in gen_results]
        job_status_storage[job_id].update({
            "generation_context_text_for_llm": gen_ctx_text,
            "generation_context_snippets_metadata": gen_ctx_meta
        })

        job_status_storage[job_id]["message"] = "Generating initial question..."
        q_text, q_evals, convo_hist, ans_ctx_meta = generate_and_evaluate_question_once(
            job_id, params, gen_ctx_text, [], "", [doc_id_qdrant], job_id, job_specific_temp_dir, 1
        )
        job_status_storage[job_id].update({
            "status": "awaiting_feedback",
            "message": "Initial question generated. Please review.",
            "current_question": q_text, "current_evaluations": q_evals,
            "answerability_context_snippets_metadata": ans_ctx_meta,
            "conversation_history_for_qgen": convo_hist,
            "regeneration_attempts_made": 1,
            "max_regeneration_attempts": MAX_INTERACTIVE_REGENERATION_ATTEMPTS,
        })

    except Exception as e:
        logger.error(f"[{job_id}] Critical error during initial processing/first question: {e}", exc_info=True)
        job_status_storage[job_id].update({
            "status": "error", "message": f"Job setup failed: {str(e)}", "error_details": str(e)
        })
        if job_specific_temp_dir.exists(): shutil.rmtree(job_specific_temp_dir, ignore_errors=True)

def generate_and_evaluate_question_once(
    job_id: str, params: QuestionGenerationRequest, gen_ctx_text: str,
    convo_hist: List[Dict[str, Any]], user_feedback: str, doc_ids_filter: List[str],
    session_id_filter: str, job_specific_temp_dir: Path, attempt_num: int
) -> Tuple[str, Dict[str, Any], List[Dict[str,Any]], List[Dict]]:
    
    logger.info(f"[{job_id}] Generating/Evaluating Question Attempt {attempt_num}")
    placeholders = {
        "academic_level": params.academic_level, "major": params.major, "course_name": params.course_name,
        "taxonomy_level": params.taxonomy_level, "taxonomy": params.taxonomy_level,
        "marks_for_question": params.marks_for_question, "topics_list": params.topics_list,
        "bloom_guidance": get_bloom_guidance(params.taxonomy_level, job_id),
        "blooms_taxonomy_descriptions": get_bloom_guidance(params.taxonomy_level, job_id),
        "retrieved_context": gen_ctx_text, "feedback_on_previous_attempt": user_feedback, "num_questions": "1"
    }
    prompt_path = job_specific_temp_dir / f"qgen_prompt_job_{job_id}_attempt_{attempt_num}.txt"
    user_prompt_qgen = fill_template_string(FINAL_USER_PROMPT_PATH, placeholders, job_id)
    prompt_path.write_text(user_prompt_qgen, encoding='utf-8')

    sys_prompt_qgen = "You are an expert AI specializing in educational content..." # As before
    llm_resp_block = get_gemini_response(job_id, sys_prompt_qgen, user_prompt_qgen, convo_hist)
    
    updated_hist = list(convo_hist)
    updated_hist.append({"role": "user", "parts": [{"text": user_prompt_qgen}]})

    if llm_resp_block.startswith("Error:"):
        return ("Error: LLM failed to generate question.",
                {"error_message": llm_resp_block, "generation_status_message": "LLM API error during question generation."},
                updated_hist, [])

    updated_hist.append({"role": "model", "parts": [{"text": llm_resp_block}]})
    parsed_q_list = parse_questions_from_llm_response(job_id, llm_resp_block, 1)
    if not parsed_q_list:
        return ("Error: Failed to parse question from LLM.",
                {"error_message": "LLM response was unparsable.", "generation_status_message": "Could not parse question from LLM output."},
                updated_hist, [])
    
    curr_q_text = parsed_q_list[0]
    qsts = evaluate_question_qsts(job_id, curr_q_text, gen_ctx_text)
    qual_evals = evaluate_question_qualitative_llm(
        job_id, curr_q_text, gen_ctx_text, params.academic_level, params.major,
        params.course_name, params.taxonomy_level, params.marks_for_question
    )
    is_ans, ans_reason, ans_ctx_meta = evaluate_question_answerability_llm(
        job_id, curr_q_text, params.academic_level, params.major, params.course_name,
        params.taxonomy_level, params.marks_for_question, doc_ids_filter, session_id_filter
    )
    if prompt_path.exists(): prompt_path.unlink(missing_ok=True)

    eval_metrics = {
        "qsts_score": qsts, "qualitative_metrics": qual_evals,
        "llm_answerability": {"is_answerable": is_ans, "reasoning": ans_reason},
        "generation_status_message": f"Question evaluation for attempt {attempt_num} complete."
    }
    return curr_q_text, eval_metrics, updated_hist, ans_ctx_meta

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
    generate_diagrams: bool = Form(False)
):
    job_id = str(uuid.uuid4())
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file. Only PDF allowed.")

    job_temp_dir = TEMP_UPLOAD_DIR / job_id
    job_temp_dir.mkdir(parents=True, exist_ok=True)
    safe_fname = "".join(c for c in Path(file.filename).name if c.isalnum() or c in ('-', '_', '.')) or "upload.pdf"
    temp_pdf = job_temp_dir / f"input_{safe_fname}"

    try:
        content = await file.read()
        if not content: raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        with temp_pdf.open("wb") as buf: buf.write(content)
        logger.info(f"[{job_id}] File '{file.filename}' ({len(content)}B) saved to '{temp_pdf}'")
    except Exception as e:
        if job_temp_dir.exists(): shutil.rmtree(job_temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}") from e
    finally: await file.close()

    req_params = QuestionGenerationRequest(**locals()) # Collect params from Form fields
    job_status_storage[job_id] = {
        "status": "queued", "message": "Job queued for initial processing.",
        "job_params": req_params.model_dump(), "original_filename": file.filename,
        "regeneration_attempts_made": 0, "max_regeneration_attempts": MAX_INTERACTIVE_REGENERATION_ATTEMPTS,
    }
    background_tasks.add_task(process_document_and_generate_first_question,
                              job_id, temp_pdf, file.filename, req_params, job_temp_dir)
    return JobCreationResponse(job_id=job_id, message="Job successfully queued.")

def _get_job_status_response_dict(job_id: str, job_data: Dict) -> Dict:
    """Helper to construct the dictionary for JobStatusResponse, ensuring job_params is typed."""
    response_dict = {k: job_data.get(k) for k in JobStatusResponse.model_fields.keys() if k != 'job_id'}
    job_params_data = job_data.get("job_params")
    if isinstance(job_params_data, dict):
        try:
            response_dict['job_params'] = QuestionGenerationRequest(**job_params_data)
        except Exception: response_dict['job_params'] = None
    elif not isinstance(job_params_data, QuestionGenerationRequest):
        response_dict['job_params'] = None
    return response_dict

@app.post("/regenerate-question/{job_id}", response_model=JobStatusResponse)
async def regenerate_question_endpoint(job_id: str, regen_request: RegenerationRequest):
    if job_id not in job_status_storage: raise HTTPException(status_code=404, detail="Job not found.")
    job_data = job_status_storage[job_id]
    
    status, attempts, max_attempts = job_data.get("status"), job_data.get("regeneration_attempts_made",0), job_data.get("max_regeneration_attempts", MAX_INTERACTIVE_REGENERATION_ATTEMPTS)
    if status not in ["awaiting_feedback", "max_attempts_reached"]:
        raise HTTPException(status_code=400, detail=f"Cannot regenerate from status: {status}")
    if attempts >= max_attempts:
        job_data["status"], job_data["message"] = "max_attempts_reached", "Max regeneration attempts reached."
        return JobStatusResponse(job_id=job_id, **_get_job_status_response_dict(job_id, job_data))

    job_data["status"], job_data["message"] = "regenerating_question", "Regenerating question..."
    job_temp_dir = TEMP_UPLOAD_DIR / job_id; job_temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        params = QuestionGenerationRequest(**job_data["job_params"])
        q_text, q_evals, convo_hist, ans_meta = generate_and_evaluate_question_once(
            job_id, params, job_data["generation_context_text_for_llm"],
            job_data["conversation_history_for_qgen"], regen_request.user_feedback,
            [job_data["document_id_for_qdrant"]], job_id, job_temp_dir, attempts + 1
        )
        job_data.update({
            "current_question": q_text, "current_evaluations": q_evals,
            "conversation_history_for_qgen": convo_hist,
            "answerability_context_snippets_metadata": ans_meta,
            "regeneration_attempts_made": attempts + 1, "error_details": None
        })
        if job_data["regeneration_attempts_made"] >= max_attempts:
            job_data["status"], job_data["message"] = "max_attempts_reached", "Max attempts reached."
        else:
            job_data["status"], job_data["message"] = "awaiting_feedback", "Question regenerated."
    except Exception as e:
        logger.error(f"[{job_id}] Error during regeneration: {e}", exc_info=True)
        job_data.update({
            "status": "awaiting_feedback", "message": f"Regeneration error: {str(e)}",
            "error_details": str(e)
        })
        if isinstance(job_data.get("current_evaluations"), dict): job_data["current_evaluations"]["error_message_regeneration"] = str(e)
        else: job_data["current_evaluations"] = {"error_message_regeneration": str(e)}
        
    return JobStatusResponse(job_id=job_id, **_get_job_status_response_dict(job_id, job_data))

@app.post("/finalize-question/{job_id}", response_model=JobStatusResponse)
async def finalize_question_endpoint(job_id: str, finalize_request: FinalizeRequest):
    if job_id not in job_status_storage: raise HTTPException(status_code=404, detail="Job not found.")
    job_data = job_status_storage[job_id]
    if job_data.get("status") not in ["awaiting_feedback", "max_attempts_reached"]:
        raise HTTPException(status_code=400, detail=f"Cannot finalize from status: {job_data.get('status')}")
    
    current_q = job_data.get("current_question")
    if not current_q or current_q.startswith("Error:"):
         raise HTTPException(status_code=400, detail="Cannot finalize with an errored/missing question.")
    if finalize_request.final_question != current_q:
        logger.warning(f"[{job_id}] Finalizing question mismatch. Using server's current question.")

    job_data["status"], job_data["message"] = "finalizing", "Finalizing job..."
    final_payload = {
        "job_id": job_id, "original_filename": job_data.get("original_filename"),
        "parameters": job_data.get("job_params"), "generated_question": current_q,
        "evaluation_metrics": job_data.get("current_evaluations"),
        "generation_context_snippets_metadata": job_data.get("generation_context_snippets_metadata"),
        "answerability_context_snippets_metadata": job_data.get("answerability_context_snippets_metadata"),
        "processed_markdown_path_relative": job_data.get("processed_markdown_path_relative"),
        "processed_markdown_filename_on_server": job_data.get("processed_markdown_filename_on_server"),
        "total_regeneration_attempts_made": job_data.get("regeneration_attempts_made"),
    }
    result_path = JOB_DATA_DIR / job_id / f"{job_id}_final_interactive_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    try: result_path.write_text(json.dumps(final_payload, indent=2, cls=PathEncoder), encoding="utf-8")
    except Exception as e: logger.error(f"[{job_id}] Failed to save final result JSON: {e}", exc_info=True)

    job_data.update({"final_result": final_payload, "status": "completed", "message": "Job finalized by user."})
    job_temp_dir = TEMP_UPLOAD_DIR / job_id
    if job_temp_dir.exists(): shutil.rmtree(job_temp_dir, ignore_errors=True)
    
    return JobStatusResponse(job_id=job_id, **_get_job_status_response_dict(job_id, job_data))

@app.get("/job-status/{job_id}", response_model=JobStatusResponse)
async def get_job_status_endpoint(job_id: str):
    job_info = job_status_storage.get(job_id)
    if not job_info: raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusResponse(job_id=job_id, **_get_job_status_response_dict(job_id, job_info))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Interactive Question Generation API with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8002)
