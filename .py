# surya_ai_full.py

"""
Surya AI - Full prototype with persistent learning
Filename: surya_ai_full.py
"""

import os
import time
import json
import sqlite3
import datetime
from typing import List, Dict, Optional

# Optional imports with fallback for environments without the dependencies
try:
    import openai
except ImportError:
    openai = None
try:
    import numpy as np
except ImportError:
    np = None
try:
    import faiss
except ImportError:
    faiss = None
try:
    import speech_recognition as sr
except ImportError:
    sr = None
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
try:
    from sympy import sympify, SympifyError
except ImportError:
    sympify = None
    SympifyError = Exception

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if openai and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

FAISS_INDEX_PATH = "surya_faiss.index"
SQLITE_DB_PATH = "surya_knowledge.db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
OPENAI_CHAT_MODEL = "gpt-4o-mini"
TTS_RATE_MULTIPLIER = 0.95

# The rest of the classes can now safely check for None before using external modules.
