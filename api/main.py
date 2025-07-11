#!/usr/bin/env python3
"""
Advanced Genshin Impact Lore System
Multi-source scraper with semantic chunking, caching, web interface, and real-time updates
"""
import asyncio
import os
import sqlite3
import logging
# Set logging level to DEBUG to get more detailed output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import re
import time
import hashlib
import requests
from datetime import datetime, timedelta
from functools import lru_cache
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque # Added for BFS crawling

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from pathlib import Path

# FastAPI related imports
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware # Import CORS middleware

# Pydantic for data validation
from pydantic import BaseModel

# Third-party libraries
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import aiohttp
import redis.asyncio as aioredis_client
import redis as redis_sync_client
import praw
import yt_dlp
import schedule
import faiss
import numpy as np
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import tiktoken
import json
import google.generativeai as genai

# BeautifulSoup for web scraping
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# --- Configuration ---
# API Keys (from environment variables)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# API Key for protected endpoints (NEW)
API_KEY = os.getenv("API_KEY")

# Scraper settings
MAX_CONCURRENT_WIKI_REQUESTS = 5 # Max concurrent requests for wiki
MAX_CONCURRENT_REDDIT_REQUESTS = 5
MAX_CONCURRENT_YOUTUBE_REQUESTS = 3
TIMEOUT_SECONDS = 45 # Increased timeout to 45 seconds for robustness

# Database settings
DB_NAME = "genshin_lore.db"
VECTOR_DIMENSION = 384
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Security Dependency for API Key (NEW) ---
async def get_api_key(x_api_key: str = Header(...)):
    if API_KEY is None:
        logger.error("API_KEY environment variable is not set.")
        raise HTTPException(status_code=500, detail="Server configuration error: API key not set.")
    if x_api_key != API_KEY:
        logger.warning(f"Unauthorized access attempt with key: {x_api_key[:5]}...")
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key")
    return x_api_key

# --- Data Models ---

@dataclass
class LoreEntry:
    entry_id: str
    title: str
    content: str
    source_type: str
    source_url: str
    last_updated: str

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class QueryResponse(BaseModel):
    results: List[LoreEntry]
    message: str = "Query successful"

class SourceSummary(BaseModel):
    source_type: str
    count: int
    last_scraped: Optional[str]
    next_scrape: Optional[str]

class StatsResponse(BaseModel):
    total_entries: int
    total_chunks: int
    source_details: List[SourceSummary]

class ScrapeRequest(BaseModel):
    action: str
    source_type: Optional[str] = None

class ScrapeResponse(BaseModel):
    message: str
    status: str

class ChatRequest(BaseModel):
    # Changed 'message' to 'query' to match frontend expectation
    query: str

class ChatResponse(BaseModel):
    response: str
    source_lore: List[Dict[str, Any]] = []

# --- Caching with Redis ---

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0):
        self._sync_client = None
        self._async_client = None
        self.host = host
        self.port = port
        self.db = db
        try:
            # Initialize synchronous client
            self._sync_client = redis_sync_client.StrictRedis(host=self.host, port=self.port, db=self.db, decode_responses=True)
            self._sync_client.ping()
            logger.info("Redis synchronous client connected successfully")
        except redis_sync_client.exceptions.ConnectionError as e:
            logger.error(f"Could not connect to synchronous Redis: {e}")
            self._sync_client = None

    async def _get_async_client(self):
        if not self._async_client:
            try:
                # Initialize async client only when first needed
                self._async_client = aioredis_client.Redis(host=self.host, port=self.port, db=self.db, decode_responses=True)
                await self._async_client.ping()
                logger.info("Redis asynchronous client connected successfully")
            except Exception as e:
                logger.error(f"Could not connect to asynchronous Redis: {e}")
                self._async_client = None
        return self._async_client

    async def get(self, key: str) -> Optional[str]:
        client = await self._get_async_client()
        if client:
            try:
                return await client.get(key)
            except Exception as e:
                logger.error(f"Error getting from Redis cache: {e}")
                return None
        return None

    async def set(self, key: str, value: str, ttl: int = 3600):
        client = await self._get_async_client()
        if client:
            try:
                await client.setex(key, ttl, value)
            except Exception as e:
                logger.error(f"Error setting to Redis cache: {e}")

    def get_sync(self, key: str) -> Optional[str]:
        if self._sync_client:
            try:
                return self._sync_client.get(key)
            except Exception as e:
                logger.error(f"Error getting from sync Redis cache: {e}")
                return None
        return None

    def set_sync(self, key: str, value: str, ttl: int = 3600):
        if self._sync_client:
            try:
                self._sync_client.setex(key, ttl, value)
            except Exception as e:
                self._sync_client = None # Invalidate client on error
                logger.error(f"Error setting to sync Redis cache: {e}")

# --- Database Management ---

class LoreDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lore_entries (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_url TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                content_hash TEXT NOT NULL
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                embedding BLOB,
                FOREIGN KEY (entry_id) REFERENCES lore_entries (id) ON DELETE CASCADE
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scrape_schedule (
                source_type TEXT PRIMARY KEY,
                last_scraped TEXT,
                next_scrape TEXT
            );
        """)
        conn.commit()
        conn.close()

    def add_or_update_entry(self, entry: LoreEntry) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_hash = hashlib.sha256(entry.content.encode()).hexdigest()
        
        cursor.execute("SELECT content_hash FROM lore_entries WHERE id = ?", (entry.entry_id,))
        result = cursor.fetchone()
        existing_hash = result[0] if result else None

        if existing_hash and existing_hash == current_hash:
            logger.debug(f"Entry '{entry.title}' with ID '{entry.entry_id}' content hash matches existing. Skipping update.")
            conn.close()
            return False
            
        cursor.execute("""
            INSERT OR REPLACE INTO lore_entries (id, title, content, source_type, source_url, last_updated, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (entry.entry_id, entry.title, entry.content, entry.source_type, entry.source_url, entry.last_updated, current_hash))
        
        conn.commit()
        conn.close()
        logger.debug(f"Entry '{entry.title}' with ID '{entry.entry_id}' was added or updated.")
        return True

    def get_all_entries_with_chunks(self) -> List[Tuple[str, str, bytes]]: # Changed return type to include bytes for embedding
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, chunk_text, embedding FROM chunks WHERE embedding IS NOT NULL")
        chunks = cursor.fetchall()
        conn.close()
        return chunks
        
    def get_entry_by_id(self, entry_id: str) -> Optional[LoreEntry]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, content, source_type, source_url, last_updated FROM lore_entries WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return LoreEntry(
                entry_id=row[0],
                title=row[1],
                content=row[2],
                source_type=row[3],
                source_url=row[4],
                last_updated=row[5]
            )
        return None

    def get_chunks_for_entry(self, entry_id: str) -> List[Tuple[int, str, Optional[bytes]]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, chunk_text, embedding FROM chunks WHERE entry_id = ? ORDER BY chunk_index", (entry_id,))
        chunks = cursor.fetchall()
        conn.close()
        return chunks

    def delete_chunks_for_entry(self, entry_id: str): # Corrected: 'entry_id' is the parameter
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunks WHERE entry_id = ?", (entry_id,)) # Corrected: use 'entry_id' directly
        conn.commit()
        conn.close()
        logger.debug(f"Deleted existing chunks for entry ID '{entry_id}'.")

    def add_chunk(self, entry_id: str, chunk_text: str, chunk_index: int, embedding: np.ndarray):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        embedding_bytes = embedding.tobytes() if embedding is not None else None
        cursor.execute("""
            INSERT INTO chunks (entry_id, chunk_text, chunk_index, embedding)
            VALUES (?, ?, ?, ?)
        """, (entry_id, chunk_text, chunk_index, embedding_bytes))
        conn.commit()
        conn.close()
        
    def get_all_chunks_with_embeddings(self) -> List[Tuple[int, str, bytes]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, chunk_text, embedding FROM chunks WHERE embedding IS NOT NULL")
        chunks = cursor.fetchall()
        conn.close()
        return chunks

    def update_scrape_schedule(self, source_type: str, last_scraped: datetime, next_scrape: datetime):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO scrape_schedule (source_type, last_scraped, next_scrape)
            VALUES (?, ?, ?)
        """, (source_type, last_scraped.isoformat(), next_scrape.isoformat()))
        conn.commit()
        conn.close()
        
    def get_scrape_schedule(self) -> List[Tuple[str, Optional[str], Optional[str]]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT source_type, last_scraped, next_scrape FROM scrape_schedule")
        schedule_data = cursor.fetchall()
        conn.close()
        return schedule_data

# --- Vector Database (FAISS) ---

class EnhancedVectorDB:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.index_path = self.data_dir / "lore_faiss.index"
        self.metadata_path = self.data_dir / "lore_faiss.metadata"
        self.index = None
        self.chunk_ids = []

        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Loading SentenceTransformer: all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info(f"Using device: {self.embedding_model.device}")
        self.vector_dimension = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Vector dimension: {self.vector_dimension}")

    def _create_index(self, embeddings: np.ndarray):
        if embeddings.size == 0:
            self.index = None
            self.chunk_ids = []
            return

        self.index = faiss.IndexFlatL2(self.vector_dimension)
        self.index.add(embeddings)
        
    def _save_index(self):
        if self.index:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, 'w') as f:
                json.dump(self.chunk_ids, f)
            logger.info("Vector index and metadata saved.")

    def _load_index(self):
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, 'r') as f:
                    self.chunk_ids = json.load(f)
                logger.info("Vector index and metadata loaded successfully.")
                return True
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                return False
        return False

    def load_or_create_index(self, db_path: Path):
        if self._load_index():
            return

        logger.info("Vector index not found. Rebuilding from database.")
        self.rebuild_index_from_db(db_path)

    def rebuild_index_from_db(self, db_path: Path):
        db = LoreDatabase(db_path)
        chunks_data = db.get_all_chunks_with_embeddings()
        
        if not chunks_data:
            logger.info("No chunks in database to build index from.")
            self.index = None
            self.chunk_ids = []
            self._save_index()
            return
            
        self.chunk_ids = [chunk[0] for chunk in chunks_data]
        embeddings = np.array([np.frombuffer(chunk[2], dtype=np.float32) for chunk in chunks_data])
        
        self._create_index(embeddings)
        self._save_index()
        logger.info(f"Rebuilt FAISS index with {len(self.chunk_ids)} chunks.")

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        if not self.index:
            logger.warning("FAISS index not loaded. Cannot perform search.")
            return []

        query_embedding = self.embedding_model.encode(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        D, I = self.index.search(query_embedding, k)
        
        results = []
        for i, distance in zip(I[0], D[0]):
            if i == -1:
                continue
            chunk_db_id = self.chunk_ids[i]
            results.append((chunk_db_id, float(distance)))
        return results

# --- Multi-Source Scraper ---

class MultiSourceScraper:
    def __init__(self, data_dir: Path, embedding_model: SentenceTransformer):
        self.data_dir = data_dir
        self.db_path = self.data_dir / DB_NAME
        self.db = LoreDatabase(self.db_path)
        self.redis_cache = RedisCache(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        self.logger = logger
        self.embedding_model = embedding_model

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Reddit API setup
        if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT:
            self.reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT
            )
            logger.info("Reddit API client initialized.")
        else:
            self.reddit = None
            logger.warning("Reddit API credentials not set. Reddit scraping disabled.")

        # NLTK setup
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK 'punkt' tokenizer...")
            nltk.download('punkt')

        # SpaCy setup
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully.")
        except OSError:
            logger.warning("SpaCy model not found. Using NLTK fallback.")
            
        self._initialize_scrape_schedule()
            
    def _initialize_scrape_schedule(self):
        schedule_data = self.db.get_scrape_schedule()
        existing_sources = {s[0] for s in schedule_data}
        
        default_frequencies = {
            'wiki': 24,
            'reddit': 2,
            'youtube': 12
        }

        for source_type, freq_hours in default_frequencies.items():
            if source_type not in existing_sources:
                now = datetime.now()
                next_scrape = now + timedelta(hours=freq_hours)
                self.db.update_scrape_schedule(source_type, now, next_scrape)
                logger.info(f"Initialized scrape schedule for {source_type}")

    def _get_updated_entry_ids(self, entries: List[LoreEntry]) -> List[str]:
        """Helper to get a list of entry IDs that were potentially updated/added."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        updated_ids = []
        for entry in entries:
            cursor.execute("SELECT 1 FROM lore_entries WHERE id = ? AND content_hash = ?", 
                           (entry.entry_id, hashlib.sha256(entry.content.encode()).hexdigest()))
            if not cursor.fetchone(): # If no matching hash, it was updated or new
                updated_ids.append(entry.entry_id)
        conn.close()
        return updated_ids

    # --- Wiki Scraping (USING AIOHTTP AND BEAUTIFULSOUP) ---

    async def _fetch_wiki_page_content_bs4(self, session: aiohttp.ClientSession, page_title: str, semaphore: asyncio.Semaphore) -> Tuple[Optional[str], Optional[str], List[str]]:
        """
        Fetches the content of a Fandom wiki page using aiohttp and parses it with BeautifulSoup.
        Uses a semaphore to limit concurrent requests.
        Returns the page content, its URL, and a list of internal links found on the page.
        """
        base_url = "https://genshin-impact.fandom.com/wiki/"
        page_url = urljoin(base_url, page_title.replace(" ", "_"))
        
        content = None
        internal_links = []

        async with semaphore: # Acquire a semaphore slot before making the request
            try:
                async with session.get(page_url, timeout=TIMEOUT_SECONDS) as response:
                    response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
                    html_content = await response.text()
                    
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Attempt to find the main content area. Common Fandom content divs:
                    content_div = soup.find('div', class_='mw-parser-output') or \
                                  soup.find('div', id='mw-content-text') or \
                                  soup.find('div', class_='WikiaArticle') or \
                                  soup.find('div', class_='article-content')

                    if content_div:
                        # Remove unwanted elements (e.g., navigation, infoboxes, sidebars, tables of contents, edit buttons)
                        for tag_class in ['mw-editsection', 'toc', 'infobox', 'portable-infobox', 'aside', 'gallerybox', 'tabber', 'reference', 'reflist', 'mw-references-columns']:
                            for tag in content_div.find_all(class_=tag_class):
                                tag.decompose()
                        
                        # Remove script and style tags
                        for script_or_style in content_div(['script', 'style', 'noscript']):
                            script_or_style.decompose()

                        # Extract text, stripping excessive whitespace
                        content = content_div.get_text(separator=' ', strip=True)
                        content = re.sub(r'\s+', ' ', content).strip() # Normalize whitespace
                        self.logger.debug(f"Extracted content length for '{page_title}': {len(content)}")

                        # Extract internal links
                        for link in content_div.find_all('a', href=True):
                            href = link['href']
                            # Check if it's an internal wiki link and not a special page/file/category
                            if href.startswith('/wiki/') and not any(href.startswith(f'/wiki/{prefix}') for prefix in ["Category:", "File:", "Special:", "MediaWiki:", "Help:", "User:", "Template:"]):
                                link_title = href.replace('/wiki/', '').replace('_', ' ')
                                if link_title: # Ensure title is not empty after cleaning
                                    internal_links.append(link_title)
                    else:
                        self.logger.warning(f"Could not find main content div for {page_url}")

            except aiohttp.ClientError as e:
                self.logger.error(f"HTTP error fetching {page_url}: {e}")
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout fetching {page_url}")
            except Exception as e:
                self.logger.error(f"Error parsing {page_url}: {e}", exc_info=True)
                
            return content, page_url, list(set(internal_links)) # Return unique links

    async def scrape_wiki_async(self) -> List[LoreEntry]:
        self.logger.info("Starting comprehensive wiki scraping with BeautifulSoup...")
        all_entries_collected = [] # Collect all entries here
        
        page_queue = deque()
        seen_for_processing = set() 

        # Initial seed pages for the crawl - EXPANDED FOR MORE WORLD LORE, QUESTS, EVENTS
        initial_seed_pages = [
            # Core Characters & Regions (already present)
            "Ganyu", "Raiden Shogun", "Zhongli", "Paimon", "Traveler", "Nahida", "Xiao", "Klee", "Venti", "Albedo", "Hu Tao",
            "Liyue", "Mondstadt", "Inazuma", "Sumeru", "Fontaine", "Natlan", "Snezhnaya", "Khaenri'ah", "Celestia",
            # Core Lore Concepts (already present)
            "Abyss Order", "Vision", "Archons", "Fatui", "Seven Archons", "Harbingers", "Aether", "Lumine", "Dainsleif",
            "Elemental Sight", "Ley Lines", "Teyvat", "Lore",
            # Elements (already present)
            "Dendro", "Anemo", "Electro", "Geo", "Hydro", "Pyro", "Cryo",
            # Specific Locations/Concepts (already present)
            "Primordial One", "Phanes", "Enkanomiya", "Dragonspine", "The Chasm", "Honkai Impact 3rd",
            # Expanded World Lore
            "History of Teyvat", "Timeline (Genshin Impact)", "Cosmology (Genshin Impact)", "Gods (Genshin Impact)",
            "The Seven Sovereigns", "Dragon (Genshin Impact)", "Adeptus", "Seelie", "Hilichurl", "Fatui",
            "Ruins (Genshin Impact)", "Domains (Genshin Impact)", "Abyss (Genshin Impact)",
            # Expanded Quests
            "Archon Quest", "Story Quest", "World Quest", "Hangout Event", "Commission (Genshin Impact)",
            # Expanded Events
            "Events", "Lantern Rite", "Windblume Festival", "Moonchase Festival", "Summer Fantasia", "Irodori Festival",
            "Golden Apple Archipelago", "Chalk Prince and the Dragon", "Perilous Trail",
            # General Game Concepts
            "Weapons", "Artifacts", "Elements", "Combat", "Exploration", "Achievements",
            # NEW: In-game Books and related categories
            "Books", "Collected Miscellany", "Series of Books", "In-game Books", "Lore Books",
            "A Drunkard's Tale", "The Byakuyakoku Collection", "Records of the Fall", "The Pale Princess and the Six Pygmies",
            "Rex Incognito", "Vernal Winds of the New World", "The Legend of Vennessa", "The Winding River of Time",
            "The Yakshas: The Guardian Adepti", "Moonlit Bamboo Forest", "A Thousand Questions With Paimon",
            "Heart's Desire", "The Boar Princess", "The Fox in the Dandelion Sea", "The Saga of the Frostbearer",
            "The Great Thunderbird", "The Tale of Shirikoro Peak", "The Sunken Pearl", "The Dragon-Devouring Deep",
            "The Vishap and the Serpent", "The Divine Will", "The Price of Freedom", "The Moon-Bathed Deep",
            "The Narzissenkreuz Ordo", "The Seven Sages", "The History of the Knights of Favonius",
            "The Records of Jueyun", "The Stone Tablet Compendium", "The Story of the Stone", "The Tale of the Cruising Cloud",
            "The Legend of the Shattered Halberd", "The Customs of Liyue", "The Art of Negotiation",
            "The Glaze Lily and the Traveler", "The Rite of Parting", "The Solitary Sea-Beast", "The Tale of the Flaming Heart",
            "The Adepti and the Yakshas", "The Story of the God of Dust", "The History of the Guili Assembly",
            "The Legend of the Geo Archon", "The Origin of the Abyss", "The Cat's Tail", "The Teyvat Travel Guide",
            "The Adventurers' Guild Guide", "The Favonius Handbook", "The Knights of Favonius Handbook",
            "The Mondstadt History of Wine", "The Liyue Cuisine Guide", "The Inazuma Cuisine Guide",
            "The Sumeru Cuisine Guide", "The Fontaine Cuisine Guide", "The Natlan Cuisine Guide",
            "The Snezhnaya Cuisine Guide"
        ]
        
        for title in initial_seed_pages:
            if title not in seen_for_processing:
                page_queue.append(title)
                seen_for_processing.add(title)

        self.logger.info(f"Starting crawl with {len(initial_seed_pages)} seed pages.")
        
        async with aiohttp.ClientSession() as session:
            pages_processed = 0
            # Increased max_pages_to_crawl to allow for more extensive book scraping
            max_pages_to_crawl = 2000 # Increased from 1000 to 2000
            batch_size = 50 # Save and rebuild index every 50 pages

            # Semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_WIKI_REQUESTS)

            tasks = []
            while page_queue and pages_processed < max_pages_to_crawl:
                # Add tasks to the list until semaphore limit or queue is empty
                while page_queue and len(tasks) < MAX_CONCURRENT_WIKI_REQUESTS and pages_processed < max_pages_to_crawl:
                    current_page_title = page_queue.popleft()
                    self.logger.info(f"Queueing page: {current_page_title} (pages processed: {pages_processed}/{max_pages_to_crawl}, queue size: {len(page_queue)})")
                    tasks.append(asyncio.create_task(self._fetch_wiki_page_content_bs4(session, current_page_title, semaphore)))
                    
                if not tasks: # If no tasks were added and queue is empty, break
                    break

                # Wait for tasks to complete
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                
                for task in done:
                    try:
                        content, source_url, discovered_links = await task
                        if content and len(content) >= 200:
                            entry_id = f"wiki_{hashlib.sha256(content.encode()).hexdigest()}" 
                            title_from_url = source_url.split('/')[-1].replace('_', ' ')
                            
                            entry = LoreEntry(
                                entry_id=entry_id,
                                title=title_from_url,
                                content=content,
                                source_type="wiki",
                                source_url=source_url,
                                last_updated=datetime.now().isoformat()
                            )
                            all_entries_collected.append(entry)
                            pages_processed += 1
                            self.logger.debug(f"Added entry for '{entry.title}'. Total entries collected: {len(all_entries_collected)}, Pages processed: {pages_processed}")
                            
                            for link_title in discovered_links:
                                if link_title not in seen_for_processing and pages_processed + len(page_queue) < max_pages_to_crawl:
                                    page_queue.append(link_title)
                                    seen_for_processing.add(link_title)

                            # Save entries in batches and rebuild index
                            if pages_processed % batch_size == 0:
                                self.logger.info(f"Saving {batch_size} wiki entries and rebuilding index (total processed: {pages_processed})...")
                                new_entries, new_chunks = self.save_entries_with_chunks(all_entries_collected[-batch_size:])
                                if new_entries > 0 or new_chunks > 0:
                                    self.vector_db.rebuild_index_from_db(self.db_path)
                                self.logger.info(f"Batch save complete. Added {new_entries} entries, {new_chunks} chunks.")
                        else:
                            self.logger.warning(f"Skipping page due to insufficient content or fetch error.")
                    except Exception as e:
                        self.logger.error(f"Error processing completed task: {e}", exc_info=True)

                # Update tasks list to include only pending tasks
                tasks = list(pending)

            # After the loop, save any remaining entries
            if all_entries_collected and pages_processed % batch_size != 0:
                self.logger.info(f"Saving remaining wiki entries and rebuilding index (total processed: {pages_processed})...")
                new_entries, new_chunks = self.save_entries_with_chunks(all_entries_collected[-(pages_processed % batch_size):])
                if new_entries > 0 or new_chunks > 0:
                    self.vector_db.rebuild_index_from_db(self.db_path)
                self.logger.info(f"Final batch save complete. Added {new_entries} entries, {new_chunks} chunks.")

        self.logger.debug(f"scrape_wiki_async completed. Returning {len(all_entries_collected)} entries.")
        return all_entries_collected

    # --- Reddit Scraping ---

    async def scrape_reddit_async(self) -> List[LoreEntry]:
        self.logger.info("Starting Reddit scraping...")
        if not self.reddit:
            self.logger.warning("Reddit API not configured.")
            return []

        all_entries = []
        # Updated subreddit list to include r/Genshin_Impact_Lore
        subreddit_names = ["Genshin_Impact_Lore", "Genshin_Lore", "GenshinImpact", "Genshin_Impact_Leaks"]
        
        try:
            for sub_name in subreddit_names:
                subreddit = await asyncio.to_thread(self.reddit.subreddit, sub_name)
                
                # Fetch top posts (trending) - Increased limit to 100
                top_posts = await asyncio.to_thread(lambda: list(subreddit.top(time_filter="month", limit=100)))
                # Fetch new posts - Increased limit to 100
                new_posts = await asyncio.to_thread(lambda: list(subreddit.new(limit=100)))

                # Combine and deduplicate posts
                processed_ids = set()
                for submission in top_posts + new_posts:
                    # Filter for "Genshin lore" or "Genshin analysis" in title or selftext
                    if submission.id not in processed_ids and submission.is_self and submission.selftext and len(submission.selftext) > 100:
                        combined_text = f"{submission.title} {submission.selftext}".lower()
                        if "genshin lore" in combined_text or "genshin analysis" in combined_text:
                            entry_id = f"reddit_{submission.id}"
                            
                            entry = LoreEntry(
                                entry_id=entry_id,
                                title=submission.title,
                                content=submission.selftext,
                                source_type="reddit",
                                # Use permalink as source_url
                                source_url=f"https://www.reddit.com{submission.permalink}",
                                last_updated=datetime.now().isoformat()
                            )
                            all_entries.append(entry)
                            processed_ids.add(submission.id)
                        
        except Exception as e:
            self.logger.error(f"Error during Reddit scraping: {e}", exc_info=True)
            
        self.logger.info(f"Reddit scraping completed. Found {len(all_entries)} entries.")
        return all_entries

    # --- YouTube Scraping ---

    async def scrape_youtube_async(self) -> List[LoreEntry]:
        self.logger.info("Starting YouTube scraping...")
        all_entries = []
        
        # General search terms for trending/new lore videos - Added new search term
        search_terms = [
            "Genshin Impact lore explained", 
            "Genshin Impact story analysis", 
            "Genshin Impact new lore", 
            "Genshin Impact theories",
            "Genshin Impact Lore Analysis Explain" # NEW Search Term
        ]
        
        # Specific YouTube channel URLs or IDs
        channel_urls = [
            "https://www.youtube.com/@Ashikai/videos", 
            "https://www.youtube.com/@MyNameForNow/videos",
            "https://www.youtube.com/@Minslief/videos",
            "https://www.youtube.com/@DragonMJE/videos"
        ]

        # Combine all sources to scrape
        sources_to_scrape = search_terms + channel_urls
        
        processed_video_ids = set() # To avoid duplicate entries

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_YOUTUBE_REQUESTS) as executor:
            loop = asyncio.get_event_loop()
            
            for source in sources_to_scrape:
                try:
                    ydl_opts = {
                        'quiet': True,
                        'extract_flat': True, # Extract flat list of videos (for channels/playlists)
                        'default_search': 'ytsearch10' if source in search_terms else None, # Search for 10 videos if it's a search term
                        'skip_download': True,
                        'playlist_items': '1:50' if source in channel_urls else None, # Get up to 50 recent videos from channels
                        'format': 'bestaudio/best', # Minimal format to get metadata
                        'noplaylist': True, # Do not download playlist if it's a single video URL
                    }
                    
                    info_dict = await loop.run_in_executor(
                        executor, 
                        lambda: yt_dlp.YoutubeDL(ydl_opts).extract_info(source, download=False)
                    )
                    
                    if info_dict:
                        entries_to_process = []
                        if 'entries' in info_dict:
                            entries_to_process.extend(info_dict['entries'])
                        elif 'id' in info_dict: # Single video result from a direct URL or search
                            entries_to_process.append(info_dict)

                        for entry_data in entries_to_process:
                            if entry_data and 'id' in entry_data and entry_data['id'] not in processed_video_ids:
                                video_id = entry_data['id']
                                title = entry_data.get('title', '')
                                url = entry_data.get('webpage_url', f"https://youtube.com/watch?v={video_id}")
                                description = entry_data.get('description', '')
                                
                                combined_content = f"{title}\n\n{description}"
                                
                                # Attempt to get transcript for the video
                                transcript = await self._get_youtube_transcript(url)
                                if transcript:
                                    combined_content += f"\n\nTranscript:\n{transcript}"
                                    self.logger.debug(f"Successfully retrieved transcript for video: {title}")
                                else:
                                    self.logger.warning(f"No transcript found for video: {title} ({url})")

                                # Filter for "Genshin lore" or "Genshin analysis" in title or description or transcript
                                if len(combined_content) > 100: # Ensure meaningful content
                                    lower_combined_content = combined_content.lower()
                                    # Enhanced keyword check for better relevance
                                    if (("genshin" in lower_combined_content and "lore" in lower_combined_content) or
                                        ("genshin" in lower_combined_content and "analysis" in lower_combined_content) or
                                        ("genshin impact lore" in lower_combined_content)): # Explicitly include the full phrase
                                        entry = LoreEntry(
                                            entry_id=f"youtube_{video_id}",
                                            title=title,
                                            content=combined_content,
                                            source_type="youtube",
                                            source_url=url,
                                            last_updated=datetime.now().isoformat()
                                        )
                                        all_entries.append(entry)
                                        processed_video_ids.add(video_id)
                                    else:
                                        self.logger.debug(f"Skipping YouTube video '{title}' (ID: {video_id}) - no relevant keywords.")
                                else:
                                    self.logger.debug(f"Skipping YouTube video '{title}' (ID: {video_id}) - content too short.")
                                    
                except Exception as e:
                    self.logger.error(f"Error scraping YouTube for '{source}': {e}", exc_info=True)
                    
        self.logger.info(f"YouTube scraping completed. Found {len(all_entries)} entries.")
        return all_entries

    async def _get_youtube_transcript(self, video_url: str) -> Optional[str]:
        """Fetches the transcript for a given YouTube video URL."""
        try:
            ydl_opts = {
                'writesubtitles': True,
                'subtitleslangs': ['en'],
                'skip_download': True,
                'quiet': True,
                'force_generic_extractor': True,
                'outtmpl': 'temp_transcript' # Dummy output template
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # This will download the subtitle file to temp_transcript.en.vtt
                info = await asyncio.to_thread(ydl.extract_info, video_url, download=True)
                # Read the VTT file
                subtitle_file_path = Path(f"temp_transcript.en.vtt")
                if subtitle_file_path.exists():
                    with open(subtitle_file_path, 'r', encoding='utf-8') as f:
                        vtt_content = f.read()
                    subtitle_file_path.unlink() # Clean up temp file
                    # Parse VTT content (very basic parsing, just extract text)
                    lines = vtt_content.split('\n')
                    transcript_lines = []
                    for line in lines:
                        if '-->' not in line and not line.startswith('WEBVTT') and not line.startswith('Kind:') and line.strip():
                            transcript_lines.append(line.strip())
                    return " ".join(transcript_lines)
        except Exception as e:
            logger.error(f"Error getting transcript for {video_url}: {e}")
        return None

    def save_entries_with_chunks(self, entries: List[LoreEntry]) -> Tuple[int, int]:
        new_entries_count = 0
        new_chunks_count = 0
        self.logger.debug(f"save_entries_with_chunks called with {len(entries)} entries.")

        for entry in entries:
            is_new_or_updated = self.db.add_or_update_entry(entry)
            
            if is_new_or_updated:
                new_entries_count += 1
                self.db.delete_chunks_for_entry(entry.entry_id)
                
                chunks = self._chunk_text(entry.content)
                for i, chunk_text in enumerate(chunks):
                    embedding = self.embed_text(chunk_text)
                    self.db.add_chunk(entry.entry_id, chunk_text, i, embedding)
                    new_chunks_count += 1
                    
                self.logger.info(f"Added/Updated: '{entry.title}' with {len(chunks)} chunks")
            else:
                self.logger.debug(f"Entry '{entry.title}' (ID: {entry.entry_id}) was not new or updated, skipping chunking.")
                
        self.logger.info(f"Saved {new_entries_count} entries, {new_chunks_count} chunks")
        return new_entries_count, new_chunks_count

    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []

        try:
            # Use tiktoken for better chunking
            token_encoder = tiktoken.get_encoding("cl100k_base")
            tokens = token_encoder.encode(text)
            
            chunks = []
            for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_tokens = tokens[i : i + CHUNK_SIZE]
                chunk_text = token_encoder.decode(chunk_tokens)
                chunks.append(chunk_text)
                
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error chunking text: {e}")
            # Simple fallback chunking if tiktoken fails
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) > CHUNK_SIZE and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else []
                    current_length = sum(len(w) for w in current_chunk)
                
                current_chunk.append(word)
                current_length += len(word) + 1
                
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                
            return chunks

    @lru_cache(maxsize=1000)
    def embed_text(self, text: str) -> np.ndarray:
        return self.embedding_model.encode(text)

# --- Scheduled Scraper ---

class ScheduledScraper:
    def __init__(self, scraper: MultiSourceScraper, vector_db: EnhancedVectorDB):
        self.scraper = scraper
        self.vector_db = vector_db
        self._scheduler_thread = None
        self._running = False
        self._lock = threading.Lock()

    def start_scheduler(self):
        with self._lock:
            if self._running:
                logger.info("Scheduler already running.")
                return
            self._running = True
            self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self._scheduler_thread.start()
            logger.info("Scheduled scraper started.")

    def _run_scheduler(self):
        schedule.clear()
        
        # Set up periodic tasks 
        schedule.every(24).hours.do(self._safe_scrape_and_update, 'wiki')
        schedule.every(2).hours.do(self._safe_scrape_and_update, 'reddit') # Now scheduled
        schedule.every(12).hours.do(self._safe_scrape_and_update, 'youtube') # Now scheduled

        while self._running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def _safe_scrape_and_update(self, source_type: str):
        """Thread-safe wrapper for scraping"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._scrape_and_update(source_type))
        except Exception as e:
            logger.error(f"Error in scheduled scrape for {source_type}: {e}", exc_info=True)
        finally:
            loop.close()

    async def _scrape_and_update(self, source_type: str):
        logger.info(f"Starting scheduled scraping for {source_type}")
        entries = []
        
        try:
            if source_type == 'wiki':
                entries = await self.scraper.scrape_wiki_async()
                logger.debug(f"scrape_wiki_async returned {len(entries)} entries.") # New log
            elif source_type == "reddit":
                entries = await self.scraper.scrape_reddit_async()
            elif source_type == "youtube":
                entries = await self.scraper.scrape_youtube_async()
            else:
                self.logger.error(f"Invalid source_type for scheduled scrape: {source_type}")
                return

        except Exception as e:
            self.logger.error(f"Error during scheduled {source_type} scrape: {e}", exc_info=True)
            return

        # Note: save_entries_with_chunks and rebuild_index_from_db are now handled
        # in batches within scrape_wiki_async for wiki source type.
        # For other source types, it still saves and rebuilds once at the end.
        if source_type != 'wiki': # Only do this if not wiki (as wiki handles its own saving/rebuilding)
            new_entries, new_chunks_count = scraper.save_entries_with_chunks(entries)
            if new_entries > 0 or new_chunks_count > 0:
                self.vector_db.rebuild_index_from_db(self.scraper.db_path)
            self.logger.info(f"Finished scheduled scraping for {source_type}. Added {new_entries} entries, {new_chunks_count} chunks.")
        else:
            self.logger.info(f"Finished scheduled scraping for {source_type}.") # Wiki's saving is handled internally

        # Update scrape schedule in DB
        now = datetime.now()
        # Determine frequency based on source_type, fallback to defaults
        if source_type == 'wiki':
            current_freq_hours = 24
        elif source_type == 'reddit':
            current_freq_hours = 2
        elif source_type == 'youtube':
            current_freq_hours = 12
        else:
            current_freq_hours = 24 # Fallback, should not happen with validation above

        next_scrape_time = now + timedelta(hours=current_freq_hours)
        self.scraper.db.update_scrape_schedule(source_type, now, next_scrape_time)
        self.logger.info(f"Next run for {source_type} due at {next_scrape_time.isoformat()}")

    def stop_scheduler(self):
        with self._lock:
            self._running = False
            if self._scheduler_thread and self._scheduler_thread.is_alive():
                self._scheduler_thread.join(timeout=5)
                logger.info("Scheduled scraper stopped.")

    def reschedule_all_scrapes(self):
        with self._lock:
            logger.info("Rescheduling all scrapes...")
            self.stop_scheduler()
            
            # Clear existing entries and re-initialize to set next_scrape to now + freq
            conn = sqlite3.connect(self.scraper.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM scrape_schedule")
            conn.commit()
            conn.close()
            
            self.scraper._initialize_scrape_schedule() # Re-initializes with current time + freq
            self.start_scheduler()
            logger.info("All scraping jobs rescheduled and scheduler restarted.")

# --- FastAPI App ---

app = FastAPI()

# Add CORS middleware to allow requests from your frontend (NEW)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins. For production, replace with your Vercel domain.
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers, including X-API-Key
)

# Get the directory where api/main.py is located
# This will be '/var/task/api' in Vercel Lambda
current_dir = os.path.dirname(os.path.abspath(__file__))

# The project root is the parent directory of 'api'
# This will be '/var/task' in Vercel Lambda
project_root = os.path.dirname(current_dir)

# Log the calculated project_root for debugging
logger.info(f"Calculated project_root for static files: {project_root}")

# Explicitly serve index.html from the project root
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # index.html is at the project root, one level up from api/main.py
    index_path = os.path.join(project_root, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Mount other static files (like zhcn.ttf) under a /static prefix
# This assumes the 'static' folder is at the project root.
app.mount("/static", StaticFiles(directory=os.path.join(project_root, "static")), name="static")


# Global instances (initialized on startup)
scraper: MultiSourceScraper = None
vector_db: EnhancedVectorDB = None
scheduler: ScheduledScraper = None
gemini_model = None # Global instance for Gemini model

@app.on_event("startup")
async def startup_event():
    global scraper, vector_db, scheduler, gemini_model
    logger.info("Application starting up...")
    # MODIFIED: Changed data_dir to point to 'data/genshin_data'
    # This path is relative to the directory where main.py is executed,
    # which is likely the /api directory in the Vercel Lambda.
    # So, 'data/genshin_data' would be 'api/data/genshin_data'.
    # If 'data' folder is at the project root, you'd need:
    data_dir = Path(os.path.join(project_root, "data", "genshin_data"))
    data_dir.mkdir(parents=True, exist_ok=True) # Ensure data directory exists
    
    vector_db = EnhancedVectorDB(data_dir) # Initialize vector_db first to get embedding_model
    scraper = MultiSourceScraper(data_dir, vector_db.embedding_model) # Pass embedding_model to scraper
    
    # Load/rebuild vector index during startup (can be time-consuming)
    logger.info("Loading or rebuilding vector database...")
    await asyncio.to_thread(vector_db.load_or_create_index, scraper.db_path)

    # Configure Gemini API
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        # Changed model from 'gemini-pro' to 'gemini-2.0-flash'
        gemini_model = genai.GenerativeModel('gemini-2.0-flash') 
        logger.info("Gemini LLM model initialized.")
    else:
        logger.warning("GEMINI_API_KEY not set. Gemini LLM functionality will be disabled.")
    
    scheduler = ScheduledScraper(scraper, vector_db)
    scheduler.start_scheduler()
    logger.info("Application startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    global scheduler
    logger.info("Application shutting down...")
    if scheduler:
        scheduler.stop_scheduler()
    logger.info("Application shutdown complete.")


@app.post("/query", response_model=QueryResponse)
async def query_lore(request: QueryRequest):
    try:
        if not vector_db or not vector_db.index:
            raise HTTPException(status_code=503, detail="Vector database not ready. Please wait for initialization/scraping to complete.")
            
        # Search the FAISS index
        search_results = vector_db.search(request.query, request.k)
        
        if not search_results:
            return QueryResponse(results=[], message="No relevant lore found.")

        # Use context manager for database connection
        with sqlite3.connect(scraper.db_path) as conn:
                cursor = conn.cursor()
                chunk_ids = [res[0] for res in search_results]
                
                # Fetch chunk text directly and entry_id
                placeholders = ','.join('?' * len(chunk_ids))
                cursor.execute(f"SELECT id, chunk_text, entry_id FROM chunks WHERE id IN ({placeholders})", chunk_ids)
                chunk_data_rows = cursor.fetchall()
                chunk_data_map = {row[0]: (row[1], row[2]) for row in chunk_data_rows} # {chunk_id: (chunk_text, entry_id)}
                
                # Fetch full lore entries for unique entry_ids
                unique_entry_ids = list(set(entry_id for _, (_, entry_id) in chunk_data_map.items()))
                placeholders_entries = ','.join('?' * len(unique_entry_ids))
                cursor.execute(f"SELECT id, title, content, source_type, source_url, last_updated FROM lore_entries WHERE id IN ({placeholders_entries})", unique_entry_ids)
                entry_data_map = {row[0]: LoreEntry(entry_id=row[0], title=row[1], content=row[2], source_type=row[3], source_url=row[4], last_updated=row[5]) for row in cursor.fetchall()}

        # Combine results, ensuring order based on search_results
        final_results = []
        for chunk_db_id, distance in search_results:
            if chunk_db_id in chunk_data_map:
                chunk_text, entry_id = chunk_data_map[chunk_db_id]
                full_entry = entry_data_map.get(entry_id)
                if full_entry:
                    # For simplicity, returning the full entry. 
                    # You might want to return just the relevant chunk + metadata.
                    final_results.append(full_entry) 
            
        return QueryResponse(results=final_results, message="Query successful")
    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing your query: {e}")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Provides statistics about the lore database."""
    try:
        with sqlite3.connect(scraper.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM lore_entries")
            total_entries = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]

            cursor.execute("SELECT source_type, last_scraped, next_scrape FROM scrape_schedule")
            schedule_data = cursor.fetchall()

            source_details = []
            for source_type, last_scraped, next_scrape in schedule_data:
                source_details.append(SourceSummary(
                    source_type=source_type,
                    count=0,
                    last_scraped=last_scraped,
                    next_scrape=next_scrape
                ))

            # Populate count per source_type
            cursor.execute("SELECT source_type, COUNT(*) FROM lore_entries GROUP BY source_type")
            source_counts = dict(cursor.fetchall())

            for detail in source_details:
                detail.count = source_counts.get(detail.source_type, 0)

        return StatsResponse(
            total_entries=total_entries,
            total_chunks=total_chunks,
            source_details=source_details
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {e}")

@app.post("/scrape", response_model=ScrapeResponse)
async def manage_scrape(request: ScrapeRequest, api_key: str = Depends(get_api_key)): # Added API Key dependency
    try:
        if not scheduler or not scraper or not vector_db:
            raise HTTPException(status_code=503, detail="Scraper components not initialized. Please wait for application startup.")

        if request.action == "start":
            scheduler.start_scheduler()
            return ScrapeResponse(message="Scraping scheduler started.", status="success")
        elif request.action == "stop":
            scheduler.stop_scheduler()
            return ScrapeResponse(message="Scraping scheduler stopped.", status="success")
        elif request.action == "reschedule":
            scheduler.reschedule_all_scrapes()
            return ScrapeResponse(message="All scraping jobs rescheduled.", status="success")
        elif request.action == "run_now":
            source_type = request.source_type
            if source_type == "wiki":
                entries = await scraper.scrape_wiki_async()
                logger.debug(f"scrape_wiki_async returned {len(entries)} entries.") # New log
            elif source_type == "reddit":
                entries = await scraper.scrape_reddit_async()
            elif source_type == "youtube":
                entries = await scraper.scrape_youtube_async()
            else:
                raise HTTPException(status_code=400, detail="Invalid source_type. Must be 'wiki', 'reddit', or 'youtube'.")
            
            # For wiki, saving and rebuilding is now handled in batches within scrape_wiki_async
            # For other sources, it's still handled here once at the end of the scrape.
            if source_type != 'wiki': # Only do this if not wiki (as wiki handles its own saving/rebuilding)
                new_entries, new_chunks_count = scraper.save_entries_with_chunks(entries)
                if new_entries > 0 or new_chunks_count > 0:
                    vector_db.rebuild_index_from_db(scraper.db_path)
                return ScrapeResponse(message=f"Manual scrape for {source_type} completed. Added {new_entries} entries, {new_chunks_count} chunks.", status="success")
            else:
                return ScrapeResponse(message=f"Manual scrape for {source_type} completed. Entries saved and index rebuilt in batches.", status="success")
        else:
            raise HTTPException(status_code=400, detail="Invalid action. Must be 'start', 'stop', 'reschedule', or 'run_now'.")

    except Exception as e:
        logger.error(f"Error during scrape operation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing scrape request: {e}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_lore(request: ChatRequest):
    try:
        if not gemini_model:
            raise HTTPException(status_code=503, detail="Gemini LLM model not initialized. Please ensure GEMINI_API_KEY is set.")
        if not vector_db or not vector_db.index:
            raise HTTPException(status_code=503, detail="Vector database not ready. Please wait for initialization/scraping to complete.")

        # Access the query field from the request object
        user_query = request.query 
        
        # 1. Search local lore database for relevant context
        # Increased k from 5 to 10 to provide more context to Gemini
        search_results_tuples = vector_db.search(user_query, k=10) 
        
        context_lore_entries = []
        if search_results_tuples:
            with sqlite3.connect(scraper.db_path) as conn:
                cursor = conn.cursor()
                chunk_ids = [res[0] for res in search_results_tuples]
                
                # Fetch chunk text directly and entry_id
                placeholders = ','.join('?' * len(chunk_ids))
                cursor.execute(f"SELECT id, chunk_text, entry_id FROM chunks WHERE id IN ({placeholders})", chunk_ids)
                chunk_data_rows = cursor.fetchall()
                chunk_data_map = {row[0]: (row[1], row[2]) for row in chunk_data_rows} # {chunk_id: (chunk_text, entry_id)}
                
                # Fetch full lore entries for unique entry_ids
                unique_entry_ids = list(set(entry_id for _, (_, entry_id) in chunk_data_map.items()))
                placeholders_entries = ','.join('?' * len(unique_entry_ids))
                cursor.execute(f"SELECT id, title, content, source_type, source_url, last_updated FROM lore_entries WHERE id IN ({placeholders_entries})", unique_entry_ids)
                entry_data_map = {row[0]: LoreEntry(entry_id=row[0], title=row[1], content=row[2], source_type=row[3], source_url=row[4], last_updated=row[5]) for row in cursor.fetchall()}

            for chunk_db_id, distance in search_results_tuples:
                if chunk_db_id in chunk_data_map:
                    chunk_text, entry_id = chunk_data_map[chunk_db_id]
                    full_entry = entry_data_map.get(entry_id)
                    if full_entry:
                        context_lore_entries.append({
                            "title": full_entry.title,
                            "content_excerpt": chunk_text, # Use the specific chunk content
                            "source_url": full_entry.source_url,
                            "source_type": full_entry.source_type # Add source_type for better prompt
                        })

        # 2. Construct prompt for Gemini with enhanced instructions
        context_string = ""
        if context_lore_entries:
            context_string = (
                "You are an expert on Genshin Impact lore. Your task is to provide comprehensive, precise, and detailed answers "
                "to user questions, utilizing all relevant information from the provided lore snippets. "
                "Structure your responses clearly using Markdown (headings, bullet points, bold text). "
                "Avoid filler phrases, general knowledge, or stating that you are an AI. Focus solely on delivering accurate "
                "lore-based information. If the provided lore is insufficient to answer a specific detail, state that the "
                "information is not available in your current database for that specific point, but still provide a general answer "
                "based on what you do have.\n\n"
                "Here are some relevant Genshin Impact lore snippets from my database:\n\n"
            )
            for i, lore in enumerate(context_lore_entries):
                context_string += f"--- Lore Snippet {i+1} (Source: {lore['source_type']}) ---\n"
                context_string += f"Title: {lore['title']}\n"
                context_string += f"Content: {lore['content_excerpt']}\n"
                context_string += f"Source URL: {lore['source_url']}\n\n"
            context_string += "Please use this information to answer the following question in detail. Ensure your answer is well-structured and directly addresses the user's query.\n\n"
        else:
            context_string = (
                "You are an expert on Genshin Impact lore. Your task is to provide comprehensive, precise, and detailed answers "
                "to user questions. Structure your responses clearly using Markdown (headings, bullet points, bold text). "
                "Avoid filler phrases, general knowledge, or stating that you are an AI. Focus solely on delivering accurate "
                "lore-based information. No relevant lore snippets were found in the database for this query, so please answer "
                "based on your general knowledge of Genshin Impact lore.\n\n"
            )

        full_prompt = f"{context_string}User Question: {user_query}"
        
        logger.info(f"Sending prompt to Gemini: {full_prompt[:500]}...") # Log first 500 chars

        # 3. Call Gemini API
        # Using the direct genai.GenerativeModel.generate_content method as per previous successful usage
        gemini_response = await asyncio.to_thread(gemini_model.generate_content, full_prompt)
        
        # Extract text response
        response_text = ""
        if gemini_response and gemini_response.candidates:
            for part in gemini_response.candidates[0].content.parts:
                response_text += part.text
        elif isinstance(gemini_response, str): # Fallback if generate_content returns a string (e.g., error message)
            response_text = gemini_response
        else:
            response_text = "I could not generate a response at this time."
        
        return ChatResponse(response=response_text, source_lore=context_lore_entries)

    except Exception as e:
        logger.error(f"Error during chat with LLM: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {e}")
