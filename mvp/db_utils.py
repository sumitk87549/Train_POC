"""
Database Utilities for ReadLyte MVP
PostgreSQL CRUD operations for books, sections, translations, summaries, and audio
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import os

# Database configuration
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "readlyte",
    "user": "postgres",
    "password": "0000"
}

@contextmanager
def get_connection():
    """Get database connection with context manager."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
    finally:
        if conn:
            conn.close()

def init_database():
    """Initialize database tables if they don't exist."""
    schema_sql = """
    -- Books table
    CREATE TABLE IF NOT EXISTS books (
        id SERIAL PRIMARY KEY,
        title VARCHAR(500) NOT NULL,
        author VARCHAR(500),
        filename VARCHAR(500),
        created_at TIMESTAMP DEFAULT NOW()
    );

    -- Sections table
    CREATE TABLE IF NOT EXISTS book_sections (
        id SERIAL PRIMARY KEY,
        book_id INTEGER REFERENCES books(id) ON DELETE CASCADE,
        section_number INTEGER,
        section_title VARCHAR(500),
        content TEXT,
        word_count INTEGER,
        created_at TIMESTAMP DEFAULT NOW()
    );

    -- Translations table
    CREATE TABLE IF NOT EXISTS translations (
        id SERIAL PRIMARY KEY,
        section_id INTEGER REFERENCES book_sections(id) ON DELETE CASCADE,
        language VARCHAR(50) DEFAULT 'hindi',
        tier VARCHAR(20),
        model_name VARCHAR(100),
        translated_text TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(section_id, language, tier, model_name)
    );

    -- Summaries table
    CREATE TABLE IF NOT EXISTS summaries (
        id SERIAL PRIMARY KEY,
        section_id INTEGER REFERENCES book_sections(id) ON DELETE CASCADE,
        source_type VARCHAR(20) DEFAULT 'original',
        source_id INTEGER,
        tier VARCHAR(20),
        length_type VARCHAR(20),
        model_name VARCHAR(100),
        summary_text TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );

    -- Audio files table
    CREATE TABLE IF NOT EXISTS audio_files (
        id SERIAL PRIMARY KEY,
        section_id INTEGER REFERENCES book_sections(id) ON DELETE CASCADE,
        source_type VARCHAR(20),
        source_id INTEGER,
        tier VARCHAR(20),
        model_name VARCHAR(100),
        audio_data BYTEA,
        file_path VARCHAR(500),
        duration_seconds FLOAT,
        created_at TIMESTAMP DEFAULT NOW()
    );

    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_sections_book ON book_sections(book_id);
    CREATE INDEX IF NOT EXISTS idx_translations_section ON translations(section_id);
    CREATE INDEX IF NOT EXISTS idx_summaries_section ON summaries(section_id);
    CREATE INDEX IF NOT EXISTS idx_audio_section ON audio_files(section_id);
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()
    print("✅ Database initialized successfully")

# ============ BOOKS ============

def save_book(title: str, author: str = None, filename: str = None) -> int:
    """Save a new book and return its ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO books (title, author, filename) VALUES (%s, %s, %s) RETURNING id",
                (title, author, filename)
            )
            book_id = cur.fetchone()[0]
        conn.commit()
    return book_id

def get_books() -> List[Dict]:
    """Get all books with section counts."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT b.*, COUNT(s.id) as section_count
                FROM books b
                LEFT JOIN book_sections s ON b.id = s.book_id
                GROUP BY b.id
                ORDER BY b.created_at DESC
            """)
            return cur.fetchall()

def get_book(book_id: int) -> Optional[Dict]:
    """Get a single book by ID."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM books WHERE id = %s", (book_id,))
            return cur.fetchone()

def delete_book(book_id: int):
    """Delete a book and all its sections."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM books WHERE id = %s", (book_id,))
        conn.commit()

# ============ SECTIONS ============

def save_section(book_id: int, section_number: int, section_title: str, content: str) -> int:
    """Save a book section and return its ID."""
    word_count = len(content.split()) if content else 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO book_sections (book_id, section_number, section_title, content, word_count) 
                   VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                (book_id, section_number, section_title, content, word_count)
            )
            section_id = cur.fetchone()[0]
        conn.commit()
    return section_id

def get_sections(book_id: int) -> List[Dict]:
    """Get all sections for a book."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM book_sections WHERE book_id = %s ORDER BY section_number",
                (book_id,)
            )
            return cur.fetchall()

def get_section(section_id: int) -> Optional[Dict]:
    """Get a single section by ID."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM book_sections WHERE id = %s", (section_id,))
            return cur.fetchone()

# ============ TRANSLATIONS ============

def save_translation(section_id: int, tier: str, model_name: str, 
                     translated_text: str, language: str = "hindi") -> int:
    """Save a translation. Uses upsert to handle duplicates."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO translations (section_id, language, tier, model_name, translated_text)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (section_id, language, tier, model_name) 
                   DO UPDATE SET translated_text = EXCLUDED.translated_text, created_at = NOW()
                   RETURNING id""",
                (section_id, language, tier, model_name, translated_text)
            )
            trans_id = cur.fetchone()[0]
        conn.commit()
    return trans_id

def get_translation(section_id: int, tier: str, model_name: str, 
                    language: str = "hindi") -> Optional[Dict]:
    """Get existing translation if available."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT * FROM translations 
                   WHERE section_id = %s AND language = %s AND tier = %s AND model_name = %s""",
                (section_id, language, tier, model_name)
            )
            return cur.fetchone()

def get_section_translations(section_id: int) -> List[Dict]:
    """Get all translations for a section."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM translations WHERE section_id = %s ORDER BY created_at DESC",
                (section_id,)
            )
            return cur.fetchall()

# ============ SUMMARIES ============

def save_summary(section_id: int, tier: str, length_type: str, model_name: str,
                 summary_text: str, source_type: str = "original", source_id: int = None) -> int:
    """Save a summary. Uses upsert to handle duplicates."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Handle NULL source_id in unique constraint
            cur.execute(
                """INSERT INTO summaries (section_id, source_type, source_id, tier, length_type, model_name, summary_text)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (section_id, source_type, COALESCE(source_id, 0), tier, length_type, model_name)
                   DO UPDATE SET summary_text = EXCLUDED.summary_text, created_at = NOW()
                   RETURNING id""",
                (section_id, source_type, source_id, tier, length_type, model_name, summary_text)
            )
            summary_id = cur.fetchone()[0]
        conn.commit()
    return summary_id

def get_summary(section_id: int, tier: str, length_type: str, model_name: str,
                source_type: str = "original", source_id: int = None) -> Optional[Dict]:
    """Get existing summary if available."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if source_id is None:
                cur.execute(
                    """SELECT * FROM summaries 
                       WHERE section_id = %s AND source_type = %s AND source_id IS NULL
                       AND tier = %s AND length_type = %s AND model_name = %s""",
                    (section_id, source_type, tier, length_type, model_name)
                )
            else:
                cur.execute(
                    """SELECT * FROM summaries 
                       WHERE section_id = %s AND source_type = %s AND source_id = %s
                       AND tier = %s AND length_type = %s AND model_name = %s""",
                    (section_id, source_type, source_id, tier, length_type, model_name)
                )
            return cur.fetchone()

def get_section_summaries(section_id: int) -> List[Dict]:
    """Get all summaries for a section."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM summaries WHERE section_id = %s ORDER BY created_at DESC",
                (section_id,)
            )
            return cur.fetchall()

# ============ AUDIO ============

def save_audio(section_id: int, source_type: str, tier: str, model_name: str,
               audio_data: bytes = None, file_path: str = None, 
               duration_seconds: float = None, source_id: int = None) -> int:
    """Save audio file data or path."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO audio_files (section_id, source_type, source_id, tier, model_name, 
                   audio_data, file_path, duration_seconds)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (section_id, source_type, source_id, tier, model_name, 
                 psycopg2.Binary(audio_data) if audio_data else None, file_path, duration_seconds)
            )
            audio_id = cur.fetchone()[0]
        conn.commit()
    return audio_id

def get_audio(section_id: int, source_type: str, tier: str, model_name: str,
              source_id: int = None) -> Optional[Dict]:
    """Get existing audio if available."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if source_id is None:
                cur.execute(
                    """SELECT * FROM audio_files 
                       WHERE section_id = %s AND source_type = %s AND source_id IS NULL
                       AND tier = %s AND model_name = %s""",
                    (section_id, source_type, tier, model_name)
                )
            else:
                cur.execute(
                    """SELECT * FROM audio_files 
                       WHERE section_id = %s AND source_type = %s AND source_id = %s
                       AND tier = %s AND model_name = %s""",
                    (section_id, source_type, source_id, tier, model_name)
                )
            return cur.fetchone()

def get_section_audio(section_id: int) -> List[Dict]:
    """Get all audio files for a section."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, section_id, source_type, source_id, tier, model_name, file_path, duration_seconds, created_at FROM audio_files WHERE section_id = %s ORDER BY created_at DESC",
                (section_id,)
            )
            return cur.fetchall()


if __name__ == "__main__":
    # Test database connection
    try:
        init_database()
        print("✅ Database connection successful!")
    except Exception as e:
        print(f"❌ Database error: {e}")
        print("\n💡 Make sure PostgreSQL is running and database 'readlyte' exists:")
        print("   createdb -U postgres readlyte")
