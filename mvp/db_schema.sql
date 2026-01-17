-- ReadLyte MVP Database Schema
-- PostgreSQL 

-- Drop tables if exist (for fresh start)
DROP TABLE IF EXISTS audio_files CASCADE;
DROP TABLE IF EXISTS summaries CASCADE;
DROP TABLE IF EXISTS translations CASCADE;
DROP TABLE IF EXISTS book_sections CASCADE;
DROP TABLE IF EXISTS books CASCADE;

-- Books table
CREATE TABLE books (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    author VARCHAR(500),
    filename VARCHAR(500),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Sections (segregated content from books)
CREATE TABLE book_sections (
    id SERIAL PRIMARY KEY,
    book_id INTEGER REFERENCES books(id) ON DELETE CASCADE,
    section_number INTEGER,
    section_title VARCHAR(500),
    content TEXT,
    word_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Translations (per section, per model/tier combo)
CREATE TABLE translations (
    id SERIAL PRIMARY KEY,
    section_id INTEGER REFERENCES book_sections(id) ON DELETE CASCADE,
    language VARCHAR(50) DEFAULT 'hindi',
    tier VARCHAR(20),
    model_name VARCHAR(100),
    translated_text TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(section_id, language, tier, model_name)
);

-- Summaries (per section, per model/tier/length combo)
CREATE TABLE summaries (
    id SERIAL PRIMARY KEY,
    section_id INTEGER REFERENCES book_sections(id) ON DELETE CASCADE,
    source_type VARCHAR(20) DEFAULT 'original',  -- 'original' or 'translation'
    source_id INTEGER,  -- translation_id if source_type='translation'
    tier VARCHAR(20),
    length_type VARCHAR(20),
    model_name VARCHAR(100),
    summary_text TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(section_id, source_type, source_id, tier, length_type, model_name)
);

-- Audio files (per section or translation/summary)
CREATE TABLE audio_files (
    id SERIAL PRIMARY KEY,
    section_id INTEGER REFERENCES book_sections(id) ON DELETE CASCADE,
    source_type VARCHAR(20),  -- 'original', 'translation', 'summary'
    source_id INTEGER,        -- translation_id or summary_id (NULL for original)
    tier VARCHAR(20),
    model_name VARCHAR(100),
    audio_data BYTEA,
    file_path VARCHAR(500),   -- Alternative: store path instead of BYTEA
    duration_seconds FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for faster queries
CREATE INDEX idx_sections_book ON book_sections(book_id);
CREATE INDEX idx_translations_section ON translations(section_id);
CREATE INDEX idx_summaries_section ON summaries(section_id);
CREATE INDEX idx_audio_section ON audio_files(section_id);
