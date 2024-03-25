-- Your SQL goes here
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    chunk_number INTEGER,
    title TEXT NOT NULL,
    content TEXT DEFAULT '',
    embedding VECTOR(768)
)