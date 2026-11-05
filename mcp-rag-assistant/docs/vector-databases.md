# Vector databases and similarity search

## Embedding dimensions
An embedding model maps text to a vector of fixed length. The popular sentence-transformers model all-MiniLM-L6-v2 produces 384-dimensional embeddings, while larger models produce 768 or more dimensions. Higher dimensions can capture finer meaning but cost more memory and time per comparison.

## Exact versus approximate search
Exact brute-force search compares the query with every stored vector. It is simple and perfectly accurate, and it is perfectly adequate for up to roughly tens of thousands of vectors. Beyond that, approximate nearest neighbour indexes such as HNSW trade a tiny loss of recall for very large speedups.

## Common tools
FAISS is a library from Meta for fast similarity search, and it runs inside your own process. Chroma is an open-source embedding database with persistent storage and a simple API, which makes it a good choice for quick prototypes. The pgvector extension adds vector columns and indexes to PostgreSQL, so embeddings can live next to the rest of the application data.

## Choosing
For a small personal project, a SQLite table of vectors and a brute-force scan is enough, and it avoids another dependency. Move to FAISS, Chroma or pgvector when the collection grows, when several processes need to share the index, or when query latency becomes a problem.
