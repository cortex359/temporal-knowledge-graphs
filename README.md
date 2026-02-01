# Temporal Knowledge Graph RAG System

A powerful RAG (Retrieval-Augmented Generation) system that uses a temporal knowledge graph for document storage and retrieval with temporal awareness.

## Disclaimer

Große Teile der Codebasis und der Dokumentation wurden in Absprache mit Jan-Niklas Schagen mit Hilfsmitteln wie 
Claude Code, Gemini, Codex und GitHub Copilot erstellt.
Die Verwendung generativer KI hat uns eine umfangreiche Untersuchung des Themas _Temporal Knowledge Graphs_, den
verschiedenen Herausforderungen und den praktischen Lösungsansätzen ermöglicht, da wir in kürzester Zeit ganz 
verschiedene Ansätze prototypisch umsetzen und evaluieren konnten.
Unsere Prüfungsleistung besteht somit nicht in dem vorliegenden Code, sondern in den Untersuchungen, die wir in unserem
Vortrag vorstellen und einordnen werden. Der Code dient der Reproduktion der Ergebnisse und einer möglichen
Weiterverwendung der implementierten Systeme.    

## Features

- **Temporal Knowledge Graph**: Track how information evolves over time with document versioning, chunk supersession, and temporal relationships
- **Hybrid Retrieval**: Combine vector similarity search with graph traversal for comprehensive retrieval
- **Neo4j Vector Index**: Native vector search capabilities integrated with graph queries
- **Multi-format Document Support**: Ingest PDF, Markdown, HTML, and text documents
- **Entity Extraction**: Automatic entity recognition and relationship mapping
- **Three Web Interfaces**:
  - Graph Visualization Explorer
  - Chunk Retrieval Interface
  - RAG Chatbot with Source Citations

## Architecture

The system implements temporality at three levels:
1. **Document-level**: Creation and modification timestamps
2. **Chunk-level**: Version history with supersession tracking
3. **Relationship-level**: Valid_from and valid_to dates on all relationships

### Technology Stack

- **Graph Database**: Neo4j 5.15+ with vector index
- **Embeddings**: OpenAI Embeddings API
- **LLM**: LiteLLM Proxy (OpenAI-compatible)
- **Web Framework**: Streamlit
- **Python**: 3.10+
- **Libraries**: LangChain, LangGraph, Neo4j Python driver, spaCy

## Quick Start

**TL;DR**: See [QUICKSTART.md](QUICKSTART.md) for a setup guide.

### 1. Prerequisites

- Docker and Docker Compose
- Python 3.10 or higher
- External LiteLLM service (for LLM and embeddings)
- `uv` package manager (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### 2. Clone and Setup

```bash
# Clone the repository
cd temporal-knowledge-graphs

# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

Required environment variables:
```bash
# Neo4j (default values work with docker-compose)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LiteLLM (external service for both LLM and embeddings)
LITELLM_API_BASE=http://your-litellm-url:4000
LITELLM_API_KEY=your-litellm-api-key

# Embedding Configuration (via LiteLLM)
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
EMBEDDING_DIMENSIONS=4096
```

### 3. Start Services

```bash
# Start Neo4j
docker-compose up -d

# Wait for Neo4j to be ready (check http://localhost:7474)
# Default credentials: neo4j/password
```

**Note**: Make sure your external LiteLLM service is running and accessible.

### 4. Install Python Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with uv
uv pip install -r requirements.txt
```

### 5. Initialize Database

```bash
# Initialize Neo4j schema (constraints, indexes, vector index)
uv run python scripts/init_db.py

# Verify schema
uv run python scripts/init_db.py --verify-only

# Show current schema
uv run python scripts/init_db.py --show-schema
```

## Project Structure

```
temporal-knowledge-graphs/
├── docker-compose.yml              # Services orchestration
├── .env.example                    # Environment template
├── requirements.txt                # Python dependencies
│
├── src/temporal_kg_rag/           # Core library
│   ├── config/                    # Configuration management
│   ├── models/                    # Data models (Pydantic)
│   ├── graph/                     # Neo4j operations
│   ├── embeddings/                # Embedding generation
│   ├── temporal/                  # Temporal query building
│   ├── ingestion/                 # Document processing
│   ├── retrieval/                 # Hybrid search
│   ├── rag/                       # RAG workflow
│   └── utils/                     # Utilities
│
├── scripts/
│   ├── init_db.py                 # Database initialization
│   ├── ingest_documents.py        # Document ingestion CLI
│   └── sample_data.py             # Generate test data
│
└── apps/                          # Streamlit applications
    ├── 1_graph_visualization.py   # Graph explorer
    ├── 2_chunk_retrieval.py       # Search interface
    └── 3_chatbot.py               # RAG chatbot
```

## Database Schema

### Node Labels

- **Document**: Source documents with metadata
- **Chunk**: Text segments with embeddings
- **Entity**: Named entities (PERSON, ORG, LOCATION, etc.)
- **Topic**: Document/chunk topics
- **TimeSnapshot**: Temporal markers

### Key Relationships

- `(Document)-[:HAS_CHUNK]->(Chunk)`
- `(Chunk)-[:SUPERSEDES]->(Chunk)` - Version history
- `(Chunk)-[:MENTIONS {valid_from, valid_to}]->(Entity)` - Temporal mentions
- `(Entity)-[:RELATES_TO {valid_from, valid_to}]->(Entity)` - Entity relationships

### Indexes

- Unique constraints on all node IDs
- Vector index on `Chunk.embedding` (1536 dimensions, cosine similarity)
- Performance indexes on temporal fields (created_at, updated_at, is_current)
- Full-text indexes on Document.title and Chunk.text

## Usage Examples

### Initialize Database

```bash
# First time setup
uv run python scripts/init_db.py

# Force re-initialization (WARNING: drops existing schema)
uv run python scripts/init_db.py --force

# Verify schema is correct
uv run python scripts/init_db.py --verify-only
```

### Document Ingestion

```bash
# Generate sample documents for testing
uv run python scripts/sample_data.py --output-dir ./sample_data

# Ingest a single document
uv run python scripts/ingest_documents.py --path sample_data/artificial_intelligence_2023.txt

# Ingest all documents in a directory
uv run python scripts/ingest_documents.py --path sample_data/ --pattern "*.txt"

# Ingest with custom metadata
uv run python scripts/ingest_documents.py \
    --path document.pdf \
    --title "Annual Report 2024" \
    --metadata '{"author": "Jane Doe", "department": "Engineering"}'

# Ingest recursively with statistics
uv run python scripts/ingest_documents.py \
    --path docs/ \
    --recursive \
    --pattern "*.md" \
    --show-stats

# Quick ingestion (skip entity extraction for speed)
uv run python scripts/ingest_documents.py --path doc.txt --no-entities
```

### Programmatic Usage

```python
from temporal_kg_rag.ingestion.pipeline import get_ingestion_pipeline

# Get pipeline instance
pipeline = get_ingestion_pipeline()

# Ingest a document
document = pipeline.ingest_document(
    file_path="path/to/document.pdf",
    title="My Document",
    metadata={"author": "John Doe"},
    extract_entities=True,
    generate_embeddings=True,
)

print(f"Document ID: {document.id}")
print(f"Chunks created: {len(pipeline.get_statistics()['chunks'])}")
```

### Test Neo4j Connection

```python
from temporal_kg_rag.graph.neo4j_client import get_neo4j_client

# Get client and verify connectivity
client = get_neo4j_client()
if client.verify_connectivity():
    print("✓ Connected to Neo4j")

    # Get database statistics
    stats = client.get_database_stats()
    print(f"Documents: {stats['documents']}")
    print(f"Chunks: {stats['chunks']}")
    print(f"Entities: {stats['entities']}")
```

### Query the Graph

```python
from temporal_kg_rag.graph.operations import get_graph_operations

ops = get_graph_operations()

# Get a document
document = ops.get_document(document_id)

# Get all chunks for a document
chunks = ops.get_document_chunks(document_id, current_only=True)

# Get entities mentioned in a chunk
entities = ops.get_chunk_entities(chunk_id)
```

### Retrieval and Search

```bash
# Test retrieval system with demo
uv run python scripts/test_retrieval.py --demo

# Search with different methods
uv run python scripts/test_retrieval.py --query "artificial intelligence" --method hybrid
uv run python scripts/test_retrieval.py --query "OpenAI" --method graph
uv run python scripts/test_retrieval.py --query "machine learning" --method vector

# Temporal search
uv run python scripts/test_retrieval.py --query "AI in 2023" --temporal
uv run python scripts/test_retrieval.py --query "quantum computing" --year 2023

# Search with context expansion
uv run python scripts/test_retrieval.py --query "climate change" --expand-context

# Compare all search methods
uv run python scripts/test_retrieval.py --query "neural networks" --compare
```

### Programmatic Retrieval

```python
from temporal_kg_rag.retrieval.hybrid_search import get_hybrid_search
from temporal_kg_rag.retrieval.temporal_retrieval import get_temporal_retrieval
from temporal_kg_rag.retrieval.context_expansion import get_context_expander
from temporal_kg_rag.models.temporal import TemporalFilter
from datetime import datetime

# Hybrid search (combines vector + graph)
hs = get_hybrid_search()
results = hs.search(
    query="artificial intelligence",
    top_k=10,
)

# Temporal search with auto-detection
tr = get_temporal_retrieval()
temporal_results = tr.search_with_temporal_context(
    query="AI developments in 2023",
    auto_detect_temporal=True,
)

# Point-in-time search
point_in_time_results = tr.search_at_time(
    query="quantum computing",
    timestamp=datetime(2023, 12, 31),
    top_k=5,
)

# Expand results with context
ce = get_context_expander()
expanded = ce.expand_results(
    results,
    include_neighboring_chunks=True,
    include_entities=True,
    neighboring_chunk_window=1,
)

# Get context summary for RAG
context_summary = ce.build_context_summary(expanded)
print(context_summary)
```

### RAG System

```bash
# Test RAG system with demo
uv run python scripts/test_rag.py --demo

# Test LangChain RAG with custom query
uv run python scripts/test_rag.py --query "What is artificial intelligence?" --method langchain

# Test LangGraph RAG with temporal query
uv run python scripts/test_rag.py --query "AI developments in 2023" --method langgraph --temporal

# Compare both implementations
uv run python scripts/test_rag.py --query "machine learning" --compare

# Test streaming response
uv run python scripts/test_rag.py --query "climate change" --stream

# Run conversation demo
uv run python scripts/test_rag.py --conversation-demo
```

### Programmatic RAG Usage

```python
from temporal_kg_rag.rag.chain import get_rag_chain
from temporal_kg_rag.rag.graph import get_rag_graph

# LangChain RAG Chain
chain = get_rag_chain()
result = chain.query(
    "What is quantum computing?",
    top_k=5,
    use_temporal_detection=True,
    expand_context=True,
)

print(result["answer"])
print(f"Sources: {len(result['sources'])}")

# With conversation history
history = [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is..."},
]
result = chain.query_with_history(
    "When was it developed?",
    conversation_history=history,
    top_k=5,
)

# Streaming response
for chunk in chain.stream_query("Explain neural networks", top_k=5):
    print(chunk, end="", flush=True)

# LangGraph RAG Workflow
graph = get_rag_graph()
result = graph.query("Compare AI and ML", top_k=10)

print(result["answer"])
print(f"Query type: {result['metadata']['query_type']}")
print(f"Temporal detected: {result['metadata']['temporal_detected']}")
print(f"Verified: {result['metadata']['verified']}")
```

### Web Applications

The project includes three Streamlit applications for easy interaction with the system:

```bash
# Quick start - run all apps at once
./scripts/run_all_apps.sh

# Or run individually:
streamlit run apps/1_graph_visualization.py                    # Port 8501
streamlit run apps/2_chunk_retrieval.py --server.port 8502     # Port 8502
streamlit run apps/3_chatbot.py --server.port 8503              # Port 8503

# Stop all apps
./scripts/stop_all_apps.sh
```

**Application 1: Graph Visualization Explorer** (Port 8501)
- Explore the temporal knowledge graph interactively
- Search and visualize entity neighborhoods
- View document structures and chunk relationships
- Export graph data
- Execute custom Cypher queries

**Application 2: Chunk Retrieval Interface** (Port 8502)
- Test different retrieval strategies (Vector, Graph, Hybrid, Temporal)
- Compare search methods side-by-side
- Apply temporal and entity filters
- View performance metrics
- Export search results

**Application 3: RAG Chatbot** (Port 8503)
- Chat interface with conversation history
- Choose between LangGraph and LangChain implementations
- View source citations for all answers
- See temporal context and entity detection
- Export conversations
- Streaming responses (LangChain mode)

See [apps/README.md](apps/README.md) for detailed documentation.

### Using Configuration

```python
from temporal_kg_rag.config.settings import get_settings

settings = get_settings()
print(f"Neo4j URI: {settings.neo4j_uri}")
print(f"Chunk size: {settings.chunk_size}")
print(f"Embedding model: {settings.openai_embedding_model}")
```

## Configuration

All configuration is managed through environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j connection URI | bolt://localhost:7687 |
| `NEO4J_PASSWORD` | Neo4j password | password |
| `OPENAI_API_KEY` | OpenAI API key | (required) |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | text-embedding-3-small |
| `CHUNK_SIZE` | Chunk size in tokens | 1000 |
| `CHUNK_OVERLAP` | Overlap between chunks | 100 |
| `HYBRID_SEARCH_ALPHA` | Vector vs graph weight | 0.5 |

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/temporal_kg_rag
```

### Code Formatting

```bash
# Format code with black
black src/ tests/ scripts/

# Lint with ruff
ruff check src/ tests/ scripts/

# Type checking with mypy
mypy src/
```

## Troubleshooting

### Neo4j Connection Issues

```bash
# Check if Neo4j is running
docker-compose ps

# View Neo4j logs
docker-compose logs neo4j

# Access Neo4j browser
# Open http://localhost:7474 in your browser
```

### Vector Index Issues

```python
# Check vector index status
from temporal_kg_rag.graph.neo4j_client import get_neo4j_client

client = get_neo4j_client()
query = """
SHOW INDEXES
YIELD name, type, state
WHERE name = 'chunk_embeddings'
RETURN name, type, state
"""
result = client.execute_query(query)
print(result)
```

### Python Import Issues

Make sure you're in the project root and have activated your virtual environment:

```bash
# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or install in development mode
pip install -e .
```


## License

MIT License - see LICENSE file for details

