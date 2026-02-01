# Temporal Knowledge Graph RAG System

A powerful RAG (Retrieval-Augmented Generation) system that uses a temporal knowledge graph for document storage and retrieval with temporal awareness. Designed for financial document analysis (earnings call transcripts) with fiscal period-based temporal filtering.

## Features

- **Temporal Knowledge Graph**: Track how information evolves over time with fiscal period metadata (year/quarter), document versioning, chunk supersession, and temporal relationships
- **Hybrid Retrieval**: Combine vector similarity search with graph traversal using Reciprocal Rank Fusion (RRF)
- **PPR-based Graph Traversal**: Personalized PageRank for intelligent entity-based retrieval
- **Neo4j Vector Index**: Native vector search capabilities integrated with graph queries
- **Multi-format Document Support**: Ingest PDF, Markdown, HTML, text documents, and ECT-QA JSONL datasets
- **Entity Extraction & Relation Extraction**: LLM-based NER and semantic relationship mapping
- **Entity Deduplication**: Intelligent entity consolidation using embeddings, string similarity, and LLM validation
- **Four Web Interfaces**:
  - Graph Visualization Explorer
  - Chunk Retrieval Interface
  - RAG Chatbot with Source Citations
  - Visual Search Interface

## Architecture

The system implements temporality at multiple levels:
1. **Document-level**: Fiscal year and quarter metadata from document content
2. **Chunk-level**: Fiscal period fields (`fiscal_year`, `fiscal_quarter`, `fiscal_period_end`) and version history with supersession tracking
3. **Relationship-level**: `valid_from` and `valid_to` dates on all relationships
4. **Bi-temporal Model**: Separation of event time (when content refers to) and transaction time (when ingested)

### Technology Stack

- **Graph Database**: Neo4j 5.15+ with vector index
- **Embeddings**: LiteLLM Proxy (configurable models, e.g., `text-embedding-3-large` or `Qwen/Qwen3-Embedding-8B`)
- **LLM**: LiteLLM Proxy (OpenAI-compatible, for entity extraction, relation extraction, RAG generation)
- **Web Framework**: Streamlit
- **Python**: 3.10+
- **Package Manager**: uv (NOT pip)
- **Libraries**: LangChain, LangGraph, Neo4j Python driver, tiktoken, Pydantic

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
DEFAULT_LLM_MODEL=default

# Embedding Configuration (via LiteLLM)
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072

# Or for Qwen embeddings:
# EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
# EMBEDDING_DIMENSIONS=4096

# PPR-based Traversal (optional)
ENABLE_PPR_TRAVERSAL=true
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
├── docker-compose.yml              # Services orchestration (Neo4j)
├── .env.example                    # Environment template
├── requirements.txt                # Python dependencies
├── CLAUDE.md                       # Developer guidance for Claude Code
│
├── src/temporal_kg_rag/           # Core library
│   ├── config/                    # Configuration management (settings.py)
│   ├── models/                    # Data models (document, chunk, entity, temporal)
│   ├── graph/                     # Neo4j operations (client, schema, operations, consolidation)
│   ├── embeddings/                # Embedding generation via LiteLLM (generator, cache)
│   ├── temporal/                  # Temporal query building (query_builder, versioning, time_travel)
│   ├── ingestion/                 # Document processing (loader, chunker, entity_extractor,
│   │                              #   relation_extractor, entity_deduplication, ectqa_loader, pipeline)
│   ├── retrieval/                 # Search (vector_search, graph_search, hybrid_search,
│   │                              #   ppr_traversal, temporal_retrieval, context_expansion)
│   ├── rag/                       # RAG workflow (chain, graph, prompts, context_builder)
│   └── utils/                     # Utilities (logger)
│
├── scripts/
│   ├── init_db.py                 # Database initialization
│   ├── ingest_documents.py        # Document ingestion CLI
│   ├── consolidate_graph.py       # Entity deduplication and graph consolidation
│   ├── migrate_fiscal_periods.py  # Migrate fiscal period data to chunks
│   ├── evaluate_tkg.py            # Evaluate TKG answers from JSONL
│   ├── test_retrieval.py          # Test retrieval methods
│   ├── test_rag.py                # Test RAG system
│   ├── run_all_apps.sh            # Start all Streamlit apps
│   └── stop_all_apps.sh           # Stop all Streamlit apps
│
├── apps/                          # Streamlit applications
│   ├── 1_graph_visualization.py   # Graph explorer (Port 8501)
│   ├── 2_chunk_retrieval.py       # Search interface (Port 8502)
│   ├── 3_chatbot.py               # RAG chatbot (Port 8503)
│   └── 4_visual_search.py         # Visual entity search (Port 8504)
│
└── data/                          # Data files (JSONL datasets, evaluation results)
```

## Database Schema

### Node Labels

- **Document**: Source documents with metadata (`meta_year`, `meta_quarter`, `metadata_json`)
- **Chunk**: Text segments with embeddings and fiscal period fields (`fiscal_year`, `fiscal_quarter`, `fiscal_period_end`, `is_current`)
- **Entity**: Named entities (PERSON, ORG, LOCATION, PRODUCT, EVENT, etc.)

### Key Relationships

- `(Document)-[:HAS_CHUNK]->(Chunk)` - Document to chunk relationship
- `(Chunk)-[:SUPERSEDES]->(Chunk)` - Version history
- `(Chunk)-[:MENTIONS {valid_from, valid_to, confidence}]->(Entity)` - Temporal entity mentions
- `(Entity)-[:RELATES_TO {relationship, timestamp, description, valid_from, valid_to}]->(Entity)` - Semantic entity relationships (temporal quadruples)

### Chunk Properties (Fiscal Period-based Temporal Filtering)

- `fiscal_year`: Integer (e.g., 2021) - The fiscal year the content refers to
- `fiscal_quarter`: String (e.g., "Q1", "Q2", "Q3", "Q4") - The fiscal quarter
- `fiscal_period_end`: DateTime - End date of the fiscal period
- `is_current`: Boolean - Whether this is the current version of the chunk

### Indexes

- Unique constraints on all node IDs
- Vector index on `Chunk.embedding` (configurable dimensions: 3072 or 4096, cosine similarity)
- Performance indexes on fiscal period fields (`fiscal_year`, `fiscal_quarter`, `is_current`)
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
# Ingest a single document
uv run python scripts/ingest_documents.py --path document.pdf --title "Document Title"

# Ingest all documents in a directory
uv run python scripts/ingest_documents.py --path docs/ --pattern "*.pdf"

# Ingest with fiscal period metadata (for temporal filtering)
uv run python scripts/ingest_documents.py \
    --path earnings_call.txt \
    --title "Q2 2021 Earnings Call" \
    --metadata '{"year": 2021, "quarter": "Q2", "company": "Skechers"}'

# Ingest recursively with statistics
uv run python scripts/ingest_documents.py \
    --path docs/ \
    --recursive \
    --pattern "*.md" \
    --show-stats

# Quick ingestion (skip entity extraction for speed)
uv run python scripts/ingest_documents.py --path doc.txt --no-entities

# Ingest ECT-QA dataset (earnings call transcripts)
# Use the ectqa_loader for JSONL format with fiscal period metadata
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
from temporal_kg_rag.retrieval.ppr_traversal import get_ppr_traversal
from temporal_kg_rag.models.temporal import TemporalFilter

# Hybrid search (combines vector + graph with RRF)
hs = get_hybrid_search()
results = hs.search(
    query="quarterly sales performance",
    top_k=10,
)

# Temporal search with auto-detection
tr = get_temporal_retrieval()
temporal_results = tr.search_with_temporal_context(
    query="Skechers revenue in Q2 2021",
    auto_detect_temporal=True,
)

# Fiscal period-based search
fiscal_filter = TemporalFilter.create_fiscal_period(year=2021, quarter="Q2")
filtered_results = hs.search(
    query="revenue growth",
    top_k=10,
    temporal_filter=fiscal_filter,
)

# Fiscal range search
range_filter = TemporalFilter.create_fiscal_range(start_year=2020, end_year=2022)
range_results = hs.search(query="financial performance", top_k=10, temporal_filter=range_filter)

# PPR-based graph traversal (if enabled)
ppr = get_ppr_traversal()
ppr_results = ppr.search(seed_entities=["Skechers", "revenue"], top_k=10)

# Expand results with context
ce = get_context_expander()
expanded = ce.expand_results(results, include_neighboring_chunks=True, include_entities=True)
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

The project includes four Streamlit applications for easy interaction with the system:

```bash
# Quick start - run all apps at once
./scripts/run_all_apps.sh

# Or run individually:
uv run streamlit run apps/1_graph_visualization.py                    # Port 8501
uv run streamlit run apps/2_chunk_retrieval.py --server.port 8502     # Port 8502
uv run streamlit run apps/3_chatbot.py --server.port 8503             # Port 8503
uv run streamlit run apps/4_visual_search.py --server.port 8504       # Port 8504

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

**Application 4: Visual Search Interface** (Port 8504)
- Visual entity-based search
- Interactive graph exploration
- Entity relationship visualization

See [apps/README.md](apps/README.md) for detailed documentation.

### Using Configuration

```python
from temporal_kg_rag.config.settings import get_settings

settings = get_settings()
print(f"Neo4j URI: {settings.neo4j_uri}")
print(f"Chunk size: {settings.chunk_size}")
print(f"Embedding model: {settings.embedding_model}")
print(f"Embedding dimensions: {settings.embedding_dimensions}")
print(f"PPR enabled: {settings.enable_ppr_traversal}")
```

## Configuration

All configuration is managed through environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j connection URI | bolt://localhost:7687 |
| `NEO4J_PASSWORD` | Neo4j password | password |
| `LITELLM_API_BASE` | LiteLLM proxy URL | http://localhost:4000 |
| `LITELLM_API_KEY` | LiteLLM API key | sk-1234 |
| `DEFAULT_LLM_MODEL` | Default LLM model | default |
| `EMBEDDING_MODEL` | Embedding model | text-embedding-3-large |
| `EMBEDDING_DIMENSIONS` | Embedding vector size | 3072 |
| `CHUNK_SIZE` | Chunk size in tokens | 1000 |
| `CHUNK_OVERLAP` | Overlap between chunks | 100 |
| `HYBRID_SEARCH_ALPHA` | Vector vs graph weight (0=graph, 1=vector) | 0.5 |
| `ENABLE_PPR_TRAVERSAL` | Enable Personalized PageRank | true |
| `ENABLE_EMBEDDING_CACHE` | Enable embedding caching | true |
| `VECTOR_SIMILARITY_THRESHOLD` | Minimum similarity score | 0.7 |

## Development

### Running Tests

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/temporal_kg_rag
```

### Code Formatting

```bash
# Format code with black
uv run black src/ tests/ scripts/ apps/

# Lint with ruff
uv run ruff check src/ tests/ scripts/ apps/

# Type checking with mypy
uv run mypy src/
```

### Data Migration

If you need to update existing data (e.g., after adding new chunk fields):

```bash
# Migrate fiscal period data from documents to chunks
uv run python scripts/migrate_fiscal_periods.py --verify    # Check current state
uv run python scripts/migrate_fiscal_periods.py --dry-run   # Preview changes
uv run python scripts/migrate_fiscal_periods.py             # Apply migration

# Consolidate entities (deduplication)
uv run python scripts/consolidate_graph.py --dry-run
uv run python scripts/consolidate_graph.py
```

### Evaluation

```bash
# Evaluate TKG answers from JSONL dataset
uv run python scripts/evaluate_tkg.py --input data/questions.jsonl --output data/evaluated.jsonl
uv run python scripts/evaluate_tkg.py --input data/questions.jsonl --output data/evaluated.jsonl --limit 10
uv run python scripts/evaluate_tkg.py --comparison-only --input data/questions.jsonl --output data/evaluated.jsonl
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
result = client.execute_read_transaction(query, {})
print(result)
```

### Dimension Mismatch Errors

If you see vector dimension errors, your embeddings don't match the vector index:

```bash
# Re-initialize database with correct dimensions
uv run python scripts/init_db.py --force

# Re-ingest all documents
uv run python scripts/ingest_documents.py --path your_docs/
```

### LiteLLM Connection Issues

```bash
# Check if LiteLLM is accessible
curl $LITELLM_API_BASE/health
```

### Missing Fiscal Period Data

If temporal queries return no results, chunks may be missing fiscal period fields:

```bash
# Check and migrate fiscal period data
uv run python scripts/migrate_fiscal_periods.py --verify
uv run python scripts/migrate_fiscal_periods.py --show-docs
uv run python scripts/migrate_fiscal_periods.py
```

### Python Import Issues

With `uv`, you don't need to manually set PYTHONPATH:

```bash
# Use uv run for all commands
uv run python scripts/init_db.py

# Or add src to PYTHONPATH manually
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## License

MIT License - see LICENSE file for details

