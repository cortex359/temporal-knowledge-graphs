# Temporal Knowledge Graph RAG System

A powerful RAG (Retrieval-Augmented Generation) system that uses a temporal knowledge graph for document storage and retrieval with temporal awareness.

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

**TL;DR**: See [QUICKSTART.md](QUICKSTART.md) for a 15-minute setup guide.

### 1. Prerequisites

- Docker and Docker Compose
- Python 3.10 or higher
- OpenAI API key
- LiteLLM Proxy running (or use the included Docker setup)

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

# OpenAI
OPENAI_API_KEY=sk-your-key-here

# LiteLLM (configure based on your setup)
LITELLM_API_BASE=http://localhost:4000
LITELLM_API_KEY=sk-1234
```

### 3. Start Services

```bash
# Start Neo4j and LiteLLM
docker-compose up -d

# Wait for Neo4j to be ready (check http://localhost:7474)
# Default credentials: neo4j/password
```

### 4. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model for entity extraction
python -m spacy download en_core_web_sm
```

### 5. Initialize Database

```bash
# Initialize Neo4j schema (constraints, indexes, vector index)
python scripts/init_db.py

# Verify schema
python scripts/init_db.py --verify-only

# Show current schema
python scripts/init_db.py --show-schema
```

## Project Status

### ✅ Phase 1: Foundation (COMPLETED)

The foundation is complete with:

**Infrastructure:**
- ✅ Docker Compose setup with Neo4j and LiteLLM
- ✅ Environment configuration (.env.example)
- ✅ Project structure with proper package layout

**Core Library:**
- ✅ Configuration management (pydantic-settings)
- ✅ Structured logging (JSON and text formats)
- ✅ Neo4j client with connection pooling and retry logic
- ✅ Schema initialization (constraints, indexes, vector index)
- ✅ Data models (Document, Chunk, Entity, Temporal)

**Validation:**
- Neo4j services running: ✅
- Schema initialized: ✅
- Vector index created (1536 dimensions): ✅
- Python imports working: ✅

### ✅ Phase 2: Document Ingestion (COMPLETED)

The complete document ingestion pipeline is now operational:

**Document Processing:**
- ✅ Document loader for PDF, Markdown, HTML, and text files
- ✅ Semantic chunking with sentence boundaries (1000 tokens, 100 overlap)
- ✅ Fixed-size chunking fallback
- ✅ Token counting with tiktoken

**Embeddings:**
- ✅ OpenAI embedding generation with batch processing
- ✅ File-based embedding cache to avoid redundant API calls
- ✅ Automatic retry logic for API failures
- ✅ Cost estimation

**Entity Extraction:**
- ✅ spaCy NER integration (PERSON, ORG, LOCATION, etc.)
- ✅ Entity deduplication and aggregation
- ✅ Entity mention tracking with context

**Graph Operations:**
- ✅ CRUD operations for documents, chunks, and entities
- ✅ Batch operations for efficient storage
- ✅ Relationship creation (HAS_CHUNK, MENTIONS)

**Temporal Features:**
- ✅ Version management with SUPERSEDES relationships
- ✅ Temporal query building
- ✅ Point-in-time queries
- ✅ Time travel capabilities

**CLI Tools:**
- ✅ `scripts/ingest_documents.py` - Full-featured ingestion script
- ✅ `scripts/sample_data.py` - Generate test documents

**Validation:**
- Can ingest multi-format documents: ✅
- Chunks stored with embeddings: ✅
- Entities extracted and linked: ✅
- Temporal metadata captured: ✅

### ✅ Phase 3: Retrieval System (COMPLETED)

The complete hybrid retrieval system with temporal awareness is now operational:

**Vector Search:**
- ✅ Neo4j vector index integration
- ✅ Cosine similarity search
- ✅ Configurable similarity thresholds
- ✅ Query embedding generation
- ✅ Context window expansion

**Graph Search:**
- ✅ Entity-based retrieval
- ✅ Entity co-occurrence patterns
- ✅ Multi-hop graph traversal
- ✅ Entity type filtering
- ✅ Full-text search fallback

**Hybrid Search:**
- ✅ Reciprocal Rank Fusion (RRF) algorithm
- ✅ Configurable alpha weighting (vector vs graph)
- ✅ Result deduplication
- ✅ Score normalization
- ✅ Comparative analysis tools

**Temporal Retrieval:**
- ✅ Automatic temporal context detection
- ✅ Point-in-time queries ("as of 2023")
- ✅ Time range queries ("between 2020 and 2024")
- ✅ Historical evolution analysis
- ✅ Version-aware retrieval

**Context Expansion:**
- ✅ Neighboring chunk retrieval
- ✅ Entity graph neighborhoods
- ✅ Related chunk discovery
- ✅ Document context enrichment
- ✅ Context summary generation

**Testing Tools:**
- ✅ `scripts/test_retrieval.py` - Comprehensive test suite
- ✅ Demo mode with example queries
- ✅ Comparison tools for different methods

**Validation:**
- Vector search working: ✅
- Graph traversal working: ✅
- Hybrid RRF combining correctly: ✅
- Temporal queries filtering properly: ✅
- Context expansion enriching results: ✅

### ✅ Phase 4: RAG System (COMPLETED)

The complete RAG system with both LangChain and LangGraph implementations is now operational:

**Prompt Engineering:**
- ✅ Comprehensive prompt templates for different query types
- ✅ System prompts with temporal awareness
- ✅ Factual, temporal, comparison, evolution, and exploratory prompts
- ✅ Source citation formatting
- ✅ Context and entity relationship formatting

**Context Building:**
- ✅ Context assembly from retrieval results
- ✅ Metadata enrichment (sources, entities, temporal info)
- ✅ Context truncation and summarization
- ✅ Conversation history integration
- ✅ Prompt template selection based on query type

**LangChain RAG Chain:**
- ✅ Simple RAG chain implementation
- ✅ Temporal detection integration
- ✅ Context expansion
- ✅ Conversation history support
- ✅ Streaming response capability
- ✅ Error handling and fallbacks

**LangGraph Workflow:**
- ✅ Multi-node RAG pipeline with 5 stages
- ✅ Node 1: Query understanding (temporal, entities, query type)
- ✅ Node 2: Retrieval (hybrid search with temporal filtering)
- ✅ Node 3: Context building (formatting with metadata)
- ✅ Node 4: Generation (LLM answer with appropriate prompts)
- ✅ Node 5: Verification (quality checks and validation)
- ✅ State management across workflow
- ✅ Dynamic prompt selection based on query characteristics

**Testing Tools:**
- ✅ `scripts/test_rag.py` - Comprehensive RAG testing suite
- ✅ Demo mode with example queries
- ✅ Comparison tools for LangChain vs LangGraph
- ✅ Streaming response testing
- ✅ Conversation history testing

**Validation:**
- RAG chain generates relevant answers: ✅
- LangGraph workflow processes queries correctly: ✅
- Source citations working properly: ✅
- Temporal queries handled correctly: ✅
- Different query types use appropriate prompts: ✅
- Verification checks functioning: ✅

### ✅ Phase 5: Web Interfaces (COMPLETED)

Three comprehensive Streamlit applications for interacting with the temporal knowledge graph:

**Graph Visualization Explorer:**
- ✅ Interactive graph exploration with streamlit-agraph
- ✅ Database statistics dashboard
- ✅ Entity search and neighborhood visualization
- ✅ Document explorer with chunk relationships
- ✅ Temporal filtering controls (point-in-time, date ranges)
- ✅ Custom Cypher query interface
- ✅ Graph export to JSON

**Chunk Retrieval Interface:**
- ✅ Multiple search strategies (Vector, Graph, Hybrid, Temporal)
- ✅ Search method comparison tool
- ✅ Result highlighting with query terms
- ✅ Temporal and entity filtering
- ✅ Performance metrics display
- ✅ Advanced options (similarity threshold, alpha weighting)
- ✅ Result export functionality

**RAG Chatbot:**
- ✅ Conversational chat interface
- ✅ Both LangGraph and LangChain implementations
- ✅ Streaming response support
- ✅ Source citations with expandable details
- ✅ Conversation history management
- ✅ Temporal query detection indicators
- ✅ Entity extraction display
- ✅ Answer verification warnings
- ✅ Quick query buttons
- ✅ Conversation export

**Validation:**
- All three apps launch successfully: ✅
- Graph visualization renders correctly: ✅
- Search retrieves and displays results: ✅
- Chatbot provides coherent answers with sources: ✅
- Temporal filtering works across all interfaces: ✅
- UI is intuitive and responsive: ✅

### ⏳ Phase 6: Testing & Documentation (NOT STARTED)

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
│   ├── graph/                     # Neo4j operations ✅
│   ├── embeddings/                # Embedding generation
│   ├── temporal/                  # Temporal query building
│   ├── ingestion/                 # Document processing
│   ├── retrieval/                 # Hybrid search
│   ├── rag/                       # RAG workflow
│   └── utils/                     # Utilities ✅
│
├── scripts/
│   ├── init_db.py                 # Database initialization ✅
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
python scripts/init_db.py

# Force re-initialization (WARNING: drops existing schema)
python scripts/init_db.py --force

# Verify schema is correct
python scripts/init_db.py --verify-only
```

### Document Ingestion

```bash
# Generate sample documents for testing
python scripts/sample_data.py --output-dir ./sample_data

# Ingest a single document
python scripts/ingest_documents.py --path sample_data/artificial_intelligence_2023.txt

# Ingest all documents in a directory
python scripts/ingest_documents.py --path sample_data/ --pattern "*.txt"

# Ingest with custom metadata
python scripts/ingest_documents.py \
    --path document.pdf \
    --title "Annual Report 2024" \
    --metadata '{"author": "Jane Doe", "department": "Engineering"}'

# Ingest recursively with statistics
python scripts/ingest_documents.py \
    --path docs/ \
    --recursive \
    --pattern "*.md" \
    --show-stats

# Quick ingestion (skip entity extraction for speed)
python scripts/ingest_documents.py --path doc.txt --no-entities
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
python scripts/test_retrieval.py --demo

# Search with different methods
python scripts/test_retrieval.py --query "artificial intelligence" --method hybrid
python scripts/test_retrieval.py --query "OpenAI" --method graph
python scripts/test_retrieval.py --query "machine learning" --method vector

# Temporal search
python scripts/test_retrieval.py --query "AI in 2023" --temporal
python scripts/test_retrieval.py --query "quantum computing" --year 2023

# Search with context expansion
python scripts/test_retrieval.py --query "climate change" --expand-context

# Compare all search methods
python scripts/test_retrieval.py --query "neural networks" --compare
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
python scripts/test_rag.py --demo

# Test LangChain RAG with custom query
python scripts/test_rag.py --query "What is artificial intelligence?" --method langchain

# Test LangGraph RAG with temporal query
python scripts/test_rag.py --query "AI developments in 2023" --method langgraph --temporal

# Compare both implementations
python scripts/test_rag.py --query "machine learning" --compare

# Test streaming response
python scripts/test_rag.py --query "climate change" --stream

# Run conversation demo
python scripts/test_rag.py --conversation-demo
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

## Roadmap

- [x] Phase 1: Foundation and core infrastructure
- [x] Phase 2: Document ingestion pipeline
- [x] Phase 3: Hybrid retrieval system
- [x] Phase 4: RAG workflow with LangGraph
- [x] Phase 5: Streamlit web interfaces
- [ ] Phase 6: Testing and optimization

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation in `/docs`
- Review the plan file in `.claude/plans/`

---

**Status**: Phases 1-5 Complete ✅✅✅✅✅ | Last Updated: 2026-01-06
