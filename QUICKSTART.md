# Temporal Knowledge Graph RAG System - Quick Start Guide

This guide will help you get the system up and running in 15 minutes.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.10 or higher
- External LiteLLM service (for LLM and embeddings)
- `uv` package manager
- 4GB RAM minimum

## Step-by-Step Setup

### 1. Environment Configuration (2 minutes)

```bash
# Clone and navigate to the project
cd temporal-knowledge-graphs

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

Required settings in `.env`:
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
LITELLM_API_BASE=http://your-litellm-url:4000
LITELLM_API_KEY=your-litellm-api-key
DEFAULT_LLM_MODEL=default
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072
ENABLE_PPR_TRAVERSAL=true
```

### 2. Start Docker Services (1 minute)

```bash
# Start Neo4j
docker-compose up -d

# Check service is running
docker-compose ps

# Wait for Neo4j to be ready (check http://localhost:7474)
```

**Note**: LiteLLM is provided externally. Make sure your LiteLLM service is running and accessible.

### 3. Install Python Dependencies (3 minutes)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with uv
uv pip install -r requirements.txt
```

### 4. Initialize Database (1 minute)

```bash
# Initialize Neo4j schema (constraints, indexes, vector index)
uv run python scripts/init_db.py

# Verify schema
uv run python scripts/init_db.py --verify-only
```

Expected output:
```
✓ Neo4j connectivity verified
✓ Constraints created
✓ Indexes created
✓ Vector index created (3072 dimensions)
```

### 5. Ingest Documents (5 minutes)

```bash
# Ingest documents into the knowledge graph
uv run python scripts/ingest_documents.py --path your_docs/ --pattern "*.pdf"

# Or ingest with fiscal period metadata (for temporal filtering)
uv run python scripts/ingest_documents.py \
  --path earnings_call.txt \
  --title "Q2 2021 Earnings Call" \
  --metadata '{"year": 2021, "quarter": "Q2"}'
```

This will:
- Process documents into chunks with fiscal period metadata
- Generate embeddings (via LiteLLM)
- Extract entities and relations (using LLM via LiteLLM)
- Build the temporal knowledge graph

### 6. Test the System (2 minutes)

```bash
# Test retrieval
uv run python scripts/test_retrieval.py --demo

# Test RAG system
uv run python scripts/test_rag.py --demo
```

### 7. Launch Web Interfaces (1 minute)

```bash
# Start all four Streamlit apps
./scripts/run_all_apps.sh
```

Access the applications:
- **Graph Visualization**: http://localhost:8501
- **Chunk Retrieval**: http://localhost:8502
- **RAG Chatbot**: http://localhost:8503
- **Visual Search**: http://localhost:8504

## Quick Test Workflow

### 1. Test Graph Visualization (App 1)
1. Open http://localhost:8501
2. Click "Connect to Neo4j" in sidebar
3. Search for "OpenAI" in Entity Explorer
4. Click "Visualize Graph" to see connections

### 2. Test Chunk Retrieval (App 2)
1. Open http://localhost:8502
2. Enter query: "What is artificial intelligence?"
3. Select "Hybrid Search" method
4. Click "Search" and review results

### 3. Test RAG Chatbot (App 3)
1. Open http://localhost:8503
2. Ask: "What were the main AI developments in 2023?"
3. Review answer and source citations
4. Follow up: "Tell me more about GPT-4"

## Common Commands

```bash
# Ingest a new document
uv run python scripts/ingest_documents.py --path document.pdf --title "My Document"

# Ingest with fiscal period metadata
uv run python scripts/ingest_documents.py --path report.pdf --metadata '{"year": 2022, "quarter": "Q3"}'

# Search with different methods
uv run python scripts/test_retrieval.py --query "quarterly revenue" --method hybrid

# Test temporal/fiscal period queries
uv run python scripts/test_rag.py --query "Revenue in Q2 2021" --method langgraph --temporal

# Migrate fiscal period data (if chunks are missing fiscal_year)
uv run python scripts/migrate_fiscal_periods.py --verify
uv run python scripts/migrate_fiscal_periods.py

# Consolidate entities (deduplication)
uv run python scripts/consolidate_graph.py

# Stop all Streamlit apps
./scripts/stop_all_apps.sh

# View Neo4j browser
# Open http://localhost:7474 (user: neo4j, password: password)

# View logs
docker-compose logs neo4j
```

## Troubleshooting

### Neo4j Connection Failed
```bash
# Check if Neo4j is running
docker-compose ps neo4j

# View logs
docker-compose logs neo4j

# Restart Neo4j
docker-compose restart neo4j
```

### LiteLLM API Errors
```bash
# Check your LiteLLM service is accessible
curl $LITELLM_API_BASE/health

# Test embeddings endpoint
curl $LITELLM_API_BASE/embeddings \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-Embedding-8B", "input": "test"}'
```

### Import Errors
```bash
# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Reinstall dependencies with uv
uv pip install -r requirements.txt
```

### Port Already in Use
```bash
# Kill processes on ports
lsof -ti:8501 | xargs kill
lsof -ti:8502 | xargs kill
lsof -ti:8503 | xargs kill
lsof -ti:8504 | xargs kill

# Or use different ports
uv run streamlit run apps/3_chatbot.py --server.port 9503
```

### Missing Fiscal Period Data
```bash
# If temporal queries return no results, migrate fiscal period data
uv run python scripts/migrate_fiscal_periods.py --verify
uv run python scripts/migrate_fiscal_periods.py --show-docs
uv run python scripts/migrate_fiscal_periods.py
```

## Next Steps

### Add Your Own Documents

```bash
# PDF documents
uv run python scripts/ingest_documents.py --path /path/to/documents/ --pattern "*.pdf"

# Markdown files
uv run python scripts/ingest_documents.py --path /path/to/docs/ --pattern "*.md" --recursive

# With fiscal period metadata (for temporal filtering)
uv run python scripts/ingest_documents.py \
  --path earnings_q2_2021.pdf \
  --title "Q2 2021 Earnings Call" \
  --metadata '{"year": 2021, "quarter": "Q2", "company": "Acme Corp"}'
```

### Explore Advanced Features

1. **Fiscal Period Queries**
   - "What were the Q2 2021 results?"
   - "Revenue changes between 2020 and 2022"
   - "Recent quarterly performance"

2. **Entity-Based Search**
   - Search for specific entities (people, organizations, products)
   - Explore entity relationships in the graph
   - PPR-based traversal from seed entities

3. **Comparison Queries**
   - "Compare Q1 2021 and Q1 2022 performance"
   - "Difference between domestic and international sales"

4. **Evolution Queries**
   - "How has revenue evolved over the quarters?"
   - "History of product launches"

### Customize the System

1. **Adjust Chunking**
   - Edit `.env`: `CHUNK_SIZE=1000` and `CHUNK_OVERLAP=100`

2. **Change Hybrid Search Weighting**
   - Edit `.env`: `HYBRID_SEARCH_ALPHA=0.5` (0=graph only, 1=vector only)

3. **Use Different LLM**
   - Configure LiteLLM proxy to use different models
   - Edit `docker-compose.yml` or use external LiteLLM

## System Architecture

```
User Query
    ↓
[Web Interface] (4 Streamlit apps)
    ↓
[RAG System] ← [Prompt Templates]
    ↓
[Hybrid Retrieval] (RRF Fusion)
    ├─ [Vector Search] (Neo4j vector index)
    ├─ [Graph Search] (Entity traversal)
    └─ [PPR Traversal] (Personalized PageRank)
    ↓
[Context Building] ← [Fiscal Period Filter]
    ↓
[LLM Generation] (via LiteLLM)
    ↓
[Answer + Sources]
```

## Performance Tips

1. **For Large Datasets**
   - Reduce `top_k` in searches (default: 10)
   - Enable fiscal period filtering to narrow results
   - Use PPR traversal for entity-focused queries

2. **For Faster Ingestion**
   - Skip entity extraction: `--no-entities`
   - Use batch mode for multiple documents

3. **For Better Answers**
   - Increase context expansion
   - Use LangGraph workflow (more thorough)
   - Enable conversation history in chatbot
   - Include fiscal period metadata when ingesting

## Getting Help

- **Documentation**: See [README.md](README.md) for full documentation
- **Application Help**: See [apps/README.md](apps/README.md)
- **Database Schema**: Run `python scripts/init_db.py --show-schema`
- **Logs**: Check `logs/` directory for application logs

## Resources

- **Neo4j Browser**: http://localhost:7474
- **LiteLLM Proxy**: (External service - configured in .env)
- **Project Plan**: `.claude/plans/generic-tinkering-cosmos.md`

## Important Notes

- **LiteLLM Service**: This system requires an external LiteLLM service for both LLM and embeddings
- **Embedding Model**: Configurable via `EMBEDDING_MODEL` (default: `text-embedding-3-large` with 3072 dimensions)
- **Package Manager**: Uses `uv` for fast Python package management - always use `uv run python ...`
- **No OpenAI Dependency**: All API calls go through LiteLLM proxy
- **Fiscal Period Filtering**: Temporal queries filter by document content time (year/quarter), not ingestion time

---

**Estimated Setup Time**: 15 minutes

**Status**: Ready for production use

**Last Updated**: 2026-02-01
