# Temporal Knowledge Graph RAG System - Quick Start Guide

This guide will help you get the system up and running in 15 minutes.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.10 or higher
- OpenAI API key
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
OPENAI_API_KEY=sk-your-actual-key-here
LITELLM_API_BASE=http://localhost:4000
LITELLM_API_KEY=sk-1234
```

### 2. Start Docker Services (2 minutes)

```bash
# Start Neo4j and LiteLLM
docker-compose up -d

# Check services are running
docker-compose ps

# Wait for Neo4j to be ready (check http://localhost:7474)
```

### 3. Install Python Dependencies (3 minutes)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model for entity extraction
python -m spacy download en_core_web_sm
```

### 4. Initialize Database (1 minute)

```bash
# Initialize Neo4j schema (constraints, indexes, vector index)
python scripts/init_db.py

# Verify schema
python scripts/init_db.py --verify-only
```

Expected output:
```
✓ Neo4j connectivity verified
✓ Constraints created
✓ Indexes created
✓ Vector index created (1536 dimensions)
```

### 5. Ingest Sample Data (5 minutes)

```bash
# Generate sample documents
python scripts/sample_data.py --output-dir sample_data

# Ingest documents into the knowledge graph
python scripts/ingest_documents.py --path sample_data/ --pattern "*.txt"
```

This will:
- Create 4 sample documents (AI, climate, quantum, healthcare)
- Process ~50-100 chunks
- Generate embeddings (using OpenAI API)
- Extract entities
- Build the temporal knowledge graph

### 6. Test the System (2 minutes)

```bash
# Test retrieval
python scripts/test_retrieval.py --demo

# Test RAG system
python scripts/test_rag.py --demo
```

### 7. Launch Web Interfaces (1 minute)

```bash
# Start all three Streamlit apps
./scripts/run_all_apps.sh
```

Access the applications:
- **Graph Visualization**: http://localhost:8501
- **Chunk Retrieval**: http://localhost:8502
- **RAG Chatbot**: http://localhost:8503

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
python scripts/ingest_documents.py --path document.pdf --title "My Document"

# Search with different methods
python scripts/test_retrieval.py --query "quantum computing" --method hybrid

# Test temporal queries
python scripts/test_rag.py --query "AI in 2023" --method langgraph --temporal

# Stop all Streamlit apps
./scripts/stop_all_apps.sh

# View Neo4j browser
# Open http://localhost:7474 (user: neo4j, password: password)

# View logs
docker-compose logs neo4j
docker-compose logs litellm
tail -f logs/chatbot.log
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

### OpenAI API Errors
```bash
# Check your API key is set
echo $OPENAI_API_KEY

# Test the API
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Reinstall dependencies
pip install -r requirements.txt
```

### Port Already in Use
```bash
# Kill processes on ports
lsof -ti:8501 | xargs kill
lsof -ti:8502 | xargs kill
lsof -ti:8503 | xargs kill

# Or use different ports
streamlit run apps/3_chatbot.py --server.port 9503
```

## Next Steps

### Add Your Own Documents

```bash
# PDF documents
python scripts/ingest_documents.py --path /path/to/documents/ --pattern "*.pdf"

# Markdown files
python scripts/ingest_documents.py --path /path/to/docs/ --pattern "*.md" --recursive

# With custom metadata
python scripts/ingest_documents.py \
  --path report.pdf \
  --title "Annual Report 2024" \
  --metadata '{"author": "John Doe", "department": "Engineering"}'
```

### Explore Advanced Features

1. **Temporal Queries**
   - "What did we know about AI in 2022?"
   - "Climate policy changes between 2020 and 2023"
   - "Recent developments in quantum computing"

2. **Entity-Based Search**
   - Search for specific entities (people, organizations)
   - Explore entity relationships in the graph
   - Filter by entity type

3. **Comparison Queries**
   - "Compare GPT-3 and GPT-4"
   - "Difference between supervised and unsupervised learning"

4. **Evolution Queries**
   - "How has AI evolved over time?"
   - "History of climate change policy"

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
[Web Interface]
    ↓
[RAG System] ← [Prompt Templates]
    ↓
[Hybrid Retrieval]
    ├─ [Vector Search] (Neo4j vector index)
    └─ [Graph Search] (Entity traversal)
    ↓
[Context Building] ← [Temporal Filter]
    ↓
[LLM Generation] (via LiteLLM)
    ↓
[Answer + Sources]
```

## Performance Tips

1. **For Large Datasets**
   - Reduce `top_k` in searches (default: 10)
   - Enable temporal filtering to narrow results
   - Use document-level filtering

2. **For Faster Ingestion**
   - Skip entity extraction: `--no-entities`
   - Use batch mode for multiple documents

3. **For Better Answers**
   - Increase context expansion
   - Use LangGraph workflow (more thorough)
   - Enable conversation history in chatbot

## Getting Help

- **Documentation**: See [README.md](README.md) for full documentation
- **Application Help**: See [apps/README.md](apps/README.md)
- **Database Schema**: Run `python scripts/init_db.py --show-schema`
- **Logs**: Check `logs/` directory for application logs

## Resources

- **Neo4j Browser**: http://localhost:7474
- **LiteLLM Proxy**: http://localhost:4000
- **Project Plan**: `.claude/plans/generic-tinkering-cosmos.md`

---

**Estimated Setup Time**: 15 minutes

**Status**: Ready for production use

**Last Updated**: 2026-01-06
