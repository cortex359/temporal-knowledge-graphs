# Streamlit Web Applications

This directory contains three Streamlit applications for interacting with the Temporal Knowledge Graph RAG system.

## Applications

### 1. Graph Visualization Explorer (`1_graph_visualization.py`)

Interactive exploration of the temporal knowledge graph.

**Features:**
- Database statistics and connection status
- Entity search and exploration
- Entity neighborhood graph visualization
- Document explorer with chunk relationships
- Temporal filtering (point-in-time, date ranges)
- Graph export to JSON
- Custom Cypher query interface

**How to run:**
```bash
streamlit run apps/1_graph_visualization.py
```

**Access:** http://localhost:8501

---

### 2. Chunk Retrieval Interface (`2_chunk_retrieval.py`)

Search and retrieve text chunks using different strategies.

**Features:**
- Multiple search methods:
  - Vector Search (semantic similarity)
  - Graph Search (entity-based traversal)
  - Hybrid Search (RRF combination)
  - Temporal Search (automatic temporal detection)
- Temporal filtering controls
- Entity-based filtering
- Result highlighting with query terms
- Search method comparison
- Performance metrics
- Result export to JSON

**How to run:**
```bash
streamlit run apps/2_chunk_retrieval.py
```

**Access:** http://localhost:8502

---

### 3. RAG Chatbot (`3_chatbot.py`)

Conversational interface with the complete RAG system.

**Features:**
- Chat interface with conversation history
- Two RAG implementations:
  - LangGraph (multi-node workflow)
  - LangChain (simple chain)
- Streaming responses (LangChain only)
- Source citations with expandable details
- Temporal query detection
- Entity extraction display
- Answer verification indicators
- Conversation export to JSON
- Quick query buttons
- Response time metrics

**How to run:**
```bash
streamlit run apps/3_chatbot.py
```

**Access:** http://localhost:8503

---

## Running All Applications

You can run all three applications simultaneously:

```bash
# Terminal 1
streamlit run apps/1_graph_visualization.py --server.port 8501

# Terminal 2
streamlit run apps/2_chunk_retrieval.py --server.port 8502

# Terminal 3
streamlit run apps/3_chatbot.py --server.port 8503
```

Or use a process manager like `tmux` or `screen`:

```bash
# Using tmux
tmux new-session -d -s streamlit1 'streamlit run apps/1_graph_visualization.py'
tmux new-session -d -s streamlit2 'streamlit run apps/2_chunk_retrieval.py --server.port 8502'
tmux new-session -d -s streamlit3 'streamlit run apps/3_chatbot.py --server.port 8503'
```

## Prerequisites

1. **Neo4j Database** must be running:
   ```bash
   docker-compose up -d neo4j
   ```

2. **LiteLLM Proxy** must be configured and running:
   ```bash
   docker-compose up -d litellm
   ```

3. **Documents ingested** into the knowledge graph:
   ```bash
   python scripts/sample_data.py --output-dir sample_data
   python scripts/ingest_documents.py --path sample_data/ --pattern "*.txt"
   ```

4. **Environment variables** configured in `.env`:
   ```bash
   NEO4J_URI=bolt://localhost:7687
   NEO4J_PASSWORD=password
   OPENAI_API_KEY=sk-your-key
   LITELLM_API_BASE=http://localhost:4000
   ```

## Configuration

### Streamlit Configuration

Create `.streamlit/config.toml` in the project root for custom Streamlit settings:

```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#4ecdc4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Application Settings

Each application has its own sidebar settings that can be adjusted in real-time:

- **Graph Visualization:** Temporal filters, graph depth, entity types
- **Chunk Retrieval:** Search method, top-k results, similarity thresholds
- **Chatbot:** RAG implementation, streaming, conversation history length

## Troubleshooting

### Connection Issues

If applications can't connect to Neo4j:

1. Check Docker containers are running:
   ```bash
   docker-compose ps
   ```

2. Verify Neo4j is accessible:
   ```bash
   curl http://localhost:7474
   ```

3. Check environment variables:
   ```bash
   cat .env
   ```

### Import Errors

If you get import errors:

```bash
# Ensure you're in the project root
cd /path/to/temporal-knowledge-graphs

# Install dependencies
pip install -r requirements.txt

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Port Conflicts

If ports are already in use, specify different ports:

```bash
streamlit run apps/1_graph_visualization.py --server.port 9501
```

### Performance Issues

For large datasets:

1. Limit the number of results (reduce top-k)
2. Use temporal filtering to narrow results
3. Enable caching in Streamlit:
   ```python
   @st.cache_data
   def expensive_computation():
       ...
   ```

## Tips for Best Experience

1. **Start with the Chatbot** - Most intuitive interface for general use
2. **Use Graph Visualization** - To explore entity relationships and understand the knowledge graph structure
3. **Use Chunk Retrieval** - For detailed search and comparison of different retrieval methods
4. **Enable Temporal Filtering** - When working with time-sensitive queries
5. **Export Results** - Save interesting searches and conversations for later analysis

## Examples

### Example Workflow

1. **Explore the Graph** (App 1):
   - Search for "OpenAI"
   - Visualize entity neighborhood
   - See related entities and documents

2. **Test Retrieval** (App 2):
   - Search "AI developments in 2023"
   - Compare Vector vs Hybrid search
   - Export top results

3. **Chat** (App 3):
   - Ask "What were the main AI breakthroughs in 2023?"
   - Review sources
   - Follow up: "Tell me more about GPT-4"

### Sample Questions for Chatbot

**Factual:**
- What is machine learning?
- Explain neural networks
- How does ChatGPT work?

**Temporal:**
- What happened in AI in 2023?
- Recent advances in quantum computing
- Evolution of climate policy

**Comparison:**
- Compare GPT-3 and GPT-4
- Difference between supervised and unsupervised learning
- Solar vs wind energy

## Development

To modify the applications:

1. Edit the Python files in `apps/`
2. Streamlit will auto-reload on file changes
3. Test changes immediately in the browser

For custom styling, modify the CSS in the `st.markdown()` blocks at the top of each file.

## Support

For issues:
- Check the logs in the Streamlit terminal
- Review Neo4j logs: `docker-compose logs neo4j`
- Verify LiteLLM is working: `curl http://localhost:4000/health`

---

**Last Updated:** 2026-01-06
