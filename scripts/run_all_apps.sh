#!/bin/bash
# Script to launch all three Streamlit applications

echo "🚀 Starting all Streamlit applications..."
echo ""

# Check if apps directory exists
if [ ! -d "apps" ]; then
    echo "❌ Error: apps directory not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Error: Streamlit is not installed"
    echo "Please install dependencies: pip install -r requirements.txt"
    exit 1
fi

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Check and kill existing processes on ports
for port in 8501 8502 8503; do
    if check_port $port; then
        echo "⚠️  Port $port is already in use. Killing existing process..."
        kill $(lsof -t -i:$port) 2>/dev/null || true
        sleep 1
    fi
done

echo ""
echo "📊 Launching applications:"
echo "  - Graph Visualization: http://localhost:8501"
echo "  - Chunk Retrieval: http://localhost:8502"
echo "  - RAG Chatbot: http://localhost:8503"
echo ""

# Create log directory
mkdir -p logs

# Launch apps in background with logging
echo "Starting Graph Visualization..."
nohup streamlit run apps/1_graph_visualization.py \
    --server.port 8501 \
    --server.headless true \
    > logs/graph_visualization.log 2>&1 &
echo $! > logs/graph_viz.pid

sleep 2

echo "Starting Chunk Retrieval..."
nohup streamlit run apps/2_chunk_retrieval.py \
    --server.port 8502 \
    --server.headless true \
    > logs/chunk_retrieval.log 2>&1 &
echo $! > logs/chunk_retrieval.pid

sleep 2

echo "Starting RAG Chatbot..."
nohup streamlit run apps/3_chatbot.py \
    --server.port 8503 \
    --server.headless true \
    > logs/chatbot.log 2>&1 &
echo $! > logs/chatbot.pid

sleep 3

echo ""
echo "✅ All applications started!"
echo ""
echo "📝 Process IDs:"
echo "  - Graph Visualization: $(cat logs/graph_viz.pid)"
echo "  - Chunk Retrieval: $(cat logs/chunk_retrieval.pid)"
echo "  - RAG Chatbot: $(cat logs/chatbot.pid)"
echo ""
echo "🌐 Access the applications:"
echo "  - Graph Visualization: http://localhost:8501"
echo "  - Chunk Retrieval: http://localhost:8502"
echo "  - RAG Chatbot: http://localhost:8503"
echo ""
echo "📋 View logs:"
echo "  - tail -f logs/graph_visualization.log"
echo "  - tail -f logs/chunk_retrieval.log"
echo "  - tail -f logs/chatbot.log"
echo ""
echo "🛑 To stop all applications, run:"
echo "  ./scripts/stop_all_apps.sh"
echo ""
