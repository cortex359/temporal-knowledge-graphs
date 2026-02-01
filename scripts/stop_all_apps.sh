#!/bin/bash
# Script to stop all running Streamlit applications

echo "🛑 Stopping all Streamlit applications..."
echo ""

# Function to stop process by PID file
stop_process() {
    local name=$1
    local pid_file=$2

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo "Stopping $name (PID: $pid)..."
            kill $pid 2>/dev/null
            sleep 1

            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                echo "  Force stopping $name..."
                kill -9 $pid 2>/dev/null
            fi

            echo "  ✓ $name stopped"
        else
            echo "  ℹ️  $name not running"
        fi
        rm -f "$pid_file"
    else
        echo "  ℹ️  No PID file for $name"
    fi
}

# Stop each application
stop_process "Graph Visualization" "logs/graph_viz.pid"
stop_process "Chunk Retrieval" "logs/chunk_retrieval.pid"
stop_process "RAG Chatbot" "logs/chatbot.pid"
stop_process "Visual Search" "logs/visual_search.pid"

# Also kill any streamlit processes on the ports
echo ""
echo "Checking for remaining processes on ports..."
for port in 8501 8502 8503 8504; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "  Killing process on port $port..."
        kill $(lsof -t -i:$port) 2>/dev/null || true
    fi
done

echo ""
echo "✅ All applications stopped!"
echo ""
