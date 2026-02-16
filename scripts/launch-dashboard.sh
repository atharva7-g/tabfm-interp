#!/bin/bash
# Launch the TabPFN Patching Dashboard

cd "$(dirname "$0")/.." || exit
echo "Starting TabPFN Patching Dashboard..."
echo "Open your browser to: http://localhost:8501"
echo ""
uv run streamlit run src/dashboard/app.py "$@"
