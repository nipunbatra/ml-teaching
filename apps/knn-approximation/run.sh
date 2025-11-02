#!/bin/bash
# Launch the KNN Approximation app

echo "🔍 KNN Approximation Apps"
echo ""

# Check if argument provided
if [ "$1" == "stepbystep" ]; then
    echo "🎬 Starting Step-by-Step Algorithm Walkthrough..."
    echo "   Best for: Teaching algorithm mechanics in detail"
    echo ""
    streamlit run app_stepbystep.py
elif [ "$1" == "comparison" ]; then
    echo "📊 Starting Comparison & Performance Analysis..."
    echo "   Best for: Understanding trade-offs and performance"
    echo ""
    streamlit run app.py
else
    echo "Usage: ./run.sh [stepbystep|comparison]"
    echo ""
    echo "Available apps:"
    echo "  stepbystep  - Step-by-step algorithm walkthrough (recommended for teaching)"
    echo "  comparison  - Compare all methods with performance metrics"
    echo ""
    echo "Example: ./run.sh stepbystep"
    echo ""
    echo "Install dependencies first:"
    echo "  pip install -r requirements.txt"
fi
