#!/bin/bash

# MRLM End-to-End Demo Runner
# This script runs the complete demo in one command

set -e  # Exit on error

echo "======================================================================="
echo "MRLM: End-to-End Demo Runner"
echo "======================================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "demo_complete.py" ]; then
    echo "Error: Please run this script from the examples/demo_e2e directory"
    exit 1
fi

# Step 1: Generate synthetic data
echo "[Step 1/3] Generating synthetic data..."
echo "-----------------------------------------------------------------------"
python synthetic_data_generator.py
echo ""

# Step 2: Validate setup
echo "[Step 2/3] Validating setup..."
echo "-----------------------------------------------------------------------"
python test_demo.py
echo ""

# Step 3: Run demo
echo "[Step 3/3] Running complete demo..."
echo "-----------------------------------------------------------------------"
python demo_complete.py

echo ""
echo "======================================================================="
echo "Demo completed successfully!"
echo "======================================================================="
echo ""
echo "Files created:"
echo "  - data/math_problems.json"
echo "  - data/code_problems.json"
echo "  - data/debate_topics.json"
echo "  - data/tool_scenarios.json"
echo ""
echo "Next steps:"
echo "  • Review README.md for customization options"
echo "  • Try larger models: Qwen/Qwen2.5-1.5B"
echo "  • Experiment with more training epochs"
echo "  • Explore other environments (code, debate, tools)"
echo ""
