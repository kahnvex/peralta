#!/bin/bash

# Script to stop Ray cluster from the head node
# Run this on tonkotsu.kahnvex.com

echo "========================================="
echo "Stopping Ray Cluster"
echo "========================================="

# Stop worker nodes via SSH
echo ""
echo "Stopping worker nodes..."

echo "  Stopping skylab0.kahnvex.com..."
ssh -i ~/.ssh/skylab_key kahnvex@skylab0.kahnvex.com "source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null || true; uv run ray stop 2>/dev/null || ray stop 2>/dev/null" && echo "  ✓ skylab0 stopped" || echo "  ✗ skylab0 might already be stopped"

echo "  Stopping skylab1.kahnvex.com..."
ssh -i ~/.ssh/skylab_key kahnvex@skylab1.kahnvex.com "source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null || true; uv run ray stop 2>/dev/null || ray stop 2>/dev/null" && echo "  ✓ skylab1 stopped" || echo "  ✗ skylab1 might already be stopped"

# Stop head node locally
echo ""
echo "Stopping Ray head node locally..."
uv run ray stop

echo ""
echo "✓ Ray cluster stopped"
