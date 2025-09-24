#!/bin/bash

# Script to start Ray cluster from the head node without SSH to localhost
# Run this on tonkotsu.kahnvex.com

set -e

echo "========================================="
echo "Starting Ray Cluster from Head Node"
echo "========================================="

# Step 1: Start Ray head node locally (no SSH)
echo ""
echo "Step 1: Starting Ray head node locally..."
uv run ray stop || true

# Get the IP address that workers will use to connect
HEAD_IP=tonkotsu.kahnvex.com
echo "  Using head IP: $HEAD_IP"

uv run ray start --head \
    --node-ip-address=$HEAD_IP \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --resources='{"head":1}' \
    --include-dashboard=true

echo "✓ Ray head node started. Dashboard available at http://${HEAD_IP}:8265"

# Step 2: Start worker nodes via SSH
echo ""
echo "Step 2: Starting worker nodes via SSH..."

# Start skylab0 (CPU worker)
echo "  Starting skylab0.kahnvex.com (CPU worker)..."
ssh -i ~/.ssh/skylab_key kahnvex@skylab0.kahnvex.com "
    cd ~/projects/peralta
    source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null || true
    uv run ray stop
    uv run ray start --address=${HEAD_IP}:6379 --resources='{\"num_cpus\":4, \"cpu_worker\": 1}' --num-cpus=4
" && echo "  ✓ skylab0 connected" || echo "  ✗ Failed to start skylab0"

# Start skylab1 (GPU worker)
echo "  Starting skylab1.kahnvex.com (GPU worker)..."
ssh -i ~/.ssh/skylab_key kahnvex@skylab1.kahnvex.com "
    cd ~/projects/peralta
    source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null || true
    uv run ray stop
    uv run ray start --address=${HEAD_IP}:6379 --resources='{\"num_gpus\":1,\"num_cpus\":4, \"gpu_worker\": 1}' --num-gpus=1 --num-cpus=4
" && echo "  ✓ skylab1 connected" || echo "  ✗ Failed to start skylab1"

# Step 3: Check cluster status
echo ""
echo "Step 3: Checking cluster status..."
echo "========================================="
sleep 2  # Give Ray a moment to stabilize
uv run ray status

echo ""
echo "Cluster setup complete!"
echo "Dashboard: http://tonkotsu.kahnvex.com:8265"
echo ""
echo "To stop the cluster, run: ./stop_cluster_from_head.sh"
