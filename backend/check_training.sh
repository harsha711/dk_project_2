#!/bin/bash

# Check training status and display results

echo "📊 YOLO Training Status Check"
echo "=============================="
echo ""

# Check if training is running
if pgrep -f "train_yolo_dental.py" > /dev/null; then
    echo "🟢 Training is currently running"
    ps aux | grep "train_yolo_dental.py" | grep -v grep
else
    echo "⚪ No training process found"
fi

echo ""

# Check for results
if [ -d "runs" ]; then
    echo "📁 Training Results:"
    find runs -name "results.png" -o -name "best.pt" | head -5
    echo ""
    
    # Show latest training run
    LATEST_RUN=$(find runs -type d -name "train*" | sort -r | head -1)
    if [ -n "$LATEST_RUN" ]; then
        echo "📈 Latest Training Run: $LATEST_RUN"
        if [ -f "$LATEST_RUN/results.csv" ]; then
            echo ""
            echo "Last 5 epochs:"
            tail -5 "$LATEST_RUN/results.csv" | column -t -s,
        fi
    fi
else
    echo "⚠️  No training results found in runs/ directory"
fi

echo ""
echo "💾 Model Files:"
ls -lh models/*.pt 2>/dev/null || echo "   No model files found in models/"

