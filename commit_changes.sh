#!/bin/bash
# Script to commit changes to git repository

cd "/Users/liuyuxiang/Desktop/Rework datathon"

# Check if git repo exists, if not initialize
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Add all files
echo "Staging files..."
git add .

# Commit with descriptive message
echo "Committing changes..."
git commit -m "Add comprehensive clustering analysis workflow

- Implement 4-phase Latent-Sparse Clustering analysis
- Add data processing script with filtering and metric calculation
- Include clustering analysis with FAMD/PCA, K-Means optimization
- Add feature importance extraction and cluster profiling
- Generate visualizations (UMAP, heatmaps, feature importance)
- Create comprehensive summary report generation
- Add documentation and requirements files"

echo "Changes committed successfully!"
echo ""
echo "To push to remote (if configured):"
echo "  git push origin main"
echo ""
echo "To see commit history:"
echo "  git log --oneline"
