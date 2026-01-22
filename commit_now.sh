#!/bin/bash
cd "/Users/liuyuxiang/Desktop/Rework datathon"
git add .
git commit -m "Fix NaN handling in clustering analysis and simplify data processing

- Improve NaN value handling in FAMD preprocessing
- Simplify redundancy pruning logic
- Fix data alignment issues between dataframes
- Update sample count tracking after row drops"

echo "✓ Changes committed successfully!"
git log -1 --stat
