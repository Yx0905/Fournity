#!/usr/bin/env python3
"""
Script to commit changes to git repository
"""
import subprocess
import os
import sys

def run_command(cmd, cwd=None):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    repo_path = "/Users/liuyuxiang/Desktop/Rework datathon"
    os.chdir(repo_path)
    
    print("="*60)
    print("Git Commit Script")
    print("="*60)
    
    # Check if git repo exists
    success, stdout, stderr = run_command("git status", cwd=repo_path)
    if not success:
        print("Git repository not found. Initializing...")
        success, stdout, stderr = run_command("git init", cwd=repo_path)
        if not success:
            print(f"Error initializing git: {stderr}")
            return
        print("Git repository initialized.")
    
    # Check status
    print("\nCurrent git status:")
    success, stdout, stderr = run_command("git status --short", cwd=repo_path)
    if success:
        if stdout.strip():
            print(stdout)
        else:
            print("No changes to commit.")
            return
    else:
        print(stderr)
    
    # Add all files
    print("\nStaging all files...")
    success, stdout, stderr = run_command("git add .", cwd=repo_path)
    if not success:
        print(f"Error staging files: {stderr}")
        return
    print("Files staged successfully.")
    
    # Commit
    commit_message = """Add comprehensive clustering analysis workflow

- Implement 4-phase Latent-Sparse Clustering analysis
- Add data processing script with filtering and metric calculation
- Include clustering analysis with FAMD/PCA, K-Means optimization
- Add feature importance extraction and cluster profiling
- Generate visualizations (UMAP, heatmaps, feature importance)
- Create comprehensive summary report generation
- Add documentation and requirements files"""
    
    print("\nCommitting changes...")
    # Escape the commit message properly
    success, stdout, stderr = run_command(
        f'git commit -m "Add comprehensive clustering analysis workflow" -m "- Implement 4-phase Latent-Sparse Clustering analysis" -m "- Add data processing script with filtering and metric calculation" -m "- Include clustering analysis with FAMD/PCA, K-Means optimization" -m "- Add feature importance extraction and cluster profiling" -m "- Generate visualizations (UMAP, heatmaps, feature importance)" -m "- Create comprehensive summary report generation" -m "- Add documentation and requirements files"',
        cwd=repo_path
    )
    
    if success:
        print("\n✓ Changes committed successfully!")
        print("\nCommit details:")
        success, stdout, stderr = run_command("git log -1 --stat", cwd=repo_path)
        if success:
            print(stdout)
    else:
        print(f"\n✗ Error committing: {stderr}")
        if "nothing to commit" in stderr.lower():
            print("No changes to commit.")
        return
    
    print("\n" + "="*60)
    print("To push to remote (if configured):")
    print("  git push origin main")
    print("\nTo see commit history:")
    print("  git log --oneline")

if __name__ == "__main__":
    main()
