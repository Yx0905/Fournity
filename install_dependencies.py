#!/usr/bin/env python3
"""
Quick script to install required dependencies
"""
import subprocess
import sys

def install_packages():
    packages = ['pandas>=2.0.0', 'openpyxl>=3.0.0', 'numpy>=1.24.0']
    
    print("Installing required packages...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\nAll packages installed successfully!")
    print("You can now run: python3 process_champions_data.py")

if __name__ == "__main__":
    install_packages()
