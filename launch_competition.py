#!/usr/bin/env python3
"""
Launch script for the Code Competition Arena

This script starts the Streamlit application with the competitive programming system.
"""

import os
import sys
import subprocess

def main():
    """Launch the competition system"""
    print("🏆 Starting Code Competition Arena...")
    print("=" * 50)
    print("Features:")
    print("- 🐛 Bug Detection with AI Agents")
    print("- 🎯 Edge Case Analysis")
    print("- ⚡ Performance Optimization")
    print("- 🔒 Security Assessment")
    print("- 🏆 Live Leaderboard")
    print("- 📊 Real-time Analytics")
    print("=" * 50)
    
    # Check if required dependencies are installed
    try:
        import streamlit
        import plotly
        import pandas
        print("✅ Dependencies verified")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return 1
    
    # Check for environment file
    if not os.path.exists('.env'):
        print("⚠️  No .env file found. Please create one with your CEREBRAS_API_KEY")
        print("Example:")
        print("CEREBRAS_API_KEY=your_api_key_here")
        print()
    
    # Launch Streamlit
    try:
        print("🚀 Launching application...")
        subprocess.run([
            sys.executable, 
            "-m", 
            "streamlit", 
            "run", 
            "app.py",
            "--server.headless", 
            "true"
        ], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to start Streamlit")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Competition ended. Thanks for participating!")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 