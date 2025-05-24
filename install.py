#!/usr/bin/env python3
"""
Installation script for Interactive Chat QnA with Direct Pinecone API
This script helps set up the environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10 or higher is recommended!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        print("⚠️  This application was tested with Python 3.10.16")
        choice = input("Continue anyway? [y/N]: ").lower().strip()
        if choice != 'y':
            return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible!")
    return True

def create_secrets_template():
    """Create secrets template if it doesn't exist"""
    secrets_dir = Path(".streamlit")
    secrets_file = secrets_dir / "secrets.toml"
    template_file = secrets_dir / "secrets.toml.template"

    if not secrets_file.exists() and template_file.exists():
        print("\n🔄 Creating secrets.toml from template...")
        try:
            secrets_dir.mkdir(exist_ok=True)
            with open(template_file, 'r') as src, open(secrets_file, 'w') as dst:
                content = src.read()
                dst.write(content)
            print("✅ Created .streamlit/secrets.toml")
            print("⚠️  Please edit .streamlit/secrets.toml with your actual API keys!")
            return True
        except Exception as e:
            print(f"❌ Failed to create secrets.toml: {e}")
            return False
    elif secrets_file.exists():
        print("✅ secrets.toml already exists")
        return True
    else:
        print("⚠️  No secrets template found")
        return True

def main():
    """Main installation process"""
    print("🚀 Interactive Chat QnA Installation Script")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Upgrade pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        print("⚠️  Pip upgrade failed, continuing anyway...")

    # Install requirements
    requirements_files = ["requirements.txt", "requirements-minimal.txt"]
    requirements_file = None

    for req_file in requirements_files:
        if Path(req_file).exists():
            requirements_file = req_file
            break

    if requirements_file:
        print(f"\n📦 Found {requirements_file}")
        choice = input("Install (f)ull requirements or (m)inimal? [f/m]: ").lower().strip()

        if choice == 'm' and Path("requirements-minimal.txt").exists():
            req_file = "requirements-minimal.txt"
        else:
            req_file = "requirements.txt"

        if not run_command(f"{sys.executable} -m pip install -r {req_file}", f"Installing dependencies from {req_file}"):
            print("❌ Installation failed!")
            sys.exit(1)
    else:
        print("❌ No requirements.txt file found!")
        sys.exit(1)

    # Create secrets template
    create_secrets_template()

    print("\n🎉 Installation completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .streamlit/secrets.toml with your API keys")
    print("2. Run: streamlit run pinecone_direct_api.py")
    print("\n🔑 Get your API keys from:")
    print("   - Pinecone: https://www.pinecone.io/")
    print("   - Google Gemini: https://makersuite.google.com/")
    print("   - 11 Labs: https://elevenlabs.io/")

if __name__ == "__main__":
    main()
