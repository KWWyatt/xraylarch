#!/usr/bin/env python
"""
Setup script to configure imports for xraylarch development.
Run this script or import it to set up the correct Python path.
"""

import sys
import os
from pathlib import Path

def setup_xraylarch_imports():
    """Add the xraylarch repository root to Python path."""
    # Get the directory where this script is located (repository root)
    repo_root = Path(__file__).parent.absolute()
    
    # Convert to string for Windows compatibility
    repo_root_str = str(repo_root)
    
    # Add to Python path if not already there
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
        print(f"Added {repo_root_str} to Python path")
    else:
        print(f"{repo_root_str} already in Python path")
    
    return repo_root_str

def test_imports():
    """Test that larch imports work correctly."""
    try:
        from larch.xafs import pre_edge, energy_align
        from larch.io import read_xdi, read_athena, is_athena_project
        print("✓ Successfully imported larch functions")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

if __name__ == "__main__":
    setup_xraylarch_imports()
    test_imports()
