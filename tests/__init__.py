"""
Test initialization file for the tests package.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path for testing
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
