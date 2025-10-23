#!/usr/bin/env python3
"""
WSL-based evaluation script with PESQ support
Run this from Windows: wsl python3 evaluate_wsl.py --enhanced output.wav --clean clean.wav
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run the regular evaluation
from evaluate import *

if __name__ == '__main__':
    main()
