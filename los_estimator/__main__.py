"""Entry point for running los_estimator as a module."""

# Add the current folder to the Python path
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cli import main

if __name__ == "__main__":
    main()
