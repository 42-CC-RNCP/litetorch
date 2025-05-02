# tests/conftest.py
import sys
import os
import pytest

# This file is used to configure pytest and set up the test environment.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
