"""Backward-compatible Streamlit entry point.

The primary application file is now app.py.
This wrapper is kept so older commands using `streamlit run kualitas_susu.py`
still work without breaking.
"""
from app import main

if __name__ == "__main__":
    main()
