"""
SSL fix for Windows Python 3.14 — injects system certificate store.
On Linux (Streamlit Cloud) this is a no-op.
"""
import sys

def apply() -> None:
    if sys.platform != "win32":
        return
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
