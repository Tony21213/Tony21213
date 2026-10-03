"""Точка входа собранного приложения (PyInstaller)."""
import sys

from casedesigner.app.main import main

if __name__ == "__main__":
    sys.exit(main())
