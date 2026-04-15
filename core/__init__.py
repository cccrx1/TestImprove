"""Top-level package for attacks, defenses, and models.

Keep the package import lightweight so users can work on one submodule
without forcing every optional dependency to load at import time.
"""

__all__ = ['attacks', 'defenses', 'models', 'utils']
