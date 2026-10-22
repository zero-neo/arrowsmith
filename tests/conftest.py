"""Shared pytest configuration."""

import matplotlib

# Draw figures off-screen so tests never open windows.
matplotlib.use("Agg")
