import sys
from pathlib import Path

# Add the workspace root so the ``tools`` package resolves as ``tools.*``.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
