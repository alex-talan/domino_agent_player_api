import sys
from pathlib import Path

# Add the workspace root so sibling packages resolve as ``agents.*`` and
# ``tools.*``. Provider modules (``llm_providers`` / ``prompt_providers``) are
# referenced only under ``TYPE_CHECKING`` and are duck-typed at runtime, so
# they are not required on the path for the tests.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
