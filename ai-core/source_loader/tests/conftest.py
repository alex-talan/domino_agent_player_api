import sys
from pathlib import Path

module_root = Path(__file__).parent.parent
workspace_root = module_root.parent

sys.path.insert(0, str(module_root))
sys.path.insert(0, str(workspace_root))
