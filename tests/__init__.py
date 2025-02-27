import pathlib
import sys

# Add the root path to the PYTHON PATH
root_path = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))