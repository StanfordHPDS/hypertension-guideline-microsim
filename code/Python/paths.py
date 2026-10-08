import os

# Resolve the repository root by walking up from this file two directories
# (code/Python/paths.py -> code/Python -> code -> repo root). Scripts depend
# on this constant to locate sibling data_and_models/ and results/ trees.
current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)
overall_folder = os.path.dirname(parent_directory)
