# Usage:  source activate.sh
# Puts this project's own Python first and isolates it from personal settings.
_EO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset PYTHONPATH                                  # personal helper folders must not leak in
export PYTHONNOUSERSITE=1                         # ignore ~/.local packages too
export PATH="$_EO_ROOT/.venv/bin:$PATH"           # project Python + tools come first
export PIP_CACHE_DIR="$_EO_ROOT/.cache/pip"       # caches stay inside the project
export CONDA_PKGS_DIRS="$_EO_ROOT/.cache/conda_pkgs"
export EO_DATA_CACHE="$_EO_ROOT/data_cache"
export MPLCONFIGDIR="$_EO_ROOT/.cache/matplotlib"     # matplotlib needs a writable cache
export XDG_CACHE_HOME="$_EO_ROOT/.cache"            # fontconfig and other tools
export HF_HOME="$_EO_ROOT/data_cache/hf"             # model weights (home is read-only; keep inside project)
echo "eo-agent env active: $(python --version) at $(command -v python)"
