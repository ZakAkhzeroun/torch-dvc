#!/usr/bin/env bash
set -euo pipefail

# Installs the runtime dependencies needed by this project.
# Default target is the `opendvc` conda environment.

ENV_NAME="${1:-opendvc}"
PYTHON_VERSION="${PYTHON_VERSION:-3.6}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/.cache}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$REPO_ROOT/.conda/pkgs}"

mkdir -p "$CACHE_ROOT" "$CONDA_PKGS_DIRS"
export XDG_CACHE_HOME="$CACHE_ROOT"
export CONDA_PKGS_DIRS

find_conda_sh() {
  local candidates=(
    "/home/alizakaria/miniconda3/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "$HOME/mambaforge/etc/profile.d/conda.sh"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -f "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

if CONDA_SH="$(find_conda_sh)"; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"

  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating conda env '$ENV_NAME' with Python ${PYTHON_VERSION}..."
    conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
  fi

  echo "Activating conda env '$ENV_NAME'..."
  conda activate "$ENV_NAME"

  REQUESTED_PYTHON_MM="$(echo "$PYTHON_VERSION" | awk -F. '{print $1"."$2}')"
  ACTIVE_PYTHON_VERSION="$(python -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))')"

  if [[ "$ACTIVE_PYTHON_VERSION" != "$REQUESTED_PYTHON_MM" ]]; then
    echo "Updating Python in '$ENV_NAME' to ${PYTHON_VERSION}..."
    conda install -y "python=${PYTHON_VERSION}"
    ACTIVE_PYTHON_VERSION="$(python -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))')"
    if [[ "$ACTIVE_PYTHON_VERSION" != "$REQUESTED_PYTHON_MM" ]]; then
      echo "ERROR: Active Python is $ACTIVE_PYTHON_VERSION but expected ${PYTHON_VERSION}."
      exit 1
    fi
  else
    echo "Python version already compatible: ${ACTIVE_PYTHON_VERSION}"
  fi

  echo "Installing dependencies in active env '$ENV_NAME'..."
  conda install -y \
    "numpy==1.19.5" \
    "scipy==1.1.0" \
    "pillow==8.4.0" \
    "tensorflow==1.12.0"

  conda install -y -c pytorch \
    "pytorch==1.10.2" "cpuonly"

  echo "Installation complete."
  echo "Active environment: $CONDA_DEFAULT_ENV"
  python -V
  if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Note: this script was executed, so activation does not persist in your parent shell."
    echo "To keep it active after installation run:"
    echo "source \"$CONDA_SH\" && conda activate \"$ENV_NAME\""
  fi
else
  echo "Conda not found. Falling back to pip in current Python environment."
  echo "Note: tensorflow==1.12.0 generally requires Python 3.6."
  python -m pip install --upgrade pip
  python -m pip install -r "$REPO_ROOT/requirements.txt"
fi
