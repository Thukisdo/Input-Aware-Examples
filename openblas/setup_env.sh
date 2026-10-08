#!/usr/bin/env bash
# You should source this script `source ./setup_env.sh` to setup the environment for these experiments

VENV_PATH="venv"

if [[ ! -d "$VENV_PATH" || ! -f "$VENV_PATH/bin/activate" ]];
then

    # First, test if python3 is available
    if ! command -v python3 &> /dev/null
    then
        echo "python3 could not be found. Please install python3 and try again."
        echo "Fedora: sudo dnf install python3"
        echo "Ubuntu: sudo apt install python3"
        exit 1
    fi

    # Test if venv module is available
    if ! python3 -m venv --help &> /dev/null
    then
        echo "python3 venv module is not available. Please install python3-venv and try again."
        echo "Fedora: sudo dnf install python3-virtualenv"
        echo "Ubuntu: sudo apt install python3-venv"
        exit 1
    fi

    echo "== Setting up virtual environment in $VENV_PATH"
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"

    echo "== Installing required packages"
    pip install pandas numpy matplotlib seaborn tqdm psutil optuna

    echo "== Setup complete. In the future, run source ./setup_env.sh or source $VENV_PATH/bin/activate to activate the virtual environment."
else
    echo "== Sourcing existing virtual environment in $VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi
