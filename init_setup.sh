echo [$(date)]: "START"
echo [$(date)]: "Creating conda env with python 3.8"
conda create --prefix ./venv python=3.8 -y
echo [$(date)]: "Activating the environment"
source activate ./venv/bin/activate
echo [$(date)]: "Installing required packages"
pip install -r requirements.txt
echo [$(date)]: "END"