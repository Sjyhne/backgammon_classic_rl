if [[ -d "./env" ]]; then
    echo "The virtualenvironment exists, and is named 'env'"
else
    echo "Virtualenvironment named 'env' does not exist, creating one"
    virtualenv env || echo "virtualenv not installed, can be installed by 'sudo apt install python-virtualenv'"
    echo "Created virtualenv, now sourcing" && exit 1
    source "./env/bin/activate"
    echo "virtualenv is now sourced, will now clone backgammon gym if not already cloned"
fi

if [[-d "./reduced_backgammon_gym"]]; then
    echo "reduced_backgammon_gym already cloned"
else
    git clone "https://github.com/sjyhne/reduced_backgammon_gym.git"
    echo "Successfully cloned the backgammon gym, will now install environment"
    cd "./reduced_backgammon_gym"
    pip install -e "."
fi

echo "Everything should be installed properly"
