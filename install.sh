#!/bin/bash

project_name

echo "Creating Virtual Env"
virtualenv --python=/usr/bin/python3 env

echo "Activating Virtual Env"
source ./env/bin/activate

echo "Installing Requirements"
pip install -r requirements.txt

ipython kernel install --user --name=nlptorch
