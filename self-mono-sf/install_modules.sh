#!/bin/bash
## added to find python3-venv
source ../venv/bin/activate

cd ./models/correlation_package
python setup.py install
cd ../forwardwarp_package
python setup.py install
cd ../..
