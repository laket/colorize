#!/bin/bash

set -x

sphinx-apidoc -F src/ -o docs/
cd docs
# edit conf.py
make html
