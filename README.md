Bandits in the Loop
===================

[![PyPI version](https://badge.fury.io/py/bitl.svg)](https://badge.fury.io/py/bitl)
[![Build Status](https://travis-ci.org/tritas/bitl.svg?branch=master)](https://travis-ci.org/tritas/bitl)
[![Documentation Status](https://readthedocs.org/projects/bandits-in-the-loop/badge/?version=latest)](http://bandits-in-the-loop.readthedocs.io/en/latest/?badge=latest)

This module provides tools to design experiments and benchmark multi-armed bandits policies.
The idea is to be able to easily evaluate a new algorithm on multiple problems.
Either generate a simulated dataset and evaluate online, or load a real dataset and evaluate offline.

The idea is to have the following workflow:
``` 
Choose a policy/policies, initialize them

Load dataset or generate synthetic samples

Initialize environment and benchmark parameters (n_runs, results_folder, ...)

Run simulation and get report as plots, comparative HTML&LaTeX table
```

## Project directory structure
* `environment`: defines global parameters, features, feedback graphs etc.
* `policies`: examples policies for different classes of bandits
* `datasets`: fetch, load, generate data
* `evaluation`: setup and run evaluation and comparison loops
* `utils`: helper functions to generate plots and output formatted results (HTML, TeX)

## Installation

### From `pip`
```
pip install bitl
```
### From source

Download the source
```
git clone https://github.com/tritas/bitl.git
```

Install some dependencies
``` 
pip -r install requirements.txt
```

Install the package
```
python setup.py install
or
python setup.py develop (--user)
```

## Examples


The `examples` folder will showcase how to use the different functionality.