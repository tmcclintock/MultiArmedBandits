# MultiArmedBandits [![Build Status](https://travis-ci.com/tmcclintock/MultiArmedBandits.svg?branch=master)](https://travis-ci.com/tmcclintock/MultiArmedBandits)[![Coverage Status](https://coveralls.io/repos/github/tmcclintock/MultiArmedBandits/badge.svg?branch=master&service=github)](https://coveralls.io/github/tmcclintock/MultiArmedBandits?branch=master&service=github)

A package to build and test bandit algorithms.

Bandit algorithms learn how to balance exploitation with exploration when attempting to maximize rewards obtained from a single-state (but possibly non-stationary) environment. This repository is for building and experimenting with various bandit algorithms.

## Installation

Install by running these two commands to install the requirements and this package
```bash
pip install -r requirements.txt
pip install .
```
If you use `conda` then you can create the environment for this package (called `mab`) from the `environment.yml` file with
```bash
conda env create -f environment.yml
```
and then install with
```bash
pip install .
```
Once installed, test your installation by running
```bash
pytest
```
by default you should have all passing tests and some skipped tests.