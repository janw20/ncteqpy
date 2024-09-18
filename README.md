# nCTEQpy

nCTEQpy is the Python analysis framework for nCTEQ Parton Distribution Function fits.

## Installation

### From PyPI or conda-forge (WIP)

Install with pip:
```sh
pip install ncteqpy
```
or conda:
```sh
conda install -c conda-forge ncteqpy
```

### From source

The source is available at http://github.com/janw20/ncteqpy. Clone the repository:
```sh
git clone https://github.com/janw20/ncteqpy.git
```
nCTEQpy is built with [poetry](https://python-poetry.org/). Install poetry e.g. with [pipx](https://pipx.pypa.io):
```sh
pipx install poetry
```
Then navigate to the directory where you cloned this repository:
```sh
cd path/to/ncteqpy
```
and install:
```sh
poetry install
```
Now you can import the `ncteqpy` module in Python:
```python
import ncteqpy as nc
```
