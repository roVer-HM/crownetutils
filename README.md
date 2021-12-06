# roveranalyzer

## System setup
We strongly recommend to use a virtual environment
```
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip3 install -r requirements.txt
```
Note: `python3.8 setup.py install` is not working, because requirements.txt contains URLs.

## Dev

### setup pre-comit hooks

install [pre-commit](https://pre-commit.com/#config-args) if missing.

```
pip3 install pre-commit
```

install in git pre-commit hook with

```
pre-commit install
```

To run all pre-commit hook for all files use

```
pre-commit run --all-files
```

Hint: if run twice the second run should show all green.

Configuration with following files:

- `.pre-commit-config.yaml` (all installed hooks)
- `.isort.cfg` (import sorting configuration)



## vadereanalyzer

...

## oppanalyzer

Small collection of tools to access `*.vec` and
`*.sca` files produced by OMNeT++. The *oppanalyzer*
wraps the `scavetool` provided by OMNeT++ to
access, filter and combine results from OMNeT++ simulation
campaigns.

### Usage

```python
from oppanalyzer import *

cfg = Config()
scavetool = ScaveTool(cfg)

scavetool.load_csv('some.csv')

```

The default configuration `Config()` will use
containrized a version of the scavetool, which means
the `$CROWNET_HOME` variable must be set to the
root of this repository. See [tutorials/README](tutorials/README.md)
for configuration setup if you dont use the containers.

### ToDo's

- [ ] Define standard transformation as helpers.
- [ ] Read Vadere output
- [ ] Integrate [suq-controler](https://gitlab.lrz.de/vadere/suq-controller)
