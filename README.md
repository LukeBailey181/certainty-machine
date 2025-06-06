# Certainty Machine

Automated theorem proving with AI


# Luke TODOs


-[x] Setup verification server on Stanford cluster
-[x] Setup loading data scripts
-[] Setup multiturn prompting


## Testing

Testing files are in `./tests`. To run tests:

```
pytest
```

## Setup

### Setting up client

```
pip install uv
pip install -e .
pre-commit install
```

### Setting up verification server

You need lean4, first pull the matblib module

```
cd mathlib4
git submodule update --init --recursive
```

Now install lean and build mathlib. This should work:

```
# Get ELAN_HOME from first argument, default to $HOME if not provided
ELAN_HOME="${1:-$HOME}"
export ELAN_HOME

# Install lean on unix
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Source elan
source "$ELAN_HOME/.elan/env"

cd lean_ideas/mathlib4
lake build
```

### Developing

To run pre-commit check:

```
pre-commit install
pre-commit run --all-files
```



