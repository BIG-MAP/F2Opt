# F2Opt

Optimizer applications for the FINALES2 materials acceleration platform (MAP).


## Optimizers

### src/ocond.py

* Optimise conductivity as a function of electrolyte formulation.
* Data is sourced from experiment and simulation.
* The surrogate model is a multi-source single-objective Gaussian process regression model.

Run `$ python -m src.ocond --help` to see required arguments.


## Requirements

Developed with Python 3.10.10.
Requirements are specified in `requirements.txt`.


## Tests

Run tests with `$ python -m unittest`.
