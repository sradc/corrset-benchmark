# To run the benchmark

- First generate `data-large.json`, as described in the project root README.md
    - Move this to `python_optimization/data/large.json`
- Then `cd python_optimization`
- `poetry install`
- `make run_benchmark`

Note the benchmark should be converted to a Python script before running, which gives better performance than when it's in a Jupyter Notebook.
