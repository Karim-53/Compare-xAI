todo update before sub
todo cite the original repository
add binder tag
python version
nb of XAI nb of tests 
todo write the power capacity of the used computer 
todo check again `requirements.txt`

[![Python Version](https://img.shields.io/badge/python-v3.8.3-blue)]()

[//]: # (<p align="center"><img src="img/banner.svg" width=700 /></p>)

`Comapre-xAI` is a library for benchmarking feature attribution / feature importance Explainable AI techniques using different unit tests. 

See our NeurIPS paper at https://arxiv.org/ todo.

You can check directly the benchmark results at https://karim-53.github.io/cxAI/

[//]: # (<p align="center"><img src="img/overview_figure.svg" width=700 /></p>)

# 1. Download
All data are located in `data/`

The list of fair unit tests is in `data/01_raw/test.csv`.

Info about explainers could be found in `data/03_experiment_output_aggregated/explainer.csv`

Results are in `data/03_experiment_output_aggregated/cross_tab.csv`

The data is also available as one SQLite database file `data/04_sql/database`

# 2. Run experiments (Sample Usage) 
Want to reproduce the results shown in [our paper](arxiv todo) ? Follow these instructions:
## 2.1 Install required packages

[//]: # (There is no specific requirements listed in `requirements.txt` you can run only a few unit tests and a few Explainer with a small set of packages. So just install what is needed on the go :&#41; )
[//]: # (`requirements.txt` contains a good start)
install the required packages using
```
pip install -r requirements.txt
```

## 2.2 Reset experiment results

Run the following command to explain the currently implemented unit-tests using the currently implemented explainers.

```
python reset_experiment.py
```
The results are written in `data/02_experiment_output/results.csv`.
Now run the following command to aggregate results in a more human-readable format.
```
python src/aggregate_data.py
```
This also generate an SQLite database used in https://karim-53.github.io/cxAI/
`data/04_sql/database` aggregate all data: information about unit tests, explainers, papers, and results of all experiments.

**Tip**: Reduce the list explainers by changing `valid_explainers` in `src/explainer.py`. Same for the unit tests, see `src/test.py`.


# 3. Contributing
To add a new Explainer algorithm or a unit test to the benchmark, please follow the instructions below.

## 3.1 Add a new Explainer

.1 Create a python script `explainers/my_explainer.py`.

.2 Create `MyExplainer` class that inherit from `Explainer` superclass. Have a look at `explainers/saabas.py` to better understand how to implement the explainer. Also do not hesitate to import a library and to add it to `requirements.txt`. 

.3 In `src/explainer.py` add `MyExplainer` to the list of `valid_explainers`.

.4 Run `reset_experiment.py` then run `src/aggregate_data.py`.
src/aggregate_data.py

## 3.2 Add a new Unit test

.1 Create a python script `explainers/my_explainer.py`.

.2 Create `MyTest` class that inherit from `Test` superclass. Have a look at `tests/cough_and_fever.py` to better understand how to implement the unit test. Also do not hesitate to import a library and to add it to `requirements.txt`. 

.3 In `src/test.py` add `MyTest` to the list of `valid_tests`.

.4 Run `reset_experiment.py` then run `src/aggregate_data.py`.


---

## More details

## Reference
The source code was inspired from https://github.com/abacusai/xai-bench and https://github.com/mtsang/archipelago
## Citation

Please cite our work if you use code from this repo:

```bibtex
 
```
