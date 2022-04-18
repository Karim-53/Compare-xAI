todo update before sub
todo cite the original repository
add binder tag
python version
nb of XAI nb of tests 

<p align="center"><img src="img/banner.svg" width=700 /></p>

`Comapre-xAI` is a library for benchmarking feature attribution techniques using synthetic data. 

See our NeurIPS paper at https://arxiv.org/ todo.

<p align="center"><img src="img/overview_figure.svg" width=700 /></p>

## Installation

To use our package, make sure you install all the dependencies in `requirements.txt` using

```
pip install -r requirements.txt
```

## Sample Usage

We use an `Experiment` to benchmark various datasets, models, explainers, metrics. This is the recommended way to access
our library.

For running several experiments across multiple datasets, use a script as shown in,

```
python reset_experiment.py
```

Each Experiment is saved after execution for checkpointing. This way, additional experiments can be run without
having to rerun previous computation.

---

## More details

## Reference
The source code was inspired from https://github.com/abacusai/xai-bench and https://github.com/mtsang/archipelago
## Citation

Please cite our work if you use code from this repo:

```bibtex
 
```
