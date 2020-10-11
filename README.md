
/* ![SymJAX logo](./docs/img/symjax_logo.png)
*/

# All dataset utilities (downloading/loading/batching/processing) in Numpy ![Continuous integration](https://github.com/RandallBalestriero/numpy-datasets/workflows/Continuous%20integration/badge.svg) ![license](https://img.shields.io/badge/license-Apache%202-blue) <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
This is an under-development research project, not an official product, expect bugs and sharp edges; please help by trying it out, reporting bugs.
[**Reference docs**](https://numpy-datasets.readthedocs.io/en/latest/)


## What is and why doing numpy-datasets ?

* First, numpy-datasets offers out-of-the-box dataset download and loading only based on Numpy and core Python libraries.
* Second, numpy-datasets offers utilities such as (mini-)batching a.k.a looping through a dataset one chunk at a time, or preprocessing techniques that are highly suited for machine learning and deep learning pipelines.
* Third, numpy-datasets offers many options to transparently deal with very large datasets. For example, automatic mini-batching with a priori caching of the next batch, online preprocessing, and the likes.
* Fourth, numpy-datasets does not only focus on computer vision datasets but also offers plenty in time-series datasets, with a constantly groing collection of implemented datasets.

## Minimal Example

```python
import sys
import symjax as sj
import symjax.tensor as T

# create our variable to be optimized
mu = T.Variable(T.random.normal((), seed=1))

# create our cost
cost = T.exp(-(mu-1)**2)

# get the gradient, notice that it is itself a tensor that can then
# be manipulated as well
g = sj.gradients(cost, mu)
print(g)

# (Tensor: shape=(), dtype=float32)

# create the compield function that will compute the cost and apply
# the update onto the variable
f = sj.function(outputs=cost, updates={mu:mu-0.2*g})

for i in range(10):
    print(f())

# 0.008471076
# 0.008201109
# 0.007946267
# ...
```

## Installation

Installation is direct with pip as described in this [**guide**](https://numpy-datasets.readthedocs.io/en/latest/user/installation.html).
