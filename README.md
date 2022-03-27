# Scalable Gaussian Processes @ GPRV

This repository contains the materials for my March 28, 2022 "Scalable Gaussian
Processes" tutorial at the GPRV workshop in Oxford, UK.

The file `ajac5176t3_mrt.txt` is the machine-readable version of Table 3 from
[Zhao et al.
(2022)](https://iopscience.iop.org/article/10.3847/1538-3881/ac5176/meta), and
it should not be used without the appropriate citation.

The notebooks can be executed locally following the instructions below, or they
can be run on Google Colab, if you can't or don't want to set up the local
environment.

## Local environment

If you want to run these notebooks locally, you'll need to set up a Python
environment with the usual scientific stack (`numpy`, `scipy`, and `matplotlib`)
installed, as well as a Jupyter client. Besides these standard dependencies,
you'll also need `jax`, `jaxopt`, and `tinygp` installed. I released a new
version of `tinygp` the day before this workshop, and I think you'll need at
least that version installed.

For a CPU-only build, the best way to get these non-standard dependencies
installed is with `pip`:

```bash
python -m pip instal -U "jax[cpu]" jaxopt tinygp
```

If you want to install a GPU-accelerated version of JAX, follow the instructions
in the [JAX README](https://github.com/google/jax).

For the `real-data.ipynb` notebook, you'll also need to have `astropy`
installed, and that can also be installed using `conda` or `pip`.

## The notebooks

A good place to start is the `intro-to-jax.ipynb` notebook which includes a very
brief introduction to the `jax` library which is the core dependency of
