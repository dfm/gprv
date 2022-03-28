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

If you want to install a GPU-accelerated version of `jax`, follow the instructions
in the [jax README](https://github.com/google/jax).

For the `real-data.ipynb` notebook, you'll also need to have `astropy`
installed, and that can also be installed using `conda` or `pip`.

Once you have your environment set up, you can clone this repository

```bash
git clone https://github.com/dfm/gprv.git
```

and open it in your favorite Jupyter environment.

## The notebooks

At the workshop I will live code the first two notebooks, but this repository
includes cleaned up versions of where we will (hopefully!) end up, as well as
some extra explanations and suggested extensions.

1. A good place to start is the `intro-to-jax.ipynb` notebook which includes a
   very brief introduction to the `jax` library which is the main dependency of
   `tinygp`. You'll mostly use `jax` a lot like `numpy`, but there are some
   fundamental programming concepts that will be useful to know. You can [open this
   notebook in Google Colab][intro-to-jax].

2. The next notebook is `intro-to-tinygp.ipynb`, where I go through a simple
   example use case for `tinygp` applied to simulated data. This includes some
   suggested exercises and extensions near the end. You can [open this notebook
   in Google Colab][intro-to-tinygp].

3. The last notebook shows an example of a `tinygp` model fit to real data.
   *Disclaimer:* this particular notebook is by no means meant as a suggestion
   for how to actually use `tinygp` for RV data analysis, that's a discussion
   for the rest of the workshop! You can [open this notebook
   in Google Colab][real-data].

[intro-to-jax]: https://colab.research.google.com/github/dfm/gprv/blob/main/intro-to-jax.ipynb
[intro-to-tinygp]: https://colab.research.google.com/github/dfm/gprv/blob/main/intro-to-tinygp.ipynb
[real-data]: https://colab.research.google.com/github/dfm/gprv/blob/main/real-data.ipynb
