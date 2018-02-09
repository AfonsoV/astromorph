# README file for astromorph
## A python library for galaxy morphology

A general purpose library for galaxy morphology research implemented purely in python and containing several widely used methods in astronomical research.

**This library is currently under heavy development and below you can find a list of modules that are planned to be part of this package.**

### Parametric morphology

This package includes a module to run [GALFIT](https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html) using [SExtractor](https://www.astromatic.net/software/sextractor) results as first guesses for batch fitting of a list of galaxies. It requires that both GALFIT and SExtractor are installed on your computer and in your _PATH_.

### Non-parametric morphology

A list of implemented non-parametric statistics:
* CAS - [Conselice (2003)](http://adsabs.harvard.edu/abs/2004AJ....128..163L)
* Gini-M20 - [Lotz et al. (2004)](http://adsabs.harvard.edu/abs/2004AJ....128..163L)
* $\Psi$,I - [Law et al. (2007)](http://adsabs.harvard.edu/abs/2007ApJ...656....1L)
* F - [Matsuda et al. (2011)](http://adsabs.harvard.edu/abs/2011MNRAS.410L..13M)
* MID - [Freeman et al. (2013)](http://adsabs.harvard.edu/abs/2013MNRAS.434..282F)
* T - [Ribeiro et al. (2016)](http://adsabs.harvard.edu/abs/2016A%26A...593A..22R)

### General utility tools

A collection of python functions that are useful and common to many of this quantities, including for visualisation, creation of segmentation masks, aperture photometry, among others.
