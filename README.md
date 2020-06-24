# ICML-boomerang

This repository contains all code required to generate (random variations of) the graphics in the paper *The Boomerang Sampler* by Joris Bierkens, Sebastiano Grazzi, Kengo Kamatani and Gareth Robers, ICML 2020.

The folder `julia` contains the Julia code to carry out the experiments of Sections 4.1 and 4.3.
The folder `R` contains R scripts to subsequently generate the corresponding graphics.

The folder `BoomerangDiffusionBridge` contains Julia code to carry out the diffusion bridge simulations of Section 4.2.
This code is based for a large extent upon code in the folder ZZDiffusionBridge from the Public directory  [SebaGraz/ZZDiffusionBridge](https://github.com/SebaGraz/ZZDiffusionBridge) (see pre-print [A piecewise deterministic Monte Carlo method for diffusion bridges](https://arxiv.org/abs/2001.05889)). We build up from that implementation and add the Boomerang sampler for the problem of sampling diffusion bridges.
