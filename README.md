# fkguidance

Post-training Feynman–Kac guidance for stochastic generative models. Supply a terminal potential, generated endpoints, forward noising, and continuation callbacks.

Initialize the official CTSM-v dependency before using `ConditionalPathPotential`:

```bash
git submodule update --init vendor/dre_prob_paths
```
