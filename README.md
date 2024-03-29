# Density Functional Theory and the Fractional Quantum Hall Effect
Based on the work of Jain el. [1]

## Abstract

Fractional Quantum Hall Effect (FQHE) is a typical strong-correlated system.
Jain proposed his composite fermion (CF) theory in 1989 in which CF is introduced as a weakly coupled degree of freedom.
For a pedagogical review of the CF theory, see David Tong's lecture [2].
We use Density Functional Theory (DFT) to compute FQHE in CF theory in this work.
Local density approximation (LDA) is used in CFT computation.

## Programme Structure

The following figure shows the self-consistency procedure and the usage of most of the functions in `Fqhe.py`.

![programme_structure.png](https://raw.githubusercontent.com/WhymustIhaveaname/FQHE_media/main/programme_structure.png)

The dependency relation of python scripts is

```
Fconsts ---> Futil ---> Fqhe ---> Flll
```

* Functions for DFT are in `Fqhe.py`
* `Futil.py` contains some utility funtions such as
    * for generating init states: `eig_to_n`, `gen_initst_*`, `gen_LL`
    * save and load partial results: `save_state`, `load_initst`
    * visualization: `heatmap`, `plot_profile`, `view_A`
* `Fconsts.py` defines constants such as electron number, filling fraction, etc. as well as `n_posi` and `ewt_tab` for computing Coulomb force.
* `Ubuntu_mono` is a font used in visualization

## Todo List

- [x] Check the codes for computing ground state again.
- [ ] Rewrite the Columb potential `Vc` part in C/C++. 
- [ ] Try different `nu` and `m` to probe the limit of this algorithm
- [ ] Put a point charge above the disk, both at the center and a little away from the center, to study how to excite anyons
- [ ] Braiding
- [ ] How does Jain get his `Vxc`?

## References

1. Yayun Hu and J. K. Jain. Kohn-sham theory of the fractional quantum hall effect. Phys. Rev. Lett., 123:176802, Oct 2019
2. David Tong. The Quantum Hall Effect. Jan 2016.
