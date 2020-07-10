# DPE

Code used in *"Optimal Algorithms for Multiplayer Multi-Armed Bandits" - Po-An Wang, Alexandre Proutiere, Kaito Ariu, Yassir Jedra, Alessio Russo, Proceedings AISTATS 2020*.

Code Author: Alessio Russo

## License

Our code is released under the MIT License (refer to the [LICENSE](https://github.com/rssalessio/dpe/blob/master/LICENSE.md) file for details).

## Requirements

- Python 3.7
- numpy 1.16.1
- matplotlib
- cython (for pyximport)


## Simulations

To do the simulations, first create scores/ and figures/ repositories. Then run the notebook file.
If you decide to run a new simulations with different parameters, first delete all the content in the
 \"scores\" folder, or copy it somewhere else if you need it.

## Cite

If you find this code useful in your research, please, consider citing it:
>@misc{dpe2020,
>  author       = {Alessio Russo},
>  title        = {DPE Algorithms},
>  year         = 2020,
>  doi          = {10.5281/zenodo.3783611},
>  url          = { https://doi.org/10.5281/zenodo.3783611 }
>}

and/or cite the paper:
>@inproceedings{wang2020optimal,
>  title={Optimal algorithms for multiplayer multi-armed bandits},
>  author={Wang, Po-An and Proutiere, Alexandre and Ariu, Kaito and Jedra, Yassir and Russo, Alessio},
>  booktitle={International Conference on Artificial Intelligence and Statistics},
>  pages={4120--4129},
>  year={2020}
>}


[![DOI](https://zenodo.org/badge/242978531.svg)](https://zenodo.org/badge/latestdoi/242978531)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/rssalessio/dpe/blob/master/LICENSE.md)
