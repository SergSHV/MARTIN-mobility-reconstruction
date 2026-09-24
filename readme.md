# MARTIN: Reconstructing higher-order human mobility from temporal contact networks



This repository contains the code, data and numerical experiments accompanying the manuscript.



MARTIN reconstructs finite-order Markov mobility processes from temporal contact statistics.



## Code



The `code/` folder contains the reconstruction methods.


* `martin.py` contains `recover_markov_chain` for exact reconstruction and `recover_markov_chain_numerical` for numerical reconstruction.

* `utils.py` contains supporting functions used by the reconstruction methods and numerical experiments.



## Example



The `example/` folder contains a complete example of exact reconstruction with MARTIN.

The notebook imports a ground-truth Markov mobility process, computes the corresponding stationary contact statistics, reconstructs the mobility process from those statistics and compares the reconstructed solutions with the ground-truth process.

The ground-truth transition matrix is used only to generate the contact statistics and is not provided to MARTIN during reconstruction.



## Data



The `data/` folder contains the Markov chains used in the numerical experiments, including heterogeneous mobility processes and mobility processes of different Markov orders.



## Experiments



The `experiments/` folder contains the notebooks and associated files used to reproduce Figures 3, 5 and 6 of the manuscript.

Each figure has a separate folder containing the corresponding analysis and plotting code. Precomputed results are included where rerunning the complete calculation is computationally expensive.



## Requirements



Install the required Python packages with:



```bash

pip install -r requirements.txt

```



Exact reconstruction additionally requires [MPSolve](https://numpi.dm.unipi.it/scientific-computing-libraries/mpsolve/) for multiprecision polynomial root finding. MPSolve must be installed separately and available from the command line.



## Reproducing the results



The exact-reconstruction example is provided in the `example/` folder.



The numerical results reported in Figures 3, 5-6 can be reproduced from the corresponding folders in `experiments/`.



Run the notebooks in the corresponding folders to reproduce the analyses and figures.





## Contact



Sergey Shvydun

s.shvydun at tudelft.nl

