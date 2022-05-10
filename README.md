# estoiles-public
This repository contains all the code for the figures and analysis in [Wang et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021PhRvD.103h4007W/abstract) and Wang et al. (2022) (in prep).

The [`estoiles/`](estoiles/) directory contains the actual calculation code.

The [`drivers/`](drivers/) directory contains separate folders for each paper with scripts/notebooks for making the figures, generating the necessary data and running example analysis.  

To create the figures and run example analysis, first run `python setup.py install`. Then run the scripts in the [`drivers/`](drivers/) directory.
