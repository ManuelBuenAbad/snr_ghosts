# snr_ghosts
A Python code to compute the axion dark matter echo of supernova remnants.

Written by Manuel A. Buen-Abad and Chen Sun, 2021

Requirements
-----------------------------------------

1. Python  
2. numpy  
3. scipy  
4. astropy
5. healpy
6. astropy_healpix
7. astroquery

Bibtex entry
-----------------------------------------

If you use this code or part of it, or if you find it in any way useful for your research, please cite [Buen-Abad, Fan, & Sun (2021)](https://arxiv.org/abs/2110.13916). The BibTeX entry is:

	@article{Buen-Abad:2021qvj,
	    author = "Buen-Abad, Manuel A. and Fan, JiJi and Sun, Chen",
	    title = "{Axion Echos from the Supernova Graveyard}",
	    eprint = "2110.13916",
	    archivePrefix = "arXiv",
	    primaryClass = "hep-ph",
	    month = "10",
	    year = "2021"
	}


Nota Bene
-----------------------------------------

In the computation of the signal and noise powers in interferometry mode, we have not accounted for the fact that only active telescopes contribute to the reception area, instead having used the total area. As such, the individual values for the signal and noise powers cannot be trusted. However, since both scale the same way with the reception area, their ratio can. This ratio is what we use in all our results.

We will account for this area correction for the individual signal and noise powers in a future version of our code.
