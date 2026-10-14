# squasim-homeostasis
This repository contains the code to run simulations and produce figures for the biophysical modelling part of the paper:
Engelman, D.*, Hochstetter, J.*, Bauduin*, A. et al. Sexual dimorphism in epidermal stem cell dynamics. Nature (2026). https://doi.org/10.1038/s41586-026-11159-2

Full details of the model can be found in the Supplementary Note.

Github repository: [https://github.com/joelhochstetter/squasim-homeostasis](https://github.com/joelhochstetter/squasim-homeostasis)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21997810.svg)](https://doi.org/10.5281/zenodo.21997810)


# Repository contributions
Joel Hochstetter (under the PhD supervision of Professor Ben Simons) is the sole author of all code in this repository. No AI-generated code was used in this study. AI tools were occasionally used for code autocomplete (Copilot) and debugging (LLMs).

The experimental data collected for this study that is modelled is included in this repository. This data was acquired by Dan Engelman and Anais Baudin, under the supervision of Cedric Blanpain.

For questions on the codebase contact Joel Hochstetter [jthochstetter@gmail.com](mailto:jthochstetter@gmail.com). 


# Overview of repository
- sim/: code to simulate 2D Voronoi model of tissues ('tissue' object in [tissue.py](sim/tissue.py)) and 0D tissue simulation ('tissue0D' object in [tissue.py](sim/tissue.py)), with predefined models defined in [sim_model.py](sim/sim_model.py).
- fit/: code to fit 'tissue'/'tissue0D' models to experimental data.
    - Use the 'optimizer' class (see [tiss_optimize.py](fit/tiss_optimize.py)) for grid-search or specified list of parameter sets.
    - Use the 'sim_anneal' class (subclass of 'optimizer') for simulated annealing (as well as grid search) (see [sim_anneal.py](fit/sim_anneal.py)). 
    - Experimental and simulated datasets are stored and manipulated as 'dataset' objects during fitting (see [datasets.py](fit/datasets.py)). 
    - The user selects the metrics for fitting from metrics defined in [metrics.py](fit/metrics.py) and [custom_metrics.py](fit/custom_metrics.py).
- experiment/: contains the csv files storing raw experimental data for each experiment, and ipynb which process the data and save in a pkl format that is used in the fitting pipeline.
- plot/: plotting functionality to both compare experimental data to simulations ([ds_plot.py](plot/ds_plot.py)) and visualise 2D tissue simulations ([tissplot.py](plot/tissplot.py))
- notebooks/: notebooks to plot clonal data analysis, and reproduce the computatational modelling part of the manuscript (full details are described below).
- analysis/: detailed clonal analysis pipeline ([clones.py](analysis/clones.py)) and further analysis functions ([analysis.py](analysis/analysis.py)).
- demo/: demonstration on how to run 0D/2D tissue simulations ([STD_spatial_sims.ipynb](demo/STD_spatial_sims.ipynb)) and demo on how to use the fitting pipeline ([SD_model_demo_fit.ipynb](demo/SD_model_demo_fit.ipynb)).




# Requirements
## Hardware requirements
To run simulations requires sufficient RAM (~1GB should be sufficient for individual simulations).



## Software requirements
### OS requirements
Code is operating system agnostic (Python code). The code has been tested on Ubuntu 24.04.4.



### Python dependencies
- python3.12.8 was used for the code.

The following packages were used. The versions are listed. Compatibility with other versions is not guaranteed.

| Package          | Version |
| ---------------- | ------- |
| numpy            | 2.0.2   |
| pandas           | 2.2.3   |
| matplotlib       | 3.10.0  |
| scipy            | 1.14.1  |
| joblib           | 1.4.2   |
| numba            | 0.60.0  |
| sparse           | 0.15.4  |
| moviepy          | 1.0.3   |
| sortedcontainers | 2.4.0   |

Full requirements are provided in [requirements.txt](requirements.txt) and [environment.yml](environment.yml).


### Other dependencies
To run simulations in parallel on the cluster: GNU parallel was used (version 20210822).



# Installation guide
No installation required: this is python code. A yml file ([environment.yml](environment.yml)) and requirements file ([requirements.txt](requirements.txt)) have been included to reproduce the conda environment.

The installation time is the time to install all python packages: < 10 minutes.



# Details of demo
This repository includes the following demos as ipynb:
- Run an example simulation and plot snapshots, movies: [demo/STD_spatial_sims.ipynb](demo/STD_spatial_sims.ipynb) (expected runtime: 30 minutes total)
- Example of fitting a model to experimental data with grid search or simulated annealing: [demo/SD_model_demo_fit.ipynb](demo/SD_model_demo_fit.ipynb) (expected runtime: 10-15 minutes total on 5 cores, ~50-75 minutes on 1 core)

The expected outputs are shown in the ipynb. Note, that for the fitting example these are not full fits: to run complete fits of data see below.


# Instructions to use
The instructions here allow one to reproduce the analysis and figures present in the paper. The run time of the code to reproduce all figures is >1000 CPU hours (we recommend using a cluster). This is because the grid-search and simulated annealing procedure is computationally intensive as it requires running a large number of parameter sets for each model. 



## Process experimental data
To process experimental data for use in python. Run in full the relevant notebooks for the type of data provided (data is ear-skin unless otherwise stated):

- Homeostasis (K14-Cre): [HomeoTimelapse.ipynb](experiment/homeo/HomeoTimelapse.ipynb)
- Castrated males (K14-Cre): [CastratedMice.ipynb](experiment/castre/CastratedMice.ipynb)
- Ovariectomized females (K14-Cre): [Ovx_data.ipynb](experiment/ovx/Ovx_data.ipynb)
- Wound healing (K14-Cre): [plot_WH.ipynb](experiment/wh/plot_WH.ipynb)
- Back and tail skin homeostasis (K14-Cre): [back_tail.ipynb](experiment/back_tail/back_tail.ipynb)
- Notch1-Cre homeostasis: [Notch1_data.ipynb](experiment/notch/Notch1_data.ipynb)
- Lgr6-Cre homeostasis: [Lgr6_data.ipynb](experiment/lgr6/Lgr6_data.ipynb)
- Past skin/oesophagus clonal data: [Old_studies.ipynb](experiment/past-studies/Old_studies.ipynb)

This will produce pickle files that are used in the fitting code.



## Run simulations and fit data
Code to run fitting pipeline (grid-search & simulated annealing) for each model mentioned in Supplementary Note is provided in the folder "modelling":
- SD model: 0D ([FitSD0D.py](modelling/FitSD0D.py)), 2D ([FitSD2D_full.py](modelling/FitSD2D_full.py)), 0D early time data ([FitSD_early_fits.py](modelling/FitSD_early_fits.py))
- STD model: 0D/2D ([FitSTD.py](modelling/FitSTD.py))
- SPD model (multiple rounds of TA asymmetric division): 0D/2D ([FitSPD.py](modelling/FitSPD.py))


Instructions to run code are provided as comments at the start of each py file. Examples are provided here for an SD model with args defined in the relevant file:

```bash
# Run sweep with:
parallel --max-procs 50 python3.12 FitSD0D.py {1} 1 0 0 ::: {0..22000} & #M, SD
parallel --max-procs 50 python3.12 FitSD0D.py {1} 1 0 1 ::: {0..22000} & #F, SD

# Combine sims with:
python3.12 FitSD0D.py -1000 1 0 0 &
python3.12 FitSD0D.py -1000 1 0 1 &

# Run parallel simulated annealing with:
parallel --max-procs 40 python3.12 FitSD0D.py {1} 1 0 0 ::: {-40..-1} & #M, SD
parallel --max-procs 40 python3.12 FitSD0D.py {1} 1 0 1 ::: {-40..-1} & #F, SD

# Combine sim anneal simulations with:
python3.12 FitSD0D.py -2000 1 0 0 &
python3.12 FitSD0D.py -2000 1 0 1 &
```



## Analyse data and plot figures
To reproduce all results from the paper, run the linked notebooks:

- Plot combined M/F data (EDFig 2a-c): [1.1_Combined_mouse_data.ipynb](notebooks/1_clonal_analysis/1.1_Combined_mouse_data.ipynb)
- Mouse-by-mouse analysis: plotting clonal dynamics (SF 5) and statistics (Supplementary Note): [1.2_mouse2mouse_variability.ipynb](notebooks/1_clonal_analysis/1.2_mouse2mouse_variability.ipynb)
- Analysis of the full clone-size distributions, including statistical analysis of the enhancement of even clone sizes (Figure 1e-h, EDFig 2e,f,h, Supplementary Note): [1.3_statistical_analysisM2M.ipynb](notebooks/1_clonal_analysis/1.3_statistical_analysisM2M.ipynb), [1.4_EvenOdd_by_sex.ipynb](notebooks/1_clonal_analysis/1.4_EvenOdd_by_sex.ipynb)
- Determination of mean-field kinetic parameters (Supplementary Note): [2.1_Kinetic_parameters_mean_field.ipynb](notebooks/2_SD_model/2.1_Kinetic_parameters_mean_field.ipynb)
- Fitting S model to experimental data, and plotting figures (Supplementary Note): [2.2_S_model_fits.ipynb](notebooks/2_SD_model/2.2_S_model_fits.ipynb)
- Theoretical analysis of SD model for varying stem cell asymmetry and sister coordination (SFig2  & Supplementary Note): [2.3_Theory_of_even_odd.ipynb](notebooks/2_SD_model/2.3_Theory_of_even_odd.ipynb)
- Fitting SD model to experimental data, and plotting figures (EDFig 3): [2.4_S_D_model_fits.ipynb](notebooks/2_SD_model/2.4_S_D_model_fits.ipynb)
- Fitting STD model to experimental data and plotting figures (Fig. 1h-n, EDFig 4): [3.1_STD_model_fits_paramater_uncertainties.ipynb](notebooks/3_STD_model/3.1_STD_model_fits_paramater_uncertainties.ipynb)
- Fitting SPD model (STD model with multiple rounds of TA asymmetric division permitted) to experimental data and plotting figures (Supplementary Note): [3.2_SPD_model_fits.ipynb](notebooks/3_STD_model/3.2_SPD_model_fits.ipynb)
- Fitting STD model with TA-D sister correlations permitted (Supplementary Note): [3.3_STD_model_fits_cTD=1.ipynb](notebooks/3_STD_model/3.3_STD_model_fits_cTD=1.ipynb)
- Plotting spatial snapshots and movies for STD model (Fig. 1n, Supplementary Videos): [STD_spatial_sims.ipynb](demo/STD_spatial_sims.ipynb) (note different clones and seed are used to paper)
- Fitting castrated male data with STD model (Fig. 3i, EDFig 6a-e, k): [4.1_castrated_STD_model.ipynb](notebooks/4_perturbations/4.1_castrated_STD_model.ipynb)
- Fitting ovariectomized female data with STD model (Fig. 3m, EDFig 6f-k): [4.2_perturbed_STD_model.ipynb](notebooks/4_perturbations/4.2_perturbed_STD_model.ipynb)
- Fitting Lgr6 and Notch1 data with STD model (ED Fig 9-10): [5_Lgr6_Notch.ipynb](notebooks/5_targeted_promters/5_Lgr6_Notch.ipynb)
- Plotting wound healing data (Fig. 2h): [plot_WH.ipynb](experiment/wh/plot_WH.ipynb)
- Plotting EdU pseudoclone size distributions (ED Fig 2h): [1.4_EvenOdd_by_sex.ipynb](notebooks/1_clonal_analysis/1.4_EvenOdd_by_sex.ipynb)
- Plotting clone size distributions from old studies (SFig 4): [Old_studies.ipynb](experiment/past-studies/Old_studies.ipynb).


# References
The tissue simulation (contents of sim/) and fitting (contents of fit/) code in this work builds on a prior codebase (see [Zenodo](https://doi.org/10.5281/zenodo.12629079) and [Github](https://github.com/joelhochstetter/2024-Cell-Sarate-Hochstetter-Valet-et-al)) written by Joel Hochstetter for the study below:
Dynamic regulation of tissue fluidization controls skin repair during wound healing, Sarate R. (co‑first), Hochstetter J. (co‑first), Valet M. (co‑first), Hallou A., Song Y., Bansaccal N., Ligare M., Aragona M., Campas O., Simons B.D., Blanpain C., Cell, 187, 1–18, [https://doi.org/10.1016/j.cell.2024.07.031](https://doi.org/10.1016/j.cell.2024.07.031).



# License
This code is available under a [Attribution-NonCommercial-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-nc-sa/4.0/) 

This license strictly prohibits commercial use of the code or derivatives. This includes the prohibition to use the code for the training of AI models for any commercial purpose. Any research, academic, or personal use is permitted.



# Copyright 
Copyright (C) 2026 Joel Hochstetter.
