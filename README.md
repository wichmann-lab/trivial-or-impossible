## Trivial or Impossible — dichotomous data difficulty masks model differences (on ImageNet and beyond)
#### OpenReview: https://openreview.net/forum?id=C_vsGwEIjAr
The code is structured as follows:
1. "00_train.ipynb": example notebook that showcases how models were trained in the different conditions
2. "01_build_arrays.ipynb": shows how data files with model responses were re-formed into arrays for the analysis
3. "02_figure_2.ipynb" - "06_figure_6.ipynb": example notebooks that show how the plots in the main paper were generated
4. "99_figures_appendix.ipynb": example notebook showing how plots in the appendix were generated

We have also attached "run_econ_res18.sh" and "conv_econ.sh" to show how models were trained for the ResNet-18 on ImageNet condition. If you have a SLURM based cluster, you should be able to reproduce our results using these scripts. Sadly, none of the notebooks are able to be executed as they are since the data files and resulting arrays exceed the Github file size limit. If you require any of the data files please contact: luca.schulze-buschoff@student.uni-tuebingen.de
