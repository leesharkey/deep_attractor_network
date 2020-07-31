# Neural Random Field with Latent Variables

I needed to build an energy-based model with latent variables that could work with natural images. Previous 
EBMs with latent variables hadn't demonstrated that
they could do that. 

## Linux setup
###Requirements
Requires Python 3.X with

First, download the repository.

Then, in the repo directory, run the following commands in the terminal:
> mkdir data exps exps/exp_data exps/models exps/samples exps/tblogs analysis_results

If you're on the Euler cluster at ETHZurich, load the 
correct python module
> module load new gcc/4.8.2 python/3.7.1

> mkdir 