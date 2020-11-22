# Neural Random Field with Latent Variables

NOTE: This README is a work in progress: some of the setup instructions might not yet work as stated.

Code for my MSc thesis entitled "On the computational function of context-dependent gamma synchrony". [Here is a link](https://drive.google.com/file/d/1h04hH7Ov0ojVmm-LAokibgKHt0VV68oT/view?usp=sharing) to the thesis.

**Abstract**:
Cortical computation is carried out by transiently activated sets of oscillating neurons. A classic and surprising result in cortical neuroscience revealed that gamma oscillations of nearby neurons in primary visual cortex with the same orientation preference synchronise when stimulated with collinear stimuli. Synchrony of neural activity reflects a temporal correlation between neural responses. Why synchronise neurons that represent collinear stimuli? A popular qualitative hypothesis is that natural scenes frequently contain collinear edges and it makes sense for correlated environment variables to be represented by correlated neural activity. But so far no quantitative model has bridged the gap between the statistical correlation of edge representations and synchronous neural dynamics. Recently, it has been suggested that cortical gamma oscillations may implement efficient probabilistic inference through dynamics that resemble Hamiltonian Markov Chain Monte Carlo. This thesis investigates whether this probabilistic account of cortical oscillations can extend to context-dependent gamma synchrony. If such synchrony were to emerge in a model trained to represent natural scenes that performs Hamiltonian sampling over its dependent latent variables, as is hypothesised in cortex, then it would represent quantitative evidence for the qualitative correlational hypothesis of context-dependent gamma synchrony. Here we train such a model. The model captures some of the statistics of natural scenes and exhibits oscillatory responses, a lag between inhibition and excitation, and contrast-dependent transient overshoots. However, we find no synchrony, suggesting that context-dependent synchrony results from factors beyond mere correlation using Hamiltonian sampling.

## Linux setup
### Requirements
Requires Python 3.5>.

First, download the repository. e.g.:
> git clone https://github.com/leesharkey/deep_attractor_network.git

Then, in the repo directory, run the following commands in the terminal:
> mkdir data exps exps/exp_data exps/models exps/samples exps/tblogs analysis_results

Then set up the pipenv environment and install the required python modules:
> pipenv install

The datasets (CIFAR10 and MNIST) should download automatically upon first use and be saved in the data directory. 


Then you should be good to go for training and visualisation. Training takes a very long time because every weight update involves many, many forward and backward passes through the network. 

From the repo directory, this command with these settings is a reasonable starting point for any other exploratory modifications you might want to do:

> python main.py --use_cuda --batch_size 128 --network_type NRFLV
--dataset CIFAR10 --architecture NRFLV_CIFAR_4SL_small_densebackw_wideSL1base
--initter_network_weight_norm --model_weight_norm --initializer ff_init
--initter_network_lr 5e-4 --l2_reg_energy_param 1.0 --num_it_neg 500
--num_it_pos 500 --img_logging_interval 10 --scalar_logging_interval 200
--log_histograms --histogram_logging_interval 3000 --sample_buffer_prob 0.95
--lr 5e-4 --lr_decay_gamma 0.95 --sampling_step_size 0.1
--states_activation hardtanh --activation swish --model_save_interval 50
--sigma 0.00 --state_optimizer sghmc --momentum_param 0.1 --non_diag_inv_mass
--num_burn_in_steps 100 --scale_grad 1.0 --max_sq_sigma 1e-4 1e-4 1e-4 1e-4
--min_sq_sigma 9e-5 --viz --viz_type standard --num_it_viz 5000
--viz_img_logging_step_interval 25 --viz_start_layer 1 --weights_optimizer adam





