# Neural Random Field with Latent Variables

Code for my MSc thesis entitled "On the computational function of context-dependent gamma synchrony". [Here is a link](https://drive.google.com/file/d/1h04hH7Ov0ojVmm-LAokibgKHt0VV68oT/view?usp=sharing) to the thesis.

**Abstract**:
Cortical computation is carried out by transiently activated sets of oscillating neurons. A classic and surprising result in cortical neuroscience revealed that gamma oscillations of nearby neurons in primary visual cortex with the same orientation preference synchronise when stimulated with collinear stimuli. Synchrony of neural activity reflects a temporal correlation between neural responses. Why synchronise neurons that represent collinear stimuli? A popular qualitative hypothesis is that natural scenes frequently contain collinear edges and it makes sense for correlated environment variables to be represented by correlated neural activity. But so far no quantitative model has bridged the gap between the statistical correlation of edge representations and synchronous neural dynamics. Recently, it has been suggested that cortical gamma oscillations may implement efficient probabilistic inference through dynamics that resemble Hamiltonian Markov Chain Monte Carlo. This thesis investigates whether this probabilistic account of cortical oscillations can extend to context-dependent gamma synchrony. If such synchrony were to emerge in a model trained to represent natural scenes that performs Hamiltonian sampling over its dependent latent variables, as is hypothesised in cortex, then it would represent quantitative evidence for the qualitative correlational hypothesis of context-dependent gamma synchrony. Here we train such a model. The model captures some of the statistics of natural scenes and exhibits oscillatory responses, a lag between inhibition and excitation, and contrast-dependent transient overshoots. However, we find no synchrony, suggesting that context-dependent synchrony results from factors beyond mere correlation using Hamiltonian sampling.

## Linux setup
### Requirements
Requires Python 3.5>.

First, download the repository.

Then, in the repo directory, run the following commands in the terminal:
> mkdir data exps exps/exp_data exps/models exps/samples exps/tblogs analysis_results

Then set up the pipenv environment and install the required python modules:
> pipenv install torch torchvision tensorboard numpy matplotlib pandas

The datasets (CIFAR10 and MNIST) should download automatically upon first use and be saved in the data directory. 


Then you should be good to go for training, visualisation, generating the 
experimental stimuli, and recording the experimental data. All of these 
processes take place in the script `main.py`. They each have a separate 
manager class, (found in `managers.py`) which contain the methods required for
each process. In the script `main.py`, the different processes can be turned on
or off using CLI flags. The following sections provide a walkthrough of the 
processes that occur in `main.py`. 

## Training

Training occurs on line 1836 of `main.py` and is managed by the 
class `managers.TrainingManager`. 

Training takes a very long time because every weight update involves many, 
many forward and backward passes through the network. 

From the repo directory, this command with these settings is a reasonable 
starting point for any other exploratory modifications you might want to do:

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
--min_sq_sigma 9e-5 --weights_optimizer adam

This will create a model in the directory `exps/models`.

Since that will take a long time, it's recommended to load a pretrained model. 
You can load the model used in the thesis by adding 
> --load_model=20200922-152859__rndidx_22892_at_47500_its

to the above command. Even though we load this model, a new model is still
created so as not to overwrite the model that we loaded. 

Since this model is already trained, you can skip training in `main.py` 
by adding the argument
> --no_train_model

. Training is on by default. After training, or if training is skipped, 
the script simply proceeds to 'visualisation'. 

## Visualisation

Visualisation occurs on line 1841 of `main.py` and is managed by the
class `managers.VisualizationManager`. 

> --viz 

activates the visualisation process following the (potentially skipped) 
training process. It also requires:
> --num_it_viz 5000 --viz_img_logging_step_interval 25 --viz_start_layer 1

There are several types of visualisation and we need to specify it before 
visualisation will work. 
 

Usually, in the negative training phase, we let the network find an activity
configuration is low energy across all parts of the network. To recreate this
process without training the network, add 
> --viz_type standard 

Visualisation permits us to find activity configurations that are low energy
only from the perspective of parts of the network. For example, figure 2.6 (right)
was produced using this method and illustrates what the 1st state layer considers
low energy. This is analagous to asking a cortical V1 neuron 
'what stimulus do you find least surprising/lowest energy?'
> --viz_type channels_energy

You will need to run `main.py` twice, using a different `--viz_type` flag each time
if you want to run more than one type of visualisation. 

Visualisation will deposit images in the directory 
`exps/samples/<model name>/`

## Experimental stimuli generation

After visualisation, the `main.py` script runs the methods that generate the 
experimental stimuli.

This requires the CLI argument 
> --gen_exp_stim

to be added to `main.py`'s arguments. 

These methods will create gabor filter-like images in the 
directory `data/gabor_filters` and its subdirectories.

## Recording experimental data

Using the experimental stimuli that have just been created, the next section of
`main.py` records the experimental data. This is managed by the class 
`managers.ExperimentsManager`.

It is required to add the argument 
> --experiment

in order to actually record the experimental data.

Because this will generate several hundred GB of data, it is recommended to 
use a large storage space. It is therefore also required to add the location
of this storages space as a CLI argument:
> --exp_data_root_path <path to storage>

## Analysis

There is just one script that consecutively runs all the analyses. This will
take over a week to run, so be patient. 

You can run the analysis script using:

> python analysis_main.py --root_path <path to storage>. 

Assuming you stored the experimental data on an EHD, adding the argument
> --ehd_results_storage

will save the analysis results in a directory on that storage device too. 
This is recommended, because analysis also consumes several gigabytes of 
storage (but not hundreds). The directory used to store the analyses will
have the same name as the model