import torch
import pandas as pd
import os
import re
import random
from datetime import datetime
from torch import optim
import lib.custom_components.custom_swish_activation as cust_actv
import lib.custom_components.sghmc as sghmc


def save_configs_to_csv(args, model_name, session_name=None, loading=False, results_dict=None):
    """"""
    arg_dict = vars(args).copy()
    arg_dict = {**arg_dict, **{'unique_id': model_name}}

    # Combine dicts
    if results_dict is not None:
        arg_dict = {**arg_dict, **results_dict}

    # Convert any lists into string so csvs can take them
    for k, v in arg_dict.items():
        arg_dict[k] = str(v) if type(v) ==list or type(v) ==dict else v

    # Check there isn't already a df for this model(session); adjust model name if so
    if os.path.isfile("exps/params_and_results_%s.csv" % model_name) and loading:
        model_name = model_name + 'loaded_' + session_name

    # Create a df with a single row with new info
    full_df = pd.DataFrame(arg_dict, index=[model_name])

    # Create a new csv if one doesn't already exist and save the new data
    full_df.to_csv("exps/params_and_results_%s.csv" % model_name)
    print("Created new params_and_results csv for %s." % model_name)

    print("Saved params and results csv.")

def combine_all_csvs(directory_str, base_csv_name='params_and_results.csv',
                     remove_old_csvs=True):
    # Define some fixed variables
    archive_dir = 'archive_csvs'
    archive_path = os.path.join(directory_str, archive_dir)
    dfs = []
    files = [f for f in os.listdir(directory_str) if
             os.path.isfile(os.path.join(directory_str, f))]

    # Make archive file if keeping old csvs and it doesn't already exist
    if remove_old_csvs and not os.path.isdir(archive_path):
        os.mkdir(archive_path)

    # Loop through files, checking if they're a csv, and adding their pandas
    # dataframe to the list of dfs
    full_df = None
    for f in files:
        ext_filename = os.path.join(directory_str, f) #extended_filename
        print(ext_filename)
        if f.endswith('.csv'):
            new_df = pd.read_csv(ext_filename, header=0, index_col=0)
            dfs += [new_df]
            if full_df is not None:
                full_df = pd.concat([full_df, new_df], axis=0, sort=True)
            else:
                full_df = new_df
            if remove_old_csvs:
                os.rename(ext_filename, os.path.join(archive_path, f))

    # Concatenate all the dfs together, remove any duplicate rows, and save
    if dfs==[]:
        raise FileNotFoundError("No CSVs in the list to be merged. Check" +
                                " that your path names are correct and that" +
                                " the folder contains CSVs to merge.")
    full_df = full_df.drop_duplicates()
    full_df.to_csv(os.path.join(directory_str, base_csv_name))

def conv_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Thanks to DuaneNielsen
    Link to source: https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    """
    from math import floor
    h = floor( ((h_w + (2 * padding) - (kernel_size - 1) - 1 )/ stride) + 1)
    return h

def conv_t_output_shape(h_w, kernel_size=1, stride=1, padding=0,
                        output_padding=0):
    return (h_w -1)*stride - 2*padding + kernel_size + output_padding

def generate_random_states(args, shapes, device,
                           initter_network=None): #TODO if states_activation is relu then this needs to be multiplied by something like the variance of the negative phase states.


    rand_states = []
    for shape in shapes:
        if len(shape)==4:
            rs = torch.rand(shape[0],
                            shape[1],
                            shape[2],
                            shape[3],
                            device=device)
        elif len(shape)==2:
            rs = torch.rand(shape[0],
                            shape[1],
                            device=device)
        else:
            raise ValueError("Shape error. " +
                             "Must be either 4 (for conv) or 2 (for FC).")
        rand_states.append(rs)

    if args.states_activation == 'hardtanh':
        rand_states = [2 * (rs - 0.5) for rs in rand_states]

    # Use the initter network to initialize states close to the
    # 'physiological regime'
    if initter_network is not None and args.neg_ff_init:
        # Get the random pixels
        random_means = torch.rand(rand_states[0].shape[0:2]) * 0.7
        random_means = [random_means] * torch.prod(torch.tensor(rand_states[0].shape[2:])).item()
        random_means = torch.stack(random_means, dim=-1)
        random_means = random_means.view(rand_states[0].shape)
        random_means = random_means.to(device)

        random_pixels = (torch.randn_like(rand_states[0], device=device) * 0.3)
        random_pixels = random_means + random_pixels
        itr_rand_states = [random_pixels]

        # Use the initializer and feed it the random pixels to init each state
        #initter_network.train()
        itr_rand_states_new = initter_network.forward(itr_rand_states[0], x_id=None)
        itr_rand_states.extend(itr_rand_states_new)

        # Randomly shuffle the pixels and channels in each image to protect
        # against attractors in the initter network
        batch_size = itr_rand_states[0].shape[0]

        new_itr_rand_states = []
        for rs in itr_rand_states:
            shape = rs.shape
            num_ele = torch.prod(torch.tensor(rs.shape[1:]))
            rand_inds = torch.randperm(num_ele)
            shuff_rs = rs.view(batch_size, -1)[:,rand_inds].view(shape)
            new_itr_rand_states.append(shuff_rs)


        rand_states = itr_rand_states



    rand_states = [rsi.clone().detach() for rsi in rand_states]

    return rand_states


def random_arg_generator(parser, args):
    """Randomly reassigns specified arguments with certain specified values.

        Takes the parser and command line arguments, which specify the
        arguments that need to be randomised. The values from which they will
        be randomly selected are specified in the argument help string. This
        was easier than implementing it as a 'choice', for which argparse has
        functionality, because we only want certain choice restrictions
        when the arguments are being randomized. To implement them as a choice
        would mean that we'd always have to select from those choices even when
        not randomizing. """
    rand_args = args.randomize_args  # A list of arg names to randomize
    search_str = "Options: "

    # Go through rand_args and get possible options from help string
    for name in rand_args:
        arg = parser._option_string_actions['--' + name]  # Gets all arg info
        argtype = type(vars(args)[name])  # Gets the type of variable
        arghelpstr = arg.help  # Gets the helpstring of the arg

        if argtype == bool:
            new_arg_val = random.choice([True, False])
        elif argtype == str:
            arg_opts = extract_possible_values(search_str, arghelpstr)
            new_arg_val = random.choice(arg_opts) # Samples from possible vals
        elif argtype == int:
            arg_opts = extract_possible_values(search_str, arghelpstr)
            arg_opts = [int(opt) for opt in arg_opts]
            assert len(arg_opts) == 2, "Require exactly 2 numbers in %s"% name
            arg_opts = list(range(min(arg_opts), max(arg_opts)))
            new_arg_val = random.choice(arg_opts)
        elif argtype == float:
            arg_opts = extract_possible_values(search_str, arghelpstr)
            arg_opts = [float(opt) for opt in arg_opts]
            assert len(arg_opts) == 2, "Require exactly 2 numbers in %s" % name
            power = random.uniform(min(arg_opts), max(arg_opts))
            new_arg_val = 10. ** power
            # Rounds to n significant figures (n defined in string formatter)
            new_arg_val = float('%s' % float('%.4g' % new_arg_val))
        elif argtype == list and name == 'alphas':
            # This is just for searching for a list of alphas, because that
            # is awkward in the above code
            # #REMOVE WHEN PUBLISHING

            arg_opts1 = extract_possible_values(search_str, arghelpstr)
            arg_opts1 = [float(opt) for opt in arg_opts1]
            # print(arg_opts1)
            assert len(arg_opts1) == 2, "Require exactly 2 numbers in %s" % name
            power = random.uniform(min(arg_opts1), max(arg_opts1))
            base_rate = 10. ** power

            search_str = "Opt2: "
            arg_opts2 = extract_possible_values(search_str, arghelpstr, second_opt=True)
            arg_opts2 = [float(opt) for opt in arg_opts2]
            multiplier = random.uniform(min(arg_opts2), max(arg_opts2))
            # print('MULT: ' + str(multiplier))
            # print('Base: ' + str(base_rate))
            new_arg_val = [base_rate * (multiplier ** i)
                           for i in range(len(args.size_layers)-1)]
        else:
            raise TypeError("Unrecognised type of random argument given in " +
                            "command line arguments. ")

        vars(args)[name] = new_arg_val
        print("Random value for %s is %r" % (name, new_arg_val))
    return args

def extract_possible_values(search_str, arg_helpstr, second_opt=False):
    if second_opt:
        arg_opts = re.search(search_str + "\{(.*)\}", arg_helpstr).group()
    else:
        arg_opts = re.search(search_str + "\[(.*)\]", arg_helpstr).group()
    arg_opts = arg_opts[len(search_str) + 1:-1].split(sep=",")
    arg_opts = [opt.strip() for opt in arg_opts]
    arg_opts = [opt for opt in arg_opts if 'Opt2' not in opt] #REMOVE WHEN PUBLISHING
    return arg_opts

def requires_grad(parameters, flag=True):
    """Sets parameters to require grad"""
    for p in parameters:
        p.requires_grad = flag


def get_activation_function(args, states_activation=False, inplace=False):
    # Define whether it's getting the state_activation or the activation
    # for use within the multi-layer perceptrons
    if states_activation:
        activation = args.states_activation
    else:
        activation = args.activation

    if  activation == 'leaky_relu':
        act = torch.nn.LeakyReLU()
    elif activation == "relu":
        act = torch.nn.ReLU()
    elif activation == "swish":
        act = cust_actv.Swish_module()
    elif activation == "identity":
        act = torch.nn.Identity()
    elif activation == "hardsig":
        act = torch.nn.Hardtanh(min_val=0.0)
    elif activation == "hardtanh":
        act = torch.nn.Hardtanh()
    elif activation == "relu":
        act = torch.nn.ReLU(inplace=inplace)
    return act

def get_state_optimizers(args, params):
    """Gets an optimizer for each state_layer values (the gradients themselves
    may pass through multiple networks unless an energy mask is placed on them
    and the gradient is 0 at the mask value)."""
    if args.state_optimizer == 'langevin':
        return None
    if args.state_optimizer == 'sghmc':

        return [sghmc.SGHMC([prm],
                            lr=args.sampling_step_size[i],
                            noise=args.sigma[i],
                            mdecay=args.momentum_param[i],
                            num_burn_in_steps=args.num_burn_in_steps,
                            scale_grad=args.scale_grad,
                            min_sq_sigma=args.min_sq_sigma,
                            max_sq_sigma=args.max_sq_sigma[i],
                            args=args,
                            state_layer=i) for (i,prm) in enumerate(params)]
    if args.state_optimizer == 'sgd':
        return [optim.SGD([prm], lr=args.sampling_step_size[i]) for (i,prm) in enumerate(params)]
    if args.state_optimizer == 'sgd_momentum':
        return [optim.SGD([prm], args.sampling_step_size,
                         momentum=args.momentum_param,
                         dampening=args.dampening_param) for prm in params]
    if args.state_optimizer == 'nesterov':
        return [optim.SGD([prm], args.sampling_step_size,
                         momentum=args.momentum_param, nesterov=True) for prm in params]
    if args.state_optimizer == 'adam':
        return [optim.Adam([prm], args.sampling_step_size, betas=(0.9,0.999)) for prm in params]


def datetimenow():
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")



shapes = lambda x : [y.shape for y in x]
nancheck = lambda x : (x != x).any()
listout = lambda y : [x for x in y]
gradcheck = lambda  y : [x.requires_grad for x in y]
leafcheck = lambda  y : [x.is_leaf for x in y]
existgradcheck = lambda  y : [(x.grad is not None) for x in y]
existgraddatacheck = lambda  y : [(x.grad.data is not None) for x in y]
divl = lambda l1, l2: torch.prod(torch.tensor(l1)).float()/torch.prod(torch.tensor(l2)).float()



