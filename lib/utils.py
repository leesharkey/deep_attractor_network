import torch
import pandas as pd
import os
import re
import random
from datetime import datetime

def save_configs_to_csv(args, model_name, results_dict=None):
    """"""
    arg_dict = vars(args).copy()
    arg_dict = {**arg_dict, **{'unique_id': model_name}}


    # Combine dicts
    if results_dict is not None:
        arg_dict = {**arg_dict, **results_dict}

    # Convert any lists into string so csvs can take them
    for k, v in arg_dict.items():
        arg_dict[k] = str(v) if type(v) ==list else v

    # Create a df with a single row with new info
    new_df = pd.DataFrame(arg_dict, index=[model_name])

    # Check there isn't already a df for this model
    if os.path.isfile("exps/params_and_results_%s.csv" % model_name):
        # If there is, get the colnames and make sure to include them
        old_df = pd.read_csv("exps/params_and_results_%s.csv" % model_name,
                             header=0, index_col=0)
        full_df = pd.merge(old_df, new_df, how='left')
        full_df = full_df.set_index(new_df.index)
    else:
        full_df = new_df




    # Create a new csv if one doesn't already exist and save the new data
    # if not os.path.isfile("exps/params_and_results_%s.csv" % model_name):
    full_df.to_csv("exps/params_and_results_%s.csv" % model_name)
    print("Created new params_and_results csv for %s." % model_name)

    # If a csv already exists, concat the new df to the old df and save
    # else:
    #     old_df = pd.read_csv("exps/params_and_results.csv", header=0,
    #                          index_col=0)
    #     # If a model by this name already exists in the df, drop it to
    #     # overwrite it (useful for saving configs early and results later)
    #     if model_name in old_df.index:
    #         old_df = old_df.drop(model_name)
    #     full_df = pd.concat([old_df, new_df], axis=0, sort=True)
    #     full_df.to_csv("exps/params_and_results.csv")
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
        if f.endswith('.csv'):
            new_df = pd.read_csv(ext_filename, header=0, index_col=0)
            dfs += [new_df]
            if full_df is not None:
                full_df = pd.concat([full_df, new_df], axis=0, sort=True)
            else:
                full_df = new_df
            if remove_old_csvs:
                os.rename(ext_filename, os.path.join(archive_path, f))
            print(ext_filename)

    # Concatenate all the dfs together, remove any duplicate rows, and save
    if dfs==[]:
        raise FileNotFoundError("No CSVs in the list to be merged. Check" +
                                " that your path names are correct and that" +
                                " the folder contains CSVs to merge.")
    # full_df = pd.concat(dfs, axis=0, sort=True)
    full_df = full_df.drop_duplicates()
    full_df.to_csv(os.path.join(directory_str, base_csv_name))


def generate_random_states(shapes, device):
    return [torch.rand(shapes[i][0],
                         shapes[i][1],
                         shapes[i][2],
                         shapes[i][3],
                         device=device)
             for i in range(len(shapes))]


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
    # all_arg_names = list(parser._option_string_actions.keys())
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
            # is awkward in the above code #TODO REMOVE WHEN PUBLISHING
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
    arg_opts = [opt for opt in arg_opts if 'Opt2' not in opt] #TODO REMOVE WHEN PUBLISHING
    return arg_opts

def datetimenow():
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")
