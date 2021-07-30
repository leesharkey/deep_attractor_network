import os
import numpy as np


class DataManager:
    def __init__(self, root_path, model_exp_name, var_names,
                 state_layers=None, batches=None, channels=None, hw=None,
                 timesteps=None, save_extracted=True):
        self.root_path = root_path
        self.model_exp_name = model_exp_name
        self.var_names = var_names # list of variables (e.g. energy, state)
        self.state_layers = state_layers
        self.batches = batches # a list of idxs for batchs to select. All if None
        self.channels = channels # a list of idxs for channels to select. All if None
        self.hw = hw # a list of len==2 of the topleft and bottomright corners

        self.state_sizes = [[128,3,32,32],[128, 32, 32, 32]]

        self.filepaths = {}  # for each var in var_names
        self.data = {}
        self.data_name = '%s_%s_%s_%s_%s_%s' % (str(state_layers),
                                                str(var_names),
                                                str(batches),
                                                str(channels),
                                                str(hw),
                                                str(timesteps))

        # Get timestep range by counting the number of files in any variable's
        # data dir
        # Could be any variable dir, all should have the same num of files
        get_ts_key = var_names[0] + '_1'
        get_ts_full_path = os.path.join(self.root_path,
                                        self.model_exp_name,
                                        get_ts_key)
        print("Getting %s" % get_ts_full_path)
        (_, _, filenames) = next(os.walk(get_ts_full_path))
        if timesteps == None:
            self.timesteps = range(len(
                filenames))
        else:
            self.timesteps = timesteps

        # Check whether data of the same specifications has been extracted
        # first, since extraction takes quite a while from a EHD.
        pre_extracted_bool = self.check_for_pre_extracted_data()

        # If data have not been pre-extracted, then extract
        if not pre_extracted_bool:
            for sl in state_layers:
                sl = str(sl)
                for var in var_names:
                    # Make the path to the data for this variable and statelayer
                    key = var + '_' + sl  # e.g. 'energy_1'
                    full_path = os.path.join(self.root_path,
                                             self.model_exp_name,
                                             key)

                    # Get list of all filenames for this variable and statelayer
                    # and add to dict of filepaths
                    (_, _, filenames) = next(os.walk(full_path))

                    # Only selects the files for the specified timesteps or selects all if timesteps==None
                    filenames = [fnm for fnm in filenames
                                 if fnm in ['%.6i.npy' % n for n in self.timesteps]]
                    filenames = sorted(filenames)
                    # TODO consider adding an or fnm ['%.6i.png' % n for n in self.timesteps] to get images
                    filenames = [os.path.join(full_path, fnm) for fnm in filenames]
                    self.filepaths.update({key: filenames})
                    self.data.update({key :self.get_data(var, sl)}) #TODO if this stays here, change args to be just key as arg to get_data

            if save_extracted:
                print("Saving extracted data...")
                np.save(os.path.join(root_path, model_exp_name, self.data_name),
                        self.data)


    def get_data(self, var_name, state_layer_idx):
        """For the arg var_name, gets all the arrays from storage and collates
        into a single numpy array with dims [ts, b, ch, h, w].

        The specific batch, channel, or hw idxs can be specified and the func
        will only return.

        returns an array of dims [ts, b, ch, h, w]"""
        key = var_name + '_' + str(state_layer_idx)
        arrs = []
        for fnm in self.filepaths[key]:
            print("Loading and filtering %s" % fnm)
            arr = np.load(fnm)
            arr = arr.reshape(self.state_sizes[int(state_layer_idx)])
            if self.batches is not None:
                arr = arr.take(self.batches, axis=0)
                if type(self.batches)==int:
                    arr = np.expand_dims(arr, axis=0)
            if self.channels is not None:
                arr = arr.take(self.channels, axis=1)
                #arr = np.expand_dims(arr, axis=1)
            if self.hw is not None:
                x_inds = [corner[0] for corner in self.hw]
                y_inds = [corner[1] for corner in self.hw]

                x = [min(x_inds), max(x_inds)]
                y = [min(y_inds), max(y_inds)]

                arr = arr[:,:,y[0]:y[1],x[0]:x[1]]

            arrs.append(arr)

        return np.stack(arrs)

    def check_for_pre_extracted_data(self):
        full_path = os.path.join(self.root_path,
                                 self.model_exp_name,
                                 self.data_name) + '.npy'
        if os.path.exists(full_path):
            print("Getting preextracted data")
            self.data = np.load(full_path, allow_pickle=True).item()
            return True
        else:
            print("No preextracted data. Extracting ab initio...")
            return False