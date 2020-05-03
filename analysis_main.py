"""To use this script, you'll need data. You can generate data from a trained
model in the script main.py by using the experiment functions found in the
experiment manager. For even a modest size of architecture, saving the values
of the states, their energies, their gradients, and their momenta takes up
a lot of storage space e.g. tens or hundreds of megabytes per timestep, and
you'll need hundreds or thousands of timesteps. Recommend using an EHD. """

import argparse
import os
import numpy as np
import pandas as pd
import scipy.signal as sps
from lib.data import SampleBuffer, Dataset
import lib.utils
import seaborn as sns
import matplotlib.pyplot as plt


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

        self.state_size = [128, 32, 32, 32]

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
        get_ts_key = var_names[0] + '_0'
        get_ts_full_path = os.path.join(self.root_path,
                                        self.model_exp_name,
                                        get_ts_key)
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
            arr = arr.reshape(self.state_size)
            if self.batches is not None:
                arr = arr.take(self.batches, axis=0)
            if self.channels is not None:
                arr = arr.take(self.channels, axis=1)
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






class AnalysisManager:
    def __init__(self, args):
        self.args = args
        self.primary_model_exp_name = "20200428-125116__rndidx_15777_loaded20200423-154227__rndidx_15605/observeCIFAR10"
        pass

    def find_oscillating_neurons(self):
        """Looks at a target subset of the data and returns the indices of the
        neurons whose responses qualify as oscillating."""
        model_exp_name = self.primary_model_exp_name
        var_names = ['energy']
        var_label_pairs = [('energy_1', 'Energy')]
        dm = DataManager(root_path=self.args.root_path,
                         model_exp_name=model_exp_name,
                         var_names=var_names,
                         state_layers=[1],
                         batches=[0],
                         channels=None,
                         hw=[[10,10],[18,18]],
                         timesteps=range(250, 1000))

        # Process data before putting in a dataframe
        reshaped_data = [np.arange(len(dm.timesteps)).reshape(
            len(dm.timesteps), 1)]
        reshaped_data.extend([
                    dm.data['energy_1'].reshape(len(dm.timesteps), -1)])
        num_energy_cols = reshaped_data[1].shape[1]
        arr_sh = dm.data['energy_1'].shape[2:]
        inds = np.meshgrid(np.arange(arr_sh[0]),
                           np.arange(arr_sh[1]),
                           np.arange(arr_sh[2]),
                           sparse=False, indexing='ij')
        inds = [i.reshape(-1) for i in inds]
        energy_names = []
        for i,j,k in zip(*inds):
            idx = str(i) + '_' + str(j) + '_' + str(k)
            energy_names.append('energy_1__' + idx)


        data = np.concatenate(reshaped_data, axis=1)
        colnames = ['Timestep']
        colnames.extend(energy_names)
        df = pd.DataFrame(data, columns=colnames)

        c = 201
        sdata = data[:, c]
        f, t, zxx = sps.stft(sdata, axis=0, nperseg=100, nfft=100,
                             window=sps.windows.gaussian(100, 1), boundary='constant')
        plt.pcolormesh(t, f, np.abs(zxx))
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.savefig("spectrum.png")


        # Make one plot per variable
        for j, colname in enumerate(colnames[1:]):
            sns.set(style="darkgrid")
            _, _, plot1, plot2 = plt.acorr(df[colname],
                                           detrend=lambda x: x - np.mean(x),
                                           maxlags=749)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))  # Uses sci notation for units
            plt.tight_layout()  # Stops y label being cut off
            fig = plot1.get_figure()
            fig.savefig('plot_osc_search_%s.png' % colname)
            fig.clf()

    def single_neuron_dynamics_plot_Ham_case(self):
        """The oscillatory dynamics of a randomly selected neuron in the HD
        networks. But also include activations of the momentum variables."""
        model_exp_name = self.primary_model_exp_name
        var_names = ['energy', 'state', 'momenta', 'grad']
        var_label_pairs = [('energy_1', 'Energy'),
                           ('momenta_1', 'Momenta'),
                           ('state_1', 'State'),
                           ('grad_1', 'Gradient')]

        # Extract data from storage into DataManager
        dm = DataManager(root_path=self.args.root_path,
                         model_exp_name=model_exp_name,
                         var_names=var_names,
                         state_layers=[1],
                         batches=[0],#range(6,11),
                         channels=[3],
                         hw=[[14,14],[15,15]],
                         timesteps=range(0, 6999))

        # Process data before putting in a dataframe
        reshaped_data = [np.arange(len(dm.timesteps)).reshape(
            len(dm.timesteps), 1)]
        reshaped_data.extend([
                    dm.data[var].reshape(len(dm.timesteps), -1) for var, _ in var_label_pairs])
        data = np.concatenate(reshaped_data, axis=1)
        colnames = ['Timestep']
        colnames.extend([label for var, label in var_label_pairs])
        df = pd.DataFrame(data, columns=colnames)

        # Make one plot per variable
        for var, label in var_label_pairs:
            sns.set(style="darkgrid")
            plot = sns.lineplot(x="Timestep", y=label, data=df)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))  # Uses sci notation for units
            plt.tight_layout()  # Stops y label being cut off
            fig = plot.get_figure()
            fig.savefig('plot_%s.png' % label)
            fig.clf()  # Clears figure so plots aren't put on top of one another

        # Make pairs plot
        g = sns.PairGrid(df)
        g.map(sns.lineplot)
        g.savefig('pairs plot.png')
        plt.close()
        # plot two plots, one for state/energy and the other for momenta


    def single_neuron_dynamics_plot_LD_case(self):
        """The oscillatory dynamics of a randomly selected neuron in the LD
        networks."""
        raise NotImplementedError()

    def single_neuron_autocorrelation(self):
        """The oscillatory dynamics of a randomly selected neuron in the HD
        networks. But also include activations of the momentum variables."""
        model_exp_name = self.primary_model_exp_name
        var_names = ['energy', 'state', 'momenta', 'grad']
        var_label_pairs = [('energy_1', 'Energy'),
                           ('momenta_1', 'Momenta'),
                           ('state_1', 'State'),
                           ('grad_1', 'Gradient')]

        # Extract data from storage into DataManager
        dm = DataManager(root_path=self.args.root_path,
                         model_exp_name=model_exp_name,
                         var_names=var_names,
                         state_layers=[1],
                         batches=[0],#range(6,11),
                         channels=[3],
                         hw=[[14,14],[15,15]],
                         timesteps=range(0, 6999))

        # Process data before putting in a dataframe
        reshaped_data = [np.arange(len(dm.timesteps)).reshape(
            len(dm.timesteps), 1)]
        reshaped_data.extend([
                    dm.data[var].reshape(len(dm.timesteps), -1) for var, _ in var_label_pairs])
        data = np.concatenate(reshaped_data, axis=1)
        colnames = ['Timestep']
        colnames.extend([label for var, label in var_label_pairs])
        df = pd.DataFrame(data, columns=colnames)

        # Make one autocorrelation plot per variable
        for var, label in var_label_pairs:
            sns.set(style="darkgrid")
            _, _, plot1, plot2 = plt.acorr(df[label],
                                           detrend=lambda x: x - np.mean(x),
                                           maxlags=2000)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))  # Uses sci notation for units
            plt.tight_layout()  # Stops y label being cut off
            fig = plot1.get_figure()
            fig.savefig('plot_acf_%s.png' % label)
            fig.clf()

    def two_neuron_autocorrelation(self):
        """The oscillatory dynamics of a randomly selected neuron in the HD
        networks. But also include activations of the momentum variables."""
        model_exp_name = self.primary_model_exp_name
        var_names = ['energy', 'state', 'momenta', 'grad']
        var_label_pairs = [('energy_1', 'Energy'),
                           ('momenta_1', 'Momenta'),
                           ('state_1', 'State'),
                           ('grad_1', 'Gradient')]

        # Extract data from storage into DataManager
        dm1 = DataManager(root_path=self.args.root_path,
                         model_exp_name=model_exp_name,
                         var_names=var_names,
                         state_layers=[1],
                         batches=[0],#range(6,11),
                         channels=[3],
                         hw=[[14,14],[15,15]],
                         timesteps=range(0, 6999))
        dm2 = DataManager(root_path=self.args.root_path,
                         model_exp_name=model_exp_name,
                         var_names=var_names,
                         state_layers=[1],
                         batches=[0],#range(6,11),
                         channels=[5],
                         hw=[[20,20],[21,21]],
                         timesteps=range(0, 6999))


        # Process data before putting in a dataframe
        reshaped_data1 = [np.arange(len(dm1.timesteps)).reshape(
            len(dm1.timesteps), 1)]
        reshaped_data1.extend([
                    dm1.data[var].reshape(len(dm1.timesteps), -1) for var, _ in var_label_pairs])
        data1 = np.concatenate(reshaped_data1, axis=1)

        reshaped_data2 = [np.arange(len(dm2.timesteps)).reshape(
            len(dm2.timesteps), 1)]
        reshaped_data2.extend([
            dm2.data[var].reshape(len(dm2.timesteps), -1) for var, _ in
            var_label_pairs])
        data2 = np.concatenate(reshaped_data2, axis=1)

        colnames = ['Timestep']
        colnames.extend([label for var, label in var_label_pairs])
        df1 = pd.DataFrame(data1, columns=colnames)
        df2 = pd.DataFrame(data2, columns=colnames)

        # Make one autocorrelation plot per variable
        for var, label in var_label_pairs:
            fig, ax = plt.subplots()
            sns.set(style="darkgrid")
            ax.set_title("%s cross correlation for random neurons" % label)
            _, _, plot1, plot2 = plt.xcorr(x=df1[label],
                                           y=df2[label],
                                           detrend=lambda x: x - np.mean(x),
                                           maxlags=5000)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))  # Uses sci notation for units
            plt.tight_layout()  # Stops y label being cut off
            fig = plot1.get_figure()
            fig.savefig('plot_xcf3_%s.png' % label)
            fig.clf()

    def plot_energy_timeseries(self):
        """Plot the decrease in energy as time progresses. Should be faster
        in the HD case than in the LD case."""

        # TODO
        #  still need to run the LD experiments
        #  then need to confirm this is working
        #  Confirm what the error bars in their figure are and include them
        var_names = ['energy']
        var_label_pairs = [('energy_1', 'Energy (HD)'),
                           ('energy_1', 'Energy (LD)')]
        hd_model_exp_name = self.primary_model_exp_name
        # TODO change the LD model name when you've done the inference
        ld_model_exp_name = self.primary_model_exp_name

        # Extract data from storage into DataManager
        hd_dm = DataManager(root_path=self.args.root_path,
                         model_exp_name=hd_model_exp_name,
                         var_names=var_names,
                         state_layers=[1],
                         batches=None,  # range(6,11),
                         channels=None,
                         hw=None,
                         timesteps=range(1, 6999))
        ld_dm = DataManager(root_path=self.args.root_path,
                         model_exp_name=ld_model_exp_name,
                         var_names=var_names,
                         state_layers=[1],
                         batches=None,
                         channels=None,
                         hw=None,
                         timesteps=range(1, 6999))

        # Process data before putting in a dataframe
        hd_energysum = hd_dm.data['energy_1'].sum(axis=(1, 2, 3, 4))
        ld_energysum = ld_dm.data['energy_1'].sum(axis=(1, 2, 3, 4))

        reshaped_data = [np.arange(len(hd_dm.timesteps)).reshape(
            len(hd_dm.timesteps), 1)]
        reshaped_data.extend([
            hd_energysum.reshape(len(hd_dm.timesteps), -1)])
        reshaped_data.extend([
            ld_energysum.reshape(len(ld_dm.timesteps), -1)])

        data = np.concatenate(reshaped_data, axis=1)
        colnames = ['Timestep']
        colnames.extend([label for var, label in var_label_pairs])
        df = pd.DataFrame(data, columns=colnames)

        # Plot the data in the dataframe, overlaying the HD and LD cases in
        # one plot
        sns.set(style="darkgrid")
        plot1 = sns.lineplot(x="Timestep", y='Energy (HD)', data=df)
        plot2 = sns.lineplot(x="Timestep", y='Energy (LD)', data=df)
        plt.ticklabel_format(axis="y", style="sci",
                             scilimits=(0, 0))  # Uses sci notation for units
        plt.tight_layout()  # Stops y label being cut off
        fig = plot1.get_figure()
        fig.savefig('plot_2%s.png' % 'HDLD_energy')

    def timeseries_EI_balance(self):
        """As in Fig 5 of Aitcheson and Lengyel (2016)."""
        model_exp_name = self.primary_model_exp_name
        var_names = ['energy', 'state', 'momenta', 'grad']
        num_batches = 128 # i.e. num of trials
        dm = DataManager(root_path=self.args.root_path,
                         model_exp_name=model_exp_name,
                         var_names=var_names,
                         state_layers=[1],
                         batches=None,
                         channels=[3],
                         hw=[[14,14],[15,15]],
                         timesteps=range(0, 6999))

        var_label_pairs = [('momenta_1', 'Momenta'),
                           ('energy_1', 'Energy'),
                           ('state_1', 'State'),
                           ('grad_1', 'Gradient')]

        # Put data into a df
        data = [dm.data[var].sum(axis=(2,3,4))
                              for var, _ in var_label_pairs
                if var in ['momenta_1', 'energy_1']] #gets only mom and energy and unsqueezes
        data = [v - v.mean(axis=0) for v in data]
        reshaped_data = [rsd.reshape(len(dm.timesteps), num_batches)
                              for rsd in data] # make into columns
        summed_data = reshaped_data[0] + reshaped_data[1]  # Sum momenta and energy

        num_ts = len(dm.timesteps)
        timesteps = [np.arange(num_ts).reshape(num_ts, 1)] * num_batches #time idxs
        timesteps = np.concatenate(timesteps, axis=0)
        summed_data = summed_data.reshape(num_ts * num_batches, 1)
        df_data = np.concatenate([timesteps, summed_data], axis=1)
        colnames = ['Timestep', 'Mean-centred sum of momentum and energy']
        df = pd.DataFrame(df_data, columns=colnames)

        # Plot the data in the dataframe
        sns.set(style="darkgrid")
        plot = sns.lineplot(x=colnames[0],
                            y=colnames[1],
                            data=df)
        #plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))  # Uses sci notation for units
        plt.tight_layout()  # Stops y label being cut off
        fig = plot.get_figure()
        fig.savefig('plot_%s vs %s.png' % (colnames[1], colnames[0]))
        fig.clf()  # Clears figure so plots aren't put on top of one another


    def xcorr_EI_lag(self):
        """As in Fig 5 of Aitcheson and Lengyel (2016).

        It's supposed to show that the excitation leads the inhibition slightly
        by showing a peak offset in the Xcorr plot. But """
        model_exp_name = self.primary_model_exp_name
        var_names = ['energy', 'state', 'momenta', 'grad']
        num_batches = 128 # i.e. num of trials
        dm = DataManager(root_path=self.args.root_path,
                         model_exp_name=model_exp_name,
                         var_names=var_names,
                         state_layers=[1],
                         batches=None,
                         channels=[3],
                         hw=[[14,14],[15,15]],
                         timesteps=range(0, 6999))

        var_label_pairs = [('momenta_1', 'Momenta'),
                           ('energy_1', 'Energy')]

        # Process data before plotting
        data = [dm.data[var].sum(axis=(2,3,4))
                              for var, _ in var_label_pairs]  #unsqueezes
        data = [v - v.mean(axis=0) for v in data]  # mean of timeseries to centre
        reshaped_data = [rsd.reshape(len(dm.timesteps), num_batches)
                              for rsd in data]  # make into columns
        mean_data = [np.mean(rsdata, axis=1) for rsdata in reshaped_data]  # takes mean across batches

        # Plot
        sns.set(style="darkgrid")
        fig, ax = plt.subplots()
        ax.set_title("Cross correlation avg. momentum and avg. energy")
        _, _, plot1, plot2 = plt.xcorr(x=mean_data[1],#1 is energy
                                       y=mean_data[0],#0 is momentum
                                       detrend=lambda x: x - np.mean(x),
                                       maxlags=300)
        plt.ticklabel_format(axis="y", style="sci",
                             scilimits=(0, 0))  # Uses sci notation for units
        plt.tight_layout()  # Stops y label being cut off
        fig = plot1.get_figure()
        fig.savefig('plot_xcf_avgmom_avgenergy.png')
        fig.clf()


# analysis manager (contains analysis main)
# init:
# # sets up data manager class
# # has a Root_dir_str

# has a line plotting func
# # 2d plotting function (with convenient axes, title, and nice formatting)
#
# for each desired plot,
# # loads desired data using data manager
# # uses analysis functions to calculate statistics
# #
shapes = lambda x: [y.shape for y in x]

def main():

    parser = argparse.ArgumentParser(description='Analysis of inference dynamics.')
    sgroup = parser.add_argument_group('Model selection')
    sgroup.add_argument('--root_path', type=str, default='/media/lee/DATA/DDocs/AI_neuro_work/DAN/exp_data/',
                        help='The path to the recorded experimental data.' +
                             'session that the data were generated by. ' +
                             'Default: %(default)s.')

    args = parser.parse_args()


    # sgroup.add_argument('--modelexp_name', type=str, default='20200422-145311__rndidx_25238_loaded20200417-165337__rndidx_70768/observeCIFAR10',
    #                     help='The name of the model and the visualisation ' +
    #                          'session that the data were generated by. ' +
    #                          'Default: %(default)s.')
    # root_path = '/media/lee/DATA/DDocs/AI_neuro_work/DAN/exp_data/'
    # modelexp_path = '20200422-145311__rndidx_25238_loaded20200417-165337__rndidx_70768/observeCIFAR10'

    am = AnalysisManager(args)
    am.find_oscillating_neurons()
    # am.single_neuron_dynamics_plot_Ham_case()
    # am.single_neuron_autocorrelation()
    # am.plot_energy_timeseries()
    # #am.single_neuron_trial_avg_EI_balance() #unused
    # am.timeseries_EI_balance()
    # am.xcorr_EI_lag()
    # am.two_neuron_autocorrelation()
if __name__ == '__main__':
    main()






# Wont use but functional:
# def single_neuron_trial_avg_EI_balance(self):
#     """As in Fig 5 of Aitcheson and Lengyel (2016)."""
#     model_exp_name = '20200425-143441__rndidx_14197_loaded20200423-154227__rndidx_15605/observeCIFAR10'
#     var_names = ['energy', 'state', 'momenta', 'grad']
#     num_batches = 128 # i.e. num of trials
#     dm = DataManager(root_path=self.args.root_path,
#                      model_exp_name=model_exp_name,
#                      var_names=var_names,
#                      state_layers=[1],
#                      batches=None,
#                      channels=[3],
#                      hw=[[14,14],[15,15]],
#                      timesteps=range(1300, 1600))
#
#     var_label_pairs = [('momenta_1', 'Avg. Momenta'),
#                        ('energy_1', 'Negative Avg. Energy'),
#                        ('state_1', 'Avg. State'),
#                        ('grad_1', 'Avg. Gradient')]
#
#     # Put data into a df
#     reshaped_data = [np.arange(num_batches)]
#     dm.data['momenta_1'] = np.abs(dm.data['momenta_1'])
#     #dm.data['grad_1'] = np.abs(dm.data['grad_1'])
#     dm.data['energy_1'] = - dm.data['energy_1']
#     reshaped_data.extend([dm.data[var].mean(axis=(0,2,3,4))
#                           for var, _ in var_label_pairs])
#     reshaped_data = [rsd.reshape(num_batches, 1) for rsd in reshaped_data] # make into columns
#     data = np.concatenate(reshaped_data, axis=1)
#     colnames = ['Trial']
#     colnames.extend([label for var, label in var_label_pairs])
#     df = pd.DataFrame(data, columns=colnames)
#
#     for var, label in var_label_pairs[1:]:
#         sns.set(style="darkgrid")
#         plot = sns.scatterplot(x=label, y='Avg. Momenta', data=df)
#         #plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))  # Uses sci notation for units
#         plt.tight_layout()  # Stops y label being cut off
#         fig = plot.get_figure()
#         fig.savefig('plot_%s_vs_mom_avg.png' % label)
#         fig.clf()  # Clears figure so plots aren't put on top of one another
