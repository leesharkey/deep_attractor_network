import argparse
import os
import numpy as np
import pandas as pd
import scipy.signal as sps
import scipy.stats as spst
from scipy.signal import butter, filtfilt, detrend, welch
from lib.data import SampleBuffer, Dataset
import lib.utils
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as implt
from torchvision import transforms, utils
from PIL import Image
import torch
from scipy import optimize
import lib.analysis.datamanager as datamanager




class AnalysisManager:
    def __init__(self, args, session_name):
        self.args = args
        self.primary_model = '20200508-115243__rndidx_37562_loaded20200423-154227__rndidx_15605'
        self.just_angles_model = '20200512-075955__rndidx_62111_loaded20200423-154227__rndidx_15605'
        exp_stem = '/orientations_present_single_gabor'
        self.primary_model_exp_name = self.primary_model + exp_stem
        self.just_angles_exp = self.just_angles_model + exp_stem + '_just_angle'
        self.session_name = session_name

        # Make base analysis results dir if it doesn't exist
        if not os.path.isdir('analysis_results'):
            os.mkdir('analysis_results')

        # Make dir for this specific analysis session
        self.session_dir = os.path.join('analysis_results', self.session_name)
        self.base_analysis_results_dir = 'analysis_results'
        if not os.path.isdir(self.session_dir):
            os.mkdir(self.session_dir)

        pass

    def butter_lowpass_filter(self, data, cutoff, fs=1, order=2):
        normal_cutoff = cutoff / 0.5 #nyquist freq=0.5=1/2
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def one_d_gabor_function(self, p, x):
        return (1 / (np.sqrt(2*np.pi) * p[0])) \
               * np.exp(-(x**2)/(2*p[0])) * p[2] * \
               np.cos(2 * np.pi * p[1] * x + p[3]) + p[4] * x

    def gabor_fitting_func(self, p, x):
        return self.one_d_gabor_function(p, x)


    def print_stimuli(self):
        model_exp_name = self.primary_model_exp_name
        var_names = ['state']
        dm = datamanager.DataManager(root_path=self.args.root_path,
                                     model_exp_name=model_exp_name,
                                     var_names=var_names,
                                     state_layers=[0],
                                     batches=None,
                                     channels=None,
                                     hw=None,
                                     timesteps=[0, 51, 3051])
        num_imgs = dm.data['state_0'].shape[0]
        for i in range(num_imgs):
            im = dm.data['state_0'][i]
            im = torch.tensor(im)

            utils.save_image(im,
                             self.session_dir + \
                             'stimuli_during_%s_at_ts%i.png' % (dm.data_name, i),
                             nrow=16, normalize=True, range=(0, 1))

    def find_active_neurons(self, exp_type='primary'):
        print("Finding orientation preferences of neurons")

        if exp_type == 'just_angles':
            model_exp_name = self.just_angles_exp
            angles = np.linspace(start=0.0, stop=np.pi * 2, num=128)
            contrasts = [2.4]
        elif exp_type == 'primary':
            model_exp_name = self.primary_model_exp_name
            angles = [0.0, 0.393, 0.785, 1.178, 1.571, 1.963, 2.356, 2.749,
                      3.142, 3.534, 3.927, 4.32, 4.712, 5.105, 5.498, 5.89]
            contrasts = [0., 0.4, 0.8, 1.2, 1.6, 2., 2.4, 2.8]

        # Prepare variables for data managers
        var_names = ['state']
        h = 11
        w = 11
        hh = 21
        ww = 21

        h = 9
        hh = 9
        w = 22
        ww = 22

        # Prepare varaibles used for processing data
        angle_contrast_pairs = []
        for a in angles:
            for c in contrasts:
                angle_contrast_pairs.append((a, c))
        stim_on_start = 1050
        stim_on_stop = 3550
        num_ch = 32

        activity_df_created = False

        for ch in range(num_ch):
            print("Channel %s" % str(ch))
            # Gets data
            dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=model_exp_name,
                                         var_names=var_names,
                                         state_layers=[1],
                                         batches=None,
                                         channels=[ch],
                                         hw=[[h, hh], [w, ww]],
                                         timesteps=None)
            # Processes data
            dm.data['state_1'] = dm.data['state_1'].squeeze() # get rid of ch dim
            reshaped_activities = [np.arange(len(dm.timesteps)).reshape(
                len(dm.timesteps), 1)]  # make TS data
            reshaped_activities.extend([
                dm.data['state_1'].reshape(len(dm.timesteps), -1)])  # extend TS data with actual data
            arr_sh = dm.data['state_1'].shape[1:]

            # Make column names from indices of batches, height, and width
            inds = np.meshgrid(np.arange(arr_sh[0]),
                               np.arange(arr_sh[1]),
                               np.arange(arr_sh[2]),
                               sparse=False, indexing='ij')
            inds = [i.reshape(-1) for i in inds]

            colnames = ['Timestep']
            for i, j, k in zip(*inds):
                colnames.append(
                    'state_1__b%s_h%s_w%s' % (i, h + j, hh + k))

            activities    = np.concatenate(reshaped_activities, axis=1)
            activities_df = pd.DataFrame(activities, columns=colnames)

            # Separate data during stim from data when stim isn't present
            during_stim = activities_df[stim_on_start:stim_on_stop]
            outside_stim = activities_df.drop(
                activities_df.index[stim_on_start:stim_on_stop])
            outside_stim = outside_stim.drop(activities_df.index[0:50])

            # Test whether mean during stim is signif different than outside
            # of stim and if it is higher, then it is considered 'active'
            ttest_result = spst.ttest_ind(a=during_stim,
                                          b=outside_stim,
                                          equal_var=False,
                                          axis=0)
            mean_diffs = during_stim.mean(axis=0) - outside_stim.mean(axis=0)
            mean_results = mean_diffs > 0
            t_results = ttest_result.pvalue > 0.0005
            comb_results = mean_results & t_results
            comb_results = pd.DataFrame(comb_results).T

            # Save the results of activity tests
            if not activity_df_created:
                activity_df = pd.DataFrame(data=comb_results,
                                           columns=comb_results.columns)
                activity_df_created = True
            else:
                activity_df = activity_df.append(comb_results,
                                                 ignore_index=True)

        print("Done finding active neurons")
        # Finish adding b, ch, h, w labels
        activity_df = activity_df.T
        activity_df = activity_df.drop('Timestep')
        activity_df['batch_idx'] = inds[0]
        activity_df['height'] = [h + j for j in inds[1]]
        activity_df['width'] = [hh + j for j in inds[2]]
        activity_df.to_pickle(os.path.join(
            self.base_analysis_results_dir, 'neuron_activity_results_%s.csv') % exp_type)

    def find_orientation_preferences(self):
        print("Finding orientation preferences of neurons")

        just_angles_exp = "/media/lee/DATA/DDocs/AI_neuro_work/DAN/exp_data/20200512-075955__rndidx_62111_loaded20200423-154227__rndidx_15605/orientations_present_single_gabor_just_angle"
        model_exp_name = just_angles_exp  #self.primary_model_exp_name
        # var_names = ['energy']
        # var_label_pairs = [('energy_1', 'Energy')]
        var_names = ['state']
        var_label_pairs = [('state_1', 'State')]
        angle_contrast_pairs = []
        # angles = [0.0, 0.393, 0.785, 1.178, 1.571, 1.963, 2.356, 2.749, 3.142,
        #           3.534, 3.927, 4.32, 4.712, 5.105, 5.498, 5.89] #todo LEE TODAY update for new exp settings
        # contrasts = [0., 0.4, 0.8, 1.2, 1.6, 2., 2.4, 2.8]
        angles = np.linspace(start=0.0, stop=np.pi*2, num=128)
        contrasts = [2.4]
        for a in angles:
            for c in contrasts:
                angle_contrast_pairs.append((a, c))

        activity_df = pd.read_pickle(os.path.join(
            self.base_analysis_results_dir, 'neuron_activity_results_just_angles.csv'))


        # Create a new dataframe that sums over h and w (so we only have batch
        # info per channel)
        sum_cols = list(activity_df.columns).append(['angle', 'contrast'])
        patch_activity_df = pd.DataFrame(columns=sum_cols)
        # Get sum for each neuron (h,w) so that the df just contains sums
        # over the whole image patch for each channel in each batch element.
        # Each batch element corresponds to a particular (angle, contrast) pair
        for b in range(128):
            batch_df = activity_df.loc[activity_df['batch_idx']==b].sum(axis=0)
            batch_df['angle']    = angle_contrast_pairs[b][0]
            batch_df['contrast'] = angle_contrast_pairs[b][1]
            patch_activity_df = patch_activity_df.append(batch_df.T,
                                                         ignore_index=True)

        # Drop the trials that use very low contrasts
        patch_activity_df = patch_activity_df.drop(
            patch_activity_df[(patch_activity_df['contrast'] == 0) |
                           (patch_activity_df['contrast'] == 0.4)].index)

        # Create a new dataframe that sums over all remaining contrasts so that
        # only channel response per angle remains
        sum_angles_df = pd.DataFrame(columns=patch_activity_df.columns)
        for a in angles:
            sum_angles_df = sum_angles_df.append(
                patch_activity_df.loc[patch_activity_df['angle'] == a].sum(axis=0),
            ignore_index=True)
        sum_angles_df = sum_angles_df.drop(columns=['width','height','contrast', 'batch_idx'])

        # Plot the orientation preference plots
        fig, ax = plt.subplots(4,8, sharey=True, sharex=True)
        fig.set_size_inches(23, 8)
        k = 0
        angle_axis = [round((180/np.pi) * angle, 1) for angle in angles]
        orient_prefs = pd.DataFrame(columns=['amplitude',
                                             'phaseshift',
                                             'mean'])
        for i in range(4):
            for j in range(8):
                print("%i, %i" % (i,j))
                normed_data = sum_angles_df[k] - np.mean(sum_angles_df[k])
                normed_data = normed_data / \
                              np.linalg.norm(normed_data)

                amplitude0  = 2.0
                phaseshift0 = 0
                mean0  = 0.
                params0 = [amplitude0, phaseshift0, mean0]
                opt_func = lambda x: x[0] * np.cos(2*angles + x[1]) + x[
                    2] - normed_data

                est_params = \
                    optimize.leastsq(opt_func, params0)[0]

                ax[i,j].plot(angle_axis, normed_data)

                if est_params[0] < 0.:
                    est_params[0] = est_params[0] * -1
                    est_params[1] += np.pi

                if est_params[1] < 0.:
                    est_params[1] += 2 * np.pi


                fitted_curve = est_params[0] * np.cos(2*angles + est_params[1]) \
                               + est_params[2]
                if k!=17:  # because nans
                    ax[i,j].plot(angle_axis, fitted_curve)


                if i==3:
                    ax[i, j].set_xlabel('Angles (degree)')
                    ax[i, j].set_xticks([0,90,180,270])
                if j==0:
                    ax[i,j].set_ylabel('Normed activity [a.u.]')

                # Log orientation pref
                orient_prefs.loc[len(orient_prefs)] = est_params
                k += 1
        plt.savefig(os.path.join(
            self.session_dir, 'orientation_prefs.png'))
        plt.savefig(os.path.join(
            self.base_analysis_results_dir, 'orientation_prefs.png')) #for dev
        plt.close()
        print(orient_prefs)


        # Save the results
        exp_type = '_just_angles'
        orient_prefs.to_pickle(os.path.join(
            self.base_analysis_results_dir, 'orientation_pref_results%s.csv' % exp_type))
        patch_activity_df.to_pickle(os.path.join(
            self.base_analysis_results_dir, 'patch_activity_results%s.csv'   % exp_type))
        activity_df.to_pickle(os.path.join(
            self.base_analysis_results_dir, 'neuron_activity_results%s.csv'  % exp_type))

    def print_activity_map(self):
        nrnact = pd.read_pickle(
            os.path.join(self.base_analysis_results_dir,
                         'neuron_activity_results_primary.csv'))

        im_start = 9
        im_dim = 13
        # Reorganises activity dataframe so we can sum over pixels
        map_act = pd.melt(nrnact, id_vars=['batch_idx', 'height', 'width'],
                          value_vars=list(range(32)))
        map_act = map_act.rename(columns={'variable': 'channel', 'value':'active'})
        pixel_inds = [(i, j) for i in list(range(im_dim)) for j in range(im_dim)]

        # Plot for each channel over all batches
        for ch in range(32):
            im = np.zeros([im_dim, im_dim])
            pch = map_act['channel'] == ch
            for (i,j) in pixel_inds:
                pi = map_act['height'] == i + im_start
                pj = map_act['width']  == j + im_start
                cond = pi & pj & pch
                sum_b_ch = map_act.loc[cond]['active'].sum()
                im[i][j] = sum_b_ch
            im = im / im.max()
            plt.imshow(im)
            plt.savefig(
                os.path.join(self.base_analysis_results_dir,
                             "neuron activity locations ch%i.png" % ch))

        # Plot for each batch
        for b in range(127):
            im = np.zeros([im_dim, im_dim])
            pb = map_act['batch_idx'] == b
            for (i,j) in pixel_inds:
                pi = map_act['height'] == i + im_start
                pj = map_act['width']  == j + im_start
                cond = pi & pj & pb
                sum_b_ch = map_act.loc[cond]['active'].sum()
                im[i][j] = sum_b_ch
            im = im / im.max()
            plt.imshow(im)
            plt.savefig(
                os.path.join(self.base_analysis_results_dir,
                             "neuron activity locations b%i.png" % b))

    def plot_pixel_vs_activity(self):
        nrnact = pd.read_pickle(
            os.path.join(self.base_analysis_results_dir,
                         'neuron_activity_results_primary.csv'))

        im_start = 9
        im_dim = 13
        # Reorganises activity dataframe so we can sum over pixels
        map_act = pd.melt(nrnact, id_vars=['batch_idx', 'height', 'width'],
                          value_vars=list(range(32)))
        map_act = map_act.rename(columns={'variable': 'channel', 'value':'active'})
        pixel_inds = [(i, j) for i in list(range(im_dim)) for j in range(im_dim)]




        # Get the experimental images
        exp_stim_path_base = "data/gabor_filters/single/"
        exp_stim_stem = "contrast_and_angle"
        exp_stim_path = exp_stim_path_base + exp_stim_stem
        (_, _, filenames) = next(os.walk(exp_stim_path))
        filenames = sorted(filenames)
        exp_stims = []
        for flnm in filenames:
            im = Image.open(os.path.join(exp_stim_path, flnm))
            im = transforms.functional.to_tensor(im)
            exp_stims.append(im)
        exp_stims = torch.stack(exp_stims) # should have generated only 128 images
        exp_stims = exp_stims.mean(dim=1) # makes image black and white
        exp_stims = exp_stims.numpy()

        # Make column names from indices of batches, height, and width
        inds = np.meshgrid(np.arange(exp_stims.shape[0]),
                           np.arange(exp_stims.shape[1]),
                           np.arange(exp_stims.shape[2]),
                           sparse=False, indexing='ij')
        inds = [i.reshape(-1) for i in inds]

        # Make a df for pixel intensities
        pix_intensities = exp_stims.reshape(-1)
        df_data = np.stack([pix_intensities, inds[0], inds[1], inds[2]]).T
        pix_int_df = pd.DataFrame(data=df_data, columns=['Pixel intensity',
                                                         'batch_idx',
                                                         'height',
                                                         'width'])

        # Combine the dataframes
        pix_int_df = pd.merge(pix_int_df, map_act, how='inner')



        # Plot for each channel over all batches
        for ch in range(32):
            # Group the df data by pixel index (i.e. get the probability of
            pch = pix_int_df['channel'] == ch
            values_for_channel = pix_int_df.loc[pch]
            values_for_channel = \
                values_for_channel.groupby(['Pixel intensity'])['active'].mean()
            values_for_channel = pd.DataFrame(values_for_channel)
            plt.scatter(values_for_channel.index,
                     values_for_channel['active'],)
            plt.xlabel('Pixel intensity')
            plt.savefig(os.path.join(self.base_analysis_results_dir,
                        'activity proportion vs pixel intensity for ch %i.png' % ch))
            plt.close()

    def explore_oscillations(self):
        #### Older code (KEEP it but save it in a different function as
        # something like 'explore_oscillations_in_channel'.

        model_exp_name = self.primary_model_exp_name

        # var_names = ['energy']
        # var_label_pairs = [('energy_1', 'Energy')]
        var_names = ['state']
        var_label_pairs = [('state_1', 'State')]

        dm = datamanager.DataManager(root_path=self.args.root_path,
                                     model_exp_name=model_exp_name,
                                     var_names=var_names,
                                     state_layers=[1],
                                     batches=[0],
                                     channels=None,
                                     hw=[[13, 13], [18, 18]],
                                     timesteps=range(0, 6550))

        # Process data before putting in a dataframe
        reshaped_data = [np.arange(len(dm.timesteps)).reshape(
            len(dm.timesteps), 1)]
        reshaped_data.extend([
            dm.data['state_1'].reshape(len(dm.timesteps), -1)])
        arr_sh = dm.data['state_1'].shape[2:]
        inds = np.meshgrid(np.arange(arr_sh[0]),
                           np.arange(arr_sh[1]),
                           np.arange(arr_sh[2]),
                           sparse=False, indexing='ij')
        inds = [i.reshape(-1) for i in inds]
        energy_names = []
        for i, j, k in zip(*inds):
            idx = str(i) + '_' + str(j) + '_' + str(k)
            energy_names.append('state_1__' + idx)

        data = np.concatenate(reshaped_data, axis=1)

        colnames = ['Timestep']
        colnames.extend(energy_names)
        df = pd.DataFrame(data, columns=colnames)
        for name in colnames[1:]:
            fig, ax = plt.subplots(4)
            fig.set_size_inches(14.5, 8.5)
            spec_data = df[name]
            # spec_data = (df[name] - np.mean(df[name]))
            # spec_data = spec_data / np.linalg.norm(spec_data)

            # Denoise
            # spec_data = self.butter_lowpass_filter(spec_data,cutoff=0.1,fs=100)

            ax[0].plot(df['Timestep'], spec_data)
            ax[0].set_ylabel('State [a.u.]')
            ax[0].set_xlabel('Timesteps')

            ax[1].specgram(spec_data, NFFT=512, Fs=100, Fc=0, detrend=None,
                           noverlap=511, xextent=None, pad_to=None,
                           sides='default',
                           scale_by_freq=True, scale='dB', mode='default',
                           cmap=plt.get_cmap('hsv'))
            ax[1].set_ylabel('Frequency [a.u.]')

            lags1, acorrs1, plot11, plot12 = ax[2].acorr(df[name][1050:3550],
                                                         detrend=plt.mlab.detrend_linear,
                                                         # lambda x: x - np.mean(x)
                                                         maxlags=2499)
            lags2, acorrs2, plot21, plot22 = ax[3].acorr(df[name][3550:6050],
                                                         detrend=plt.mlab.detrend_linear,
                                                         maxlags=2499)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(
            0, 0))  # Uses sci notation for units
            plt.tight_layout()  # Stops y label being cut off

            # plt.title('STFT Magnitude')
            plt.savefig(
                self.session_dir + '/' + "spectrum__single_gabor%s.png" % name)
            plt.close()

            # p0 = {'sigma': 1.0,
            #      'omega':  1.0,
            #      'amp':    1.0,
            #      'bias1':  0.0,
            #      'bias2':  0.0}

            p0 = [555.0, 550.0, 1e20, 0.0, 0.0]

            p1, success = optimize.leastsq(self.gabor_fitting_func, p0,
                                           args=(lags1))
            # plt.plot(lags1, acorrs1)
            plt.plot(lags1, self.one_d_gabor_function(p1, lags1))
            plt.savefig(self.session_dir + '/' + "gabors.png")
            plt.close()

        # Make one plot per variable
        # for j, colname in enumerate(colnames[1:]):
        #     sns.set(style="darkgrid")
        #     _, _, plot1, plot2 = plt.acorr(df[colname],
        #                                    detrend=lambda x: x - np.mean(x),
        #                                    maxlags=749)
        #     plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))  # Uses sci notation for units
        #     plt.tight_layout()  # Stops y label being cut off
        #     fig = plot1.get_figure()
        #     fig.savefig('plot_osc_search_%s.png' % colname)
        #     fig.clf()



    def find_oscillating_neurons(self):
        """Looks at a target subset of the data and returns the indices of the
        neurons whose responses qualify as oscillating."""

        # for every neuron in a certain image patch,
        #     Load the same data as you did to determine which neurons were active
        #
        #     Calculate the acorr plot
        #     Fit the acorr plot with a gabor function
        #     calculate the frequency of oscillation regardless of whether the
        #       neuron is on or off and
        #     save it to the map_act df as a new column

        print("Finding oscillating neurons")

        # Make dir to save plots for this experiment
        exp_dir = os.path.join(self.base_analysis_results_dir,
                               'Fitted gabors and PSD plot')
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)


        # Prepare general variables
        model_exp_name = self.primary_model_exp_name

        during_or_outside_names = ['during', 'outside']
        param_names = ['sigma', 'freq', 'amplitude', 'cost']

        angles = [0.0, 0.393, 0.785, 1.178, 1.571, 1.963, 2.356, 2.749,
                  3.142, 3.534, 3.927, 4.32, 4.712, 5.105, 5.498, 5.89]
        contrasts = [0., 0.4, 0.8, 1.2, 1.6, 2., 2.4, 2.8]

        angle_contrast_pairs = []
        for a in angles:
            for c in contrasts:
                angle_contrast_pairs.append((a, c))


        # Prepare variables for data managers
        var_names = ['state']
        h = 11
        w = 11
        hh = 21
        ww = 21

        h = 9
        hh = 9
        w = 22
        ww = 22

        # Prepare variables used for processing data
        stim_on_start = 1050
        stim_on_stop = 3550
        num_ch = 32
        activity_df_created = False

        # Get info on which neurons are active
        print("Loading priorly processed data...")
        nrnact = pd.read_pickle(
            os.path.join(self.base_analysis_results_dir,
                         'neuron_activity_results_primary.csv'))
        ori_pref = pd.read_pickle(
            os.path.join(self.base_analysis_results_dir,
                         'orientation_pref_results_just_angles.csv'))
        print("Done loading presaved data.")

        # Create empty columns in df to save fitted gabor func params
        nrnact["sigma_during"] = np.nan
        nrnact["freq_during"] = np.nan
        nrnact["amplitude_during"] = np.nan
        nrnact["cost_during"] = np.nan
        nrnact["sigma_outside"] = np.nan
        nrnact["freq_outside"] = np.nan
        nrnact["amplitude_outside"] = np.nan
        nrnact["cost_outside"] = np.nan



        # Rearrange activity info so that we can identify channels and
        # pixels that are active for a given stim
        nrnact = pd.melt(nrnact, id_vars=['batch_idx', 'height', 'width'],
                          value_vars=list(range(32)))
        nrnact = nrnact.rename(columns={'variable': 'channel',
                                          'value':'active'})

        # # Count the activities for each channel for each stim
        # batch_ch_activity = pd.DataFrame(columns=['batch_idx', 'channel',
        #                                           'sum_pixels'])
        # for b in range(128):
        #     for ch in range(32):
        #         b_cond  = nrnact['batch_idx'] == b
        #         ch_cond = nrnact['channel']   == ch
        #         cond = b_cond & ch_cond
        #         sum_pixels = nrnact.loc[cond]['active'].sum()
        #         results = [b, ch, sum_pixels]
        #         batch_ch_activity.loc[len(batch_ch_activity)] = results


        for ch in range(1,num_ch):

            print("Channel %s" % str(ch))
            if ch == 17:
                # The issue was that no neurons were active for 17 so it caused
                # an error. 17 is a weird channel.
                print("Not doing channel 17 but CHANGE THIS LATER") #TODO if the model starts having a channel 17, change this
                continue

            # Get data
            dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=model_exp_name,
                                         var_names=var_names,
                                         state_layers=[1],
                                         batches=None,
                                         channels=[ch],
                                         hw=[[h, hh], [w, ww]],
                                         timesteps=None)
            # Process data
            dm.data['state_1'] = dm.data['state_1'].squeeze() # get rid of ch dim
            reshaped_activities = [np.arange(len(dm.timesteps)).reshape(
                len(dm.timesteps), 1)]  # make TS data
            reshaped_activities.extend([
                dm.data['state_1'].reshape(len(dm.timesteps), -1)])  # extend TS data with actual data
            arr_sh = dm.data['state_1'].shape[1:]

            # Make column names from indices of batches, height, and width
            inds = np.meshgrid(np.arange(arr_sh[0]),
                               np.arange(arr_sh[1]),
                               np.arange(arr_sh[2]),
                               sparse=False, indexing='ij')
            inds = [i.reshape(-1) for i in inds]

            colnames = ['Timestep']
            for i, j, k in zip(*inds):
                colnames.append(
                    'state_1__b%s_h%s_w%s' % (i, h + j, hh + k))

            activities    = np.concatenate(reshaped_activities, axis=1)
            activities_df = pd.DataFrame(activities, columns=colnames)

            # Separate data during stim from data when stim isn't present
            during_stim = activities_df[stim_on_start:stim_on_stop]
            outside_stim = activities_df.drop(
                activities_df.index[stim_on_start:stim_on_stop])
            outside_stim = outside_stim.drop(activities_df.index[0:50])

            # Find the subset of the nrnact df for this channel only and get
            # the h, w values for on neurons in each batch
            nrnact_ch = nrnact.loc[nrnact['channel'] == ch]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['active']]
            nrnact_ch = nrnact_ch.reset_index()
            on_colnames = []
            on_col_inds = []
            for i in range(len(nrnact_ch)):
                bhw = tuple(nrnact_ch.loc[i][1:4])
                on_colname = 'state_1__b%i_h%i_w%i' % bhw
                on_colnames.append(on_colname)
                on_col_inds.append(bhw)


            # For the on neurons in the above-defined subset of nrnact, get
            # their timeseries from the arrays during_stim and outside_stim
            on_during_stim  = during_stim[on_colnames]
            on_outside_stim = outside_stim[on_colnames]

            # # Calculate the acorr plots for the active neurons' timeseries
            # # from the arrays during_stim and outside_stim
            # on_during_stim  = detrend(on_during_stim, axis=0)
            # on_outside_stim = detrend(on_outside_stim, axis=0)
            for bhw, col in zip(on_col_inds, on_colnames):
                print(col)
                for i, ts in enumerate([on_during_stim,on_outside_stim]):
                    # For the on neurons in the above-defined subset of nrnact, get
                    # their timeseries from the arrays during_stim and outside_stim
                    durout_name = during_or_outside_names[i]
                    states = ts[col]

                    # Calculate the acorr plots for the active neurons' timeseries
                    # from the arrays during_stim and outside_stim
                    states  = detrend(states, axis=0)
                    acorr_states = np.correlate(states, states, mode='full')

                    # Fit a gabor curve to the acorr results to determine the
                    # freq
                    normed_acorr_states = acorr_states - np.mean(acorr_states)
                    normed_acorr_states = normed_acorr_states / \
                                            np.max(normed_acorr_states)

                    # Remove middle k timesteps because they're always 1 and
                    # very high
                    pre_despike_acorr_states = normed_acorr_states.copy()
                    # fillerval = np.mean(normed_acorr_states[
                    #     len(normed_acorr_states)//2-200:
                    #     len(normed_acorr_states)//2+200])
                    fillerval = np.max(normed_acorr_states[
                        len(normed_acorr_states)//2+200:
                        len(normed_acorr_states)//2+900])

                    normed_acorr_states[
                        len(normed_acorr_states)//2-100:
                        len(normed_acorr_states)//2+100] = fillerval

                    x = np.arange(start=-len(normed_acorr_states)/2,
                                  stop=len(normed_acorr_states)/2) # x values

                    p01 = [7e2, -2.1, 200.0]
                    p02 = [7e2, -2.5, 200.0]
                    p03 = [7e2, -2.9, 200.0]
                    p04 = [7e2, -3.2, 200.0]
                    init_params = [p01, p02, p03, p04]

                    #TESTING SPECTRAL DENSITY INSTEAD OF GABOR FITTING
                    freqs, psd = welch(
                        normed_acorr_states[len(normed_acorr_states)//2+100 :],
                        scaling='spectrum',
                        nperseg=2400)

                    plt.figure(figsize=(5, 4))
                    plt.semilogx(freqs, psd)
                    plt.title('PSD: power spectral density')
                    plt.xlabel('Frequency')
                    plt.ylabel('Power')
                    plt.tight_layout()
                    plt.savefig(os.path.join(exp_dir,
                                             "SPD ch%s col%s %s.png" % (
                                             ch, col, durout_name)))
                    plt.close()
                    ##################################################

                    gabor_func = lambda p: ((1 / (np.sqrt(2*np.pi) * p[0])) \
                                         * np.exp(-(x**2)/(2*p[0]**2)) * p[2] * \
                                         np.cos(2 * np.pi * (10**p[1]) * x))
                    opt_func = lambda p: gabor_func(p) - normed_acorr_states
                    # Run three optimization processes to see which initial
                    # parameters lead to the lowest cost
                    costs = []
                    est_params = []
                    for p0x in init_params:
                        opt_result = optimize.least_squares(opt_func, p0x)
                        est_params.append(opt_result.x)
                        costs.append(opt_result.cost)
                    print(np.argmin(costs))
                    cost = costs[np.argmin(costs)]
                    est_params = list(est_params[np.argmin(costs)])
                    est_params.append(cost) # add cost to list


                    plt.plot(x, pre_despike_acorr_states)
                    plt.plot(x, normed_acorr_states)
                    plt.plot(x, gabor_func(init_params[np.argmin(costs)]))

                    plt.plot(x, gabor_func(est_params))
                    plt.savefig(os.path.join(exp_dir,
                         "acorr ch%s col%s %s.png" % (ch, col, durout_name)))
                    plt.close()

                    b,h,w = bhw
                    for param_name, val in zip(param_names, est_params):
                        col_assign_name = param_name + '_' + durout_name
                        cond1 = nrnact['batch_idx'] == b
                        cond2 = nrnact['height']    == h
                        cond3 = nrnact['width']     == w
                        cond4 = nrnact['channel']   == ch
                        loc_index = nrnact.loc[cond1&cond2&cond3&cond4].index[0]
                        nrnact.loc[loc_index, col_assign_name] = val

                    # Save the full df periodically
                    nrnact.to_pickle(os.path.join(
                        self.base_analysis_results_dir,
                        'neuron act and osc.pkl'))

        # Save the full df one last time
        nrnact.to_pickle(os.path.join(
            self.base_analysis_results_dir, 'neuron act and osc.pkl'))
        print('burp')
        # Then move on to a new function plot_contrast_frequency_plots()





    def plot_contrast_frequency_plots(self):
        # Load df with freq and activity info


        # Load ori pref info


        # Add a column to the freq and activity df on whether the stim
        # presented during this batch was this channel's/neuron's preferred
        # orientation


        # Just check that there is the no unexpected relationship between
        # ori pref and oscillations (I want there to be more oscillating for
        # preferred orientations and less for non preferred)



        # Get the active neurons and their frequencies and get the sum for
        # each contrast level (may wish to normalise in some way)


        # Plot contrast on x, freq on y, ideally should see an increase


        # Then move on to synchrony code

        raise NotImplementedError


    def find_synchronous_neurons(self):
        #TODO first need to generate the right stimuli then run that experiment




        # load oscillating neuron df calculated in
        raise NotImplementedError





    def single_neuron_dynamics_plot_Ham_case(self):
        """The oscillatory dynamics of a randomly selected neuron in the HD
        networks. But also include activations of the momentum variables."""
        print("Plotting single neuron dynamics (Hamiltonian case)")
        model_exp_name = self.primary_model_exp_name
        var_names = ['energy', 'state', 'momenta', 'grad']
        var_label_pairs = [('energy_1', 'Energy'),
                           ('momenta_1', 'Momenta'),
                           ('state_1', 'State'),
                           ('grad_1', 'Gradient')]

        # Extract data from storage into DataManager
        dm = datamanager.DataManager(root_path=self.args.root_path,
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
            fig.savefig(self.session_dir + '/' + 'plot_%s.png' % label)
            fig.clf()  # Clears figure so plots aren't put on top of one another

        # Make pairs plot
        g = sns.PairGrid(df)
        g.map(sns.lineplot)
        g.savefig(self.session_dir + '/' + 'pairs plot.png')
        plt.close()
        # plot two plots, one for state/energy and the other for momenta


    def single_neuron_dynamics_plot_LD_case(self):
        """The oscillatory dynamics of a randomly selected neuron in the LD
        networks."""
        raise NotImplementedError()

    def single_neuron_autocorrelation(self):
        """The oscillatory dynamics of a randomly selected neuron in the HD
        networks. But also include activations of the momentum variables."""
        print("Plotting the autocorrelation of a single neuron")
        model_exp_name = self.primary_model_exp_name
        var_names = ['energy', 'state', 'momenta', 'grad']
        var_label_pairs = [('energy_1', 'Energy'),
                           ('momenta_1', 'Momenta'),
                           ('state_1', 'State'),
                           ('grad_1', 'Gradient')]

        # Extract data from storage into DataManager
        dm = datamanager.DataManager(root_path=self.args.root_path,
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
            fig.savefig(self.session_dir + '/' + 'plot_acf_%s.png' % label)
            fig.clf()

    def two_neuron_crosscorrelation(self):
        """The oscillatory dynamics of a randomly selected neuron in the HD
        networks. But also include activations of the momentum variables."""
        print("Plotting the cross correlation between two neurons")
        model_exp_name = self.primary_model_exp_name
        var_names = ['energy', 'state', 'momenta', 'grad']
        var_label_pairs = [('energy_1', 'Energy'),
                           ('momenta_1', 'Momenta'),
                           ('state_1', 'State'),
                           ('grad_1', 'Gradient')]

        # Extract data from storage into DataManager
        dm1 = datamanager.DataManager(root_path=self.args.root_path,
                                      model_exp_name=model_exp_name,
                                      var_names=var_names,
                                      state_layers=[1],
                                      batches=[0],#range(6,11),
                                      channels=[3],
                                      hw=[[14,14],[15,15]],
                                      timesteps=range(0, 6999))
        dm2 = datamanager.DataManager(root_path=self.args.root_path,
                                      model_exp_name=model_exp_name,
                                      var_names=var_names,
                                      state_layers=[1],
                                      batches=[0],#range(6,11),
                                      channels=[3],
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
                                           maxlags=2000)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))  # Uses sci notation for units
            plt.tight_layout()  # Stops y label being cut off
            fig = plot1.get_figure()
            fig.savefig(self.session_dir + '/' + 'plot_xcf3_%s.png' % label)
            fig.clf()

    def plot_energy_timeseries(self):
        """Plot the decrease in energy as time progresses. Should be faster
        in the HD case than in the LD case."""
        print("Plotting the decrease in energy over time for both HD and LD")

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
        hd_dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=hd_model_exp_name,
                                         var_names=var_names,
                                         state_layers=[1],
                                         batches=None,  # range(6,11),
                                         channels=None,
                                         hw=[[14,14],[15,15]],
                                         timesteps=range(1, 100))
        ld_dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=ld_model_exp_name,
                                         var_names=var_names,
                                         state_layers=[1],
                                         batches=None,
                                         channels=None,
                                         hw=[[14,14],[15,15]],
                                         timesteps=range(1, 100))

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
        fig.savefig(self.session_dir + '/' + 'plot_2%s.png' % 'HDLD_energy')

    def timeseries_EI_balance(self):
        """As in Fig 5 of Aitcheson and Lengyel (2016)."""
        print("Plotting the timeseries of the sum of the potential energy"+
              " and the kinetic energy to demonstrate EI balance")
        model_exp_name = self.primary_model_exp_name
        var_names = ['energy', 'state', 'momenta', 'grad']
        num_batches = 128 # i.e. num of trials
        dm = datamanager.DataManager(root_path=self.args.root_path,
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
        fig.savefig(self.session_dir + '/' + 'plot_%s vs %s.png' % (colnames[1], colnames[0]))
        fig.clf()  # Clears figure so plots aren't put on top of one another


    def xcorr_EI_lag(self):
        """As in Fig 5 of Aitcheson and Lengyel (2016).

        It's supposed to show that the excitation leads the inhibition slightly
        by showing a peak offset in the Xcorr plot. But """
        print("Plotting the cross correlation between energy and momentum")
        model_exp_name = self.primary_model_exp_name
        var_names = ['energy', 'state', 'momenta', 'grad']
        num_batches = 128 # i.e. num of trials
        dm = datamanager.DataManager(root_path=self.args.root_path,
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
        fig.savefig(self.session_dir + '/' + 'plot_xcf_avgmom_avgenergy.png')
        fig.clf()
