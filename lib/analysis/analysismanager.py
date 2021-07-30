import os
import numpy as np
import pandas as pd
import scipy.stats as spst
from scipy.signal import detrend, welch, find_peaks
from scipy.stats import shapiro
import math
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch
from scipy import optimize
import lib.analysis.datamanager as datamanager
import array2gif as a2g
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats


class AnalysisManager:
    def __init__(self, args, session_name):
        self.args = args

        # Define model name paths
        # TODO: Replace the path below with the path to whatever experimental
        #  data you wish to analyze.
        self.primary_model = "20200929-025034__rndidx_55061_loaded20200922-152859__rndidx_22892_at_47500_its"
        self.just_angles_model = self.primary_model
        self.double_stim_synchr_model = self.primary_model
        self.long_just_angles_model = self.primary_model
        self.just_angles_few_angles_model = self.primary_model

        # Define paths for individual experiments (each with diff stims)
        single_exp_stem = '/orientations_present_single_gabor'
        self.primary_model_exp_name = self.primary_model + single_exp_stem + '_contrast_and_angle'
        self.just_angles_exp_name = self.just_angles_model + single_exp_stem + '_just_angle'
        self.just_angles_few_angles_exp_name = self.just_angles_few_angles_model + single_exp_stem + '_just_angle_few_angles'
        self.long_just_angles_exp_name = self.long_just_angles_model + single_exp_stem + '_long_just_fewangles'

        self.session_name = self.primary_model  # session_name

        double_exp_stem = '/orientations_present_double_gabor'
        self.double_stim_synchr_exp_name = self.double_stim_synchr_model + double_exp_stem + '_fewlocs_and_fewerangles'

        self.exp_names_dct = {'primary': self.primary_model_exp_name,
                              'just_angle': self.just_angles_exp_name,
                              'just_angle_few_angles': self.just_angles_few_angles_exp_name,
                              'long_just_fewangles': self.long_just_angles_exp_name,
                              'double_stim': self.double_stim_synchr_exp_name}

        self.st_var_name = ['state']

        # Make base analysis results dir if it doesn't exist
        if not os.path.isdir('analysis_results'):
            os.mkdir('analysis_results')

        # Make dir for this specific analysis session
        if self.args.ehd_results_storage:
            self.base_analysis_results_dir = \
                os.path.join(self.args.root_path, 'analysis_results')
            if not os.path.isdir(self.base_analysis_results_dir):
                os.mkdir(self.base_analysis_results_dir)
        else:
            self.base_analysis_results_dir = 'analysis_results'

        self.session_dir = os.path.join(self.base_analysis_results_dir,
                                        self.session_name)
        if not os.path.isdir(self.session_dir):
            os.mkdir(self.session_dir)

        # Set some general experiment variables
        self.burn_in_len = 100
        self.exp_stim_phase_lens = [self.burn_in_len, 1000, 1000, 600]
        self.full_trace_len = sum(self.exp_stim_phase_lens)

        self.primary_stim_start = 1100
        self.primary_stim_stop  = 2100
        self.autocorr_phase_len = 1000 - 1

        self.num_ch = 32
        self.num_batches = 128

        # Extent of the architecture's horizontal connections in statelayer1
        self.double_stim_horizextent = 6  # = 11//2 + 1

        # Dimensions of the extracted SL1 variables used for analysis
        self.extracted_im_size = 32
        self.top_left_pnt = [0,0]
        self.bottom_right_pnt = [p+self.extracted_im_size
                              for p in self.top_left_pnt]

        # The size of the central patch that is measured during analysis when
        # more than just the central neuron is being aggregated
        self.central_patch_min = 15
        self.central_patch_max = 17
        self.central_patch_size = 3 * 3


        # Define the contrast and angles (and their pairs) that were used in
        # to generate the stimuli in each experiment

        ## (Primary) contrast & angle exp
        self.contrast_and_angle_angles = \
            [0.0, 0.393, 0.785, 1.178, 1.571, 1.963, 2.356, 2.749,
             3.142, 3.534, 3.927, 4.32, 4.712, 5.105, 5.498, 5.89]
        self.contrast_and_angle_contrasts = \
            [0., 0.4, 0.8, 1.2, 1.6, 2., 2.4, 2.8]
        self.contrast_and_angle_angle_contrast_pairs = []
        for a in self.contrast_and_angle_angles:
            for c in self.contrast_and_angle_contrasts:
                self.contrast_and_angle_angle_contrast_pairs.append((a, c))

        ## Just angle exp
        self.just_angle_angles = np.linspace(start=0.0,
                                             stop=np.pi * 2,
                                             num=128)
        self.just_angle_contrasts = [2.4]
        self.just_angle_angle_contrast_pairs = []
        for a in self.just_angle_angles:
            for c in self.just_angle_contrasts:
                self.just_angle_angle_contrast_pairs.append((a, c))

        ## Just angles Few angles exp
        angles = np.arange(0, 2*np.pi, (np.pi * 2) / 8)
        angles = list(angles)
        self.just_angles_few_angles_batchgroupangles = angles.copy()

        angles = angles * 16
        angles = sorted(angles)
        self.just_angles_few_angles_angles = angles.copy()

        self.just_angles_few_angles_batchgroupstarts = \
            [i for i in range(128) if (i % 16)==0]
        self.just_angles_few_angles_batchgroups = \
            [list(range(bg_start,bg_start+16)) for bg_start in \
             self.just_angles_few_angles_batchgroupstarts]


        self.just_angles_few_angles_contrasts = [2.4]
        self.just_angles_few_angles_angle_contrast_pairs = []
        for a in self.just_angles_few_angles_angles:
            for c in self.just_angles_few_angles_contrasts:
                self.just_angles_few_angles_angle_contrast_pairs.append((a, c))


        ## Long just angle few angles exp
        self.long_just_angles_few_angles_batchgroupangles = self.just_angles_few_angles_batchgroupangles.copy()
        self.long_just_angle_angles = self.just_angles_few_angles_angles.copy()
        self.long_just_angle_contrasts = [2.4]
        self.long_just_angle_angle_contrast_pairs = []
        for a in self.long_just_angle_angles:
            for c in self.long_just_angle_contrasts:
                self.long_just_angle_angle_contrast_pairs.append((a, c))

        locs = list(range(1, 32, 2))
        locs.extend(list(range(16-7, 16 + 7)))
        locs = list(set(sorted(locs)))
        self.rclocs = {}
        rclocs_x = [[16, x] for x in locs]
        rclocs_y = [[y, 16] for y in locs]
        angles = self.long_just_angles_few_angles_batchgroupangles #for brevity
        rclocs_keys = [angle for angle in angles]
        for key in rclocs_keys: #initialise dict
            self.rclocs[key] = None

        self.rclocs[angles[0]] = [[16, x] for x in locs]
        self.rclocs[angles[1]] = [[y, x] for x, y in zip(locs, reversed(locs))]
        self.rclocs[angles[2]] = [[y, 16] for y in locs]
        self.rclocs[angles[3]] = [[y, x] for x, y in zip(locs, locs)]
        self.rclocs[angles[4]] = [[16, x] for x in locs]
        self.rclocs[angles[5]] = [[y, x] for x, y in zip(reversed(locs), locs)]
        self.rclocs[angles[6]] = [[y, 16] for y in locs]
        self.rclocs[angles[7]] = [[y, x] for x, y in zip(reversed(locs), reversed(locs))]


        ## Double stim exp
        ## ensure these settings are the same as in
        # managers.ExperimentalStimuliGenerationManager.\
        # generate_double_gabor_dataset__fewlocs_and_fewerangles

        static_angles = np.arange(0, 2*np.pi, np.pi/2)
        static_angles = list(static_angles) * 32
        static_angles = sorted(static_angles)
        static_loc_x = 0
        static_loc_y = 0


        mobile_dists = [3,5,7,10]
        colin_locs_high = [0+x for x in mobile_dists]
        colin_locs_low  = [0-x for x in mobile_dists]
        angles_starts = np.arange(0, 2*np.pi, np.pi/2) # the four 90degr angles
        angles_starts = list(angles_starts)

        angles = []
        for angst in angles_starts:
            pair = [angst]*4
            pair.extend([angst + np.pi/2]*4)
            pair = pair * 4
            angles.extend(pair)

        i = 0
        locs = []
        new_locs = [[[static_loc_x, l]]*8 for l in colin_locs_high]
        for nl in new_locs:
            locs.extend(nl)
        i += 1
        new_locs = [[[l, static_loc_y]]*8 for l in colin_locs_low]
        for nl in new_locs:
            locs.extend(nl)
        i += 1
        new_locs = [[[static_loc_x, l]]*8 for l in colin_locs_low]
        for nl in new_locs:
            locs.extend(nl)
        i += 1
        new_locs = [[[l, static_loc_y]]*8 for l in colin_locs_high]
        for nl in new_locs:
            locs.extend(nl)

        self.double_stim_static_x = 16
        self.double_stim_static_y = 16
        self.double_stim_static_x_0centre = self.double_stim_static_x - \
                                            self.extracted_im_size//2
        self.double_stim_static_y_0centre = self.double_stim_static_y - \
                                            self.extracted_im_size//2
        self.double_stim_contrasts = [2.4]
        self.double_stim_static_angles = static_angles
        self.double_stim_angles = angles
        self.double_stim_batchgroupangles = [a for (i, a) in enumerate(angles) if i%4==0]
        self.double_stim_batchgroupangles_static = [self.double_stim_static_angles[i] for i in range(128) if (i%4)==0]

        self.double_stim_locs = locs
        self.double_stim_batchgroupstarts = \
            [i for i in range(128) if (i % 4)==0]
        self.double_stim_batchgroups = \
            [list(range(bg_start,bg_start+4)) for bg_start in \
             self.double_stim_batchgroupstarts]

        self.double_stim_batchgroup_locs = [self.double_stim_locs[i] for i in self.double_stim_batchgroupstarts]


        # General
        self.get_contrasts_from_images()
        self.specgram_nfft = 256
        self.acorr_mid_len = 50  # was 150
        self.acorr_x = np.arange(start=-self.acorr_mid_len,
                                 stop=self.acorr_mid_len) # x values

    def timesteps_to_ms(self, ts, round=100):
        ms = ts * (70. / 17.)
        ms = np.round(ms, round)
        return ms

    def ms_to_timesteps(self, ms, round=100):
        ts = ms * (17. / 70.)
        ts = np.round(ts, round)
        return ts


    def get_contrasts_from_images(self):
        print("Calculating true contrasts from images")
        exp_type = 'primary'
        true_contrasts = []

        for exp_type in ['just_angles', 'primary']:
            if exp_type == 'just_angles':
                exp_stim_path = 'data/gabor_filters/single/just_angle'
            elif exp_type == 'primary':
                exp_stim_path = 'data/gabor_filters/single/contrast_and_angle'

            # Get the experimental images

            (_, _, filenames) = next(os.walk(exp_stim_path))
            filenames = sorted(filenames)
            exp_stims = []
            for flnm in filenames:
                im = Image.open(os.path.join(exp_stim_path, flnm))
                im = transforms.functional.to_tensor(im)
                exp_stims.append(im)
            exp_stims = torch.stack(exp_stims) # should have generated only
            # 128 images

            # Calculate true contrasts
            pixel_intensities = exp_stims.mean(dim=1)
            max_intens = pixel_intensities.max(dim=1)[0].max(dim=1)[0]
            min_intens = pixel_intensities.min(dim=1)[0].min(dim=1)[0]
            true_contrast = (max_intens - min_intens)/(max_intens + min_intens)
            true_contrasts.append(true_contrast)

        self.true_just_angle_contrasts = \
            list(np.array(true_contrasts[0][:8]))
        self.true_contrast_and_angle_contrasts = \
            list(np.array(true_contrasts[1][:8]))


    def find_active_neurons(self, exp_type='primary'):
        print("Finding active neurons")

        if exp_type == 'just_angles': #TODO convert this into function and use elsewhere
            model_exp_name = self.just_angles_exp_name
            angles = self.just_angle_angles
            contrasts = self.just_angle_contrasts
            angle_contrast_pairs = self.just_angle_angle_contrast_pairs
        elif exp_type == 'primary':
            model_exp_name = self.primary_model_exp_name
            angles = self.contrast_and_angle_angles
            contrasts = self.contrast_and_angle_contrasts
            angle_contrast_pairs = self.contrast_and_angle_angle_contrast_pairs
        elif exp_type == 'just_angles_few_angles':
            model_exp_name = self.just_angles_few_angles_exp_name
            angles = self.just_angles_few_angles_angles
            contrasts = self.just_angles_few_angles_contrasts
            angle_contrast_pairs = \
                self.just_angles_few_angles_angle_contrast_pairs
        elif exp_type == 'long_just_fewangles':
            model_exp_name = self.long_just_angles_exp_name
            angles = self.long_just_angle_angles
            contrasts = self.long_just_angle_contrasts
            angle_contrast_pairs = self.long_just_angle_angle_contrast_pairs
        elif exp_type == 'double_stim':
            model_exp_name = self.double_stim_synchr_exp_name
            angles = self.double_stim_angles
            contrasts = self.contrast_and_angle_contrasts
            angle_contrast_pairs = self.contrast_and_angle_angle_contrast_pairs

        # Prepare variables used for processing data
        activity_df_created = False

        results_colnames = ['batch_idx', 'height', 'width', 'active',
                            'channel', 'mean_act_during', 'mean_act_outside',
                            'ttest_p_value',
                            'shapiro_W_during','shapiro_W_outside',
                            'shapiro_p_during', 'shapiro_p_outside']
        results_df_2 = pd.DataFrame(columns=results_colnames)


        for ch in range(self.num_ch):

            print("Channel %s" % str(ch))
            # Gets data
            dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=model_exp_name,
                                         var_names=self.st_var_name,
                                         state_layers=[1],
                                         batches=None,
                                         channels=[ch],
                                         hw=[self.top_left_pnt,
                                             self.bottom_right_pnt],
                                         timesteps=None)
            # Processes data
            activities_df, arr_sh, colnames, inds = \
                self.convert_ch_data_to_activity_df(dm)

            # Separate data during stim from data when stim isn't present
            during_stim = activities_df[
                          self.primary_stim_start:self.primary_stim_stop]
            outside_stim = activities_df.drop(activities_df.index[
                          self.primary_stim_start:self.primary_stim_stop])
            outside_stim = outside_stim.drop(activities_df.index[
                          0:self.burn_in_len])

            # Test for normality (Shapiro Wilk test). If p<0.05, then rejecc
            # hypothesis that the data are normally distributed
            during_cols  = during_stim.columns[1:]
            #[1:] slice to remove timesteps

            outside_cols = outside_stim.columns[1:]
            shap_test_during = [shapiro(during_stim[col])
                                for col in during_cols]
            shap_test_outside = [shapiro(outside_stim[col])
                                for col in outside_cols]
            shap_W_during, shap_p_during = zip(*shap_test_during)
            shap_W_outside, shap_p_outside = zip(*shap_test_outside)

            shap_W_outside = np.array(shap_W_outside)
            shap_W_during  = np.array(shap_W_during)
            shap_p_outside = np.array(shap_p_outside)
            shap_p_during  = np.array(shap_p_during)

            # Test whether mean during stim is signif different than outside
            # of stim and if it is higher, then it is considered 'active'
            ttest_result = spst.ttest_ind(a=during_stim,
                                          b=outside_stim,
                                          equal_var=False, #Therefore uses
                                          # Welch's t-test, not Student's
                                          axis=0)

            # Determine when states are significantly higher during stim
            mean_act_during = during_stim.mean(axis=0)
            mean_act_outside = outside_stim.mean(axis=0)
            mean_diffs = mean_act_during - mean_act_outside
            mean_higher = mean_diffs > 0
            t_results = ttest_result.pvalue < 0.05
            comb_results = t_results & mean_higher
            comb_results = pd.DataFrame(comb_results).T
            comb_results.columns = colnames

            # Make a nicer df # TODO clean out the old df and places that use it
            results_df_2['batch_idx'] = inds[0]
            results_df_2['height']    = inds[1] + self.top_left_pnt[0]
            results_df_2['width']     = inds[2] + self.top_left_pnt[1]
            results_df_2['channel']   = np.array([ch]*len(inds[0]))
            results_df_2['active'] = np.array(comb_results[comb_results.columns[1:]]).squeeze()
            results_df_2['mean_act_during']  = np.array(mean_act_during[1:])
            results_df_2['mean_act_outside'] = np.array(mean_act_outside[1:])
            results_df_2['ttest_p_value'] = ttest_result.pvalue[1:]
            results_df_2['shapiro_W_during']  = shap_W_during
            results_df_2['shapiro_W_outside'] = shap_W_outside
            results_df_2['shapiro_p_during']  = shap_p_during
            results_df_2['shapiro_p_outside'] = shap_p_outside

            # Save the results of activity tests
            if not activity_df_created:
                activity_df = pd.DataFrame(data=comb_results,
                                           columns=comb_results.columns)
                full_results_df = pd.DataFrame(data=results_df_2.copy(),
                                               columns=results_colnames)
                activity_df_created = True
            else:
                activity_df = activity_df.append(comb_results,
                                                 ignore_index=True)
                full_results_df = full_results_df.append(results_df_2.copy(),
                                                         ignore_index=True)
        print("Done finding active neurons")

        # Finish adding b, ch, h, w labels
        activity_df = activity_df.T
        activity_df = activity_df.drop('Timestep')
        activity_df['batch_idx'] = inds[0]
        activity_df['height'] = [self.top_left_pnt[0] + j for j in inds[1]]
        activity_df['width'] = [self.top_left_pnt[1] + j for j in inds[2]]
        activity_df.to_pickle(
            os.path.join(self.session_dir,
            'neuron_activity_results_%s.pkl') % exp_type)

        full_results_df.to_pickle(
            os.path.join(self.session_dir,
            'neuron_activity_results_alternativeformat_%s.pkl') % exp_type)



    def print_activity_maps_by_batch_ch(self, exp_name='primary'):

        # Make dir to save plots for this experiment
        maps_dir, exp_dir = self.prepare_dirs(self.session_dir,
                                               'activity maps',
                                               exp_name)

        # Load data and prepare variables
        pixel_inds = [(i, j) for i in list(range(self.extracted_im_size))
                      for j in range(self.extracted_im_size)]
        full_data = pd.read_pickle(os.path.join(self.session_dir,
            'neuron_activity_results_alternativeformat_%s.pkl' % exp_name))

        for b in range(self.num_batches):
            for ch in range(self.num_ch):
                print("Batch %s  ; Channel %s" % (b, ch))
                im = np.zeros([self.extracted_im_size, self.extracted_im_size])
                pch = full_data['channel'] == ch
                pb = full_data['batch_idx'] == b
                if self.extracted_im_size == 32:
                    cond = pch & pb
                    avg_activity = full_data.loc[cond]['mean_act_during'] - \
                                   full_data.loc[cond]['mean_act_outside']
                    avg_activity = np.array(avg_activity)
                    im = avg_activity.reshape((32,32))
                else:
                    for (i, j) in pixel_inds:
                        pi = full_data['height'] == i + self.top_left_pnt[0]
                        pj = full_data['width'] == j + self.top_left_pnt[1]

                        cond = pi & pj & pch & pb
                        avg_activity = \
                            full_data.loc[cond]['mean_act_during'] - \
                            full_data.loc[cond]['mean_act_outside']
                        im[i][j] = np.array(avg_activity)

                if exp_name=='primary':
                    strong_stim_batches = pb % 8 == 0
                    maxmin_conds = pch & strong_stim_batches
                else:
                    maxmin_conds = pch

                means = np.array(
                    [full_data.loc[maxmin_conds]['mean_act_during'],
                     full_data.loc[maxmin_conds]['mean_act_outside']])
                max_mean = np.max(means)
                min_mean = np.min(means)


                plt.imshow(im, vmax=max_mean, vmin=min_mean)
                plt.colorbar()
                plt.savefig(
                    os.path.join(exp_dir,
                                 "raw ch_b neuron actv locs ch%i_b%i.png" % (
                                 ch, b)))
                plt.close()

    def print_activity_maps_by_batch_ch_double_stim(self):

        # Make dir to save plots for this experiment
        maps_dir, exp_dir = self.prepare_dirs(self.session_dir,
                                               'activity maps',
                                               'double_stim')

        # Load data and prepare variables
        pixel_inds = [(i, j) for i in list(range(self.extracted_im_size))
                      for j in range(self.extracted_im_size)]
        full_data = pd.read_pickle(os.path.join(self.session_dir,
            'neuron_activity_results_alternativeformat_double_stim.pkl'))

        for b in range(self.num_batches):  #TODO comment the below code
            for ch in range(self.num_ch):
                print("Batch %s  ; Channel %s" % (b, ch))
                im = np.zeros([self.extracted_im_size, self.extracted_im_size])
                pch = full_data['channel'] == ch
                pb = full_data['batch_idx'] == b
                if self.extracted_im_size == 32:
                    cond = pch & pb
                    avg_activity = full_data.loc[cond]['mean_act_during'] - \
                                   full_data.loc[cond]['mean_act_outside']
                    avg_activity = np.array(avg_activity)
                    im = avg_activity.reshape((32,32))
                else:
                    for (i, j) in pixel_inds:
                        pi = full_data['height'] == i + self.top_left_pnt[0]
                        pj = full_data['width'] == j + self.top_left_pnt[1]

                        cond = pi & pj & pch & pb
                        avg_activity = \
                            full_data.loc[cond]['mean_act_during'] - \
                            full_data.loc[cond]['mean_act_outside']
                        im[i][j] = np.array(avg_activity)

                means = np.array(
                    [full_data.loc[pch]['mean_act_during'],
                     full_data.loc[pch]['mean_act_outside']])
                max_mean = np.max(means)
                min_mean = np.min(means)

                plt.imshow(im, vmax=max_mean, vmin=min_mean)
                plt.colorbar()
                plt.savefig(
                    os.path.join(exp_dir,
                                 "raw ch_b neuron actv locs ch%i_b%i.png" % (
                                 ch, b)))
                plt.close()


    def print_activity_map_GIFs_by_batch_ch(self, exp_name='just_angle'):

        # Make dir to save plots for this experiment
        maps_dir, exp_dir = self.prepare_dirs(self.session_dir,
                                               'activity maps gifs',
                                               exp_name)

        # Set general variables
        model_exp_name = self.exp_names_dct[exp_name]
        pixel_inds = [(i, j) for i in list(range(self.extracted_im_size))
                      for j in range(self.extracted_im_size)]
        norm = plt.Normalize()
        batch_eles = {'just_angle': [0, 32],
                      'long_just_fewangles': [0,65]}

        for ch in range(self.num_ch):
            print("Channel %s" % ch)

            # Get data
            dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=model_exp_name,
                                         var_names=self.st_var_name,
                                         state_layers=[1],
                                         batches=None,
                                         channels=[ch],
                                         hw=[self.top_left_pnt,
                                             self.bottom_right_pnt],
                                         timesteps=None)

            # Process data
            dm.data['state_1'] = dm.data[
                'state_1'].squeeze()  # get rid of ch dim
            arr_sh = dm.data['state_1'].shape[1:]
            dm.data['state_1'] = dm.data['state_1'].transpose()

            for b in batch_eles[exp_name]:

                print("Batch element %i" % b)
                # Get data and normalize it
                traces = dm.data['state_1'][:,:,b,:]
                traces = (traces - np.min(traces))
                traces = traces / np.max(traces)
                traces = (traces * 255).astype(int)

                # Add the three color channels and the signifier frames
                signlen = 10
                ones = 1
                traces[0:11,1, 0:signlen*2] = ones
                traces[0:21,1, self.primary_stim_start-signlen:self.primary_stim_start] = ones
                traces[0:31,1, self.primary_stim_stop-signlen:self.primary_stim_stop]   = ones

                intervals = np.linspace(start=0,
                                        stop=self.full_trace_len,
                                        num=self.full_trace_len)
                intervals = [int((itv/self.full_trace_len)*32)
                             for itv in intervals]

                # GIF Progress bar
                for j, itv in enumerate(intervals):
                    traces[0:itv,0,j] = 1

                traces = np.repeat(traces[np.newaxis, :, :, :], 3, axis=0)
                traces = [traces[:,:,:,i] for i in range(traces.shape[-1])]
                title = os.path.join(exp_dir,
                                 "movie %s locs ch%i_b%i.gif" % (
                                 self.st_var_name[0], ch, b))

                a2g.write_gif(traces, title, fps=75)

    def find_orientation_preferences(self):
        print("Finding orientation preferences of neurons")


        est_param_names = ['amplitude', 'phaseshift', 'mean', 'dir_or_ori']


        model_exp_name = self.just_angles_exp_name
        var_label_pairs = [('state_1', 'State')]

        angles = self.just_angle_angles
        contrasts = self.just_angle_contrasts
        angle_contrast_pairs = self.just_angle_angle_contrast_pairs

        activity_df = pd.read_pickle(os.path.join(self.session_dir,
                      'neuron_activity_results_alternativeformat_just_angles.pkl'))
        print(activity_df.columns)
        activity_df = activity_df.drop(columns=['ttest_p_value',
           'shapiro_W_during', 'shapiro_W_outside', 'shapiro_p_during',
           'shapiro_p_outside'])

        # Create a new dataframe that sums over h and w (so we only have batch
        # info per channel)
        sum_cols = list(activity_df.columns)
        patch_activity_df = pd.DataFrame(columns=sum_cols)

        centre1 = 16
        centre2 = 16

        # Get mean_activities for central patch for each channel in each batch element.
        # Each batch element corresponds to a particular (angle, contrast) pair
        for b in range(self.num_batches):
            print(b)
            cond_b = activity_df['batch_idx']==b
            cond_x = (activity_df['width']>=centre1)&(activity_df['width']<=centre2)
            cond_y = (activity_df['height']>=centre1)&(activity_df['height']<=centre2)
            cond = cond_b & cond_x & cond_y
            batch_df = activity_df.loc[cond]
            batch_df['angle']    = angle_contrast_pairs[b][0]
            patch_activity_df = patch_activity_df.append(batch_df,
                                                         ignore_index=True)

        # Get sum for each channel for each angle
        sum_angles_df = patch_activity_df.groupby(['channel', 'angle'],
                                                  as_index=False).sum()

        # Plot the orientation preference plots
        fig, ax = plt.subplots(8,4, sharey=True, sharex=True)
        fig.set_size_inches(12,16.5)
        k = 0
        angle_axis = [round((180/np.pi) * angle, 1) for angle in angles]
        orient_prefs = pd.DataFrame(columns=est_param_names)
        for i in range(8):
            for j in range(4):
                print("%i, %i" % (i,j))

                # Normalise the data
                normed_data = sum_angles_df.loc[
                    sum_angles_df['channel']==k]['mean_act_during']
                normed_data = normed_data - np.mean(normed_data)
                normed_data = normed_data / \
                              np.linalg.norm(normed_data)

                # Fit sine functions to find direction or orientation
                # selectivity
                amplitude0 = 2.0
                phaseshift0 = 0
                mean0 = 0.
                params0 = [amplitude0, phaseshift0, mean0]
                opt_func1 = lambda x: x[0] * np.cos(angles + x[1]) + x[
                    2] - normed_data
                opt_func2 = lambda x: x[0] * np.cos(2 * angles + x[1]) + x[
                    2] - normed_data
                opt_funcs = [opt_func1, opt_func2]
                est_params = []
                costs = []
                for opt_func in opt_funcs:
                    result = \
                        optimize.least_squares(opt_func, params0)

                    costs.append(result['cost'])

                    est_params.append(result['x'])

                dir_or_ori = np.argmin(costs)
                est_params = est_params[dir_or_ori]

                # Get the right curve function to fit the lowest cost params
                fitted_curve_func1 = lambda ep: ep[0] * np.cos(
                    angles + ep[1]) + ep[2]
                fitted_curve_func2 = lambda ep: ep[0] * np.cos(
                    2 * angles + ep[1]) + ep[2]
                fitted_curve_funcs = [fitted_curve_func1, fitted_curve_func2]
                fitted_curve_func = fitted_curve_funcs[dir_or_ori]
                fitted_curve = fitted_curve_func(est_params)

                # Find the orientation selectivity near the sine peaks
                sine_peak = np.argmax(fitted_curve)
                elongated_nd = np.array([0])
                elongated_nd = np.append(elongated_nd, normed_data)
                elongated_nd = np.append(elongated_nd, np.array([0])) #in order to find edge peaks

                trace_peaks = find_peaks(elongated_nd,
                                         prominence=np.max(normed_data)/9,
                                         distance=int(128/4))[0]
                trace_peaks = trace_peaks - 1

                # Calc dist from trace peaks to sine peak
                dist_to_sn_pk0 = np.abs(trace_peaks - sine_peak)
                dist_to_sn_pk2pi = np.abs(dist_to_sn_pk0 - len(normed_data))
                dist_to_sn_pk = np.array([dist_to_sn_pk0, dist_to_sn_pk2pi])
                dist_to_sn_pk = np.min(dist_to_sn_pk, axis=0)
                min_dist_peak_ind = np.argmin(dist_to_sn_pk)
                chosen_peaks = trace_peaks[min_dist_peak_ind]


                # Plot
                ax[i,j].plot(angle_axis, normed_data)
                ax[i,j].scatter(np.array(angle_axis)[chosen_peaks],
                                np.array(normed_data)[chosen_peaks], c='red')
                ax[i,j].text(0.08, 0.19, 'Channel %s' % k)

                if not all(normed_data.isna()):  # because nans
                    ax[i,j].plot(angle_axis, fitted_curve, c='red')


                if i==7: # Put labels on the edge plots only
                    ax[i, j].set_xlabel('Angles (degree)')
                    ax[i, j].set_xticks([0,90,180,270])
                if j==0:
                    ax[i,j].set_ylabel('Normed activity [a.u.]')

                # Log orientation pref
                est_params[1] = np.array(angles)[chosen_peaks] #phaseshift
                est_params = np.append(est_params, dir_or_ori)
                orient_prefs.loc[len(orient_prefs)] = est_params
                k += 1

        orient_prefs['channel'] = np.array(orient_prefs.index)

        plt.savefig(os.path.join(
            self.session_dir, 'orientation_prefs.png'))
        plt.close()
        print(orient_prefs)


        # Plot a grid of circular plots with a circle for each channel
        # Plot the orientation preference plots
        fig, ax = plt.subplots(8, 4, sharey=True, sharex=True,
                               subplot_kw=dict(projection='polar'))
        fig.set_size_inches(8,16.5)
        k = 0
        angle_axis = [round((180 / np.pi) * angle, 1) for angle in angles]
        for i in range(8):
            for j in range(4):
                print("%i, %i" % (i, j))
                normed_data = sum_angles_df.loc[
                    sum_angles_df['channel']==k]['mean_act_during']
                normed_data = normed_data - np.mean(normed_data)
                normed_data = normed_data / \
                              np.linalg.norm(normed_data)

                amplitude0 = 2.0
                phaseshift0 = 0
                mean0 = 0.
                params0 = [amplitude0, phaseshift0, mean0]
                opt_func1 = lambda x: x[0] * np.cos(angles + x[1]) + x[
                    2] - normed_data
                opt_func2 = lambda x: x[0] * np.cos(2 * angles + x[1]) + x[
                    2] - normed_data
                opt_funcs = [opt_func1, opt_func2]
                est_params = []
                costs = []
                for opt_func in opt_funcs:
                    result = \
                        optimize.least_squares(opt_func, params0)
                    costs.append(result['cost'])
                    est_params.append(result['x'])
                dir_or_ori = np.argmin(costs)
                est_params = est_params[dir_or_ori]

                # Get the right curve function to fit the lowest cost params
                fitted_curve_func1 = lambda ep: ep[0] * np.cos(
                    angles + ep[1]) + ep[2]
                fitted_curve_func2 = lambda ep: ep[0] * np.cos(
                    2 * angles + ep[1]) + ep[2]
                fitted_curve_funcs = [fitted_curve_func1, fitted_curve_func2]
                fitted_curve_func = fitted_curve_funcs[dir_or_ori]
                fitted_curve = fitted_curve_func(est_params)

                ax[i, j].plot(angles, normed_data)
                ax[i, j].text(np.pi * 0.73, 0.39, 'Channel %s' % k)


                if not all(normed_data.isna()):  # because nans
                    ax[i, j].plot(angles, fitted_curve, c='red')
                ax[i, j].set_xticklabels([])
                ax[i, j].set_yticklabels([])

                k += 1
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.session_dir, 'orientation_prefs_circles.png'))
        plt.close()

        threshold = 0.04  # TODO do nested F?t? test to determine this

        # Plot that shows a the spread of orientations around the unit circle
        t = np.linspace(0, np.pi * 2, 100)
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(projection='polar')
        ax.set_yticklabels([])
        degree_symb = u"\u00b0"
        plt.tight_layout()
        plt.plot(t, np.ones_like(t) * threshold, linewidth=1,
                 color='r')

        # Plot lines in between the points
        thetas = orient_prefs['phaseshift']
        ampls = orient_prefs['amplitude']
        for i, (amp, ang) in enumerate(zip(ampls, thetas)):
            print(i, amp)

            ang_pi = ang + np.pi
            if ang > 0.:
                angles = [ang, ang_pi]
                lengths = [np.abs(amp), 0.]
                main_angle = ang
            else:
                angles = [ang_pi, ang]
                lengths = [np.abs(amp), 0.]
                main_angle = ang_pi

            if np.abs(amp) > threshold:
                plt.plot(np.array(angles),
                         np.array(lengths), 'b-')
                plt.plot(main_angle, np.abs(amp), 'bo')
            else:
                plt.plot(np.array(angles),
                         np.array(lengths), 'r-')
                plt.plot(main_angle, np.abs(amp), 'ro')
            ax.annotate(str(i),
                        xy=(main_angle, 1.05 * np.abs(ampls[i])), # theta, radius
                        xytext=(main_angle, 0.),  # fraction, fraction
                        textcoords='offset pixels',
                        horizontalalignment='center',
                        verticalalignment='center',
                        )
        ax.set_xticks([0, np.pi/2, np.pi, np.pi*1.5])
        ax.set_xticklabels(['0'+degree_symb,
                            '90'+degree_symb,
                            '180'+degree_symb,
                            '270'+degree_symb])
        plt.tight_layout()
        plt.savefig(os.path.join(self.session_dir,
                                 'orient_prefs_circle2.png'))
        plt.close()


        # Save the results
        exp_type = '_just_angles'
        orient_prefs.to_pickle(os.path.join(
            self.session_dir, 'orientation_pref_results%s.pkl' % exp_type))
        patch_activity_df.to_pickle(os.path.join(
            self.session_dir, 'patch_activity_results%s.pkl'   % exp_type))
        activity_df.to_pickle(os.path.join(
            self.session_dir, 'neuron_activity_results%s.pkl'  % exp_type))

    def rotate_est_params(self, est_params):
        # if est_params[0] < 0.:
        #     est_params[0] = est_params[0] * -1
        #     est_params[1] += np.pi
        # if est_params[1] < 0.:
        #     est_params[1] += 2 * np.pi
        if est_params[1] > np.pi:
            est_params[1] -= np.pi
        return est_params


    def assign_ori_info(self, exp_name='primary'): #TODO consider doing this for all the exps
        # Prepare general variables
        angle_contrast_pairs = self.contrast_and_angle_angle_contrast_pairs

        # Load info on which neurons are active
        print("Loading priorly processed data...")
        nrnact = pd.read_pickle(
            os.path.join(self.session_dir,
                         'neuron_activity_results_primary.pkl'))

        # Load ori pref info
        ori_pref = pd.read_pickle(
            os.path.join(self.session_dir,
                         'orientation_pref_results_just_angles.pkl'))
        print("Done loading presaved data.")

        # Rearrange activity info so that we can identify channels and
        # pixels that are active for a given stim
        nrnact = pd.melt(nrnact, id_vars=['batch_idx', 'height', 'width'],
                         value_vars=list(range(32)))
        nrnact = nrnact.rename(columns={'variable': 'channel',
                                        'value': 'active'})

        # Decide which channels qualify as having an orientation preference
        ori_amp_threshold = 0.04
        ori_dict = ori_pref['phaseshift'].loc[
            np.abs(ori_pref['amplitude']) > ori_amp_threshold]  # not yet a dict
        ori_dict_filt = ori_pref['phaseshift'].loc[
            np.abs(ori_pref['amplitude']) < ori_amp_threshold]
        ori_dict_filt.loc[:] = np.nan

        ori_dict_filt = ori_dict_filt.to_dict()  # becomes a dict
        ori_dict = ori_dict.to_dict()
        ori_dict = {**ori_dict_filt, **ori_dict}

        k = list(ori_dict.keys())
        v = list(ori_dict.values())
        ori_dict = [x for _, x in sorted(zip(k, v))]  # becomes list
        ori_prefs = [ori_dict[ch] for ch in list(nrnact['channel'])]

        # Decide which neurons in each channel and batch are being presented
        # orientation-matched stimuli
        stim_angle_contrasts = [angle_contrast_pairs[b] for b in
                                nrnact['batch_idx']]
        stim_angles = [sac[0] for sac in stim_angle_contrasts]
        stim_contrasts = [sac[1] for sac in stim_angle_contrasts]
        ori_diff_threshhold = 10 * np.pi / 180

        # Old code for when using orientation selectivity
        # orientation_diffs_0 = [np.abs(stim_angle - ori_pref) for
        #                        (stim_angle, ori_pref) in
        #                        zip(stim_angles, ori_prefs)] #TODO change for dir select
        # orientation_diffs_180 = [np.abs(stim_angle + np.pi - ori_pref) for
        #                          (stim_angle, ori_pref) in
        #                          zip(stim_angles, ori_prefs)]

        # orientation_diffs_min = [min([diff0, diff180]) for (diff0, diff180)
        #                          in zip(orientation_diffs_0,
        #                                 orientation_diffs_180)]
        # orientation_match = [min_diff < ori_diff_threshhold
        #                      for min_diff in
        #                      orientation_diffs_min]

        # New code for when using direction selectivity
        orientation_diffs = \
            [np.pi - np.abs(np.abs(stim_angle-ori_pref) - np.pi) for
            (stim_angle, ori_pref) in zip(stim_angles, ori_prefs)]
        orientation_match = [diff < ori_diff_threshhold
                             for diff in
                             orientation_diffs]


        # Add columns to the df for new data
        nrnact['ori_pref'] = ori_prefs
        nrnact['stim_ori'] = stim_angles
        nrnact['stim_contrast'] = stim_contrasts
        nrnact['matched_stim_ori_pref'] = orientation_match

        # Save the df
        nrnact.to_pickle(os.path.join(
            self.session_dir, 'neuron act ori.pkl'))


        # Now do the same for the full data df
        full_data = pd.read_pickle(os.path.join(self.session_dir,
           "neuron_activity_results_alternativeformat_primary.pkl"))
        full_data = full_data.merge(nrnact, on=['batch_idx', 'height', 'width',
                                                'active', 'channel']) #TODO Lee do this manually so you're sure it's correct. Find the columns that fd doesn't have that nrnact does and transfer them. But make sure to compare the other columns to ensure they're in the same order. Do this for other merges.
        full_data.to_pickle(os.path.join(self.session_dir,
           "neuron_activity_results_alternativeformat_primary_w_ori_w_match.pkl"))


    def plot_state_traces_with_spec_and_acorr(self, patch_or_idx=None):
        """Plot type may be one of :
            trace_specgram_acorr_acorr,
            contrast_specgram_comparison
            """

        # Make dir to save plots for this experiment
        exp_dir = os.path.join(self.session_dir, 'summed centre states')
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)

        if patch_or_idx == 'patch':
            centre1 = self.central_patch_min
            centre2 = self.central_patch_max
            save_name = 'patch'
        else:
            centre1 = self.top_left_pnt[0] + self.extracted_im_size / 2
            centre2 = centre1
            save_name = 'neuron'
        print("Plotting state traces for central %s in each channel" % save_name)

        # Load data
        full_data = pd.read_pickle(os.path.join(self.session_dir,
            "neuron_activity_results_alternativeformat_primary_w_ori_w_match.pkl"))

        # Prepare general variables
        model_exp_name = self.primary_model_exp_name


        for ch in range(self.num_ch):
            print("Channel %s" % str(ch))

            # Prepare/reset variables for data managers

            # Get data
            dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=model_exp_name,
                                         var_names=self.st_var_name,
                                         state_layers=[1],
                                         batches=None,
                                         channels=[ch],
                                         hw=[self.top_left_pnt,
                                             self.bottom_right_pnt],
                                         timesteps=None)

            # Process data
            activities_df, arr_sh, colnames, inds = \
                self.convert_ch_data_to_activity_df(dm)


            # Find the subset of the nrnact df for this channel only and get
            # the h, w values for on neurons in each batch
            nrnact_ch = full_data.loc[full_data['channel'] == ch]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['matched_stim_ori_pref']]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['active']]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['height'] >= centre1]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['height'] <= centre2]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['width'] >= centre1]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['width'] <= centre2]
            nrnact_ch = nrnact_ch.reset_index()


            # Get the names of the columns where the neuron is on and the
            # orientation preference and stimuli are matched

            print("Defining names of columns where the neuron is on and the "+\
                  "orientation preference and stimuli are matched")

            on_colnames, on_col_inds, batches, heights, widths, stim_oris = \
                self.get_info_lists_from_nrnactch(nrnact_ch, only_matched=True)

            if len(on_colnames) == 0:
                print("No on neurons for channel %s." % ch)
            else:
                print("%i on neurons in channel %s" % (len(on_colnames), ch))

            # For the on neurons in the above-defined subset of nrnact, get
            # their timeseries from the arrays during_stim and outside_stim
            on_nrn_acts = activities_df[on_colnames].sum(axis=1)

            # Explore the states and their oscillations for each channel
            fig, ax = plt.subplots(4)
            fig.set_size_inches(14.5, 8.5)

            ax[0].plot(on_nrn_acts.index, on_nrn_acts)
            ax[0].set_ylabel('State [a.u.]')
            ax[0].set_xlabel('Timesteps')

            ax[1].specgram(on_nrn_acts, NFFT=self.specgram_nfft,
                           Fs=100, Fc=0, detrend=None,
                           noverlap=self.specgram_nfft-1,
                           xextent=None, pad_to=None,
                           sides='default',
                           scale_by_freq=True, scale='dB', mode='default',
                           cmap=plt.get_cmap('hsv'))
            ax[1].set_ylabel('Frequency [a.u.]')

            #Modern
            lags1, acorrs1, plot11, plot12 = ax[2].acorr(
                on_nrn_acts[self.primary_stim_start:self.primary_stim_stop],
                detrend=plt.mlab.detrend_linear,
                maxlags=self.autocorr_phase_len)
            lags2, acorrs2, plot21, plot22 = ax[3].acorr(
                on_nrn_acts[self.burn_in_len:self.primary_stim_start],
                detrend=plt.mlab.detrend_linear,
                maxlags=self.autocorr_phase_len)

            plt.ticklabel_format(axis="y", style="sci", scilimits=(
                0, 0))  # Uses sci notation for units
            plt.tight_layout()  # Stops y label being cut off

            plt.savefig(
                os.path.join(exp_dir,
                             'State traces and spectrogram for central'+\
                             ' %s of ch%i on aligned ' % (save_name, ch) +\
                             'trials.png'))
            plt.close()

    def convert_ch_data_to_activity_df(self, dm, var_type='state_1'):
        """Takes a datamanager instance as argument and converts it to a df.

        The df contains the trace values of the"""
        # Process data by converting to array
        dm.data[var_type] = dm.data[var_type].squeeze()  # get rid of ch dim
        reshaped_activities = [np.arange(len(dm.timesteps)).reshape(
            len(dm.timesteps), 1)]  # make TS data
        reshaped_activities.extend([
            dm.data[var_type].reshape(len(dm.timesteps),
                                       -1)])  # extend TS data with actual data
        arr_sh = dm.data[var_type].shape[1:]

        # Make column names from indices of batches, height, and width
        inds = np.meshgrid(np.arange(arr_sh[0]),
                           np.arange(arr_sh[1]),
                           np.arange(arr_sh[2]),
                           sparse=False, indexing='ij')
        inds = [i.reshape(-1) for i in inds]

        colnames = ['Timestep']
        for i, j, k in zip(*inds):
            colnames.append(
                '%s__b%s_h%s_w%s' % (var_type, i, self.top_left_pnt[0] + j,
                                          self.top_left_pnt[1] + k))

        activities = np.concatenate(reshaped_activities, axis=1)
        activities_df = pd.DataFrame(activities, columns=colnames)

        del activities, reshaped_activities

        return activities_df, arr_sh, colnames, inds

    def plot_contrast_specgram_comparison_LFP(self, patch_or_idx='patch'):
        """The difference from the '_local' function is that this
        one calculates the sum of all the traces in all channels first,
        then plots the spectrogram."""
        print("Plotting spectrograms for each channel")


        # Make dir to save plots for this experiment
        exp_dir = os.path.join(self.session_dir,
                               'spectrogram comparisons')
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)

        if patch_or_idx == 'patch':
            centre1 = self.central_patch_min
            centre2 = self.central_patch_max
            save_name = 'patch'
        else:
            centre1 = self.top_left_pnt[0] + self.extracted_im_size / 2
            centre2 = centre1
            save_name = 'neuron'

        # Load data
        full_data = \
            pd.read_pickle(os.path.join(self.session_dir,
                "neuron_activity_results_alternativeformat_primary_w_ori_w_match.pkl"))

        # Prepare general variables
        model_exp_name = self.primary_model_exp_name

        # Get the subset of contrasts that I want to plot
        contrast_inds = [0, 3, 5, 7]
        contrasts = [self.contrast_and_angle_contrasts[i]
                     for i in contrast_inds]
        per_contr_series = [None] * len(contrasts)

        for ch in range(self.num_ch):
            print("Channel %s" % str(ch))

            # Get data
            dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=model_exp_name,
                                         var_names=self.st_var_name,
                                         state_layers=[1],
                                         batches=None,
                                         channels=[ch],
                                         hw=[self.top_left_pnt,
                                             self.bottom_right_pnt],
                                         timesteps=None)

            # Process data
            activities_df, arr_sh, colnames, inds = \
                self.convert_ch_data_to_activity_df(dm)

            # Find the subset of the nrnact df for this channel only and get
            # the h, w values for on neurons in each batch
            nrnact_ch = full_data.loc[full_data['channel'] == ch]
            nrnact_ch.index = colnames[1:]
            cond1 = nrnact_ch['matched_stim_ori_pref']
            cond3 = nrnact_ch['height'] >= centre1
            cond4 = nrnact_ch['height'] <= centre2
            cond5 = nrnact_ch['width'] >= centre1
            cond6 = nrnact_ch['width'] <= centre2
            cond = cond1&cond3&cond4&cond5&cond6

            on_colnames = list(nrnact_ch.index[cond])
            nrnact_ch = nrnact_ch.loc[cond]
            on_nrn_states = activities_df.transpose().loc[on_colnames]
            nrnact_ch = pd.merge(nrnact_ch, on_nrn_states, left_index=True,
                                 right_index=True)

            if len(on_colnames) == 0:
                print("No on neurons for channel %s." % ch)
            else:
                print("%i on neurons in channel %s" % (len(on_colnames), ch))

            # Sum up the traces from each of the channels into
            # series of different contrasts
            for n, contrast in enumerate(contrasts):
                cond_stims = nrnact_ch['stim_contrast'] == contrast
                contrast_df = nrnact_ch.loc[cond_stims]
                drop_cols = list(
                    contrast_df.columns[:-self.full_trace_len])
                contrast_df = contrast_df.drop(columns=drop_cols)
                contrast_df = contrast_df.sum(axis=0)

                if per_contr_series[n] is None:
                    per_contr_series[n] = contrast_df
                else:
                    per_contr_series[n] += contrast_df

        # Plot per-contrast spectrograms of the series
        sep_str = '    -        -        -        -        -        -        -    '
        plot_titles = ['Low contrast  -        -        -    ', sep_str,
                       sep_str, '    -        -        -  High contrast']
        timesteps_per_second = self.ms_to_timesteps(1000)


        k=0
        fig, ax = plt.subplots(1, 4, figsize=(12, 3))
        for axis, contrast, p_title, series in zip(ax, contrasts,plot_titles,per_contr_series):
            spectrum, freqs, t, im = axis.specgram(series[100:],
                                                   NFFT=self.specgram_nfft,
                                                   Fs=timesteps_per_second,
                                                   Fc=0,
                                                   detrend=None,
                                                   noverlap=self.specgram_nfft-1,
                                                   xextent=None,
                                                   pad_to=None,
                                                   sides='default',
                                                   scale_by_freq=True,
                                                   scale='dB',
                                                   mode='default',
                                                   vmin=-110.,
                                                   vmax=0.,
                                                   cmap=plt.get_cmap('hsv'))
            # Increase length of t just for axes
            t = np.arange(np.min(t), np.max(t) + 1,
                      np.mean([t[i] - t[i - 1] for i in range(1, len(t))]))
            x_axis_scaler = timesteps_per_second
            axis.axvline(x=(self.primary_stim_start-128)/x_axis_scaler, c='black',
                         linewidth=1, linestyle='dashed') # subtracting min(t) moves the line so that it's in the middle of the segment that is centred on the start of the stim
            axis.axvline(x=(self.primary_stim_stop-128)/x_axis_scaler, c='black',
                         linewidth=1, linestyle='dashed')
            axis.set_xlabel('Time before stimulus [s]')
            axis.set_frame_on(False)

            if k ==0:
                axis.set_ylabel('Frequency [Hz]')
            axis.title.set_text(p_title)
            time_range = np.arange(1+np.min(t),11+np.min(t), 1)
            axis.set_xticks([self.find_t_at(x, t)/x_axis_scaler for x in time_range])
            axis.set_xticklabels(np.round(np.arange(-3,7,1)))
            axis.set_yticks([0, 50, 100])
            axis.set_yticklabels([0, 50, 100])

            plt.tight_layout()  # Stops y label being cut off
            k += 1
        plt.savefig(
            os.path.join(exp_dir,
                         'Spectrogram for central %s for all channels .png' % (
                             save_name)))
        plt.close()

    def find_t_at(self, x, t_range_ms):
        """Helper function to find the corresponding timestep (in a.u.) for
        a time in ms"""
        if x > np.max(t_range_ms) or x < np.min(t_range_ms):
            raise ValueError("Desired time x not in range of times given")
        else:
            nearby_values = np.where(np.isclose(t_range_ms, x, 0.0006))[0]
            # print(nearby_values)
            timestep = nearby_values[len(nearby_values)//2]
            return timestep

    def plot_contrast_power_spectra_LFP(self, patch_or_idx=None):

        print("Plotting power spectra at different contrasts")


        # Make dir to save plots for this experiment
        exp_dir = os.path.join(self.session_dir,
                               'Power spectra comparisons')
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)

        centre1 = self.central_patch_min
        centre2 = self.central_patch_max
        save_name = 'patch'

        # Load data
        full_data = \
            pd.read_pickle(os.path.join(self.session_dir,
                "neuron_activity_results_alternativeformat_primary_w_ori_w_match.pkl"))

        # Prepare general variables
        model_exp_name = self.primary_model_exp_name

        # Get the subset of contrasts that I want to plot
        contrast_inds = [0, 3, 5, 7]
        contrasts = [self.contrast_and_angle_contrasts[i]
                     for i in contrast_inds]
        per_contr_series = [None] * len(contrasts)

        for ch in range(self.num_ch):
            print("Channel %s" % str(ch))

            # Get data
            dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=model_exp_name,
                                         var_names=self.st_var_name,
                                         state_layers=[1],
                                         batches=None,
                                         channels=[ch],
                                         hw=[self.top_left_pnt,
                                             self.bottom_right_pnt],
                                         timesteps=None)

            # Process data
            activities_df, arr_sh, colnames, inds = \
                self.convert_ch_data_to_activity_df(dm)

            # Find the subset of the nrnact df for this channel only and get
            # the h, w values for on neurons in each batch
            nrnact_ch = full_data.loc[full_data['channel'] == ch]
            nrnact_ch.index = colnames[1:]
            cond1 = nrnact_ch['matched_stim_ori_pref']
            cond3 = nrnact_ch['height'] >= centre1
            cond4 = nrnact_ch['height'] <= centre2
            cond5 = nrnact_ch['width'] >= centre1
            cond6 = nrnact_ch['width'] <= centre2
            cond = cond1 & cond3 & cond4 & cond5 & cond6

            on_colnames = list(nrnact_ch.index[cond])
            nrnact_ch = nrnact_ch.loc[cond]
            on_nrn_states = activities_df.transpose().loc[on_colnames]
            nrnact_ch = pd.merge(nrnact_ch, on_nrn_states, left_index=True,
                                 right_index=True)

            if len(on_colnames) == 0:
                print("No on neurons for channel %s." % ch)
            else:
                print("%i on neurons in channel %s" % (len(on_colnames), ch))

            # Sum up the traces from each of the channels into
            # series of different contrasts
            for n, contrast in enumerate(contrasts):
                cond_stims = nrnact_ch['stim_contrast'] == contrast
                contrast_df = nrnact_ch.loc[cond_stims]
                drop_cols = list(
                    contrast_df.columns[:-self.full_trace_len])
                contrast_df = contrast_df.drop(columns=drop_cols)
                # contrast_df = contrast_df.iloc[p]
                contrast_df = contrast_df.sum(axis=0)

                if per_contr_series[n] is None:
                    per_contr_series[n] = contrast_df
                else:
                    per_contr_series[n] += contrast_df


        # Plot per-contrast power spectra of the series
        #Now do unnormalized plot
        fig, ax = plt.subplots()
        ax.set_frame_on(False)
        cmap = plt.cm.Greys

        timesteps_per_second = self.ms_to_timesteps(1000)

        for contrast, series in zip(contrasts,per_contr_series):
            freqs, psd = welch(
                # series[100:],
                series[self.primary_stim_start:self.primary_stim_stop],
                fs=timesteps_per_second,
                scaling='density')

            plt.plot(psd*freqs, c=cmap(0.2*contrast + 0.3))

        plt.yscale("log")
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Power X Frequency [a.u]')
        ax.legend(['Low contrast','         -', '         -', 'High contrast'])
        plt.tight_layout()  # Stops y label being cut off
        plt.savefig(
            os.path.join(exp_dir,
                         'Power spectra for active neurons in central' +
                         ' %s for all channels .png' % (save_name)))
        plt.close()



    def plot_contrast_dependent_transients_of_active_neurons(self, patch_or_idx=None):
        """"""
        print("Plotting contrast dependent transients for each channel")


        # Make dir to save plots for this experiment
        exp_dir = os.path.join(self.session_dir,
                               'transients plots')
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)

        mean_thresholding = True

        if patch_or_idx == 'patch': #TODO lee consider increasing these to an even bigger patch to even out the noise. Do overnight. by hardcoding
            centre1 = self.central_patch_min
            centre2 = self.central_patch_max
            save_name = 'patch'
        else:
            centre1 = self.top_left_pnt[0] + self.extracted_im_size / 2
            centre2 = centre1
            save_name = 'neuron'

        # Load data
        full_data = \
            pd.read_pickle(os.path.join(self.session_dir,
                "neuron_activity_results_alternativeformat_primary_w_ori_w_match.pkl"))

        # Prepare general variables
        model_exp_name = self.primary_model_exp_name

        # Get the subset of contrasts that I want to plot
        contrast_inds = [0, 3, 5, 7]
        contrasts = [self.contrast_and_angle_contrasts[i]
                     for i in contrast_inds]
        per_contr_series = [None] * len(contrasts)
        unnorm_per_contr_series = [None] * len(contrasts)

        for ch in range(self.num_ch):
            print("Channel %s" % str(ch))

            # Get data
            dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=model_exp_name,
                                         var_names=self.st_var_name,
                                         state_layers=[1],
                                         batches=None,
                                         channels=[ch],
                                         hw=[self.top_left_pnt,
                                             self.bottom_right_pnt],
                                         timesteps=None)

            # Process data
            activities_df, arr_sh, colnames, inds = \
                self.convert_ch_data_to_activity_df(dm)

            # Find the subset of the nrnact df for this channel only and get
            # the h, w values for on neurons in each batch
            nrnact_ch = full_data.loc[full_data['channel'] == ch]
            nrnact_ch.index = colnames[1:]

            cond3 = nrnact_ch['height'] >= centre1
            cond4 = nrnact_ch['height'] <= centre2
            cond5 = nrnact_ch['width'] >= centre1
            cond6 = nrnact_ch['width'] <= centre2

            cond = cond3 & cond4 & cond5 & cond6

            on_colnames = list(nrnact_ch.index[cond])
            nrnact_ch = nrnact_ch.loc[cond]
            on_nrn_states = activities_df.transpose().loc[on_colnames]
            nrnact_ch = pd.merge(nrnact_ch, on_nrn_states, left_index=True,
                                 right_index=True)

            if len(on_colnames) == 0:
                print("No on neurons for channel %s." % ch)
            else:
                print("%i on neurons in channel %s" % (len(on_colnames), ch))

            # Sum up the traces from each of the channels into
            # series of different contrasts
            for n, contrast in enumerate(contrasts):
                cond_stims = nrnact_ch['stim_contrast'] == contrast
                contrast_df = nrnact_ch.loc[cond_stims]
                drop_cols = list(
                    contrast_df.columns[:-self.full_trace_len])
                contrast_df = contrast_df.drop(columns=drop_cols)
                num_points = (len(contrast_df.columns) - self.primary_stim_stop) + (
                            self.primary_stim_start - self.burn_in_len)
                if mean_thresholding:
                    nrn_means = contrast_df.iloc[:,self.burn_in_len:self.primary_stim_start].sum(axis=1)
                    nrn_means += contrast_df.iloc[:,self.primary_stim_stop:len(contrast_df.columns)].sum(axis=1)
                    nrn_means = nrn_means / num_points
                    nrn_means = np.column_stack([nrn_means] * \
                                         len(contrast_df.columns))
                    contrast_df = np.where(contrast_df > nrn_means, contrast_df, nrn_means)

                contrast_df = contrast_df.sum(axis=0)

                mean = sum(
                    contrast_df[self.burn_in_len:self.primary_stim_start])
                mean += sum(
                    contrast_df[self.primary_stim_stop:len(contrast_df)])

                mean = mean / num_points

                if per_contr_series[n] is None:

                    unnorm_per_contr_series[n] = contrast_df.copy() - mean
                    if contrast_df.var() == 0:
                        contrast_df = (contrast_df - contrast_df.mean())
                    else:
                        contrast_df = \
                            (contrast_df - contrast_df.mean()) / \
                            contrast_df.var()
                    per_contr_series[n] = contrast_df.copy()
                else:
                    unnorm_per_contr_series[n] += (contrast_df - mean)
                    if contrast_df.var() == 0:
                        contrast_df = (contrast_df - contrast_df.mean())
                    else:
                        contrast_df = \
                            (contrast_df - contrast_df.mean()) / \
                            contrast_df.var()
                    per_contr_series[n] += contrast_df

        cmap = plt.cm.Greys

        # Plot
        fig, ax = plt.subplots()
        ax.set_frame_on(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_xlabel('Time after stimulation (ms)')
        for contrast, series in zip(contrasts,unnorm_per_contr_series):
            trunc_series = series[self.primary_stim_start-100:self.primary_stim_start+150]
            plt.plot(trunc_series, c=cmap(0.2*contrast + 0.3))
        plt.axvline(x=100, c='black',
                         linewidth=1, linestyle='dashed')
        ax.legend(['Low contrast','         -', '         -', 'High contrast'], loc='upper left')
        ax.legend_.get_frame().set_linewidth(0.0)

        ts_per_100ms = self.ms_to_timesteps(100,8)
        diff = 100 - 4 * ts_per_100ms
        ticks_locs_ts = sorted(-np.arange(0, 4*ts_per_100ms, ts_per_100ms))[:-1]
        ticks_locs_ts.extend(list(np.arange(0, 8*ts_per_100ms, ts_per_100ms)))
        ticks_locs_ts = np.array(ticks_locs_ts)
        ticks_locs_ts = ticks_locs_ts - np.min(ticks_locs_ts) + diff

        labels_ms = np.round(np.arange(-400, 700, 100))
        ax.set_xticks(ticks=ticks_locs_ts)
        ax.set_xticklabels(labels_ms)
        ax.set_yticklabels([])
        plt.tight_layout()  # Stops y label being cut off
        plt.savefig(
            os.path.join(exp_dir,
                         'Unnormalized Transients for active neurons in ' + \
                         'central %s for all channels .png' % (save_name)))
        plt.close()

        # Now do unnormalized series at the stim-off timepoint
        fig, ax = plt.subplots()
        for contrast, series in zip(contrasts,unnorm_per_contr_series):
            trunc_series = series[self.primary_stim_stop-100:self.primary_stim_stop+150]
            plt.plot(trunc_series, c=cmap(0.2*contrast + 0.3))
        plt.axvline(x=100, c='black',
                         linewidth=1, linestyle='dashed')
        ax.set_xticks(ticks=[0, 100, 200])
        ax.set_xticklabels([-100, 0, 100])
        ax.set_yticklabels([])
        ax.legend(['Low contrast',' ', ' ', 'High contrast'], loc='upper right')

        plt.tight_layout()  # Stops y label being cut off
        plt.savefig(
            os.path.join(exp_dir,
                         'Unnormalized OFF Transients for active neurons ' +
                         'in central %s for all channels .png' % (save_name)))
        plt.close()


    def plot_state_traces(self, patch_or_idx=None):
        # Make dir to save plots for this experiment
        exp_dir = os.path.join(self.session_dir,
                               'summed centre states')
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)

        # Load data
        full_data = pd.read_pickle(os.path.join(self.session_dir,
           "neuron_activity_results_alternativeformat_primary_w_ori_w_match.pkl"))

        if patch_or_idx == 'patch':
            centre1 = self.central_patch_min
            centre2 = self.central_patch_max
            save_name = 'patch'
        else:
            centre1 = self.top_left_pnt[0] + self.extracted_im_size/2
            centre2 = centre1
            save_name = 'neuron'

        # Prepare general variables
        model_exp_name = self.primary_model_exp_name

        for ch in range(self.num_ch):

            print("Channel %s" % str(ch))

            # Get data
            dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=model_exp_name,
                                         var_names=self.st_var_name,
                                         state_layers=[1],
                                         batches=None,
                                         channels=[ch],
                                         hw=[self.top_left_pnt,
                                             self.bottom_right_pnt],
                                         timesteps=None)

            # Process data
            activities_df, arr_sh, colnames, inds = \
                self.convert_ch_data_to_activity_df(dm)

            # Find the subset of the nrnact df for this channel only and get
            # the h, w values for on neurons in each batch
            nrnact_ch = full_data.loc[full_data['channel'] == ch]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['height'] >= centre1]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['height'] <= centre2]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['width'] >= centre1]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['width'] <= centre2]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['matched_stim_ori_pref']]


            nrnact_ch = nrnact_ch.reset_index()

            # Get the names of the columns where the neuron is on and the
            # orientation preference and stimuli are matched

            print("Defining names of neurons")
            on_colnames, on_col_inds, batches, heights, widths, stim_oris = \
                self.get_info_lists_from_nrnactch(nrnact_ch, only_matched=True)

            # For the  in the above-defined subset of nrnact, get
            # their timeseries from the arrays during_stim and outside_stim
            state_traces = activities_df[on_colnames]
            state_traces = state_traces[
                           self.primary_stim_start-400:
                           self.primary_stim_start+400]
            state_traces = state_traces.transpose()
            state_traces['batch'] = batches
            state_traces['stim_oris'] = stim_oris

            stims = list(set(stim_oris))
            for stim in stims:
                # Sum across batch
                st_df = state_traces.loc[state_traces['stim_oris']==stim]
                st_df = st_df.groupby('batch').sum()
                st_df = st_df.transpose()
                st_df = st_df[:-1]  # gets rid of batch info
                st_df.columns = self.true_contrast_and_angle_contrasts


                fig, ax = plt.subplots()
                cmap = plt.cm.YlGn

                legend_contrasts = []
                k = 0
                times = st_df.index
                st_df = st_df.transpose().iterrows()
                for contrast, states in st_df:
                    if k % 2 == 0:
                        plt.plot(times, states, c=cmap(contrast+0.2))
                        legend_contrasts.append(round(contrast,2))
                    k +=1
                plt.axvline(x=self.primary_stim_start)
                ax.set_ylabel('State [a.u.]')
                ax.set_xlabel('Timesteps')
                ax.legend(legend_contrasts, loc="lower right")
                fig.savefig(os.path.join(self.session_dir,
                                         'central %s state trace for ch %i stim%f.png' % (save_name, ch, round(stim, 4))))
                plt.close()


    def find_oscillating_neurons(self):
        """Looks at a target subset of the data and returns the indices of the
        neurons whose responses qualify as oscillating."""

        print("Finding oscillating neurons")

        # Make dir to save plots for this experiment
        exp_dir = os.path.join(self.session_dir,
                               'Fitted gabors and PSD plot')
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)

        # Load data
        nrnact = pd.read_pickle(
            os.path.join(self.session_dir,
                         'neuron act ori.pkl'))

        # Prepare general variables
        model_exp_name = self.primary_model_exp_name
        during_or_outside_names = ['during', 'outside']
        result_names = ['freq_power_product', 'peak_freqs', 'peak_freq_powers']

        # Prepare variables used for processing data
        stim_on_start = self.primary_stim_start
        stim_on_stop = self.primary_stim_stop

        # Create empty columns in df to save PSD results
        for d_o_name in during_or_outside_names:
            for result_name in result_names:
                new_col_name = result_name + '_' + d_o_name
                nrnact[new_col_name] = np.nan
                nrnact[new_col_name] = nrnact[new_col_name].astype('object')

        # Perform analysis on the selected neurons in each channel
        for ch in range(self.num_ch):

            print("Channel %s" % str(ch))

            # Get data
            dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=model_exp_name,
                                         var_names=self.st_var_name,
                                         state_layers=[1],
                                         batches=None,
                                         channels=[ch],
                                         hw=[self.top_left_pnt,
                                             self.bottom_right_pnt],
                                         timesteps=None)
            # Process data
            activities_df, arr_sh, colnames, inds = \
                self.convert_ch_data_to_activity_df(dm)

            # Separate data during stim from data when stim isn't present
            during_stim = activities_df[stim_on_start:stim_on_stop]
            outside_stim = activities_df.drop(
                activities_df.index[stim_on_start:stim_on_stop])

            # Find the subset of the nrnact df for this channel only and get
            # the h, w values for on neurons in each batch
            nrnact_ch = nrnact.loc[nrnact['channel'] == ch]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['active']]
            nrnact_ch = nrnact_ch.reset_index()

            # Get the names of the columns where the neuron is on and the
            # orientation preference and stimuli are matched
            print("Defining names of columns where the neuron is on and the "+\
                  "orientation preference and stimuli are matched")
            on_colnames, on_col_inds, batches, heights, widths, stim_oris = \
                self.get_info_lists_from_nrnactch(nrnact_ch, only_matched=True)

            if len(on_colnames) == 0:
                print("No on neurons for channel %s. Moving to next channel")

            # For the on neurons in the above-defined subset of nrnact, get
            # their timeseries from the arrays during_stim and outside_stim
            on_during_stim  = during_stim[on_colnames]
            on_outside_stim = outside_stim[on_colnames]

            # Calculate the PSDs for the active neurons' timeseries
            # from the arrays during_stim and outside_stim
            left_in_channel = len(on_colnames)

            for bhw, col in zip(on_col_inds, on_colnames):
                print("Neurons left in channel %i : %i" % (ch,left_in_channel))
                # Separate out bhw
                batch_id,height,width = bhw

                # Calculate the df index for this batch_idx, height, width & ch
                # and we'll use it to decide whether to run analysis at all
                # and also for assigning data
                cond1 = nrnact['batch_idx'] == bhw[0]
                cond2 = nrnact['height'] == bhw[1]
                cond3 = nrnact['width'] == bhw[2]
                cond4 = nrnact['channel'] == ch
                loc_index = nrnact.loc[cond1 & cond2 & cond3 & cond4].index[0]
                left_in_channel -= 1
                if height >= self.central_patch_min\
                        and height <= self.central_patch_max \
                        and width >= self.central_patch_min \
                        and width <= self.central_patch_max \
                        and nrnact.at[loc_index, 'matched_stim_ori_pref']:
                     print("ch%s " % ch + col)
                else:
                    continue

                for i, ts in enumerate([on_during_stim, on_outside_stim]):
                    # For the on neurons in the above-defined subset of nrnact,
                    # get their timeseries from the arrays during_stim and
                    # outside_stim and detrend
                    durout_name = during_or_outside_names[i]
                    states = ts[col]
                    states = detrend(states, axis=0)

                    # Calculate the power spectral density plots for the
                    # active neurons' timeseries
                    # from the arrays during_stim and outside_stim
                    freqs, psd = welch(
                        states,
                        scaling='spectrum',
                        nperseg=self.exp_stim_phase_lens[1])

                    # Find the peaks
                    peaks = find_peaks(psd,
                                       rel_height=10.,
                                       prominence=np.max(psd)/2,
                                       distance=3)
                    peak_locs = peaks[0]
                    if peak_locs.size == 0:
                        peak_locs = np.array(np.argmax(psd))

                    # Save PSD results
                    results = [np.sum(freqs * psd),
                               np.array([freqs[peak_locs]]),
                               np.array([psd[peak_locs]])]

                    for result_name, val in zip(result_names, results):
                        col_assign_name = result_name + '_' + durout_name
                        nrnact.at[loc_index, col_assign_name] = val

            # Save the full df periodically at the end of every channel
            nrnact.to_pickle(os.path.join(
                self.session_dir,
                'neuron act and osc.pkl'))


    def plot_contrast_frequency_plots(self):
        print("Plotting contrast/frequency plots")
        # Prepare variables
        angles = self.contrast_and_angle_angles
        contrasts = self.contrast_and_angle_contrasts
        angle_contrast_pairs = self.contrast_and_angle_angle_contrast_pairs

        # Load df with freq and activity info
        nrn_actosc = pd.read_pickle(
            os.path.join(self.session_dir,
                         'neuron act and osc.pkl'))

        # Calculate the max peak power and the product of the peak powers
        # and peak frequencies
        zero_to_nan = lambda x: np.nan if np.array(x).size == 0 else x
        nrn_actosc['peak_freq_powers_during'] = \
            nrn_actosc['peak_freq_powers_during'].apply(zero_to_nan)
        nrn_actosc['peak_freq_powers_outside'] = \
            nrn_actosc['peak_freq_powers_outside'].apply(zero_to_nan)

        nrn_actosc['max_peak_power'] = \
            nrn_actosc['peak_freq_powers_during'].apply(np.nanmax)

        nrn_actosc['peak_freqs_power_product'] = nrn_actosc.apply(
            lambda x: np.nansum(
                x['peak_freqs_during'] * x['peak_freq_powers_during']), axis=1)

        nrn_actosc.to_pickle(os.path.join(
            self.session_dir, 'neuron act and osc with new contrast freq info.pkl'))

        # Identify the neurons that have ALL [well def ori pref; matched
        # stim and ori pref; activated; peak power above a threshold]

        cond1a = nrn_actosc['peak_freqs_power_product'] > 0.0
        cond1b = nrn_actosc['matched_stim_ori_pref']
        cond1c = nrn_actosc['active']
        cond2 = nrn_actosc['height'] <= self.bottom_right_pnt[0]
        cond3 = nrn_actosc['height'] >= self.top_left_pnt[0]
        cond4 = nrn_actosc['width'] <= self.bottom_right_pnt[1]
        cond5 = nrn_actosc['width'] >= self.top_left_pnt[1]
        cond6 = nrn_actosc['max_peak_power'] > 1e-5
        nrn_actosc_filt = \
            nrn_actosc.loc[cond1a&cond1b&cond1c&cond2&cond3&cond4&cond5&cond6]

        ### Take those neurons and group them by the contrast of their stimulus

        mean_peak_power_during  = nrn_actosc_filt.groupby(['stim_contrast'])[
                     'peak_freqs_power_product'].mean()
        mean_peak_power_outside = nrn_actosc_filt.groupby(['stim_contrast'])[
                     'peak_freqs_power_product'].mean()
        print(mean_peak_power_during)
        print(mean_peak_power_outside)


        # Plot contrast on x, freq on y, ideally should see an increase
        contrasts = np.array(mean_peak_power_outside.index)
        plt.scatter(contrasts, mean_peak_power_outside,
                    color='r')
        plt.ylim(0.0, np.max(mean_peak_power_outside) * 1.3)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(
            0, 0))
        plt.savefig(os.path.join(self.session_dir,
                                 'contrast frequency plots.png'))
        plt.close()


    def get_info_lists_from_nrnactch(self, nrnact_ch, only_matched=False):

        print("Defining names of neurons")

        # Create empty info lists
        on_colnames = []
        on_col_inds = []
        batches = []
        stim_oris = []
        heights = []
        widths = []

        for i in range(len(nrnact_ch)):
            # Extract info from nrnact_ch
            bhw = tuple(nrnact_ch.loc[i][1:4])
            stim_ori = nrnact_ch.loc[i]['stim_ori']
            on_colname = 'state_1__b%i_h%i_w%i' % bhw

            if only_matched:
                matched_ori = nrnact_ch.loc[i]['matched_stim_ori_pref']
                if matched_ori:
                    # Append info to lists only if the stim and ori pref match
                    on_colnames.append(on_colname)
                    on_col_inds.append(bhw)
                    batches.append(bhw[0])
                    heights.append(bhw[1])
                    widths.append(bhw[2])
                    stim_oris.append(stim_ori)
            else:
                # Append info to lists
                on_colnames.append(on_colname)
                on_col_inds.append(bhw)
                batches.append(bhw[0])
                heights.append(bhw[1])
                widths.append(bhw[2])
                stim_oris.append(stim_ori)

        return on_colnames, on_col_inds, batches, heights, widths, stim_oris

    def synchrony_experiment1_overlapping_rec_fields(self, patch_or_idx=None):
        """Takes the column/patch of channels at the site of the static stimulus
        and (if patch, combines the traces of each channel), then calculates
        the autocorrelation plot between each channel's (combined) trace. """

        print("Performing Synchrony Experiment 1: Overlapping Receptive " +\
              "Fields")

        # Make dir to save plots for this experiment
        exp_dir = os.path.join(self.session_dir,
                               'SynchronyExp1')
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)

        # Load data
        presaved_path = os.path.join(self.session_dir,
            "neuron_activity_results_alternativeformat_just_angles_few_angles.pkl")
        full_data = pd.read_pickle(presaved_path)
        # Add stim_orientation info to data
        stim_angles = [self.just_angles_few_angles_angles[i] for i in
                       full_data['batch_idx']]
        full_data['stim_ori'] = stim_angles


        if patch_or_idx == 'patch':
            centre1 = self.central_patch_min
            centre2 = self.central_patch_max
            central_patch_size = self.central_patch_size
            save_name = 'patch'
        else:
            centre1 = self.top_left_pnt[0] + self.extracted_im_size / 2
            centre2 = centre1
            central_patch_size = 1
            save_name = 'neuron'

        # Prepare general variables
        model_exp_name = self.just_angles_few_angles_exp_name
        full_state_traces = pd.DataFrame()

        # Go through each channel and get the traces that you need (all
        # batches)
        for ch in range(self.num_ch):

            print("Channel %s" % str(ch))

            # Get data
            dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=model_exp_name,
                                         var_names=self.st_var_name,
                                         state_layers=[1],
                                         batches=None,
                                         channels=[ch],
                                         hw=[self.top_left_pnt,
                                             self.bottom_right_pnt],
                                         timesteps=None)

            # Process data
            activities_df, arr_sh, colnames, inds = \
                self.convert_ch_data_to_activity_df(dm)

            # Find the subset of the nrnact df for this channel only and get
            # the h, w values for on neurons in each batch
            nrnact_ch = full_data.loc[full_data['channel'] == ch]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['height'] >= centre1]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['height'] <= centre2]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['width'] >= centre1]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['width'] <= centre2]
            nrnact_ch = nrnact_ch.reset_index()

            # Get the names of the columns
            on_colnames, on_col_inds, batches, heights, widths, stim_oris = \
                self.get_info_lists_from_nrnactch(nrnact_ch)

            # For the  in the above-defined subset of nrnact, get
            # their timeseries from the arrays
            state_traces = activities_df[on_colnames]
            state_traces = state_traces.transpose()
            state_traces['batch'] = batches
            state_traces['stim_oris'] = stim_oris

            # true_contrasts = self.true_contrast_and_angle_contrasts * num_stims
            # state_traces.columns = true_contrasts
            batch_set = list(set(batches))
            state_traces = state_traces.groupby(['batch']).sum()
            state_traces['stim_oris'] = \
                state_traces['stim_oris'] / central_patch_size
            state_traces['channel'] = ch
            state_traces['batch'] = batch_set

            if full_state_traces.empty:
                full_state_traces = state_traces
            else:
                full_state_traces = full_state_traces.append(state_traces)

        # Now, having collected all the state traces of the neurons of
        # interest, compute cross correlation plots between neurons in
        # different channels in the same batch element:
        for b in range(self.num_batches):


            acorr_data_during = pd.DataFrame()
            acorr_data_outside = pd.DataFrame()
            acorr_data_dfs = [acorr_data_during, acorr_data_outside]

            for ch1 in range(self.num_ch):
                for ch2 in range(self.num_ch):
                    print("%s   %s    %s" % (b, ch1, ch2))
                    # Define conditions to get traces from full df
                    cond1_1 = full_state_traces['channel'] == ch1
                    cond2_1 = full_state_traces['channel'] == ch2
                    cond_b = full_state_traces['batch']    == b
                    cond1 = cond1_1 & cond_b
                    cond2 = cond2_1 & cond_b

                    if not np.any(cond1) or not np.any(cond2):
                        print("Skipping %s   %s    %s" % (b, ch1, ch2))
                        continue

                    # Get traces from full df
                    trace_1 = full_state_traces.loc[cond1]
                    trace_2 = full_state_traces.loc[cond2]

                    # Convert to array
                    trace_1 = np.array(trace_1).squeeze()
                    trace_2 = np.array(trace_2).squeeze()

                    # Remove channel,batch information to get only traces
                    trace_1 = trace_1[0:self.full_trace_len]
                    trace_2 = trace_2[0:self.full_trace_len]

                    # Split traces into 'during stim' and 'outside stim'
                    t1_duringstim = \
                        trace_1[self.primary_stim_start:self.primary_stim_stop]
                    t2_duringstim = \
                        trace_2[self.primary_stim_start:self.primary_stim_stop]
                    traces_during = [t1_duringstim, t2_duringstim]

                    t1_outsidestim = \
                        trace_1[self.burn_in_len:self.primary_stim_start]
                    t2_outsidestim = \
                        trace_2[self.burn_in_len:self.primary_stim_start]
                    traces_outside = [t1_outsidestim, t2_outsidestim]

                    traces_d_and_o = [traces_during, traces_outside]
                    for j, (during_or_outside_label, traces) in \
                            enumerate(zip(['during','outside'],
                                          traces_d_and_o)):
                        t1 = traces[0]
                        t2 = traces[1]

                        # Run acorr funcs and save results
                        acorr_result = np.correlate(t1-np.mean(t1),
                                                    t2-np.mean(t2),
                                                    mode='full')
                        acorr_result = pd.DataFrame(acorr_result).transpose()
                        acorr_result['channel A'] = ch1
                        acorr_result['channel B'] = ch2
                        acorr_result['batch'] = b
                        acorr_data_dfs[j] = \
                            acorr_data_dfs[j].append(acorr_result)
            acorr_data_dfs = [df.reset_index() for df in acorr_data_dfs]

            # Save results periodically, per batch
            for ac_df, label in zip(acorr_data_dfs,['during', 'outside']):
                ac_df.to_pickle(os.path.join(exp_dir,
                                'cross_correlation_results_%s_%s.pkl' % \
                                             (b, label)) )


    def synchrony_experiment1_overlapping_rec_fields_fit_Gabors(self):
        """Git Gabor functions to plots as in Gray et al. (1989)."""
        # Prepare general variables and dirs
        exp_dir, acorr_dir = self.prepare_dirs(self.session_dir,
                                               'SynchronyExp1',
                                               'acorrs_gabor_fitted')

        batch_groups = self.just_angles_few_angles_batchgroups

        plotting = False
        # Set initial gabor params (gaussian components) [sigma, amplitude]
        p01 = [7e1, 400.0]
        p02 = [7e1, 500.0]
        p03 = [7e1, 600.0]
        init_params_gabor = [p01, p02, p03]


        for bg_angle_index, bg in enumerate(batch_groups):

            # Collect all the dfs for this batch group (batch group = one stim)
            dfs_all = \
                [[pd.read_pickle(os.path.join(exp_dir,
                     'cross_correlation_results_%s_%s.pkl' % (b, label)))
                for label in ['during', 'outside']] for b in bg]

            # Go through all the batchelements for that stim
            for b in bg:
                param_df = pd.DataFrame()  # df to save results
                # which is saved every batch

                for i, label in enumerate(['during', 'outside']):

                    # Select the df for this batch and this time-period
                    acorr_df = dfs_all[b-(len(bg)*bg_angle_index)][i]

                    for ch1 in range(self.num_ch):
                        for ch2 in range(self.num_ch):
                            # Select the specific xcorr plot
                            print("Batch: %s ;   Channels %s %s" % (b, ch1, ch2))
                            cond1 = acorr_df['channel A'] == ch1
                            cond2 = acorr_df['channel B'] == ch2
                            cond = cond1 & cond2

                            acorr_data = np.array(acorr_df[cond]).squeeze()
                            acorr_data = acorr_data[1:-3]
                            midpoint = len(acorr_data) // 2
                            acorr_data = acorr_data[(midpoint-self.acorr_mid_len):
                                                    (midpoint+self.acorr_mid_len)]

                            # Fit sine function first
                            ## Take a guess for lambda by counting peaks. This
                            ## works because most look pretty oscillatory to
                            ## begin with.
                            peaks = find_peaks(acorr_data,
                                               height=0,
                                               rel_height=3.,
                                               prominence=np.max(acorr_data)/5,
                                               distance=3)
                            avg_peak_dist = (self.acorr_mid_len*2) / len(peaks[0])

                            ## Set init sine params
                            max_acorr = np.max(acorr_data) * 0.7
                            amplitude0 = max_acorr
                            guess_lmda = 7/avg_peak_dist # 7 is just a constant
                            # that seemed to work

                            num_incr = 5
                            negs = num_incr // 2
                            exps = range(-negs, num_incr-negs)
                            lambdas = [guess_lmda * (1.15**n) for n in exps]
                            phaseshift0 = 0.
                            mean0 = 0.
                            init_params_sine = [[phaseshift0, mean0,
                                              lambdas[i]]
                                            for i in range(len(lambdas))]
                            sine_curve = lambda p: (amplitude0 * np.cos(self.acorr_x * p[2] + p[0]) + p[1])
                            opt_func_sine = lambda p: sine_curve(p) - acorr_data

                            # Go through each of the init param settings
                            # for fitting the SINE component of the Gabor and
                            # choose the one with the lowest resulting cost
                            costs_sine = []
                            est_params_sine = []
                            for p0x in init_params_sine:
                                opt_result = \
                                    optimize.least_squares(opt_func_sine, p0x)
                                est_params_sine.append(opt_result.x)
                                costs_sine.append(opt_result.cost)
                            print(np.argmin(costs_sine))
                            cost_sine = costs_sine[np.argmin(costs_sine)]
                            est_params_sine = \
                                list(est_params_sine[np.argmin(costs_sine)])
                            fitted_sine = sine_curve(est_params_sine)


                            # Fit Gaussian component of Gabor function
                            gabor_func = lambda p: (
                                        (1 / (np.sqrt(2 * np.pi) * p[0])) \
                                        * np.exp(-(self.acorr_x ** 2) / (2 * p[0] ** 2)) *
                                        p[1] * \
                                        fitted_sine)
                            opt_func = lambda p: gabor_func(p) - acorr_data



                            costs_gabor = []
                            est_params_gabor = []
                            for p0x in init_params_gabor:
                                opt_result = optimize.least_squares(opt_func, p0x)
                                est_params_gabor.append(opt_result.x)
                                costs_gabor.append(opt_result.cost)
                            print(np.argmin(costs_gabor))
                            orig_params = \
                                init_params_gabor[np.argmin(costs_gabor)]
                            cost_gabor = costs_gabor[np.argmin(costs_gabor)]
                            print(cost_gabor)
                            est_params_gabor = \
                                list(est_params_gabor[np.argmin(costs_gabor)])
                            print(str(est_params_gabor))
                            est_params_gabor.append(costs_gabor)

                            # Take stock of and save results
                            est_gabor_curve = gabor_func(est_params_gabor)
                            min_gab = np.min(est_gabor_curve)
                            max_gab = np.max(est_gabor_curve)
                            est_params_dict = {'channel A': ch1,
                                               'channel B': ch2,
                                               'dur_or_out': i,
                                               'batch': b,
                                               'batch_group': bg_angle_index,
                                               'max_acorr': max_acorr,
                                               'min_gabor': min_gab,
                                               'max_gabor': max_gab,
                                               'amplitude': (max_gab-min_gab)/2,
                                               'freq_scale': (1/est_params_sine[2]),
                                               'sine_phase': est_params_sine[0],
                                               'sine_mean': est_params_sine[1],
                                               'gauss_sigma':est_params_gabor[0],
                                               'gauss_amp': est_params_gabor[1],
                                               'cost_sine':  cost_sine,
                                               'cost_gabor': cost_gabor}
                            param_df = \
                                param_df.append(est_params_dict,
                                                ignore_index=True)

                            plotting = np.random.rand() > .999
                            if plotting: # Happens for 1/1000 plots
                                # Plot results
                                fig, ax = plt.subplots(1, figsize=(9, 5))
                                ax.plot(self.acorr_x, acorr_data)
                                ax.plot(self.acorr_x, est_gabor_curve )
                                ax.scatter(self.acorr_x[peaks[0]], acorr_data[peaks[0]])
                                ax.set_xlabel('lag')
                                ax.set_ylabel('correlation coefficient')

                                plt.savefig(
                                    "%s/cross correlation between %i and %i in batch %i for %s" % (
                                    acorr_dir, ch1, ch2, b,label))
                                plt.close()
                                plotting = False  # Resets bool

                df_title = "%s/estimated_gabor_params_batch_%i " % (exp_dir, b)
                param_df.to_pickle(df_title)


    def synchrony_experiment1_overlapping_rec_fields_Plot_acorrs_CI_neighbours(self):
        """Collects the 128 batch dfs together
           Collects the acorr data for each trace
           Plots per channel for all batches for that angle"""
        # Prepare general variables
        exp_dir, acorr_dir = self.prepare_dirs(self.session_dir,
                                               'SynchronyExp1',
                                               'acorrs_overlayed')

        ch_data_during  = pd.DataFrame()
        ch_data_outside = pd.DataFrame()
        batches = list(range(self.num_batches))
        max_peak = 0.5
        min_trough = -0.5
        display_len = 200

        plot_names = ['Neighbouring channels', 'Channels 2 at distance 2']
        full_plot_title = 'Mean cross correlation between neighbouring ' + \
                          'and non-neighbouring channels'
        num_plots = 5
        acorr_traces_lists = [[].copy() for x in range(num_plots)]

        # Go through each channel
        dists = range(1, num_plots+1)
        fig, ax = plt.subplots(num_plots, figsize=(8, 12))
        fig.subplots_adjust(hspace=-0.4)

        for k, dist in enumerate(dists):

            ax[k].set_ylim([min_trough, max_peak])
            ax[k].set_frame_on(False)
            timerange = np.arange(-800, 1000, 200)
            print(timerange)
            if k == 0:
                title = \
                    "Mean cross correlation between channels" + \
                    " at different channel-distances"
                ax[k].set_title(title)

            if dist == num_plots: # Add x axis on last plot only
                ax[k].set_xticklabels(timerange)
                ax[k].set_xlabel('Lag [ms]')
                ax[k].set_xticks([self.ms_to_timesteps(x) for x in timerange])
            else:
                ax[k].set_xticklabels([])
                ax[k].set_xticks([])

            ax[k].set_ylabel('Channel dist %i' % dist)
            ax[k].set_yticks([-0.25, 0, 0.25])
            ax[k].set_yticklabels([-0.25, 0, 0.25])


            for b_idx, b in enumerate(batches):

                # Collect all the dfs for this batch group (this stim)
                dfs = pd.read_pickle(
                        os.path.join(exp_dir,
                                     'cross_correlation_results_%s_during.pkl' % (
                                     b)))

                for ch1 in range(self.num_ch):
                    for ch2 in range(self.num_ch):
                        print("%s %s %s" % (b, ch1, ch2))

                        if not np.abs(ch1-ch2)==dist:
                            continue

                        # Just get the channel combination you want

                        print("bi: %i" % b_idx)
                        cond_a = dfs['channel A'] == ch1
                        cond_b = dfs['channel B'] == ch2
                        ch_diff = np.abs(dfs['channel A'] - dfs['channel B'])
                        ch_diff = ch_diff == dist

                        cond = cond_a & cond_b & ch_diff
                        dfs_filt = dfs[cond]
                        dfs_filt = np.array(dfs_filt.iloc[:, 1:-3])

                        acorr_trace = dfs_filt
                        acorr_trace = acorr_trace.squeeze()
                        mid_point = len(acorr_trace) // 2
                        acorr_trace = acorr_trace[
                             (mid_point - display_len):(mid_point + display_len)]
                        # Save the traces you'll perform tests on
                        acorr_traces_lists[k].append(acorr_trace)
                        print(ch1, ch2)

            # Collate the collected traces (for this ch-distance) into an array
            acorr_traces_lists[k] = np.array(acorr_traces_lists[k])

            # Perform statistic tests on array

            ## Gaussian confidence intervals
            mean_trace = np.mean(acorr_traces_lists[k], axis=0)
            low_intv, high_intv = spst.t.interval(0.99,
                len(acorr_traces_lists[k]) - 1,
                loc=mean_trace,
                scale=spst.sem(acorr_traces_lists[k]))

            ax[k].plot(np.arange(0 - display_len, 0 + display_len),
                       mean_trace, c='black',
                       alpha=1.0)  # /np.max(acorr_trace))
            ax[k].fill_between(np.arange(0 - display_len, 0 + display_len),
                               low_intv, high_intv, color='r', alpha=0.7)
        plt.savefig(acorr_dir + '/' + full_plot_title)
        plt.close()

    def synchrony_experiment1_overlapping_rec_fields_Plot_acorrs_individually(self): #not really used
        """ This prints 32 x 32 x 128 plots so is rarely used"""
        # Prepare general variables
        exp_dir, acorr_dir = \
            self.prepare_dirs(self.session_dir,
                              'SynchronyExp1',
                              'individual acorrs')

        for b in range(3):#128): # otherwise way too many
            acorr_data_dfs = \
                [pd.read_pickle(os.path.join(self.session_dir,
                     'cross_correlation_results_%s_%s.pkl' % (b, label)))
                     for label in ['during', 'outside']]

            max_peak = max(
                [np.max(np.max(df.iloc[:,1:-3])) for df in acorr_data_dfs])
            min_trough = min(
                [np.min(np.min(df.iloc[:,1:-3])) for df in acorr_data_dfs])

            # Plot results
            for i in range(len(acorr_data_dfs[0])):
                fig, ax = plt.subplots(2,figsize=(8,9))

                for j, during_or_outside_label in enumerate(['During', 'Outside']):
                    row_data = acorr_data_dfs[j].iloc[i]
                    ch1, ch2, _ = row_data[-3:]
                    print("%s %s %s" % (b, ch1, ch2))
                    acorr_trace = np.array(row_data)
                    acorr_trace = acorr_trace[0:-3]

                    # Plotting only middle of acorr plot
                    mid_point = len(acorr_trace) // 2
                    acorr_trace = acorr_trace[mid_point-400:mid_point+400]

                    ax[j].plot(np.arange(len(acorr_trace)),
                            acorr_trace)
                    ax[j].set_ylim([min_trough, max_peak])
                    ax[j].set_title(during_or_outside_label)
                    ax[j].set_xlabel('lag')
                    ax[j].set_ylabel('correlation coefficient')

                plt.savefig("%s/cross correlation between %i and %i in " +
                            "batch %i " % (exp_dir, ch1, ch2, b))
                plt.close()

    def synchrony_experiment1_overlapping_rec_fields_Plot_acorrs(self):
        """Collects the 128 batch dfs together
           Makes mean and var values for each channel for each angle across all
           batches
           Plots per channel"""
        # Prepare general variables
        exp_dir, acorr_dir = \
            self.prepare_dirs(self.session_dir, 'SynchronyExp1', 'acorrs')

        for b in range(128):
            acorr_data_dfs = [pd.read_pickle(os.path.join(self.session_dir,
                                'cross_correlation_results_%s_%s.pkl' % (b, label)))
                              for label in ['during', 'outside']]

            max_peak = max(
                [np.max(np.max(df.iloc[:,1:-3])) for df in acorr_data_dfs])
            min_trough = min(
                [np.min(np.min(df.iloc[:,1:-3])) for df in acorr_data_dfs])

            # Plot results
            for i in range(len(acorr_data_dfs[0])):
                fig, ax = plt.subplots(2,figsize=(8,9))

                #fig.tight_layout(pad=1.0)
                for j, during_or_outside_label in \
                        enumerate(['During', 'Outside']):
                    row_data = acorr_data_dfs[j].iloc[i]
                    ch1, ch2, _ = row_data[-3:]
                    print("%s %s %s" % (b, ch1, ch2))
                    acorr_trace = np.array(row_data)
                    acorr_trace = acorr_trace[0:-3]

                    # Plotting only middle of acorr plot
                    mid_point = len(acorr_trace) // 2
                    acorr_trace = acorr_trace[mid_point-400:mid_point+400]

                    ax[j].plot(np.arange(len(acorr_trace)),
                            acorr_trace)
                    ax[j].set_ylim([min_trough, max_peak])
                    ax[j].set_title(during_or_outside_label)
                    ax[j].set_xlabel('lag')
                    ax[j].set_ylabel('correlation coefficient')

                plt.savefig("%s/cross correlation between %i and %i in " +
                            "batch %i " % (exp_dir, ch1, ch2, b))
                plt.close()



    def synchrony_experiment1_overlapping_rec_fields_Analyze_fitted_Gabors(self):
        """Analyze the fited Gabor functions as in Gray et al. (1989).

        Needs to
         - Collect all the params in one df
         - Recreate the fitted line and count its peaks
         - Perform a one-sided t-test on the amplitudes for each stim.

        """
        print("Synch Exp 1 Analyze_fitted_Gabors")
        # Prepare general variables and dirs
        exp_dir, acorr_dir = self.prepare_dirs(self.session_dir,
                                               'SynchronyExp1',
                                               'acorrs_gabor_reconstructed')

        batch_groups = self.just_angles_few_angles_batchgroups

        plotting = False

        allb_params = pd.DataFrame()

        for bg_angle_index, bg in enumerate(batch_groups):

            # Go through all the batchelements for that stim
            for b in bg:
                df_title = "%s/estimated_gabor_params_batch_%i " % (exp_dir, b)
                batch_params_df = pd.read_pickle(df_title)
                allb_params = allb_params.append(batch_params_df,
                                                 ignore_index=True)
        allb_params['num_peaks'] = -1.

        for index, row in allb_params.iterrows():
            print("Row %i out of %i" % (index, len(allb_params)-1))
            # Create lists of params for use in sine func and gabor func
            amplitude0 = row['max_acorr'] * 0.7 # as in func that fits Gabors
            sine_params = [row['sine_phase'],
                           row['sine_mean'],
                           (1/row['freq_scale'])] # as in func that fits Gabors

            gabor_params = [row['gauss_sigma'], row['gauss_amp']]

            # Create sine curve
            sine_curve = lambda p: (
                        amplitude0 * np.cos(self.acorr_x * p[2] + p[0]) + p[1])

            # Use sine curve to create gabor curve
            fitted_sine = sine_curve(sine_params)

            gabor_func = lambda p: (
                    (1 / (np.sqrt(2 * np.pi) * p[0])) \
                    * np.exp(-(self.acorr_x ** 2) / (2 * p[0] ** 2)) *
                    p[1] * \
                    fitted_sine)

            # Calculate the curve
            gabor_curve = gabor_func(gabor_params)

            # Count peaks on fitted Gabor curve and add result to df row
            peaks = find_peaks(gabor_curve,
                               height=0,
                               prominence=0.01,
                               rel_height=3.)

            row['num_peaks'] = len(peaks[0])

            plotting = np.random.rand() > .999
            if plotting:
                print("Boop! plotting")
                # Plot results
                fig, ax = plt.subplots(1, figsize=(9, 5))
                ax.plot(self.acorr_x, gabor_curve)
                ax.scatter(self.acorr_x[peaks[0]], gabor_curve[peaks[0]])
                ax.set_xlabel('lag')
                ax.set_ylabel('correlation coefficient')
                fig.savefig("%s/diagnostic plot for fitted gabors_ch %i and %i in batch %i for %s.jpg" % (acorr_dir, int(row['channel A']), int(row['channel B']), int(row['batch']), str(row['dur_or_out'])))
                plt.close()
                plotting = False

        # Group allb_df by stim and then count mean number of peaks per stim
        grouping_vars = ['batch_group', 'dur_or_out', 'channel A', 'channel B']
        mean_num_peaks = []
        results_df_peaks = allb_params.groupby(grouping_vars, as_index=False).mean()

        # Do t-test to see if amplitude significantly different from 0
        groups = allb_params.groupby(grouping_vars, as_index=False)
        groups = [g for g in groups]
        groups = [(group[0], np.array(group[1]['amplitude']))
                  for group in groups]
        groups_results = [(g[0], spst.ttest_1samp(g[1], popmean=0.)) for g in
                          groups]
        groups_results = [list(g[0]) + list(g[1]) for g in groups_results]

        # Assign results to ttest dataframe
        ttest_df_column_names = grouping_vars
        ttest_df_column_names.extend(['stat_peaks', 'pvalue_peaks'])
        ttest_results_df = pd.DataFrame(columns=ttest_df_column_names)

        for i, name in enumerate(grouping_vars):
            nums = [g[i] for g in groups_results]
            ttest_results_df[name] = nums

        # Add mean amplitude value and num peaks to get final df
        results_df = ttest_results_df
        results_df['amplitude'] = results_df_peaks['amplitude']
        results_df['num_peaks'] = results_df_peaks['num_peaks']


        # Check if amplitude during is more than 10% of amplitude outside
        dur_or_out_groups = [g[1] for g in results_df.groupby(
            ['dur_or_out'], as_index=False)]
        dur_or_out_groups = [np.array(g['amplitude'])
                             for g in dur_or_out_groups]
        dur_out_comparison = dur_or_out_groups[0] > (0.1 * dur_or_out_groups[1])

        results_df['dur_out_amp_compar'] = -1.

        dur_index = results_df[results_df['dur_or_out'] == 0.0].index
        out_index = results_df[results_df['dur_or_out'] == 1.0].index

        results_df.loc[dur_index, 'dur_out_amp_compar'] = dur_out_comparison

        results_df.to_pickle(
            "%s/synchexp1 acorr analysis results.pkl" % self.session_dir)


    def synchrony_experiment1_overlapping_rec_fields_OriPref_vs_OscAmp(self):
        # Prepare general variables and dirs
        exp_dir = os.path.join(self.session_dir,
                               'SynchronyExp1')
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)

        bg_angles = self.just_angles_few_angles_batchgroupangles

        acorr_results_df = pd.read_pickle(
            "%s/synchexp1 acorr analysis results.pkl" % self.session_dir)

        # Exclude xcorr results for not during stimulus presentation
        acorr_results_df = acorr_results_df[acorr_results_df['dur_or_out']==0.]


        # Add ori pref data for each channel
        ori_pref = pd.read_pickle(
            os.path.join(self.session_dir,
                         'orientation_pref_results_just_angles.pkl'))

        ori_pref = ori_pref.drop(['mean', 'dir_or_ori', 'amplitude'], axis=1)
        ori_pref = ori_pref.rename(columns={'phaseshift': 'ori_pref'})

        ori_pref_a = ori_pref.rename(columns={'channel': 'channel A',
                                              'ori_pref': 'ori_pref A'})
        ori_pref_b = ori_pref.rename(columns={'channel': 'channel B',
                                              'ori_pref': 'ori_pref B'})

        # Get the ori prefs of channel A
        acorr_results_df = pd.merge(acorr_results_df, ori_pref_a)

        # Get the ori prefs of channel B
        acorr_results_df = pd.merge(acorr_results_df, ori_pref_b)

        # Calculate the difference in ori pref for each acorr comparison
        ##Old code for when using orientation selectivity
        # difference0 = acorr_results_df['ori_pref A'] - \
        #              acorr_results_df['ori_pref B']
        # difference180 = acorr_results_df['ori_pref A'] + np.pi - \
        #              acorr_results_df['ori_pref B']
        # difference = [min(d0, d180) for (d0, d180) in zip(difference0, difference180)]
        # angle_diff = np.pi - np.abs(np.abs(difference) - np.pi)

        ##New code for when using direction selectivity
        angle_diff = \
            [np.pi - np.abs(np.abs(a-b) - np.pi) for (a, b) in
                zip(acorr_results_df['ori_pref A'], acorr_results_df['ori_pref B'])]
        acorr_results_df['angle_diff'] = angle_diff

        # For coloring with neighbouring channels
        neighbours = np.abs(acorr_results_df['channel A'] - acorr_results_df['channel B'])
        acorr_results_df['neighbours'] = neighbours

        # Get data for the difference in ori pref and osc amplitude
        # not_self_cond = acorr_results_df['neighbours'] > -1
        not_self_cond = acorr_results_df['channel A'] != acorr_results_df['channel B']
        ori_pref_data = acorr_results_df['angle_diff'][not_self_cond]
        osc_amp_data = acorr_results_df['amplitude'][not_self_cond]

        ##Other secondarily important data
        num_peaks_data = acorr_results_df['num_peaks'][not_self_cond]

        # Fit a line and plot
        m, c = np.polyfit(ori_pref_data, osc_amp_data, 1)
        fig, ax = plt.subplots(1, figsize=(9, 5))
        ax.scatter(ori_pref_data, osc_amp_data)#, cmap='hsv', c=acorr_results_df['channel B'][not_self_cond])
        plt.plot(ori_pref_data, m * ori_pref_data + c, c='r')
        ax.set_xlabel('Angle difference')
        ax.set_ylabel('Oscillation amplitude')
        plt.savefig(os.path.join(exp_dir,
                                 "Angle_diff vs Oscillation amplitude.png"))
        plt.close()

        # Plot angle diff vs freq (secondary imporance)
        m, c = np.polyfit(ori_pref_data, num_peaks_data, 1)
        fig, ax = plt.subplots(1, figsize=(9, 5))
        ax.scatter(ori_pref_data, num_peaks_data)
        plt.plot(ori_pref_data, m * ori_pref_data + c, c='r')
        ax.set_xlabel('Angle difference')
        ax.set_ylabel('Num of peaks')
        plt.savefig(os.path.join(exp_dir,
                                 "Angle_diff vs Num Peaks.png"))
        plt.close()

        stim_angles = np.array([bg_angles[int(i)] for i in
                                acorr_results_df['batch_group']])

        # See whether there is a match between BOTH channels and the stim
        thresh_ang = 10

        ## Old code for when using orientation selectivity
        # match_0_A = (np.abs(acorr_results_df['ori_pref A'] - stim_angles) < (thresh_ang*np.pi/180))
        # match_180_A = (np.abs(acorr_results_df['ori_pref A'] + np.pi - stim_angles) < (thresh_ang*np.pi/180))
        # match_0_B = (np.abs(acorr_results_df['ori_pref B'] - stim_angles) < (thresh_ang * np.pi / 180))
        # match_180_B = (np.abs(acorr_results_df['ori_pref B'] + np.pi - stim_angles) < (thresh_ang * np.pi / 180))
        # orientation_match = \
        #     (match_0_A | match_180_A) & (match_0_B | match_180_B)

        ## New code for when using direction selectivity
        diff_A = acorr_results_df['ori_pref A'] - stim_angles
        match_A = (np.pi - np.abs(np.abs(diff_A) - np.pi) < (thresh_ang*np.pi/180))
        diff_B = acorr_results_df['ori_pref B'] - stim_angles
        match_B = (np.pi - np.abs(np.abs(diff_B) - np.pi) < (thresh_ang*np.pi/180))
        orientation_match = match_A & match_B



        ## Aligned channels & stims
        ori_pref_data_ali = \
            acorr_results_df['angle_diff'][not_self_cond & orientation_match]
        osc_amp_data_ali = \
            acorr_results_df['amplitude'][not_self_cond & orientation_match]


        # Calculate CIs
        # Bootstrapped CIs
        ori_pref_categs_ali = sorted(list(set(round(ori_pref_data_ali,7))))
        grouped_amps_ali = [osc_amp_data_ali[round(ori_pref_data_ali,7)==op] for op in ori_pref_categs_ali]
        bootstrap_res_ali = []
        for group in grouped_amps_ali:
            bs_res = bs.bootstrap(np.array(group), stat_func=bs_stats.mean, alpha=0.05)
            bootstrap_res_ali.append(bs_res)
        bs_means_ali = [bsr.value for bsr in bootstrap_res_ali]
        bs_low_intvs_ali = [bsr.lower_bound for bsr in bootstrap_res_ali]
        bs_high_intvs_ali = [bsr.upper_bound for bsr in bootstrap_res_ali]

        ## Unaligned channels & stims
        ori_pref_data_ua = \
            acorr_results_df['angle_diff'][not_self_cond & ~orientation_match]
        osc_amp_data_ua = \
            acorr_results_df['amplitude'][not_self_cond & ~orientation_match]

        ori_pref_categs_ua = sorted(list(set(round(ori_pref_data_ua,7))))
        grouped_amps_ua = [osc_amp_data_ua[round(ori_pref_data_ua,7)==op] for op in ori_pref_categs_ua]
        bootstrap_res_ua = []
        for group in grouped_amps_ua:
            bs_res = bs.bootstrap(np.array(group), stat_func=bs_stats.mean, alpha=0.05)
            bootstrap_res_ua.append(bs_res)
        bs_means_ua = [bsr.value for bsr in bootstrap_res_ua]
        bs_low_intvs_ua = [bsr.lower_bound for bsr in bootstrap_res_ua]
        bs_high_intvs_ua = [bsr.upper_bound for bsr in bootstrap_res_ua]

        fig, ax = plt.subplots(1,2, figsize=(9,4), sharey=True)
        m, c = np.polyfit(ori_pref_data_ali, osc_amp_data_ali, 1)
        ax[0].scatter(ori_pref_data_ali, osc_amp_data_ali, cmap='viridis', c=np.array(acorr_results_df['neighbours'][not_self_cond & orientation_match]).astype(int))
        ax[0].plot(ori_pref_data_ali, m * ori_pref_data_ali + c, c='r')
        ax[0].set_xlabel('Angle difference (rad)')
        ax[0].set_ylabel('Oscillation amplitude')

        m, c = np.polyfit(ori_pref_data_ua, osc_amp_data_ua, 1)
        ax[1].scatter(ori_pref_data_ua, osc_amp_data_ua, cmap='viridis', c=np.array(acorr_results_df['neighbours'][not_self_cond & ~orientation_match]).astype(int))
        ax[1].plot(ori_pref_data_ua, m * ori_pref_data_ua + c, c='r')
        ax[1].set_xlabel('Angle difference (rad)')
        fig.tight_layout()
        plt.savefig(os.path.join(exp_dir,
                                 "Angle_diff vs Oscillation amplitude w and wo ori match.png"))
        plt.close()


        # Prep for rad conversion:
        degree_symb = u"\u00b0"
        ticks_in_deg_left = np.arange(0, 12, 2)
        ticks_in_rad_left = [math.radians(deg) for deg in ticks_in_deg_left]

        ticks_in_deg_right = np.arange(0, 181, 20)
        ticks_in_rad_right = [math.radians(deg) for deg in ticks_in_deg_right]


        # Do CI plot
        fig, ax = plt.subplots(1,2, figsize=(9,4), sharey=True)

        ax[0].errorbar(ori_pref_categs_ali, bs_means_ali, yerr=[bs_low_intvs_ali, bs_high_intvs_ali], c='r', fmt='-o', capsize=3, elinewidth=1)
        ax[0].set_xlabel('Angle difference [%s]' % degree_symb)
        ax[0].set_xticks(ticks_in_rad_left)
        ax[0].set_xticklabels([str(deg) for deg in ticks_in_deg_left])
        ax[0].set_ylabel('Oscillation amplitude')
        ax[0].title.set_text('Aligned')


        ax[1].errorbar(ori_pref_categs_ua, bs_means_ua, yerr=[bs_low_intvs_ua, bs_high_intvs_ua], c='r', fmt='-o', capsize=2, elinewidth=0.5)
        ax[1].set_xticks(ticks_in_rad_right)
        ax[1].set_xticklabels([str(deg) for deg in ticks_in_deg_right])
        ax[1].set_xlabel('Angle difference [%s]' % degree_symb)
        ax[1].title.set_text('Unaligned')

        fig.tight_layout()
        plt.savefig(os.path.join(exp_dir,
                                 "Angle_diff vs Oscillation CI amplitude w and wo ori match.png"))
        plt.close()


    # SYNCH EXP 2
    def synchrony_experiment2_xcorrs(self, patch_or_idx=None):
        """Takes the column/patch of channels at the site of the static stimulus
        and the site of the mobile stimulus, then calculates
        the autocorrelation plot between each channel's trace. """

        print("Performing Synchrony Experiment 2: Two stimuli " +\
              "Fields")

        # Make dir to save plots for this experiment
        exp_dir = os.path.join(self.session_dir,
                               'SynchronyExp2')
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)

        # Load data
        presaved_path = os.path.join(self.session_dir,
            "neuron_activity_results_alternativeformat_double_stim.pkl")
        full_data = pd.read_pickle(presaved_path)
        # Add stim_orientation info to data
        stim_angles = [self.double_stim_angles[i] for i in
                       full_data['batch_idx']]
        full_data['stim_ori'] = stim_angles

        centre = 16

        if patch_or_idx == 'patch':
            centre1 = self.central_patch_min
            centre2 = self.central_patch_max
            central_patch_size = self.central_patch_size
            save_name = 'patch'
        else:
            centre1 = [16,16]
            centre2 = centre1
            central_patch_size = 1
            save_name = 'neuron'

        # Prepare general variables
        model_exp_name = self.double_stim_synchr_exp_name
        full_state_traces = pd.DataFrame()

        # Go through each channel and get the traces that you need (all
        # batches)
        for ch in range(self.num_ch):

            print("Channel %s" % str(ch))

            # Get data
            dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=model_exp_name,
                                         var_names=self.st_var_name,
                                         state_layers=[1],
                                         batches=None,
                                         channels=[ch],
                                         hw=[self.top_left_pnt,
                                             self.bottom_right_pnt],
                                         timesteps=None)

            # Process data
            activities_df, arr_sh, colnames, inds = \
                self.convert_ch_data_to_activity_df(dm)

            # Find the subset of the nrnact df for this channel only and get
            # the h, w values for on neurons in each batch
            # An extra condition for double stim experiments is that we only
            # select the neuron at the static stimulus location in all batches
            # but only the stimulus site for other batches

            nrnact_ch = full_data.loc[full_data['channel'] == ch]

            mobile_locs = \
                [self.double_stim_locs[i] for i in nrnact_ch['batch_idx']]
            mobile_locs_x = \
                [centre + mobile_locs[i][1] for i in range(len(mobile_locs))]
            mobile_locs_y = \
                [centre + mobile_locs[i][0] for i in range(len(mobile_locs))]

            mobile_cond = (nrnact_ch['height'] == mobile_locs_y) & \
                          (nrnact_ch['width'] == mobile_locs_x)
            static_cond = (nrnact_ch['height'] == self.double_stim_static_y) &\
                          (nrnact_ch['width'] == self.double_stim_static_x)

            cond = static_cond | mobile_cond
            nrnact_ch = nrnact_ch.loc[cond]
            nrnact_ch = nrnact_ch.reset_index()

            # Get the names of the columns
            on_colnames, on_col_inds, batches, heights, widths, stim_oris = \
                self.get_info_lists_from_nrnactch(nrnact_ch)

            # For the  in the above-defined subset of nrnact, get
            # their timeseries from the arrays
            state_traces = activities_df[on_colnames]
            state_traces = state_traces.transpose()
            state_traces['batch']     = batches
            state_traces['height']    = heights
            state_traces['width']     = widths
            state_traces['stim_oris'] = stim_oris
            state_traces['channel']   = ch

            if full_state_traces.empty:
                full_state_traces = state_traces
            else:
                full_state_traces = full_state_traces.append(state_traces)

        # Now, having collected all the state traces of the neurons of
        # interest, compute cross correlation plots between neurons in
        # different channels in the same batch element:
        for b in range(self.num_batches):


            acorr_data_during = pd.DataFrame()
            acorr_data_outside = pd.DataFrame()
            acorr_data_dfs = [acorr_data_during, acorr_data_outside]

            # Define the mobile neuron location for this batch
            mobile_loc = self.double_stim_locs[b]
            mobile_loc_x = centre + mobile_loc[1]
            mobile_loc_y = centre + mobile_loc[0]

            cond_mobilex = full_state_traces['width'] == mobile_loc_x
            cond_mobiley = full_state_traces['height'] == mobile_loc_y

            cond_staticx = \
                full_state_traces['width'] == self.double_stim_static_x
            cond_staticy = \
                full_state_traces['height'] == self.double_stim_static_y

            cond_b = full_state_traces['batch'] == b

            for ch1 in range(self.num_ch):
                for ch2 in range(self.num_ch):

                    print("%s   %s    %s" % (b, ch1, ch2))

                    # Define conditions to get traces from full df
                    cond1_ch = full_state_traces['channel'] == ch1
                    cond2_ch = full_state_traces['channel'] == ch2

                    cond1 = cond1_ch & cond_mobilex & cond_mobiley & cond_b
                    cond2 = cond2_ch & cond_staticx & cond_staticy & cond_b

                    if not np.any(cond1) or not np.any(cond2):
                        print("Skipping %s   %s    %s" % (b, ch1, ch2))
                        continue

                    # Get traces from full df
                    trace_1 = full_state_traces.loc[cond1]
                    trace_2 = full_state_traces.loc[cond2]

                    # Convert to array
                    trace_1 = np.array(trace_1).squeeze()
                    trace_2 = np.array(trace_2).squeeze()

                    # Remove channel,batch information to get only traces
                    trace_1 = trace_1[0:self.full_trace_len]
                    trace_2 = trace_2[0:self.full_trace_len]

                    # Split traces into 'during stim' and 'outside stim'
                    t1_duringstim = \
                        trace_1[self.primary_stim_start:self.primary_stim_stop]
                    t2_duringstim = \
                        trace_2[self.primary_stim_start:self.primary_stim_stop]
                    traces_during = [t1_duringstim, t2_duringstim]

                    t1_outsidestim = \
                        trace_1[self.burn_in_len:self.primary_stim_start]
                    t2_outsidestim = \
                        trace_2[self.burn_in_len:self.primary_stim_start]
                    traces_outside = [t1_outsidestim, t2_outsidestim]

                    traces_d_and_o = [traces_during, traces_outside]
                    for j, (during_or_outside_label, traces) in \
                            enumerate(zip(['during','outside'],
                                          traces_d_and_o)):
                        t1 = traces[0]
                        t2 = traces[1]

                        # Run acorr funcs and save results
                        acorr_result = np.correlate(t1-np.mean(t1),
                                                    t2-np.mean(t2),
                                                    mode='full')
                        acorr_result = pd.DataFrame(acorr_result).transpose()
                        acorr_result['channel_static'] = ch1
                        acorr_result['channel_mobile'] = ch2
                        acorr_result['mob_loc_x'] = mobile_loc_x
                        acorr_result['mob_loc_y'] = mobile_loc_y
                        acorr_result['batch'] = b
                        acorr_data_dfs[j] = acorr_data_dfs[j].append(acorr_result)
            acorr_data_dfs = [df.reset_index() for df in acorr_data_dfs]

            # Save results periodically, per batch
            for ac_df, label in zip(acorr_data_dfs,['during', 'outside']):
                ac_df.to_pickle(
                    os.path.join(exp_dir,
                        'cross_correlation_results_%s_%s.pkl' % (b, label)) )

    def synchrony_experiment2_fit_Gabors(self):
        """Fit Gabor functions to xcorr plots as in Gray et al. (1989)."""
        # Prepare general variables and dirs
        exp_dir, acorr_dir = self.prepare_dirs(self.session_dir,
                                               'SynchronyExp2',
                                               'acorrs_gabor_fitted')

        batch_groups = self.double_stim_batchgroups
        plotting = False

        # Set initial gabor params (gaussian components) [sigma, amplitude]
        p01 = [7e1, 400.0]
        p02 = [7e1, 500.0]
        p03 = [7e1, 600.0]
        init_params_gabor = [p01, p02, p03]


        for bg_angle_index, bg in enumerate(batch_groups):

            # Collect all the dfs for this batch group (batch group = one repeated stim)
            dfs_all = \
                [[pd.read_pickle(os.path.join(exp_dir,
                     'cross_correlation_results_%s_%s.pkl' % (b, label)))
                for label in ['during', 'outside']] for b in bg]

            # Go through all the batchelements for that stim
            for b in bg:
                print(b)
                param_df = pd.DataFrame()  # df to save results
                # which is saved every batch

                for i, label in enumerate(['during', 'outside']):

                    # Select the df for this batch and this time-period
                    acorr_df = dfs_all[b-(len(bg)*bg_angle_index)][i]

                    for ch1 in range(self.num_ch):
                        for ch2 in range(self.num_ch):
                            # Select the specific xcorr plot
                            print("Batch: %s ;   Channels %s %s" % (b, ch1, ch2))
                            cond1 = acorr_df['channel_static'] == ch1
                            cond2 = acorr_df['channel_mobile'] == ch2
                            cond = cond1 & cond2
                            acorr_data = acorr_df[cond]

                            x_loc = acorr_data['mob_loc_x']
                            y_loc = acorr_data['mob_loc_y']

                            acorr_data = np.array(acorr_data).squeeze()
                            acorr_data = acorr_data[1:-5]
                            midpoint = len(acorr_data) // 2
                            acorr_data = \
                                acorr_data[(midpoint-self.acorr_mid_len):
                                           (midpoint+self.acorr_mid_len)]

                            # Fit sine function first
                            ## Take a guess for lambda by counting peaks. This
                            ## works because most look pretty oscillatory to
                            ## begin with.
                            peaks = find_peaks(acorr_data,
                                               height=0,
                                               rel_height=3.,
                                               prominence=np.max(acorr_data)/5,
                                               distance=3)
                            avg_peak_dist = (self.acorr_mid_len*2) / \
                                            len(peaks[0])

                            ## Set init sine params
                            max_acorr = np.max(acorr_data) * 0.7
                            amplitude0 = max_acorr
                            guess_lmda = 7/avg_peak_dist # 7 is just a constant
                            # that seemed to work

                            num_incr = 5
                            negs = num_incr // 2
                            exps = range(-negs, num_incr-negs)
                            lambdas = [guess_lmda * (1.15**n) for n in exps]
                            phaseshift0 = 0.
                            mean0 = 0.
                            init_params_sine = [[phaseshift0, mean0,
                                              lambdas[i]]
                                            for i in range(len(lambdas))]
                            sine_curve = lambda p: (amplitude0 * np.cos(self.acorr_x * p[2] + p[0]) + p[1]) #* (1 / (np.sqrt(2 * np.pi) * base_sd)) * np.exp(-(self.acorr_x ** 2) / (2 * base_sd ** 2)) * 70
                            opt_func_sine = lambda p: sine_curve(p) - acorr_data

                            # Go through each of the init param settings
                            # for fitting the SINE component of the Gabor and
                            # choose the one with the lowest resulting cost
                            costs_sine = []
                            est_params_sine = []
                            for p0x in init_params_sine:
                                opt_result = \
                                    optimize.least_squares(opt_func_sine, p0x)
                                est_params_sine.append(opt_result.x)
                                costs_sine.append(opt_result.cost)
                            print(np.argmin(costs_sine))
                            cost_sine = costs_sine[np.argmin(costs_sine)]
                            est_params_sine = \
                                list(est_params_sine[np.argmin(costs_sine)])
                            fitted_sine = sine_curve(est_params_sine)


                            # Fit Gaussian component of Gabor function
                            gabor_func = lambda p: (
                                        (1 / (np.sqrt(2 * np.pi) * p[0]))
                                        * np.exp(-(self.acorr_x ** 2)
                                                 / (2 * p[0] ** 2)) *
                                        p[1] *
                                        fitted_sine)
                            opt_func = lambda p: gabor_func(p) - acorr_data

                            costs_gabor = []
                            est_params_gabor = []
                            for p0x in init_params_gabor:
                                opt_result = \
                                    optimize.least_squares(opt_func, p0x)
                                est_params_gabor.append(opt_result.x)
                                costs_gabor.append(opt_result.cost)
                            print(np.argmin(costs_gabor))
                            orig_params = \
                                init_params_gabor[np.argmin(costs_gabor)]
                            cost_gabor = \
                                costs_gabor[np.argmin(costs_gabor)]
                            print(cost_gabor)
                            est_params_gabor = \
                                list(est_params_gabor[np.argmin(costs_gabor)])
                            print(str(est_params_gabor))
                            est_params_gabor.append(costs_gabor)

                            # Take stock of and save results
                            est_gabor_curve = gabor_func(est_params_gabor)
                            min_gab = np.min(est_gabor_curve)
                            max_gab = np.max(est_gabor_curve)
                            est_params_dict = {'channel A': ch1,
                                               'channel B': ch2,
                                               'mob_loc_x': x_loc,
                                               'mob_loc_y': y_loc,
                                               'dur_or_out': i,
                                               'batch': b,
                                               'batch_group': bg_angle_index,
                                               'max_acorr': max_acorr,
                                               'min_gabor': min_gab,
                                               'max_gabor': max_gab,
                                               'amplitude': (max_gab-min_gab)/2,
                                               'freq_scale': (1/est_params_sine[2]),
                                               'sine_phase': est_params_sine[0],
                                               'sine_mean': est_params_sine[1],
                                               'gauss_sigma':est_params_gabor[0],
                                               'gauss_amp': est_params_gabor[1],
                                               'cost_sine':  cost_sine,
                                               'cost_gabor': cost_gabor}
                            param_df = \
                                param_df.append(est_params_dict,
                                                ignore_index=True)

                            plotting = np.random.rand() > .999
                            if plotting:
                                # Plot results
                                fig, ax = plt.subplots(1, figsize=(9, 5))
                                ax.plot(self.acorr_x, acorr_data)
                                ax.plot(self.acorr_x, est_gabor_curve )
                                ax.scatter(self.acorr_x[peaks[0]],
                                           acorr_data[peaks[0]])
                                ax.set_xlabel('lag')
                                ax.set_ylabel('correlation coefficient')
                                plt.savefig(
                                    "%s/Xcorr between static ch %i and mobile ch %i at (x%i,y%i) in batch %i for %s" % (acorr_dir, ch1, ch2, x_loc,y_loc,b,label))
                                plt.close()
                                plotting = False

                df_title = "%s/estimated_gabor_params_batch_%i " % (exp_dir, b)
                param_df.to_pickle(df_title)

    def prepare_dirs(self, root_dir, main_dir_name, subdir_name=None):
        """Creates directories for saving of data and plots

        Optionally, it creates a subdirectory.

        Returns the full string for the location of the dir(s)
        """

        # Create main dir
        main_dir = os.path.join(root_dir, main_dir_name)
        if not os.path.isdir(main_dir):
            os.mkdir(main_dir)
        if subdir_name is None:
            return main_dir

        # Create sub dir
        if subdir_name is not None:
            sub_dir = os.path.join(main_dir, subdir_name)
            if not os.path.isdir(sub_dir):
                os.mkdir(sub_dir)

            return main_dir, sub_dir

    def synchrony_experiment2_Analyze_fitted_Gabors(self):
        """Analyze the fitted Gabor functions as in Gray et al. (1989).

        Needs to
         - Collect all the params in one df
         - Recreate the fitted line and count its peaks
         - Perform a one-sided t-test on the amplitudes for each stim.
        """
        # Prepare general variables and dirs
        exp_dir, acorr_dir = self.prepare_dirs(self.session_dir,
                                               'SynchronyExp2',
                                               'acorrs_gabor_reconstructed')

        allb_params_title = \
            "%s/estimated_gabor_params_all_batches.pkl" % exp_dir
        batch_groups = self.double_stim_batchgroups
        plotting = False

        overwrite_allb_params = True
        allb_params = pd.DataFrame()
        for bg_angle_index, bg in enumerate(batch_groups):
            print(bg)
            # Go through all the batchelements for that stim and collect data
            for b in bg:
                df_title = "%s/estimated_gabor_params_batch_%i " % (exp_dir, b)
                batch_params_df = pd.read_pickle(df_title)
                allb_params = allb_params.append(batch_params_df,
                                                 ignore_index=True)
        allb_params['num_peaks'] = -1.

        for index, row in allb_params.iterrows():
            print("Row %i out of %i" % (index, len(allb_params)-1))
            # Create lists of params for use in sine func and gabor func
            amplitude0 = row['max_acorr'] * 0.7 # as in func that fits Gabors
            sine_params = [row['sine_phase'],
                           row['sine_mean'],
                           (1/row['freq_scale'])] # as in func that fits Gabors

            gabor_params = [row['gauss_sigma'], row['gauss_amp']]

            # Create sine curve
            sine_curve = lambda p: (
                        amplitude0 * np.cos(self.acorr_x * p[2] + p[0]) + p[1])

            # Use sine curve to create gabor curve
            fitted_sine = sine_curve(sine_params)

            gabor_func = lambda p: (
                    (1 / (np.sqrt(2 * np.pi) * p[0])) \
                    * np.exp(-(self.acorr_x ** 2) / (2 * p[0] ** 2)) *
                    p[1] * \
                    fitted_sine)

            # Calculate the curve
            gabor_curve = gabor_func(gabor_params)

            # Count peaks on fitted Gabor curve and add result to df row
            peaks = find_peaks(gabor_curve,
                               height=0,
                               prominence=0.01,
                               rel_height=3.)

            row['num_peaks'] = len(peaks[0])


            allb_params.loc[index] = row

            plotting = np.random.rand() > .999
            if plotting:
                # Plot results
                fig, ax = plt.subplots(1, figsize=(9, 5))
                ax.plot(self.acorr_x, gabor_curve)
                ax.scatter(self.acorr_x[peaks[0]], gabor_curve[peaks[0]])
                ax.set_xlabel('lag')
                ax.set_ylabel('correlation coefficient')

                fig.savefig(
                 "%s/diagnostic plot for fitted gabors_ch %i and %i in batch %i for %s.jpg" % (
                     acorr_dir, int(row['channel A']), int(row['channel B']),
                     int(row['batch']), str(row['dur_or_out'])))
                plt.close()
                plotting = False

        # Save results
        if overwrite_allb_params and not os.path.exists(allb_params_title):
            allb_params.to_pickle(allb_params_title)

        # Group allb_df by stim and then count mean number of peaks per stim
        grouping_vars = ['batch_group', 'dur_or_out', 'channel A', 'channel B']
        mean_num_peaks = []
        results_df_peaks = allb_params.groupby(
            grouping_vars, as_index=False).mean()

        # Do t-test to see if amplitude significantly different from 0
        groups = allb_params.groupby(grouping_vars, as_index=False)
        amp_groups = [g for g in groups]
        amp_groups = [(group[0], np.array(group[1]['amplitude']))
                      for group in amp_groups]
        amp_groups_results = [(g[0], spst.ttest_1samp(g[1], popmean=0.))
                              for g in amp_groups]
        amp_groups_results = [list(g[0]) + list(g[1])
                              for g in amp_groups_results]

        # Do another t-test to see if phaseshift significantly different from 0
        phase_groups = [g for g in groups]
        phase_groups = [(group[0], np.array(group[1]['sine_phase']))
                  for group in phase_groups]
        phase_groups_results = [(g[0], spst.ttest_1samp(g[1], popmean=0.))
                                for g in phase_groups]
        all_groups_results = [ar + list(tt[1]) for ar, tt in
                              zip(amp_groups_results,phase_groups_results)]

        # Assign results to ttest dataframe
        ttest_df_column_names = grouping_vars
        ttest_df_column_names.extend(['amp_stat_peaks',
                                      'amp_pvalue_peaks',
                                      'phase_stat_peaks',
                                      'phase_pvalue_peaks'])
        ttest_results_df = pd.DataFrame(columns=ttest_df_column_names)

        for i, name in enumerate(grouping_vars):
            nums = [g[i] for g in all_groups_results]
            ttest_results_df[name] = nums

        # Add mean amplitude value and num peaks to get final df
        results_df = ttest_results_df
        results_df['amplitude'] = results_df_peaks['amplitude']
        results_df['sine_phase'] = results_df_peaks['sine_phase']
        results_df['num_peaks'] = results_df_peaks['num_peaks']

        # Check if amplitude during is more than 10% of amplitude outside
        dur_or_out_groups = [g[1] for g in results_df.groupby(
            ['dur_or_out'], as_index=False)]
        dur_or_out_groups = [np.array(g['amplitude'])
                             for g in dur_or_out_groups]
        dur_out_comparison = dur_or_out_groups[0] > (0.1*dur_or_out_groups[1])
        results_df['dur_out_amp_compar'] = -1.

        dur_index = results_df[results_df['dur_or_out'] == 0.0].index
        out_index = results_df[results_df['dur_or_out'] == 1.0].index

        results_df.loc[dur_index, 'dur_out_amp_compar'] = dur_out_comparison

        results_df.to_pickle(
            "%s/synchexp2 acorr analysis results.pkl" % self.session_dir)


    def synchrony_experiment2_OriPref_OR_Distance_vs_OscAmp_OR_vs_Phase(self):
        print("synchrony_experiment2_OriPref_OR_Distance_vs_OscAmp_OR_vs_Phase")
        # Prepare general variables and dirs
        exp_dir, acorr_dir = self.prepare_dirs(self.session_dir,
                                               'SynchronyExp2',
                                               'acorrs_gabor_reconstructed')

        anlsys_res_title = "%s/synchexp2 acorr analysis results.pkl" % \
                           self.session_dir

        batch_groups = self.double_stim_batchgroups
        bg_angles = self.double_stim_batchgroupangles
        bg_locs = self.double_stim_batchgroup_locs

        cmap = 'spring'
        thresh_ang = 10

        acorr_results_df = pd.read_pickle(anlsys_res_title)

        # Exclude xcorr results for not during stimulus presentation
        acorr_results_df = acorr_results_df[acorr_results_df['dur_or_out']==0.]

        ori_pref = pd.read_pickle(
            os.path.join(self.session_dir,
                         'orientation_pref_results_just_angles.pkl'))
        ori_pref = ori_pref.drop(['mean', 'amplitude'], axis=1)
        ori_pref = ori_pref.rename(columns={'phaseshift': 'ori_pref'})
        ori_pref_a = ori_pref.rename(columns={'channel': 'channel A',
                                              'ori_pref': 'ori_pref A'})
        ori_pref_b = ori_pref.rename(columns={'channel': 'channel B',
                                              'ori_pref': 'ori_pref B'})

        # Get the ori prefs of channel A
        acorr_results_df = pd.merge(acorr_results_df, ori_pref_a)

        # Get the ori prefs of channel B
        acorr_results_df = pd.merge(acorr_results_df, ori_pref_b)

        # Calculate the difference in ori pref for each acorr comparison
        difference = acorr_results_df['ori_pref A'] - \
                     acorr_results_df['ori_pref B']
        angle_diff = np.pi - np.abs(np.abs(difference) - np.pi)
        acorr_results_df['angle_diff'] = angle_diff

        # Add the location data for the mobile stim in each batchgroup
        locs = [bg_locs[int(i)] for i in acorr_results_df['batch_group']]
        mob_locs_x = [loc[1] for loc in locs]
        mob_locs_y = [loc[0] for loc in locs]
        acorr_results_df['mob_loc_x'] = mob_locs_x
        acorr_results_df['mob_loc_y'] = mob_locs_y

        ## Make mobile loc relative to static loc
        acorr_results_df['mob_loc_x'] = acorr_results_df['mob_loc_x'] - \
            self.double_stim_static_x_0centre
        acorr_results_df['mob_loc_y'] = acorr_results_df['mob_loc_y'] - \
            self.double_stim_static_y_0centre

        # Calculate (Manhatten/square) distance of mobile from static stim
        # as the max distance along an axis
        xbiggery = \
            acorr_results_df['mob_loc_x'] > acorr_results_df['mob_loc_y']
        acorr_results_df['mob_sq_dist'] = acorr_results_df['mob_loc_x'].where(
            xbiggery, other=acorr_results_df['mob_loc_y'])

        # Calculate euclidian distance of mobile from static stim
        centred_x_locs = [xy[1] - self.double_stim_static_x_0centre for xy in
         self.double_stim_batchgroup_locs]
        centred_y_locs = [xy[0] - self.double_stim_static_y_0centre for xy in
            self.double_stim_batchgroup_locs]
        eucl_dists = [np.sqrt(x**2 + y**2)
                 for (x,y) in zip(centred_x_locs, centred_y_locs)]

        acorr_results_df['mob_eucl_dist'] = \
            [eucl_dists[int(i)] for i in acorr_results_df['batch_group']]

        # Assign stim angles
        stim_angles = np.array([bg_angles[int(i)] for i in
                                acorr_results_df['batch_group']])
        acorr_results_df['mob_stim_angle'] = stim_angles

        st_stim_angles = np.array([self.double_stim_batchgroupangles_static[int(i)] for i in
                                acorr_results_df['batch_group']])
        acorr_results_df['static_stim_angle'] = st_stim_angles


        # Define the conditions and their labels for plotting in a for-loop
        df = acorr_results_df  # for brevity

        # Plot the figure for the thesis.
        # X axis should be euclid distance.
        # It should compare the osc amp of colinear samples (i.e. those with
        # the same orientation as the stimulus AND below) with those that are
        # not colinear (not same orientation OR beside)

        #not_overlap_cond = df['mob_loc_x'] > -1 #for debugging
        not_overlap_cond = ~((df['mob_loc_x'] == 0) & (df['mob_loc_y']==0))

        ## colinear cond
        colinear_cond = (df['mob_stim_angle'] == df['static_stim_angle'])
        colinear_cond_label = "Static and mobile stims aligned and colinear"

        ## non colinear cond
        noncolinear_cond = (df['mob_stim_angle'] != df['static_stim_angle'])
        noncolinear_cond_label = "Static and mobile stims unaligned or not colinear"

        # Orientation matching cond (get neurons only in channels matched to the static stim)
        diff_A = df['ori_pref A'] - df['static_stim_angle']
        match_A = (np.pi - np.abs(np.abs(diff_A) - np.pi) < (
                thresh_ang * np.pi / 180))
        diff_B = df['ori_pref B'] - df['static_stim_angle']
        match_B = (np.pi - np.abs(np.abs(diff_B) - np.pi) < (
                thresh_ang * np.pi / 180))
        orientation_match_cond = match_A & match_B

        ## Get data based on conds
        colinear_indep_data     = acorr_results_df['mob_eucl_dist'][not_overlap_cond & colinear_cond & orientation_match_cond]
        colinear_dependent_data = acorr_results_df['amplitude'][not_overlap_cond & colinear_cond & orientation_match_cond]
        noncolinear_indep_data     = acorr_results_df['mob_eucl_dist'][not_overlap_cond & noncolinear_cond & orientation_match_cond]
        noncolinear_dependent_data = acorr_results_df['amplitude'][not_overlap_cond & noncolinear_cond & orientation_match_cond]

        # Bootstrap confidence intervals
        x_vals_colin = sorted(list(set(colinear_indep_data)))
        x_vals_noncolin = sorted(list(set(noncolinear_indep_data)))

        grouped_colinear_dep_data = \
            [colinear_dependent_data[colinear_indep_data==d]
             for d in x_vals_colin]
        grouped_noncolin_dep_data = \
            [noncolinear_dependent_data[noncolinear_indep_data==d]
             for d in x_vals_noncolin]

        bootstrap_res_colinear = []
        bootstrap_res_noncolin = []

        # bootstrap of colinear
        for group in grouped_colinear_dep_data:
            bs_res = bs.bootstrap(np.array(group),
                                  stat_func=bs_stats.mean,
                                  alpha=0.05)
            bootstrap_res_colinear.append(bs_res)
        bs_means_colin = \
            [bsr.value for bsr in bootstrap_res_colinear]
        bs_low_intvs_colin = \
            [bsr.lower_bound for bsr in bootstrap_res_colinear]
        bs_high_intvs_colin = \
            [bsr.upper_bound for bsr in bootstrap_res_colinear]

        # bootstrap of noncolinear
        for group in grouped_noncolin_dep_data:
            bs_res = bs.bootstrap(np.array(group),
                                  stat_func=bs_stats.mean,
                                  alpha=0.05)
            bootstrap_res_noncolin.append(bs_res)
        bs_means_noncolin = \
            [bsr.value for bsr in bootstrap_res_noncolin]
        bs_low_intvs_noncolin = \
            [bsr.lower_bound for bsr in bootstrap_res_noncolin]
        bs_high_intvs_noncolin = \
            [bsr.upper_bound for bsr in bootstrap_res_noncolin]


        ## Plot the main exp2 figures:
        m_colin, c_colin = np.polyfit(colinear_indep_data, colinear_dependent_data, 1)  # Calc line
        m_noncolin, c_noncolin = np.polyfit(noncolinear_indep_data, noncolinear_dependent_data, 1)  # Calc line

        fig, ax = plt.subplots(1,2, figsize=(11, 5), sharex=True, sharey=True)
        ax[0].title.set_text('Colinear')
        ax[0].scatter(colinear_indep_data, colinear_dependent_data, cmap=cmap,
                   c=acorr_results_df['channel A'][not_overlap_cond & colinear_cond & orientation_match_cond])
        ax[0].plot(colinear_indep_data, m_colin * colinear_indep_data + c_colin, c='r')  # Plot line
        ax[0].set_xlabel('Stimuli separation distance')
        ax[0].set_ylabel('Oscillation amplitude')


        ax[1].title.set_text('Noncolinear')
        ax[1].scatter(noncolinear_indep_data, noncolinear_dependent_data, cmap=cmap,
                   c=acorr_results_df['channel A'][not_overlap_cond & noncolinear_cond & orientation_match_cond])
        ax[1].plot(noncolinear_indep_data, m_noncolin * noncolinear_indep_data + c_noncolin, c='r')  # Plot line
        ax[1].set_xlabel('Stimuli separation distance')

        plot_title = "Oscillation amplitude vs stimuli separation distance for colinear and noncolinear stimuli for neurons with orientation preference matched to static stimulus"
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir,
                                 "SynchExp2 " + plot_title + ".png"))
        plt.close()


        # Do CI plots
        fig, ax = plt.subplots(1,2, figsize=(11, 5), sharey=True)
        ax[0].title.set_text('Colinear')
        ax[0].errorbar(sorted(list(set(colinear_indep_data))), bs_means_colin, yerr=[bs_low_intvs_colin, bs_high_intvs_colin], c='r', fmt='-o', capsize=3, elinewidth=1)
        ax[0].set_xlabel('Stimuli separation distance')
        ax[0].set_ylabel('Oscillation amplitude')

        ax[1].title.set_text('Noncolinear')
        ax[1].errorbar(sorted(list(set(noncolinear_indep_data))), bs_means_noncolin, yerr=[bs_low_intvs_noncolin, bs_high_intvs_noncolin], c='r', fmt='-o', capsize=3, elinewidth=1)
        ax[1].set_xlabel('Stimuli separation distance')
        fig.tight_layout()
        ci_plot_title = "Oscillation amplitude vs stimuli separation distance CIs for colinear and noncolinear stimuli for neurons with orientation preference matched to static stimulus"
        plt.savefig(os.path.join(exp_dir,
                                 "SynchExp2 " + ci_plot_title + ".png"))
        plt.close()


    # SYNCH EXP 3
    def synchrony_experiment3_xcorrs(self, patch_or_idx=None): #TODO fix exp 3 so that it works with the new stims
        """Takes the column/patch of channels at the site of the static stimulus
        and the site of the mobile stimulus, then calculates
        the autocorrelation plot between each channel's trace. """

        print("Performing Synchrony Experiment 3: Long stimulus")

        # Make dir to save plots for this experiment
        exp_dir = os.path.join(self.session_dir,
                               'SynchronyExp3')
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)

        # Load data
        presaved_path = os.path.join(self.session_dir,
            "neuron_activity_results_alternativeformat_long_just_fewangles.pkl")
        full_data = pd.read_pickle(presaved_path)

        # Add stim_orientation info to data
        stim_angles = [self.long_just_angle_angles[i] for i in
                       full_data['batch_idx']]
        full_data['stim_ori'] = stim_angles


        if patch_or_idx == 'patch':
            centre1 = self.central_patch_min
            centre2 = self.central_patch_max
            central_patch_size = self.central_patch_size
            save_name = 'patch'
        else:
            centre = 16
            centre1 = [16,16] # softcode
            centre2 = centre1
            central_patch_size = 1
            save_name = 'neuron'

        # Prepare general variables
        model_exp_name = self.long_just_angles_exp_name
        full_state_traces = pd.DataFrame()

        # Go through each channel and get the traces that you need (all
        # batches)
        for ch in range(self.num_ch):

            print("Channel %s" % str(ch))

            # Get data
            dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=model_exp_name,
                                         var_names=self.st_var_name,
                                         state_layers=[1],
                                         batches=None,
                                         channels=[ch],
                                         hw=[self.top_left_pnt,
                                             self.bottom_right_pnt],
                                         timesteps=None)

            # Process data
            activities_df, arr_sh, colnames, inds = \
                self.convert_ch_data_to_activity_df(dm)

            # Find the subset of the nrnact df for this channel only and get
            # the h, w values for on neurons in each batch
            # In the third synch experiment we want to grab all the neurons
            # in the central column and row.
            # An extra condition for double stim experiments is that we only
            # select the neuron at the static stimulus location in all batches
            # but only the stimulus site for other batches

            nrnact_ch = full_data.loc[full_data['channel'] == ch]


            # Select a few locations along long stimulus
            bool_dfs = []
            for angle in self.long_just_angles_few_angles_batchgroupangles:
                locs = self.rclocs[angle]
                for loc in locs:
                    bool_df = (nrnact_ch['stim_ori'] == angle) & \
                              (nrnact_ch['width'] == loc[1]) & \
                              (nrnact_ch['height'] == loc[0])
                    bool_dfs.append(bool_df)

            # Collect all selected bools into one df
            for i, bdf in enumerate(bool_dfs):
                if i==0:
                    full_bdf = bdf
                else:
                    full_bdf = full_bdf | bdf

            # Get neuron info using full bool df
            nrnact_ch = nrnact_ch.loc[full_bdf]
            nrnact_ch = nrnact_ch.reset_index()

            # Get the names of the columns
            on_colnames, on_col_inds, batches, heights, widths, stim_oris = \
                self.get_info_lists_from_nrnactch(nrnact_ch)

            # For the  in the above-defined subset of nrnact, get
            # their timeseries from the arrays
            state_traces = activities_df[on_colnames]
            # state_traces.columns = self.true_contrast_and_angle_contrasts
            state_traces = state_traces.transpose()
            state_traces['batch']     = batches
            state_traces['height']    = heights
            state_traces['width']     = widths
            state_traces['stim_oris'] = stim_oris
            state_traces['channel']   = ch

            full_state_traces = state_traces

            centre_only_cond = (full_state_traces['height'] == centre) & \
                               (full_state_traces['width']  == centre)

            for b in range(self.num_batches):
                acorr_data_during = pd.DataFrame()
                acorr_data_outside = pd.DataFrame()
                acorr_data_dfs = [acorr_data_during, acorr_data_outside]

                print("b%s   ch%s  " % (b, ch))

                # Define conditions to get traces from full df
                angle = self.long_just_angle_angles[b]
                locs = self.rclocs[angle]

                ## Select a few locations along long stimulus
                bool_dfs = []
                for loc in locs:
                    bool_df = (full_state_traces['stim_oris'] == angle) & \
                              (full_state_traces['width'] == loc[1]) & \
                              (full_state_traces['height'] == loc[0])
                    bool_dfs.append(bool_df)

                ## Collect all selected bools into one df
                for i, bdf in enumerate(bool_dfs):
                    if i == 0:
                        stim_rc_cond = bdf
                    else:
                        stim_rc_cond = stim_rc_cond | bdf

                cond_b = full_state_traces['batch'] == b
                cond_ch = full_state_traces['channel'] == ch
                cond1 = centre_only_cond & cond_b & cond_ch
                cond2 = stim_rc_cond & cond_b & cond_ch

                if not np.any(cond1) or not np.any(cond2):
                    print("Skipping %s   %s  " % (b, ch))
                    continue

                # Get traces from full df
                trace_1  = full_state_traces.loc[cond1]
                traces_2 = full_state_traces.loc[cond2]

                # Convert to array
                trace_1 = np.array(trace_1).squeeze()

                # Remove channel,batch information to get only traces
                trace_1 = trace_1[0:self.full_trace_len]

                for r in range(len(traces_2)):
                    trace_2 = traces_2.iloc[r]
                    x = trace_2['width']
                    y = trace_2['height']
                    stim_ori = traces_2['stim_oris']

                    # Convert to array
                    trace_2 = np.array(trace_2).squeeze()

                    # Remove channel,batch information to get only traces
                    trace_2 = trace_2[0:self.full_trace_len]

                    # Split traces into 'during stim' and 'outside stim'
                    t1_duringstim = trace_1[self.primary_stim_start:self.primary_stim_stop]
                    t2_duringstim = trace_2[self.primary_stim_start:self.primary_stim_stop]
                    traces_during = [t1_duringstim, t2_duringstim]

                    t1_outsidestim = trace_1[self.burn_in_len:self.primary_stim_start]
                    t2_outsidestim = trace_2[self.burn_in_len:self.primary_stim_start]
                    traces_outside = [t1_outsidestim, t2_outsidestim]

                    traces_d_and_o = [traces_during, traces_outside]
                    for j, (during_or_outside_label, traces) in \
                            enumerate(zip(['during','outside'],
                                          traces_d_and_o)):
                        t1 = traces[0]
                        t2 = traces[1]

                        # Run acorr funcs and save results
                        acorr_result = np.correlate(t1-np.mean(t1),
                                                    t2-np.mean(t2),
                                                    mode='full')
                        acorr_result = pd.DataFrame(acorr_result).transpose()
                        acorr_result['stim_ori'] = stim_ori
                        acorr_result['channel'] = ch
                        acorr_result['x'] = x
                        acorr_result['y'] = y
                        acorr_result['batch'] = b
                        acorr_data_dfs[j] = acorr_data_dfs[j].append(acorr_result)
                acorr_data_dfs = [df.reset_index() for df in acorr_data_dfs]

                # Save results periodically, per batch
                for ac_df, label in zip(acorr_data_dfs,['during', 'outside']):
                    ac_df.to_pickle(os.path.join(exp_dir,
                                    'cross_correlation_results_%s_%s_%s.pkl' % (ch, b, label)) )

    def synchrony_experiment3_fit_Gabors(self):
        """Git Gabor functions to xcorr plots as in Gray et al. (1989)."""
        # Prepare general variables and dirs
        exp_dir, acorr_dir = self.prepare_dirs(self.session_dir,
                                               'SynchronyExp3',
                                               'acorrs_gabor_fitted')
        batch_groups = self.just_angles_few_angles_batchgroups
        plotting = False

        # Set initial gabor params (gaussian components) [sigma, amplitude]
        p01 = [7e1, 400.0]
        p02 = [7e1, 500.0]
        p03 = [7e1, 600.0]
        init_params_gabor = [p01, p02, p03]

        for bg_angle_index, bg in enumerate(batch_groups):

            # Collect all the dfs for this batch group (batch group = one stim)
            dfs_all = \
                [[[pd.read_pickle(os.path.join(exp_dir,
                     'cross_correlation_results_%s_%s_%s.pkl' % (ch, b, label)))
                for label in ['during', 'outside']] for b in bg] for ch in range(self.num_ch)]

            # Go through all the batchelements for that stim
            for b in bg:
                param_df = pd.DataFrame()  # df to save results,
                # saved every batch

                for i, label in enumerate(['during', 'outside']):
                    for ch in range(self.num_ch):
                        print("Batch: %s ;   Channel %s" % (b, ch))

                        # Select the df for this batch and this time-period
                        acorr_df = dfs_all[ch][b - (len(bg) * bg_angle_index)][i]


                        for r in range(len(acorr_df)):

                            acorr_data = acorr_df.iloc[r]

                            x_loc = acorr_data['x']
                            y_loc = acorr_data['y']

                            acorr_data = np.array(acorr_data).squeeze()
                            acorr_data = acorr_data[1:-5]
                            midpoint = len(acorr_data) // 2
                            acorr_data = acorr_data[(midpoint-self.acorr_mid_len):
                                                    (midpoint+self.acorr_mid_len)]

                            # Fit sine function first
                            ## Take a guess for lambda by counting peaks. This
                            ## works because most look pretty oscillatory to
                            ## begin with.
                            peaks = find_peaks(acorr_data,
                                               height=0,
                                               rel_height=3.,
                                               prominence=np.max(acorr_data) / 5,
                                               distance=3)
                            avg_peak_dist = (self.acorr_mid_len*2) / len(peaks[0])

                            ## Set init sine params
                            max_acorr = np.max(acorr_data) * 0.7
                            amplitude0 = max_acorr
                            guess_lmda = 7/avg_peak_dist # 7 is just a constant
                            # that seemed to work

                            num_incr = 5
                            negs = num_incr // 2
                            exps = range(-negs, num_incr-negs)
                            lambdas = [guess_lmda * (1.15**n) for n in exps]
                            phaseshift0 = 0.
                            mean0 = 0.
                            init_params_sine = [[phaseshift0, mean0,
                                              lambdas[j]]
                                            for j in range(len(lambdas))]
                            sine_curve = lambda p: (amplitude0 * np.cos(self.acorr_x * p[2] + p[0]) + p[1]) #* (1 / (np.sqrt(2 * np.pi) * base_sd)) * np.exp(-(self.acorr_x ** 2) / (2 * base_sd ** 2)) * 70
                            opt_func_sine = lambda p: sine_curve(p) - acorr_data

                            ## Go through each of the init param settings
                            ## for fitting the SINE component of the Gabor and
                            ## choose the one with the lowest resulting cost
                            costs_sine = []
                            est_params_sine = []
                            for p0x in init_params_sine:
                                opt_result = \
                                    optimize.least_squares(opt_func_sine, p0x)
                                est_params_sine.append(opt_result.x)
                                costs_sine.append(opt_result.cost)
                            print(np.argmin(costs_sine))
                            cost_sine = costs_sine[np.argmin(costs_sine)]
                            est_params_sine = \
                                list(est_params_sine[np.argmin(costs_sine)])
                            fitted_sine = sine_curve(est_params_sine)


                            # Fit Gaussian component of Gabor function
                            gabor_func = lambda p: (
                                        (1 / (np.sqrt(2 * np.pi) * p[0])) \
                                        * np.exp(-(self.acorr_x ** 2) / (2 * p[0] ** 2)) *
                                        p[1] * \
                                        fitted_sine)
                            opt_func = lambda p: gabor_func(p) - acorr_data

                            costs_gabor = []
                            est_params_gabor = []
                            for p0x in init_params_gabor:
                                opt_result = optimize.least_squares(opt_func, p0x)
                                est_params_gabor.append(opt_result.x)
                                costs_gabor.append(opt_result.cost)
                            print(np.argmin(costs_gabor))
                            orig_params = init_params_gabor[np.argmin(costs_gabor)]
                            cost_gabor = costs_gabor[np.argmin(costs_gabor)]
                            print(cost_gabor)
                            est_params_gabor = list(est_params_gabor[np.argmin(costs_gabor)])
                            print(str(est_params_gabor))
                            est_params_gabor.append(costs_gabor)

                            # Take stock of and save results
                            est_gabor_curve = gabor_func(est_params_gabor)
                            min_gab = np.min(est_gabor_curve)
                            max_gab = np.max(est_gabor_curve)
                            est_params_dict = {'channel': ch,
                                               'x': x_loc,
                                               'y': y_loc,
                                               'dur_or_out': i,
                                               'batch': b,
                                               'batch_group': bg_angle_index,
                                               'max_acorr': max_acorr,
                                               'min_gabor': min_gab,
                                               'max_gabor': max_gab,
                                               'amplitude': (max_gab-min_gab)/2,
                                               'freq_scale': (1/est_params_sine[2]),
                                               'sine_phase': est_params_sine[0],
                                               'sine_mean': est_params_sine[1],
                                               'gauss_sigma':est_params_gabor[0],
                                               'gauss_amp': est_params_gabor[1],
                                               'cost_sine':  cost_sine,
                                               'cost_gabor': cost_gabor}
                            param_df = \
                                param_df.append(est_params_dict,
                                                ignore_index=True)

                            plotting = np.random.rand() > .999
                            if plotting:
                                # Plot results
                                fig, ax = plt.subplots(1, figsize=(9, 5))
                                ax.plot(self.acorr_x, acorr_data)  # /np.max(acorr_trace))
                                # ax.plot(np.arange(len(acorr_data)),
                                #                      gabor_func(orig_params) )
                                ax.plot(self.acorr_x, est_gabor_curve )
                                ax.scatter(self.acorr_x[peaks[0]], acorr_data[peaks[0]])
                                # ax.plot(self.acorr_x, orig_sine )
                                # ax.plot(np.arange(len(acorr_data)),
                                #                      sine_curve(est_params_sine) )

                                ax.set_xlabel('lag')
                                ax.set_ylabel('correlation coefficient')

                                plt.savefig(
                                    "%s/Xcorr for ch %i between centre and (x%i,y%i) in batch %i for %s" % (
                                    acorr_dir, ch, x_loc,y_loc,b,label))
                                plt.close()
                                plotting = False

                df_title = "%s/estimated_gabor_params_batch_%i " % (exp_dir, b)
                param_df.to_pickle(df_title)


    def synchrony_experiment3_Analyze_fitted_Gabors(self):
        """Analyze the fitted Gabor functions as in Gray et al. (1989).

        Needs to
         - Collect all the params in one df
         - Recreate the fitted line and count its peaks
         - Perform a one-sided t-test on the amplitudes for each stim.
        """
        # Prepare general variables and dirs
        exp_dir, acorr_dir = self.prepare_dirs(self.session_dir,
                                               'SynchronyExp3',
                                               'acorrs_gabor_reconstructed')

        allb_params_title = "%s/estimated_gabor_params_all_batches.pkl" % exp_dir
        batch_groups = self.just_angles_few_angles_batchgroups #because
        # longjustangles batch groups are the same as short stim version.
        plotting = False

        overwrite_allb_params = True
        allb_params = pd.DataFrame()
        for bg_angle_index, bg in enumerate(batch_groups):
            print(bg)
            # Go through all the batchelements for that stim and collect data
            for b in bg:
                df_title = "%s/estimated_gabor_params_batch_%i " % (exp_dir, b)
                batch_params_df = pd.read_pickle(df_title)
                allb_params = allb_params.append(batch_params_df,
                                                 ignore_index=True)
        allb_params['num_peaks'] = -1.

        for index, row in allb_params.iterrows():
            print("Row %i out of %i" % (index, len(allb_params) - 1))
            # Create lists of params for use in sine func and gabor func
            amplitude0 = row['max_acorr'] * 0.7  # as in func that fits Gabors
            sine_params = [row['sine_phase'],
                           row['sine_mean'],
                           (1 / row[
                               'freq_scale'])]  # as in func that fits Gabors
            x, y = row['x'], row['y']
            gabor_params = [row['gauss_sigma'], row['gauss_amp']]

            # Create sine curve
            sine_curve = lambda p: (
                    amplitude0 * np.cos(self.acorr_x * p[2] + p[0]) + p[1])

            # Use sine curve to create gabor curve
            fitted_sine = sine_curve(sine_params)

            gabor_func = lambda p: (
                    (1 / (np.sqrt(2 * np.pi) * p[0])) \
                    * np.exp(-(self.acorr_x ** 2) / (2 * p[0] ** 2)) *
                    p[1] * \
                    fitted_sine)

            # Calculate the curve
            gabor_curve = gabor_func(gabor_params)

            # Count peaks on fitted Gabor curve and add result to df row
            peaks = find_peaks(gabor_curve,
                               height=0,
                               prominence=0.01,
                               rel_height=3.)

            row['num_peaks'] = len(peaks[0])

            allb_params.loc[index] = row

            plotting = np.random.rand() > .999
            if plotting:
                # Plot results
                fig, ax = plt.subplots(1, figsize=(9, 5))
                ax.plot(self.acorr_x, gabor_curve)
                ax.scatter(self.acorr_x[peaks[0]], gabor_curve[peaks[0]])
                ax.set_xlabel('lag')
                ax.set_ylabel('correlation coefficient')

                fig.savefig(
                    "%s/diagnostic plot for fitted gabors_ch%i in batch %i at (x%i, y%i) for %s.jpg" % (
                        acorr_dir, int(row['channel']),
                        int(row['batch']), int(x), int(y), str(row['dur_or_out'])))
                plt.close()
                plotting = False

        # Save results
        if overwrite_allb_params and not os.path.exists(allb_params_title):
            allb_params.to_pickle(allb_params_title)

        # Group allb_df by stim and then count mean number of peaks per stim
        grouping_vars = ['batch_group', 'dur_or_out', 'channel', 'x', 'y']
        results_df_peaks = allb_params.groupby(grouping_vars,
                                               as_index=False).mean()

        # Do t-test to see if amplitude significantly different from 0
        groups = allb_params.groupby(grouping_vars, as_index=False)
        amp_groups = [g for g in groups]
        amp_groups = [(group[0], np.array(group[1]['amplitude']))
                      for group in amp_groups]
        amp_groups_results = [(g[0], spst.ttest_1samp(g[1], popmean=0.))
                              for g in amp_groups]
        amp_groups_results = [list(g[0]) + list(g[1]) for g in
                              amp_groups_results]

        # Do another t-test to see if phaseshift significantly different from 0
        phase_groups = [g for g in groups]
        phase_groups = [(group[0], np.array(group[1]['sine_phase']))
                        for group in phase_groups]
        phase_groups_results = [(g[0], spst.ttest_1samp(g[1], popmean=0.))
                                for g in phase_groups]
        all_groups_results = [ar + list(tt[1]) for ar, tt in
                              zip(amp_groups_results, phase_groups_results)]

        # Assign results to ttest dataframe
        ttest_df_column_names = grouping_vars
        ttest_df_column_names.extend(
            ['amp_stat_peaks', 'amp_pvalue_peaks', 'phase_stat_peaks',
             'phase_pvalue_peaks'])
        ttest_results_df = pd.DataFrame(columns=ttest_df_column_names)

        for i, name in enumerate(grouping_vars):
            nums = [g[i] for g in all_groups_results]
            ttest_results_df[name] = nums

        # Add mean amplitude value and num peaks to get final df
        results_df = ttest_results_df
        results_df['amplitude'] = results_df_peaks['amplitude']
        results_df['sine_phase'] = results_df_peaks['sine_phase']
        results_df['num_peaks'] = results_df_peaks['num_peaks']

        # Check if amplitude during is more than 10% of amplitude outside
        dur_or_out_groups = [g[1] for g in results_df.groupby(
            ['dur_or_out'], as_index=False)]
        dur_or_out_groups = [np.array(g['amplitude'])
                             for g in dur_or_out_groups]
        dur_out_comparison = dur_or_out_groups[0] > (
                    0.1 * dur_or_out_groups[1])
        # dur_out_comparison = np.concatenate(
        #     (dur_out_comparison, np.ones_like(dur_out_comparison) * -1))
        results_df['dur_out_amp_compar'] = -1.

        dur_index = results_df[results_df['dur_or_out'] == 0.0].index
        out_index = results_df[results_df['dur_or_out'] == 1.0].index

        results_df.loc[dur_index, 'dur_out_amp_compar'] = dur_out_comparison
        # results_df.loc[out_index, 'dur_out_amp_compar'] = np.ones_like(dur_out_comparison) * -1.

        results_df.to_pickle(
            "%s/synchexp3 acorr analysis results.pkl" % self.session_dir)

    def synchrony_experiment3_Distance_vs_OscAmp(self):

        # Prepare general variables and dirs
        exp_dir, plot_dir = self.prepare_dirs(self.session_dir,
                                               'SynchronyExp3',
                                               'OscAmp vs distance plots')


        batch_groups = self.just_angles_few_angles_batchgroups
        bg_angles = self.long_just_angles_few_angles_batchgroupangles
        ten_degrees = (np.pi / 180) * 10
        centr = 0
        vert_ang = bg_angles[1]
        horiz_ang = bg_angles[0]

        cmap = 'hsv'

        # Get xcorr data
        anlsys_res_title = "%s/synchexp3 acorr analysis results.pkl" % \
                           self.session_dir
        acorr_results_df = pd.read_pickle(anlsys_res_title)

        ori_pref = pd.read_pickle(
            os.path.join(self.session_dir,
                         'orientation_pref_results_just_angles.pkl'))
        ori_pref = ori_pref.drop(['mean', 'amplitude'], axis=1)
        ori_pref = ori_pref.rename(columns={'phaseshift': 'ori_pref'})


        # Get the ori prefs of channels
        acorr_results_df = pd.merge(acorr_results_df, ori_pref)


        ## Make xy loc relative to centre loc
        acorr_results_df['x'] = acorr_results_df['x'] - \
                                self.extracted_im_size//2
        acorr_results_df['y'] = acorr_results_df['y'] - \
                                self.extracted_im_size//2

        # Calculate euclidian distances of second neuron from centre
        eucl_dists = [np.sqrt(x**2 + y**2)
                 for (x,y) in zip(acorr_results_df['x'], acorr_results_df['y'])]
        acorr_results_df['eucl_dist'] = eucl_dists

        # Assign stim angles
        stim_angles = np.array([bg_angles[int(i)] for i in
                                acorr_results_df['batch_group']])
        acorr_results_df['stim_angle'] = stim_angles

        # Calculate the difference in ori pref of channel from stim
        difference = acorr_results_df['ori_pref'] - \
                     acorr_results_df['stim_angle']
        angle_diff = np.pi - np.abs(np.abs(difference) - np.pi)
        acorr_results_df['oripref_stim_angle_diff'] = angle_diff


        # Select the right data for dur and out plots
        not_overlap_cond = ~((acorr_results_df['x'] == 0) & \
                             (acorr_results_df['y']==0))
        ori_match_cond = acorr_results_df['oripref_stim_angle_diff'] < ten_degrees
        dur_out_labels = ['During experimental stimulus','Before experimental stimulus']
        dur_out_conds = []
        dur_out_conds.append(acorr_results_df['dur_or_out']==0)
        dur_out_conds.append(acorr_results_df['dur_or_out']==1)


        # New scatter plot for exp3
        fig, ax = plt.subplots(2, figsize=(9, 9))
        for dur_out_idx, (dur_out_lab, dur_out_cond) in \
                enumerate(zip(dur_out_labels,
                              dur_out_conds)):
            print(dur_out_lab)
            full_cond = not_overlap_cond & dur_out_cond & ori_match_cond
            # Define data using conds
            indep_data = acorr_results_df['eucl_dist'][full_cond]
            dependent_data = acorr_results_df['amplitude'][full_cond]
            colors = acorr_results_df['ori_pref'][full_cond]
            # Plot basic plot
            m, c = np.polyfit(indep_data, dependent_data, 1)  # Calc line
            ax[dur_out_idx].title.set_text(dur_out_lab)
            ax[dur_out_idx].scatter(indep_data, dependent_data, cmap=cmap,c=colors)
            ax[dur_out_idx].plot(indep_data, m * indep_data + c,c='r')  # Plot line
            ax[dur_out_idx].set_xlabel("Distance in direction of aligned orientation preference (pixels)")
            ax[dur_out_idx].set_ylabel('Oscillation amplitude')
            ax[dur_out_idx].set_ylim([0.1, 0.4])
        plt.savefig(os.path.join(plot_dir,"SynchExp3 osc amp vs distance.png"))
        plt.close()

        # New CI plot for exp3
        fig, ax = plt.subplots(2, figsize=(9, 9))
        for dur_out_idx, (dur_out_lab, dur_out_cond) in \
                enumerate(zip(dur_out_labels,
                              dur_out_conds)):
            print(dur_out_lab)
            full_cond = not_overlap_cond & dur_out_cond & ori_match_cond
            # Define data using conds
            indep_data = acorr_results_df['eucl_dist'][full_cond]
            dependent_data = acorr_results_df['amplitude'][full_cond]
            colors = acorr_results_df['ori_pref'][full_cond]

            # Get CIs
            # Bootstrapped CIs
            indep_data_categs = sorted(list(set(round(indep_data, 7))))
            grouped_dependent_data = [
                dependent_data[round(indep_data, 7) == catg] for catg in
                indep_data_categs]
            bootstrap_res = []
            for group in grouped_dependent_data:
                bs_res = bs.bootstrap(np.array(group), stat_func=bs_stats.mean, alpha=0.05)
                bootstrap_res.append(bs_res)
            bs_means = [bsr.value for bsr in bootstrap_res]
            bs_low_intvs = [bsr.lower_bound for bsr in bootstrap_res]
            bs_high_intvs = [bsr.upper_bound for bsr in bootstrap_res]

            # Plot basic plot
            ax[dur_out_idx].title.set_text(dur_out_lab)
            ax[dur_out_idx].errorbar(indep_data_categs, bs_means,
                                     yerr=[bs_low_intvs, bs_high_intvs], c='r',
                                     fmt='-o', capsize=3, elinewidth=1)

            if dur_out_idx == 1:
                ax[dur_out_idx].set_xlabel(
                    "Distance in direction of aligned orientation preference (pixels)")
            ax[dur_out_idx].set_ylabel('Oscillation amplitude')
        plt.savefig(
            os.path.join(plot_dir, "SynchExp3 osc amp vs distance with CIs.png"))
        plt.close()


    def plot_state_and_mom_trace(self, patch_or_idx=None):
        # Make dir to save plots for this experiment
        exp_dir = self.prepare_dirs(self.session_dir,
                                    'state_and_mom_trace_plots')

        # Load data
        full_data = pd.read_pickle(os.path.join(self.session_dir,
            "neuron_activity_results_alternativeformat_primary_w_ori_w_match.pkl"))

        centre1 = self.top_left_pnt[0] + self.extracted_im_size / 2
        centre2 = centre1
        print("Plotting state and momentum traces")

        # Prepare general variables
        model_exp_name = self.primary_model_exp_name
        for ch in range(self.num_ch):
            # Get data
            dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=model_exp_name,
                                         var_names=self.st_var_name,
                                         state_layers=[1],
                                         batches=None,
                                         channels=[ch],
                                         hw=[self.top_left_pnt,
                                             self.bottom_right_pnt],
                                         timesteps=None)

            # Process data
            activities_df, arr_sh, colnames, inds = \
                self.convert_ch_data_to_activity_df(dm)

            # Find the subset of the nrnact df for this channel only and get
            # the h, w values for on neurons in each batch

            ## Select only the centre neuron trace
            nrnact_ch = full_data.loc[full_data['channel'] == ch]
            nrnact_ch.index = colnames[1:]
            cond3 = nrnact_ch['height'] >= centre1
            cond4 = nrnact_ch['height'] <= centre2
            cond5 = nrnact_ch['width'] >= centre1
            cond6 = nrnact_ch['width'] <= centre2
            cond = cond3 & cond4 & cond5 & cond6
            on_colnames = list(nrnact_ch.index[cond])
            nrnact_ch = nrnact_ch.loc[cond]
            on_nrn_states = activities_df.transpose().loc[on_colnames]
            nrnact_ch = pd.merge(nrnact_ch, on_nrn_states, left_index=True,
                                 right_index=True)

            ## Select only an arbitrary (last) batch element to display
            state_trace = nrnact_ch.loc[nrnact_ch['batch_idx']==127]

            ## Cut away the meta-data about the trace
            state_trace = np.array(state_trace)[0][-self.full_trace_len:]
            state_trace_during = state_trace[self.primary_stim_start:self.primary_stim_stop]
            state_trace_before = state_trace[self.burn_in_len:self.primary_stim_start]


            # Now do the same to get the momentum trace:
            del nrnact_ch, dm

            dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=model_exp_name,
                                         var_names=['momenta'],
                                         state_layers=[1],
                                         batches=None,
                                         channels=[ch],
                                         hw=[self.top_left_pnt,
                                             self.bottom_right_pnt],
                                         timesteps=None)

            # Process data
            activities_df, arr_sh, colnames, inds = \
                self.convert_ch_data_to_activity_df(dm, var_type='momenta_1')

            # Find the subset of the nrnact df for this channel only and get
            # the h, w values for on neurons in each batch
            ## Select only the centre neuron trace
            nrnact_ch = full_data.loc[full_data['channel'] == ch]
            nrnact_ch.index = colnames[1:]
            cond3 = nrnact_ch['height'] >= centre1
            cond4 = nrnact_ch['height'] <= centre2
            cond5 = nrnact_ch['width'] >= centre1
            cond6 = nrnact_ch['width'] <= centre2
            cond = cond3 & cond4 & cond5 & cond6
            on_colnames = list(nrnact_ch.index[cond])
            nrnact_ch = nrnact_ch.loc[cond]
            on_nrn_states = activities_df.transpose().loc[on_colnames]
            nrnact_ch = pd.merge(nrnact_ch, on_nrn_states, left_index=True,
                                 right_index=True)

            ## Select only an arbitrary (last) batch element to display
            momentum_trace = nrnact_ch.loc[nrnact_ch['batch_idx']==127]

            ## Cut away the meta-data about the trace
            momentum_trace = np.array(momentum_trace)[0][-self.full_trace_len:]
            ## Cut away the meta-data about the trace
            momentum_trace_during = momentum_trace[self.primary_stim_start:self.primary_stim_stop]
            momentum_trace_before = momentum_trace[self.burn_in_len:self.primary_stim_start]

            del nrnact_ch, dm

            # Cut traces to size for plotting
            start = self.primary_stim_start+300
            stop = self.primary_stim_start+600
            state_trace_full = state_trace.copy()
            momentum_trace_full = momentum_trace.copy()


            state_trace = state_trace[start:stop]
            momentum_trace = momentum_trace[start:stop]

            state_trace_during_trunc = state_trace_during[-300:] - np.mean(state_trace_during[-300:])
            state_trace_before_trunc = state_trace_before[-300:] - np.mean(state_trace_before[-300:])
            momentum_trace_during_trunc = momentum_trace_during[-300:] - np.mean(momentum_trace_during[-300:])
            momentum_trace_before_trunc = momentum_trace_before[-300:] - np.mean(momentum_trace_before[-300:])


            # Plot state and mom in different traces
            fig, ax = plt.subplots(2, sharex=True)
            fig.set_size_inches(8.5, 8.5)
            ax[0].plot(np.array(list(range(len(state_trace)))),
                       state_trace)
            ax[0].set_ylabel('State [a.u.]')
            ax[0].set_xlabel('Timesteps')
            ax[1].plot(np.array(list(range(len(momentum_trace)))),
                       -momentum_trace)
            ax[1].set_ylabel('Negative Momentum [a.u.]')
            plt.ticklabel_format(axis="y", style="sci", scilimits=(
                0, 0))  # Uses sci notation for units
            plt.tight_layout()  # Stops y label being cut off

            plt.savefig(
                os.path.join(exp_dir,
                             'State and momentum traces for central' + \
                             ' neuron of ch%i' % (ch)))
            plt.close()

            # Plot state and mom traces overlayed
            fig = plt.figure()
            ax = plt.subplot()
            stattr = ax.plot(np.array(list(range(len(state_trace)))),
                       state_trace)
            momtr = ax.plot(np.array(list(range(len(momentum_trace)))),
                       -momentum_trace, c='r')
            ax.set_ylabel('State or negative momentum [a.u.]')
            ax.set_xlabel('Timesteps')
            ax.legend(['State', 'Negative Momentum'])
            plt.tight_layout()  # Stops y label being cut off
            plt.savefig(
                os.path.join(exp_dir,
                             'State and momentum traces overlayed for central' + \
                             ' neuron of ch%i' % (ch)))
            plt.close()

            # Plot state and mom during and before, with overlayed traces
            fig, ax = plt.subplots(2, sharex=True)
            fig.set_size_inches(8.5, 8.5)
            # Remove borders etc
            ax[0].set_frame_on(False)
            ax[0].axes.get_yaxis().set_visible(False)
            ax[0].axes.get_xaxis().set_visible(False)
            ax[1].set_frame_on(False)
            ax[1].axes.get_yaxis().set_visible(False)

            ## During
            stattr = ax[0].plot(
                np.array(list(range(len(state_trace_during_trunc)))),
                state_trace_during_trunc)
            momtr = ax[0].plot(
                np.array(list(range(len(momentum_trace_during_trunc)))),
                -momentum_trace_during_trunc, c='r')
            ax[0].set_ylabel('State or negative momentum [a.u.]')
            ax[0].set_yticks([-0.15, 0.0, 0.15])
            ax[0].set_yticklabels([-0.15, 0.0, 0.15])
            ax[0].set_xticks([])

            ##Before
            stattr = ax[1].plot(
                np.array(list(range(len(state_trace_before_trunc)))),
                state_trace_before_trunc)
            momtr = ax[1].plot(
                np.array(list(range(len(momentum_trace_before_trunc)))),
                -momentum_trace_before_trunc, c='r')
            ax[1].set_ylabel('State or negative momentum [a.u.]')
            ax[1].set_xlabel('Timesteps (ms)')
            ax[1].set_yticks([-0.15, 0.0, 0.15])
            ax[1].set_yticklabels([-0.15, 0.0, 0.15])

            # Calculate labels in milliseconds
            ts_per = self.ms_to_timesteps(200, 8)
            ticks_locs_ts = np.arange(0, 7 * ts_per, ts_per)
            labels_ms = np.round(np.arange(0, 1300, 200))

            ax[1].set_xticks(ticks_locs_ts)
            ax[1].set_xticklabels(["t+%i" % int(k) for k in labels_ms])
            ax[1].legend(['State', 'Negative Momentum'])
            ax[1].legend_.get_frame().set_linewidth(0.0)

            ## Set titles
            ax[0].title.set_text(
                'During stimulation period')
            ax[1].title.set_text(
                'Before stimulation period')

            plt.tight_layout()  # Stops y label being cut off
            plt.savefig(
                os.path.join(exp_dir,
                             'State and momentum traces dur_and_out overlayed for central' + \
                             ' neuron of ch%i' % (ch)))
            plt.close()


            # Plot the acorr plots for the whole outside and during stim periods
            ## Calc acorr data
            state_trace_acorr_result_dur = \
                np.correlate(state_trace_during-np.mean(state_trace_during),
                             state_trace_during-np.mean(state_trace_during),
                             mode='full')
            state_trace_acorr_result_before = \
                np.correlate(state_trace_before-np.mean(state_trace_before),
                             state_trace_before-np.mean(state_trace_before),
                             mode='full')
            st_mom_xcorr_result = \
                np.correlate(
                    state_trace_full-np.mean(state_trace_full),
                    momentum_trace_full - np.mean(momentum_trace_full),
                    mode='full')

            ## Cut the acorr/xcorr results down to size
            centre_outdur_plots = len(state_trace_acorr_result_dur) // 2
            centre_full_trace_plots = len(st_mom_xcorr_result) // 2
            trc_mid_len = 100
            state_trace_acorr_result_dur = \
                state_trace_acorr_result_dur[centre_outdur_plots-trc_mid_len:centre_outdur_plots+trc_mid_len]
            state_trace_acorr_result_before = \
                state_trace_acorr_result_before[centre_outdur_plots-trc_mid_len:centre_outdur_plots+trc_mid_len]
            st_mom_xcorr_result = \
                st_mom_xcorr_result[centre_full_trace_plots-trc_mid_len:centre_full_trace_plots+trc_mid_len]

            ## finally make the plots
            fig, ax = plt.subplots(3, sharex=True, sharey=True)
            fig.set_size_inches(8.5, 8.5)

            ax[i].set_frame_on(False)

            ax[0].plot(range(-trc_mid_len,trc_mid_len),
                       state_trace_acorr_result_dur)
            ax[1].plot(range(-trc_mid_len,trc_mid_len),
                       state_trace_acorr_result_before)
            ax[2].plot(range(-trc_mid_len,trc_mid_len),
                       st_mom_xcorr_result)

            # Calculate labels in milliseconds
            ts_chunk = 100
            min_range = -4
            max_range = 5
            ts_per = self.ms_to_timesteps(ts_chunk, 8)
            ticks_locs_ts = np.arange(min_range * ts_per, max_range*ts_per, ts_per)
            labels_ms = np.round(np.arange(min_range*ts_chunk, max_range*ts_chunk, ts_chunk))
            for i in range(3):
                ax[i].set_frame_on(False)
                ax[i].axvline(x=0, c='black', linewidth=1, linestyle='dashed')
                ax[i].set_ylabel('Correlation')

                ax[i].set_xticks(ticks_locs_ts)
                ax[i].set_xticklabels(["%i" % int(k) for k in labels_ms])

            ax[0].title.set_text('State autocorrelation during stimulation period')
            ax[1].title.set_text('State autocorrelation before stimulation period')
            ax[2].title.set_text('Cross correlation between state and negative momentum')



            ax[2].set_xlabel('Lag (ms)')
            plt.tight_layout()
            plt.savefig(
                os.path.join(exp_dir,
                             'State acorrs during and outside stim and State-Mom xcorrs for central' + \
                             ' neuron of ch%i' % (ch)))
            plt.close()


            print("Boop")


    def calculate_EI_lag(self):
        # Make dir to save plots for this experiment
        exp_dir = self.prepare_dirs(self.session_dir,
                                    'Calculate EI lag')

        # Load data
        full_data = pd.read_pickle(os.path.join(self.session_dir,
            "neuron_activity_results_alternativeformat_primary_w_ori_w_match.pkl"))

        centre1 = self.top_left_pnt[0] + self.extracted_im_size / 2
        centre2 = centre1
        print("Calculating EI lag")

        xcorr_results = []
        trc_mid_len = 100
        lag_results = pd.DataFrame(columns=['channel', 'mean_lag'])

        # Prepare general variables
        model_exp_name = self.long_just_angles_exp_name

        for ch in range(self.num_ch):
            # Get the state data
            dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=model_exp_name,
                                         var_names=self.st_var_name,
                                         state_layers=[1],
                                         batches=None,
                                         channels=[ch],
                                         hw=[self.top_left_pnt,
                                             self.bottom_right_pnt],
                                         timesteps=None)

            # Process data
            activities_df, arr_sh, colnames, inds = \
                self.convert_ch_data_to_activity_df(dm)

            # Find the subset of the nrnact df for this channel only and get
            # the h, w values for on neurons in each batch

            ## Select only the centre neuron trace
            nrnact_ch = full_data.loc[full_data['channel'] == ch]
            nrnact_ch.index = colnames[1:]
            cond3 = nrnact_ch['height'] >= centre1
            cond4 = nrnact_ch['height'] <= centre2
            cond5 = nrnact_ch['width'] >= centre1
            cond6 = nrnact_ch['width'] <= centre2
            cond = cond3 & cond4 & cond5 & cond6
            on_colnames = list(nrnact_ch.index[cond])
            nrnact_ch = nrnact_ch.loc[cond]
            on_nrn_states = activities_df.transpose().loc[on_colnames]
            nrnact_ch = pd.merge(nrnact_ch, on_nrn_states, left_index=True,
                                 right_index=True)

            ## Cut away the meta-data about the trace
            state_traces = np.array(nrnact_ch)[:,-self.full_trace_len:]

            ## Delete objects to make room
            del dm, nrnact_ch, activities_df, colnames, inds

            # Get the momentum data
            dm = datamanager.DataManager(root_path=self.args.root_path,
                                         model_exp_name=model_exp_name,
                                         var_names=['momenta'],
                                         state_layers=[1],
                                         batches=None,
                                         channels=[ch],
                                         hw=[self.top_left_pnt,
                                             self.bottom_right_pnt],
                                         timesteps=None)

            # Process data
            activities_df, arr_sh, colnames, inds = \
                self.convert_ch_data_to_activity_df(dm, var_type='momenta_1')

            # Find the subset of the nrnact df for this channel only and get
            # the h, w values for on neurons in each batch

            # ## Select only the centre neuron trace
            nrnact_ch = full_data.loc[full_data['channel'] == ch]
            nrnact_ch.index = colnames[1:]
            cond3 = nrnact_ch['height'] >= centre1
            cond4 = nrnact_ch['height'] <= centre2
            cond5 = nrnact_ch['width'] >= centre1
            cond6 = nrnact_ch['width'] <= centre2
            cond = cond3 & cond4 & cond5 & cond6
            on_colnames = list(nrnact_ch.index[cond])
            nrnact_ch = nrnact_ch.loc[cond]
            on_nrn_states = activities_df.transpose().loc[on_colnames]
            nrnact_ch = pd.merge(nrnact_ch, on_nrn_states, left_index=True,
                                 right_index=True)

            ## Cut away the meta-data about the trace
            momenta_traces = np.array(nrnact_ch)[:,-self.full_trace_len:]
            num_rows = state_traces.shape[0]

            ## Calculate the xcorr for corresponding state and mom traces
            for r in range(num_rows):
                print(ch," ; ", r, " out of ", num_rows)
                state_trace_full = state_traces[r,:]
                momentum_trace_full = momenta_traces[r,:]
                st_mom_xcorr_result = \
                    np.correlate(state_trace_full-np.mean(state_trace_full),
                        momentum_trace_full - np.mean(momentum_trace_full),
                        mode='full')
                ## Cut the xcorr results down to size
                centre_full_trace_plots = len(st_mom_xcorr_result) // 2

                st_mom_xcorr_result = \
                    st_mom_xcorr_result[centre_full_trace_plots-trc_mid_len:centre_full_trace_plots+trc_mid_len]
                xcorr_results.append(st_mom_xcorr_result)
            mean_lag = np.mean([np.argmax(res) for res in xcorr_results]) - \
                trc_mid_len
            res_dict = {'channel': ch,
                        'mean_lag': mean_lag}
            lag_results = lag_results.append(res_dict, ignore_index=True)

            ## Delete objects to make room
            del dm, nrnact_ch, activities_df, colnames, inds

            # Save results
            lag_results.to_pickle(
                os.path.join(exp_dir,
                             'EI lag results.pkl'))


        final_result = np.mean(lag_results['mean_lag'])
        result_string = 'EI mean lag is %s .txt' % (final_result)
        # Save final result to an easy access txt file
        with open(os.path.join(exp_dir,result_string), 'w') as f:
            print({' ':' '}, file=f)


    def make_retinotopy_figure(self, exp_name='long_just_fewangles'):
        """Plots """

        # Make dir to save plots for this experiment
        maps_dir, exp_dir = self.prepare_dirs(self.session_dir,
                                               'activity maps_retinotopy',
                                               exp_name)


        # Load data and prepare variables
        full_data = pd.read_pickle(os.path.join(self.session_dir,
            'neuron_activity_results_alternativeformat_%s.pkl' % exp_name))

        global_max_min = True

        batches = [0, 65]

        for b in batches: #TODO comment the below code
            images_per_ch = []
            pb = full_data['batch_idx'] == b

            if global_max_min:
                means = np.array(
                    [full_data.loc[pb]['mean_act_during'],
                     full_data.loc[pb]['mean_act_outside']])
                max_mean = np.max(means)
                min_mean = np.min(means)

            fig, ax = plt.subplots(8, 4, figsize=(15, 30))

            for ch, axx in zip(range(self.num_ch), ax.ravel()):
                print("Batch %s  ; Channel %s" % (b, ch))
                im = np.zeros([self.extracted_im_size, self.extracted_im_size])
                pch = full_data['channel'] == ch
                cond = pch & pb
                avg_activity = full_data.loc[cond]['mean_act_during'] - \
                               full_data.loc[cond]['mean_act_outside']
                avg_activity = np.array(avg_activity)
                im = avg_activity.reshape((32,32))
                maxmin_conds = pch
                if not global_max_min:
                    means = np.array(
                        [full_data.loc[maxmin_conds]['mean_act_during'],
                         full_data.loc[maxmin_conds]['mean_act_outside']])
                    max_mean = np.max(means)
                    min_mean = np.min(means)

                axx.imshow(im, vmax=max_mean, vmin=min_mean)
                axx.text(1.0, 3.0, 'Channel %s' % ch,
                         fontsize=20, color='white')

                fig.tight_layout()
                axx.xaxis.set_visible(False)
                axx.yaxis.set_visible(False)
            fig.savefig(
                os.path.join(exp_dir,
                             "figure for all channels_b%i.png" % (
                             b)))
            plt.close()


    def one_d_gabor_function(self, p, x):
        return (1 / (np.sqrt(2*np.pi) * p[0])) \
               * np.exp(-(x**2)/(2*p[0])) * p[2] * \
               np.cos(2 * np.pi * p[1] * x + p[3]) + p[4] * x

    def gabor_fitting_func(self, p, x):
        return self.one_d_gabor_function(p, x)