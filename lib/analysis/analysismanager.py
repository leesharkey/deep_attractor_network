import os
import numpy as np
import pandas as pd
import scipy.stats as spst
from scipy.signal import butter, filtfilt, detrend, welch, find_peaks
from scipy.stats import shapiro
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from PIL import Image
import torch
from scipy import optimize
import lib.analysis.datamanager as datamanager
import array2gif as a2g


class AnalysisManager:
    def __init__(self, args, session_name):
        self.args = args
        # self.large_storage_loc = "/media/lee/DATA/DDocs/AI_neuro_work/DAN"

        # Define model name paths
        self.primary_model = "20200731-040415__rndidx_60805_loaded20200722-144054__rndidx_25624_experiment_at_8490its"
        #self.primary_model = "20200812-223329__rndidx_31944_loaded20200808-163842__rndidx_90277"
        #self.primary_model = "20200808-114216__rndidx_11284_loaded20200723-124229__rndidx_32787"
        #self.primary_model = "20200731-040415__rndidx_60805_loaded20200722-144054__rndidx_25624_experiment_at_8490its"
        #self.primary_model = "20200728-040415__rndidx_76711_loaded20200722-144054__rndidx_25624"
        #self.primary_model  = '20200701-202929__rndidx_57703_loaded20200629-143321__rndidx_89181'
        #self.primary_model = '20200602-194603__rndidx_61473_loaded20200508-141652__rndidx_82930'
        #self.primary_model = '20200508-115243__rndidx_37562_loaded20200423-154227__rndidx_15605'

        self.just_angles_model = self.primary_model
        #self.just_angles_model = "20200728-040415__rndidx_76711_loaded20200722-144054__rndidx_25624"
        #self.just_angles_model = '20200701-225551__rndidx_84485_loaded20200629-143321__rndidx_89181'
        #self.just_angles_exp_name = "/media/lee/DATA/DDocs/AI_neuro_work/DAN/exp_data/20200701-225551__rndidx_84485_loaded20200629-143321__rndidx_89181/orientations_present_single_gabor_just_angle"

        self.double_stim_synchr_model = self.primary_model
        self.long_just_angles_model = self.primary_model
        self.just_angle_few_angles_model = self.primary_model

        # Define paths for individual experiments (each with diff stims)
        single_exp_stem = '/orientations_present_single_gabor'
        self.primary_model_exp_name = self.primary_model + single_exp_stem + '_contrast_and_angle'
        self.just_angles_exp_name = self.just_angles_model + single_exp_stem + '_just_angle'
        self.just_angles_few_angles_exp_name = self.just_angle_few_angles_model + single_exp_stem + '_just_angle_few_angles'
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
        self.just_angles_few_angles_batchgroupangles = [0.0, 0.5*np.pi]
        self.just_angles_few_angles_batchgroupstarts = \
            [i for i in range(128) if (i % 64)==0]
        self.just_angles_few_angles_batchgroups = \
            [list(range(bg_start,bg_start+64)) for bg_start in \
             self.just_angles_few_angles_batchgroupstarts]

        self.just_angles_few_angles_angles = sorted([0.0, np.pi * 0.5] * 64)
        self.just_angles_few_angles_contrasts = [2.4]
        self.just_angle_few_angles_angle_contrast_pairs = []
        for a in self.just_angles_few_angles_angles:
            for c in self.just_angles_few_angles_contrasts:
                self.just_angle_few_angles_angle_contrast_pairs.append((a, c))


        ## Long just angle few angles exp
        self.long_just_angles_few_angles_batchgroupangles = [0.0, 0.5*np.pi]
        self.long_just_angle_angles = self.just_angles_few_angles_angles
        self.long_just_angle_contrasts = [2.4]
        self.long_just_angle_angle_contrast_pairs = []
        for a in self.long_just_angle_angles:
            for c in self.long_just_angle_contrasts:
                self.long_just_angle_angle_contrast_pairs.append((a, c))

        # self.rc_locs = [15, 16, 17, 21, 23, 28] # Much faster but less informative
        rc_locs = list(range(0, 32, 2))
        rc_locs.extend(list(range(16-7, 16 + 7)))
        rc_locs = list(set(sorted(rc_locs)))
        self.rc_locs = rc_locs  #locs used in exp3


        ## Double stim exp
        ## ensure these settings are the same as in
        # managers.ExperimentalStimuliGenerationManager.\
        # generate_double_gabor_dataset__fewlocs_and_fewerangles
        self.double_stim_contrasts = [2.4]
        self.double_stim_static_y = 13
        self.double_stim_static_x = 16
        self.double_stim_static_angle = np.pi * 0.5

        self.double_stim_static_y_0centre = self.double_stim_static_y - \
                                            self.extracted_im_size//2
        self.double_stim_static_x_0centre = self.double_stim_static_x - \
                                            self.extracted_im_size//2

        angle_range = [0.0] * 8
        angle_range.extend([np.pi * 0.5] * 8)
        self.double_stim_angles = angle_range * 8

        self.double_stim_batchgroupangles = [0.0, 0.5*np.pi] * 8

        y_range = [-3, 1, 5, 9]
        x_range = [0, 7]

        self.double_stim_locs_x_range = x_range
        self.double_stim_locs_y_range = y_range
        self.double_stim_locs = [[j, i] for i in y_range for j in x_range]

        self.double_stim_batchgroup_locs = []
        for loc in self.double_stim_locs:
            self.double_stim_batchgroup_locs.append(loc)
            self.double_stim_batchgroup_locs.append(loc)

        self.double_stim_batchgroupstarts = \
            [i for i in range(128) if (i % 8)==0]
        self.double_stim_batchgroups = \
            [list(range(bg_start,bg_start+8)) for bg_start in \
             self.just_angles_few_angles_batchgroupstarts]


        # General
        self.get_contrasts_from_images()
        self.specgram_nfft = 256
        self.acorr_mid_len = 50  # was 150
        self.acorr_x = np.arange(start=-self.acorr_mid_len,
                                 stop=self.acorr_mid_len) # x values



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


    def print_stimuli(self):
        model_exp_name = self.primary_model_exp_name

        # Note: Unless you've saved the bottom layer ([0]), which you might
        # not have due to storage constraints, this function won't work
        dm = datamanager.DataManager(root_path=self.args.root_path,
                                     model_exp_name=model_exp_name,
                                     var_names=self.st_var_name,
                                     state_layers=[0],
                                     batches=None,
                                     channels=None,
                                     hw=None,
                                     timesteps=[0,
                                                self.burn_in_len+1,
                                                self.primary_stim_start,
                                                self.primary_stim_start+1,
                                                self.primary_stim_stop,
                                                self.primary_stim_stop+1])
        num_imgs = dm.data['state_0'].shape[0]
        for i in range(num_imgs):
            im = dm.data['state_0'][i]
            im = torch.tensor(im)

            utils.save_image(im,
                           self.session_dir + \
                           'stimuli_during_%s_at_ts%i.png' % (dm.data_name, i),
                           nrow=16, normalize=True, range=(0, 1))

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
                self.just_angle_few_angles_angles_contrast_pairs
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
            t_results = ttest_result.pvalue < 0.05 #0.005 #0.0005
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

            # contrast_assign_idx = list(inds[0] % len(contrasts))
            # angle_assign_idx = list(inds[0] % len(angles))
            # contrasts_assigned = [contrasts[i] for i in contrast_assign_idx]
            # angles_assigned = [angles[i] for i in angle_assign_idx]
            # results_df_2['angle'] = angles_assigned
            # results_df_2['contrast'] = contrasts_assigned

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

        for b in range(self.num_batches): #TODO comment the below code
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
                        #avg_activity = avg_activity / 2
                        im[i][j] = np.array(avg_activity)

                strong_stim_batches = pb % 8 == 0
                maxmin_conds = pch & strong_stim_batches

                means = np.array(
                    [full_data.loc[maxmin_conds]['mean_act_during'],
                     full_data.loc[maxmin_conds]['mean_act_outside']])
                max_mean = np.max(means)
                min_mean = np.min(means)


                plt.imshow(im, vmax=max_mean, vmin=min_mean)
                plt.colorbar()
                # plt.savefig(
                #     os.path.join(self.session_dir,
                #                  "raw b_ch neuron actv locs b%i_ch%i.png" % (
                #                  b, ch)))
                plt.savefig(
                    os.path.join(exp_dir,
                                 "raw ch_b neuron actv locs ch%i_b%i.png" % (
                                 ch, b)))
                plt.close()

    def print_activity_maps_by_batch_ch_double_stim(self):

        # Make dir to save plots for this experiment
        # exp_dir = os.path.join(self.session_dir,
        #                        'activity maps doublestim')
        # if not os.path.isdir(exp_dir):
        #     os.mkdir(exp_dir)

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
                        #avg_activity = avg_activity / 2
                        im[i][j] = np.array(avg_activity)


                means = np.array(
                    [full_data.loc[pch]['mean_act_during'],
                     full_data.loc[pch]['mean_act_outside']])
                max_mean = np.max(means)
                min_mean = np.min(means)


                plt.imshow(im, vmax=max_mean, vmin=min_mean)
                plt.colorbar()
                # plt.savefig(
                #     os.path.join(self.session_dir,
                #                  "raw b_ch neuron actv locs b%i_ch%i.png" % (
                #                  b, ch)))
                plt.savefig(
                    os.path.join(exp_dir,
                                 "raw ch_b neuron actv locs ch%i_b%i.png" % (
                                 ch, b)))
                plt.close()



    def print_activity_map(self):
        print("Making activity maps for channels and batches")
        nrnact = pd.read_pickle(
            os.path.join(self.session_dir,
                         'neuron_activity_results_primary.pkl'))

        # Reorganises activity dataframe so we can sum over pixels
        map_act = pd.melt(nrnact, id_vars=['batch_idx', 'height', 'width'],
                          value_vars=list(range(32)))
        map_act = map_act.rename(
            columns={'variable': 'channel', 'value': 'active'})
        pixel_inds = [(i, j) for i in list(range(self.extracted_im_size)) for j in
                      range(self.extracted_im_size)]

        # Plot for each channel over all batches
        for ch in range(self.num_ch):
            print("Plotting activity map for channel %i" % ch)
            im = np.zeros([self.extracted_im_size, self.extracted_im_size])
            pch = map_act['channel'] == ch
            # TODO comment the below code
            for (i, j) in pixel_inds:
                pi = map_act['height'] == i + self.top_left_pnt[0]
                pj = map_act['width'] == j + self.top_left_pnt[1]
                cond = pi & pj & pch
                sum_b_ch = map_act.loc[cond]['active'].sum()
                im[i][j] = sum_b_ch
            im = im / im.max()
            plt.imshow(im)
            plt.savefig(
                os.path.join(self.session_dir,
                             "neuron activity locations ch%i.png" % ch))

        # Plot for each batch
        for b in range(self.num_batches):
            print("Plotting activity map for batch %i" % b)
            im = np.zeros([self.extracted_im_size, self.extracted_im_size])
            #im_raw = np.zeros([im_dim, im_dim])

            pb = map_act['batch_idx'] == b
            for (i, j) in pixel_inds:
                pi = map_act['height'] == i + self.top_left_pnt[0]
                pj = map_act['width'] == j + self.top_left_pnt[1]
                cond = pi & pj & pb

                # im is 'active' booleans
                sum_b_ch = map_act.loc[cond]['active'].sum()
                im[i][j] = sum_b_ch

                # im_raw is just the mean activities
            im = im / im.max()
            plt.imshow(im)
            plt.savefig(
                os.path.join(self.session_dir,
                             "neuron activity locations b%i.png" % b))

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
            # reshaped_activities = [np.arange(len(dm.timesteps)).reshape(
            #     len(dm.timesteps), 1)]  # make TS data
            # reshaped_activities.extend([
            #     dm.data['state_1'].reshape(len(dm.timesteps),
            #                                -1)])  # extend TS data with actual data
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

                #colors = plt.cm.jet(norm(traces),bytes=True)
                a2g.write_gif(traces, title, fps=75)

    def find_orientation_preferences(self):
        print("Finding orientation preferences of neurons")

        full_data = pd.read_pickle(os.path.join(self.session_dir,
                      'neuron_activity_results_alternativeformat_primary.pkl'))
        est_param_names = ['amplitude', 'phaseshift', 'mean']
        # for name in est_param_names:
        #     full_data[name] = np.nan

        model_exp_name = self.just_angles_exp_name #self.primary_model_exp_name
        var_label_pairs = [('state_1', 'State')]

        angles = self.just_angle_angles
        contrasts = self.just_angle_contrasts
        angle_contrast_pairs = self.just_angle_angle_contrast_pairs
        # angle_contrast_pairs = []
        #
        # angles = np.linspace(start=0.0, stop=np.pi*2, num=self.num_batches)
        # contrasts = [2.4]
        # for a in angles:
        #     for c in contrasts:
        #         angle_contrast_pairs.append((a, c))

        activity_df = pd.read_pickle(os.path.join(
            self.session_dir, 'neuron_activity_results_just_angles.pkl'))

        # Create a new dataframe that sums over h and w (so we only have batch
        # info per channel)
        sum_cols = list(activity_df.columns).append(['angle', 'contrast'])
        patch_activity_df = pd.DataFrame(columns=sum_cols)

        # Get sum for each neuron (h,w) so that the df just contains sums
        # over the whole image patch for each channel in each batch element.
        # Each batch element corresponds to a particular (angle, contrast) pair
        for b in range(self.num_batches):
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
        sum_angles_df = sum_angles_df.drop(columns=['width','height',
                                                    'contrast', 'batch_idx'])

        # Plot the orientation preference plots
        fig, ax = plt.subplots(4,8, sharey=True, sharex=True)
        fig.set_size_inches(23, 8)
        k = 0
        angle_axis = [round((180/np.pi) * angle, 1) for angle in angles]
        orient_prefs = pd.DataFrame(columns=est_param_names)
        # TODO comment the below code
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
                ax[i,j].text(0.08, 0.27, 'Channel %s' % k)

                if est_params[0] < 0.:
                    est_params[0] = est_params[0] * -1
                    est_params[1] += np.pi

                if est_params[1] < 0.:
                    est_params[1] += 2 * np.pi

                fitted_curve = \
                    est_params[0] * np.cos(2*angles + est_params[1]) \
                        + est_params[2]
                if not all(normed_data.isna()):  # because nans
                    ax[i,j].plot(angle_axis, fitted_curve)


                if i==3: # Put labels on the edge plots only
                    ax[i, j].set_xlabel('Angles (degree)')
                    ax[i, j].set_xticks([0,90,180,270])
                if j==0:
                    ax[i,j].set_ylabel('Normed activity [a.u.]')

                # Log orientation pref
                orient_prefs.loc[len(orient_prefs)] = est_params
                # cond = full_data['channel']==k
                # idxs = full_data.loc[cond][name].index
                # for r in idxs:
                #     for n, name in enumerate(est_param_names):
                #         full_data.loc[r, name] = est_params[n]

                k += 1

        orient_prefs['channel'] = np.array(orient_prefs.index)
        #full_data = full_data.merge(orient_prefs)

        plt.savefig(os.path.join(
            self.session_dir, 'orientation_prefs.png'))
        plt.close()
        print(orient_prefs)


        # Plot a grid of circular plots with a circle for each channel
        # Plot the orientation preference plots
        fig, ax = plt.subplots(4, 8, sharey=True, sharex=True,
                               subplot_kw=dict(projection='polar'))
        fig.set_size_inches(23, 8)
        k = 0
        angle_axis = [round((180 / np.pi) * angle, 1) for angle in angles]
        for i in range(4):
            for j in range(8):
                # TODO comment the below code
                print("%i, %i" % (i, j))
                normed_data = sum_angles_df[k] - np.mean(sum_angles_df[k])
                normed_data = normed_data / np.linalg.norm(normed_data)
                amplitude0 = 2.0
                phaseshift0 = 0
                mean0 = 0.
                params0 = [amplitude0, phaseshift0, mean0]
                opt_func = lambda x: x[0] * np.cos(2 * angles + x[1]) + x[
                    2] - normed_data
                est_params = \
                    optimize.leastsq(opt_func, params0)[0]
                ax[i, j].plot(angles, normed_data)
                # ax[i,j].plot(angles, np.ones_like(angles) * 0.04,
                #          linewidth=1,
                #          color='r')
                ax[i, j].text(np.pi * 0.73, 0.39, 'Channel %s' % k)
                if est_params[0] < 0.:
                    est_params[0] = est_params[0] * -1
                    est_params[1] += np.pi
                if est_params[1] < 0.:
                    est_params[1] += 2 * np.pi
                fitted_curve = est_params[0] * np.cos(
                    2 * angles + est_params[1]) \
                               + est_params[2]
                if not all(normed_data.isna()):  # because nans
                    ax[i, j].plot(angles, fitted_curve)
                # if i==3: # Put labels on the edge plots only
                #     ax[i, j].set_xlabel('Angles (degree)')
                #     ax[i, j].set_xticks([0,90,180,270])
                # if j==0:
                #     ax[i,j].set_ylabel('Normed activity [a.u.]')
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

        # ax.set_xticklabels([str(i/(2*np.pi)) for i in range(self.num_ch)])
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        # ax.set_yticks([list(sorted(orient_prefs['phaseshift']))])
        plt.tight_layout()

        # plt.plot(t, np.ones_like(t) * np.max(orient_prefs['amplitude']),
        #          linewidth=1,
        #          color='b')
        plt.plot(t, np.ones_like(t) * threshold, linewidth=1,
                 color='r')

        # Plot lines in between the points
        thetas = orient_prefs['phaseshift']
        thetas_pi = orient_prefs['phaseshift'] + np.pi
        ampls = orient_prefs['amplitude']
        for i, (amp, ang) in enumerate(zip(ampls, thetas)):
            print(i, amp)
            if amp > threshold:
                plt.plot(np.array([thetas[i], thetas_pi[i]]),
                         np.array([ampls[i], 0.]), 'b-')
                plt.plot(thetas[i], ampls[i], 'bo')
            else:
                plt.plot(np.array([thetas[i], thetas_pi[i]]),
                         np.array([ampls[i], 0.]), 'r-')
                plt.plot(thetas[i], ampls[i], 'ro')
            ax.annotate(str(i),
                        xy=(thetas[i], 1.07 * ampls[i]),  # theta, radius
                        xytext=(thetas[i], 0.),  # fraction, fraction
                        textcoords='offset pixels',
                        horizontalalignment='center',
                        verticalalignment='center',
                        )
        ax.set_xticklabels([])
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

        # full_data.to_pickle(os.path.join(
        #     self.session_dir,
        #     'neuron_activity_results_alternativeformat_primary_w_ori.pkl'))



    def assign_ori_info(self):
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
        ori_amp_threshold = 0.04  # TODO (nested F-test?) I can't think of a principled way to choose this. I've just done it by visual inspection of the orientation preference plots but I think better would be to demonstrate that some sine fits are significantly better than a linear fit
        ori_dict = ori_pref['phaseshift'].loc[
            ori_pref['amplitude'] > ori_amp_threshold]  # not yet a dict
        ori_dict_filt = ori_pref['phaseshift'].loc[
            ori_pref['amplitude'] < ori_amp_threshold]
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
        orientation_diffs_0 = [np.abs(stim_angle - ori_pref) for
                               (stim_angle, ori_pref) in
                               zip(stim_angles, ori_prefs)]
        orientation_diffs_180 = [np.abs(stim_angle + np.pi - ori_pref) for
                                 (stim_angle, ori_pref) in
                                 zip(stim_angles, ori_prefs)]
        ori_diff_threshhold = 10 * np.pi / 180
        orientation_match = [ori_diff_0 < ori_diff_threshhold or \
                             ori_diff_180 < ori_diff_threshhold
                             for (ori_diff_0, ori_diff_180) in
                             zip(orientation_diffs_0, orientation_diffs_180)]

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

        # Load data
        # nrnact = pd.read_pickle(
        #     os.path.join(self.session_dir,
        #                  'neuron act ori.pkl'))

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

            # spec_data = (df[name] - np.mean(df[name]))
            # spec_data = spec_data / np.linalg.norm(spec_data)

            # Denoise
            # spec_data = self.butter_lowpass_filter(spec_data,cutoff=0.1,fs=100)

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
                # lambda x: x - np.mean(x)
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
        """Takes a datamanager class as argument and converts it to a df.

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

    def plot_contrast_specgram_comparison_local(self, patch_or_idx=None):
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
            #nrnact_ch = pd.merge(nrnact_ch.transpose(), activities_df)
            #nrnact_ch = nrnact_ch.transpose()
            cond1 = nrnact_ch['matched_stim_ori_pref']
            #cond2 = nrnact_ch['active']
            cond3 = nrnact_ch['height'] >= centre1
            cond4 = nrnact_ch['height'] <= centre2
            cond5 = nrnact_ch['width'] >= centre1
            cond6 = nrnact_ch['width'] <= centre2
            #cond = cond1&cond2&cond3&cond4&cond5&cond6
            cond = cond1 & cond3&cond4&cond5&cond6


            on_colnames = list(nrnact_ch.index[cond])
            nrnact_ch = nrnact_ch.loc[cond]
            on_nrn_states = activities_df.transpose().loc[on_colnames]
            nrnact_ch = pd.merge(nrnact_ch,on_nrn_states, left_index=True,
                                 right_index=True)

            if len(on_colnames) == 0:
                print("No on neurons for channel %s." % ch)
            else:
                print("%i on neurons in channel %s" % (len(on_colnames), ch))

            # Get the subset of contrasts that I want to plot
            contrast_inds = [0,2,5,7]
            contrasts = [self.contrast_and_angle_contrasts[i]
                         for i in contrast_inds]

            stims_present = list(set(nrnact_ch['stim_ori']))
            for stim in stims_present:
                # for p in range(1,7):
                fig, ax = plt.subplots(1, 4, figsize=(10, 3))
                k = 0

                for axis, contrast in zip(ax, contrasts):
                    cond_x = nrnact_ch['stim_contrast'] == contrast
                    cond_y = nrnact_ch['stim_ori'] == stim
                    cond_stims = cond_x & cond_y
                    contrast_df = nrnact_ch.loc[cond_stims]
                    drop_cols = list(contrast_df.columns[:-self.full_trace_len])
                    contrast_df = contrast_df.drop(columns=drop_cols)
                    #contrast_df = contrast_df.iloc[p]
                    contrast_df = contrast_df.sum(axis=0)


                    spectrum, freqs, t, im = axis.specgram(contrast_df,
                                   NFFT=self.specgram_nfft, Fs=100, Fc=0,
                                   detrend=None,
                                   noverlap=self.specgram_nfft-1, xextent=None, pad_to=None,
                                   sides='default',
                                   scale_by_freq=True, scale='dB',
                                   mode='default',
                                   cmap=plt.get_cmap('hsv'))

                    # axis.imshow(im)
                    axis.set_ylabel('Frequency [a.u.]')

                    plt.ticklabel_format(axis="y", style="sci", scilimits=(
                         0, 0))  # Uses sci notation for units
                    #plt.xticks(np.array(contrast_df.index))

                    # plt.tight_layout()  # Stops y label being cut off

                k+=1
                plt.savefig(
                    os.path.join(exp_dir,
                                 'Spectrogram for central %s of ch%i for stim %f.png'% (
                                     save_name, ch, round(stim, 3))))
                plt.close()

    def plot_contrast_specgram_comparison_LFP(self, patch_or_idx=None):
        """The difference from the '_local' function is that this
        one calculates the sum of all the traces in all channels first,
        then plots the spectrogram."""
        print("Plotting spectrograms for each channel")


        # Make dir to save plots for this experiment
        exp_dir = os.path.join(self.session_dir,
                               'spectrogram comparisons')
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
        contrast_inds = [0, 2, 5, 7]
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
            # nrnact_ch = pd.merge(nrnact_ch.transpose(), activities_df)
            # nrnact_ch = nrnact_ch.transpose()
            #cond1 = nrnact_ch['matched_stim_ori_pref']
            # cond2 = nrnact_ch['active']
            cond3 = nrnact_ch['height'] >= centre1
            cond4 = nrnact_ch['height'] <= centre2
            cond5 = nrnact_ch['width'] >= centre1
            cond6 = nrnact_ch['width'] <= centre2
            # cond = cond1&cond2&cond3&cond4&cond5&cond6
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
                # contrast_df = contrast_df.iloc[p]
                contrast_df = contrast_df.sum(axis=0)

                if per_contr_series[n] is None:
                    per_contr_series[n] = contrast_df
                else:
                    per_contr_series[n] += contrast_df

        # Plot per-contrast spectrograms of the series
        k=0
        fig, ax = plt.subplots(1, 4, figsize=(15, 4))
        for axis, contrast, series in zip(ax, contrasts,per_contr_series):
            spectrum, freqs, t, im = axis.specgram(series,
                                                   NFFT=self.specgram_nfft,
                                                   Fs=100,
                                                   Fc=0,
                                                   detrend=None,
                                                   noverlap=self.specgram_nfft-1,
                                                   xextent=None,
                                                   pad_to=None,
                                                   sides='default',
                                                   scale_by_freq=True,
                                                   scale='dB',
                                                   mode='default',
                                                   vmin=-120.,
                                                   vmax=0.,
                                                   cmap=plt.get_cmap(
                                                       'hsv'))
            x_axis_scaler = 100
            axis.axvline(x=self.primary_stim_start/x_axis_scaler, c='black',
                         linewidth=1, linestyle='dashed')
            axis.set_xlabel('Timesteps before stimulus')
            if k ==0:
                axis.set_ylabel('Frequency [a.u.]')

            axis.set_xticks(ticks=[6,11,16,21])#,labels=['0'])
            axis.set_xticklabels([-500,0,500,1000])

            axis.set_yticks([])
            plt.tight_layout()  # Stops y label being cut off
            k += 1
        plt.savefig(
            os.path.join(exp_dir,
                         'Spectrogram for central %s for all channels .png' % (
                             save_name)))
        plt.close()

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
        contrast_inds = [0, 2, 5, 7]
        contrasts = [self.contrast_and_angle_contrasts[i]
                     for i in contrast_inds]
        per_contr_series = [None] * len(contrasts)
        per_contr_powerspecs = [[]] * len(contrasts)
        per_contr_powerfreqs = [[]] * len(contrasts)

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
            # nrnact_ch = pd.merge(nrnact_ch.transpose(), activities_df)
            # nrnact_ch = nrnact_ch.transpose()
            #cond1 = nrnact_ch['matched_stim_ori_pref']
            # cond2 = nrnact_ch['active']
            cond3 = nrnact_ch['height'] >= centre1
            cond4 = nrnact_ch['height'] <= centre2
            cond5 = nrnact_ch['width'] >= centre1
            cond6 = nrnact_ch['width'] <= centre2
            # cond = cond1&cond2&cond3&cond4&cond5&cond6
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
                # contrast_df = contrast_df.iloc[p]
                contrast_df = contrast_df.sum(axis=0)

                if per_contr_series[n] is None:
                    per_contr_series[n] = contrast_df
                else:
                    per_contr_series[n] += contrast_df


        # Plot per-contrast power spectra of the series
        #Now do unnormalized plot
        fig, ax = plt.subplots()
        cmap = plt.cm.Greys

        for contrast, series in zip(contrasts,per_contr_series):
            freqs, psd = welch(
                series,
                scaling='spectrum',
                nperseg=self.exp_stim_phase_lens[1])
            plt.plot(psd*freqs, c=cmap(0.2*contrast + 0.3))
        plt.yscale("log")
        ax.set_xlabel('Frequency [a.u]')
        ax.set_ylabel('Power X Frequency [a.u]')
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
        contrast_inds = [0, 2, 5, 7]
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
            # nrnact_ch = pd.merge(nrnact_ch.transpose(), activities_df)
            # nrnact_ch = nrnact_ch.transpose()
            #cond1 = nrnact_ch['matched_stim_ori_pref']
            cond2 = nrnact_ch['active']
            cond3 = nrnact_ch['height'] >= centre1
            cond4 = nrnact_ch['height'] <= centre2
            cond5 = nrnact_ch['width'] >= centre1
            cond6 = nrnact_ch['width'] <= centre2
            # cond = cond1&cond2&cond3&cond4&cond5&cond6
            cond = cond2 & cond3 & cond4 & cond5 & cond6

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
                    unnorm_per_contr_series[n] = contrast_df.copy()
                    if contrast_df.var() == 0:
                        contrast_df = (contrast_df - contrast_df.mean())
                    else:
                        contrast_df = \
                            (contrast_df - contrast_df.mean()) / \
                            contrast_df.var()
                    per_contr_series[n] = contrast_df.copy()
                else:
                    unnorm_per_contr_series[n] += contrast_df
                    if contrast_df.var() == 0:
                        contrast_df = (contrast_df - contrast_df.mean())
                    else:
                        contrast_df = \
                            (contrast_df - contrast_df.mean()) / \
                            contrast_df.var()
                    per_contr_series[n] += contrast_df

        # Plot per-contrast spectrograms of the series
        #cmap = plt.cm.YlGn
        cmap = plt.cm.Greys
        fig, ax = plt.subplots()
        for contrast, series in zip(contrasts,per_contr_series):
            trunc_series = \
                series[self.primary_stim_start-100:self.primary_stim_start+100]
            plt.plot(trunc_series, c=cmap(0.2*contrast + 0.3))
        plt.axvline(x=self.primary_stim_start, c='black',
                         linewidth=1, linestyle='dashed')
        ax.set_xticks(ticks=[1000, 1100, 1200])  # ,labels=['0'])
        ax.set_xticklabels([-100, 0, 100])
        ax.set_yticklabels([])
        plt.tight_layout()  # Stops y label being cut off
        plt.savefig(
            os.path.join(exp_dir,
                         'Transients for active neurons in central ' +
                         ' %s for all channels .png' % (save_name)))
        plt.close()


        #Now do unnormalized plot
        fig, ax = plt.subplots()
        for contrast, series in zip(contrasts,unnorm_per_contr_series):
            trunc_series = series[self.primary_stim_start-100:self.primary_stim_start+150]
            plt.plot(trunc_series, c=cmap(0.2*contrast + 0.3))
        plt.axvline(x=self.primary_stim_start, c='black',
                         linewidth=1, linestyle='dashed')
        ax.set_xticks(ticks=[1000, 1100, 1200])  # ,labels=['0'])
        ax.set_xticklabels([-100, 0, 100])
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
        plt.axvline(x=self.primary_stim_stop, c='black',
                         linewidth=1, linestyle='dashed')
        ax.set_xticks(ticks=[self.primary_stim_stop-100,
                             self.primary_stim_stop,
                             self.primary_stim_stop+100])  # ,labels=['0'])
        ax.set_xticklabels([-100, 0, 100])
        ax.set_yticklabels([])
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
            #nrnact_ch = nrnact_ch.loc[nrnact_ch['active']]
            # nrnact_ch = nrnact_ch.loc[nrnact_ch['height'] == centre_pix]
            # nrnact_ch = nrnact_ch.loc[nrnact_ch['width'] == centre_pix]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['height'] >= centre1]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['height'] <= centre2]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['width'] >= centre1]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['width'] <= centre2]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['matched_stim_ori_pref']]
            #nrnact_ch = nrnact_ch.loc[nrnact_ch['stim_contrast']==2.8]


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
            #state_traces.columns = self.true_contrast_and_angle_contrasts
            state_traces = state_traces.transpose()
            state_traces['batch'] = batches
            state_traces['stim_oris'] = stim_oris


            stims = list(set(stim_oris))
            #true_contrasts = self.true_contrast_and_angle_contrasts * num_stims
            #state_traces.columns = true_contrasts

            for stim in stims:
                # Sum across batch
                st_df = state_traces.loc[state_traces['stim_oris']==stim]
                st_df = st_df.groupby('batch').sum()
                st_df = st_df.transpose()
                st_df = st_df[:-1]  # gets rid of batch info
                st_df.columns = self.true_contrast_and_angle_contrasts


                fig, ax = plt.subplots()
                cmap = plt.cm.YlGn
                #cmap = plt.cm.Greys

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
        # stim_on_start = 1050
        # stim_on_stop = 3550

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

                    #TODO if peaks are empty, take the max psd and its freq

                    ##Plot the PSD
                    # plt.figure(figsize=(6, 5))
                    # plt.semilogx(freqs, psd)
                    # plt.scatter(freqs[peaks[0]], psd[peaks[0]])
                    # plt.title('PSD: power spectral density')
                    # plt.xlabel('Frequency')
                    # plt.ylabel('Power')
                    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                    # plt.tight_layout()
                    # plt.savefig(os.path.join(exp_dir,
                    #                          "PSD ch%s col%s %s.png" % (
                    #                          ch, col, durout_name)))
                    # plt.close()

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

        # Then move on to a new function plot_contrast_frequency_plots()



    def plot_contrast_frequency_plots(self):
        print("Plotting contrast/frequency plots")
        # Prepare variables
        angles = self.contrast_and_angle_angles
        contrasts = self.contrast_and_angle_contrasts
        angle_contrast_pairs = self.contrast_and_angle_angle_contrast_pairs
        # angles = [0.0, 0.393, 0.785, 1.178, 1.571, 1.963, 2.356, 2.749,
        #           3.142, 3.534, 3.927, 4.32, 4.712, 5.105, 5.498, 5.89]
        # contrasts = [0., 0.4, 0.8, 1.2, 1.6, 2., 2.4, 2.8]
        # angle_contrast_pairs = []
        # for a in angles:
        #     for c in contrasts:
        #         angle_contrast_pairs.append((a, c))

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



        # # Fill the new columns with the right info
        # for idx_label, row_series in nrn_actosc.iterrows():
        #     print(str(idx_label) + '/' + str(len(nrn_actosc)))
        #     # Get the angle and contrast for each batch_idx and
        #     # get the orientation preference of that channel
        #     row_channel = nrn_actosc.at[idx_label, 'channel']
        #     row_batch = nrn_actosc.at[idx_label, 'batch_idx']
        #     row_stim_angle, row_stim_contrast = angle_contrast_pairs[row_batch]
        #     row_ori_pref = ori_dict[row_channel]
        #
        #     nrn_actosc.at[idx_label, 'ori_pref']      = row_ori_pref
        #     nrn_actosc.at[idx_label, 'stim_ori']      = row_stim_angle
        #     nrn_actosc.at[idx_label, 'stim_contrast'] = row_stim_contrast
        #
        #     # Decide if a channels' orientation pref is matched by the
        #     # orientation of the stimulus.
        #     # I've chose a threshold of +/- 5 degrees
        #     orientation_match = \
        #         (np.abs(row_ori_pref - row_stim_angle) < 5*np.pi/180) or \
        #         (np.abs(row_ori_pref + np.pi - row_stim_angle) < 5*np.pi/180)
        #     if orientation_match:
        #         nrn_actosc.at[idx_label, 'matched_stim_ori_pref'] = True
        #     else:
        #         nrn_actosc.at[idx_label, 'matched_stim_ori_pref'] = False
        #
        #     # Record the max_peak_power if it exists
        #     peak_freq_powers_during = \
        #         nrn_actosc.at[idx_label, 'peak_freq_powers_during']
        #     if np.array(peak_freq_powers_during).size == 0:
        #         print("B %i ; Ch %i ; H%i W%i ; Index % i is empty." % \
        #                                 (row_batch,
        #                                  row_channel,
        #                                  nrn_actosc.at[idx_label, 'height'],
        #                                  nrn_actosc.at[idx_label, 'width'],
        #                                  idx_label))
        #         continue
        #     else:
        #         nrn_actosc.at[idx_label, 'max_peak_power'] = \
        #             np.nanmax(peak_freq_powers_during)
        #
        #         # Record the mean peak power-freq product. Note that the full
        #         # power-freq product has already been calculated
        #         peak_freqs = nrn_actosc.at[idx_label, 'peak_freqs_during']
        #         peak_powers = nrn_actosc.at[idx_label, 'peak_freq_powers_during']
        #         mean_peak_power_product = np.nanmean(peak_freqs * peak_powers)
        #         nrn_actosc.at[
        #             idx_label, 'mean_peak_power_during'] = mean_peak_power_product

        nrn_actosc.to_pickle(os.path.join(
            self.session_dir, 'neuron act and osc with new contrast freq info.pkl'))

        # Identify the neurons that have ALL [well def ori pref; matched
        # stim and ori pref; activated; peak power above a threshold]
        # cond1 = nrn_actosc['peak_freqs_power_product'] > 0.0 & \
        #     nrn_actosc['matched_stim_ori_pref'] & \
        #     nrn_actosc['active']
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

        # nrn_actosc_filt['mean_peak_power_during'] = pd.to_numeric(
        #     nrn_actosc_filt['mean_peak_power_during'])
        # nrn_actosc_filt['mean_peak_power_outside'] = pd.to_numeric(
        #     nrn_actosc_filt['mean_peak_power_outside'])
        # nrn_actosc_filt['freq_power_product_during'] = nrn_actosc_filt[
        #     'freq_power_product_during'].astype(float)
        # nrn_actosc_filt['freq_power_product_outside'] = nrn_actosc_filt[
        #     'freq_power_product_outside'].astype(float)

        ### Take those neurons and group them by the contrast of their stimulus

        mean_peak_power_during  = nrn_actosc_filt.groupby(['stim_contrast'])[
                     'peak_freqs_power_product'].mean()
        mean_peak_power_outside = nrn_actosc_filt.groupby(['stim_contrast'])[
                     'peak_freqs_power_product'].mean()
        print(mean_peak_power_during)
        print(mean_peak_power_outside) #what about this is 'during'?
        # mean_peak_power_during  = nrn_actosc_filt.groupby(['stim_contrast'])[
        #              'mean_peak_power_during'].mean()
        # mean_peak_power_outside = nrn_actosc_filt.groupby(['stim_contrast'])[
        #              'mean_peak_power_outside'].mean()
        # print(mean_peak_power_during)
        # print(mean_peak_power_outside)
        # Just check that there is the no unexpected relationship between
        # ori pref and oscillations (I want there to be more oscillating for
        # preferred orientations and less for non preferred)


        # Get the active neurons and their frequencies and get the sum for
        # each contrast level (may wish to normalise in some way)


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

        # Then move on to synchrony code


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
        #"/home/lee/Documents/AI_ML_neur_projects/deep_attractor_network/analysis_results/20200731-040415__rndidx_60805_loaded20200722-144054__rndidx_25624_experiment_at_8490its"
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
        model_exp_name = self.just_angles_exp_name
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
            # nrnact_ch = nrnact_ch.loc[nrnact_ch['matched_stim_ori_pref']]
            # nrnact_ch = nrnact_ch.loc[nrnact_ch['stim_contrast']==2.8]
            nrnact_ch = nrnact_ch.reset_index()

            # Get the names of the columns
            on_colnames, on_col_inds, batches, heights, widths, stim_oris = \
                self.get_info_lists_from_nrnactch(nrnact_ch)

            # For the  in the above-defined subset of nrnact, get
            # their timeseries from the arrays
            state_traces = activities_df[on_colnames]
            # state_traces.columns = self.true_contrast_and_angle_contrasts
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

        batch_groups = [list(range(64)), list(range(64,128))]

        plotting = False#True


        # Set initial gabor params (gaussian components) [sigma, amplitude]
        p01 = [7e1, 400.0]#[5e2, 5., 1.0, 0.] ######[7e2, 0.9, 500.0, 0.]
        p02 = [7e1, 500.0]#[5e2, -0.5, 100.0, 0.]
        p03 = [7e1, 600.0]#[7e2, 0.9, 500.0, 0.]
        init_params_gabor = [p02]#[p01, p02, p03]#, p04]


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
                            print("Channels %s %s" % (ch1, ch2))
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
                                # ax.legend()

                                plt.savefig(
                                    "%s/cross correlation between %i and %i in batch %i for %s" % (
                                    acorr_dir, ch1, ch2, b,label))
                                plt.close()

                df_title = "%s/estimated_gabor_params_batch_%i " % (exp_dir, b)
                param_df.to_pickle(df_title)

        ##Fitting sinewave
        # fig, ax = plt.subplots(4,8, sharey=True, sharex=True)
        # fig.set_size_inches(23, 8)
        # k = 0
        # angle_axis = [round((180/np.pi) * angle, 1) for angle in angles]
        # orient_prefs = pd.DataFrame(columns=est_param_names)
        # for i in range(4):
        #     for j in range(8):
        #         print("%i, %i" % (i,j))
        #         normed_data = sum_angles_df[k] - np.mean(sum_angles_df[k])
        #         normed_data = normed_data / \
        #                       np.linalg.norm(normed_data)
        #
        #         amplitude0  = 2.0
        #         phaseshift0 = 0
        #         mean0  = 0.
        #         params0 = [amplitude0, phaseshift0, mean0]
        #         opt_func = lambda x: x[0] * np.cos(2*angles + x[1]) + x[
        #             2] - normed_data
        #
        #         est_params = \
        #             optimize.leastsq(opt_func, params0)[0]
        #
        #         ax[i,j].plot(angle_axis, normed_data)
        #         ax[i,j].text(0.08, 0.27, 'Channel %s' % k)
        #
        #         if est_params[0] < 0.:
        #             est_params[0] = est_params[0] * -1
        #             est_params[1] += np.pi
        #
        #         if est_params[1] < 0.:
        #             est_params[1] += 2 * np.pi
        #
        #         fitted_curve = est_params[0] * np.cos(2*angles + est_params[1]) \
        #                        + est_params[2]


    def synchrony_experiment1_overlapping_rec_fields_Plot_acorrs_overlay(self):
        """Collects the 128 batch dfs together
           Collects the acorr data for each trace
           Plots per channel for all batches for that angle"""
        # Prepare general variables
        exp_dir, acorr_dir = self.prepare_dirs(self.session_dir,
                                               'SynchronyExp1',
                                               'acorrs_overlayed')

        ch_data_during  = pd.DataFrame()
        ch_data_outside = pd.DataFrame()
        batch_groups = [list(range(64)), list(range(64,128))]
        max_peak = 28
        min_trough = -20

        for bg_angle_index, bg in enumerate(batch_groups):

            # Collect all the dfs for this batch group (this stim)
            dfs_all = \
                [[pd.read_pickle(
                    os.path.join(exp_dir,
                        'cross_correlation_results_%s_%s.pkl' % (b, label)))
                 for label in ['during', 'outside']] for b in bg]

            # Go through each channel #TODO comment the below code
            for ch1 in range(self.num_ch):
                for ch2 in range(self.num_ch):
                    print("%s %s %s" % (bg, ch1, ch2))

                    #create plots and prepare to overlay
                    fig, ax = plt.subplots(2,figsize=(8,9))
                    for k, label in enumerate(['During', 'Outside']):
                        ax[k].set_ylim([min_trough, max_peak])
                        ax[k].set_title(label)
                        ax[k].set_xlabel('lag')
                        ax[k].set_ylabel('correlation coefficient')


                    for batch_idx, b in enumerate(bg):
                        print("%s" % (b))

                        dfs = dfs_all[batch_idx].copy()

                        # Just get the channel combination you want
                        for j in range(len(dfs)):
                            cond_a = dfs[j]['channel A'] == ch1
                            cond_b = dfs[j]['channel B'] == ch2
                            cond = cond_a & cond_b
                            dfs[j] = dfs[j][cond]
                            dfs[j] = np.array(dfs[j].iloc[:, 1:-3])

                        for j, label in enumerate(['During', 'Outside']):

                            title = \
                                "Cross correlation between" + \
                                " %i and %i for stim-type %i" % \
                                (ch1, ch2, bg_angle_index)

                            acorr_trace = dfs[j]
                            acorr_trace = acorr_trace.squeeze()
                            display_len = 400
                            mid_point = len(acorr_trace) // 2
                            acorr_trace = acorr_trace[
                                 (mid_point - display_len):(mid_point + display_len)]
                            ax[j].plot(np.arange(0-display_len, 0+display_len),
                                       acorr_trace, c='blue', alpha=0.25)  # /np.max(acorr_trace))

                            # plt.savefig(acorr_dir + '/' + title)
                    plt.savefig(acorr_dir + '/' + title)
                    plt.close()

    def synchrony_experiment1_overlapping_rec_fields_Plot_acorrs_individually(self):
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

                #fig.tight_layout(pad=1.0)
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
                            acorr_trace)#/np.max(acorr_trace))
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

        for b in range(3):#128):
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
                            acorr_trace)#/np.max(acorr_trace))
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

            if plotting:
                # Plot results
                fig, ax = plt.subplots(1, figsize=(9, 5))
                ax.plot(self.acorr_x, gabor_curve)  # /np.max(acorr_trace))
                ax.scatter(self.acorr_x[peaks[0]], gabor_curve[peaks[0]])
                ax.set_xlabel('lag')
                ax.set_ylabel('correlation coefficient')
                fig.savefig(
                 "%s/diagnostic plot for fitted gabors_ch %i and %i in " +
                 "batch %i for %s.jpg" % (
                     acorr_dir, int(row['channel A']), int(row['channel B']),
                     int(row['batch']), str(row['dur_or_out'])))
                plt.close()

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
        # dur_out_comparison = np.concatenate(
        #     (dur_out_comparison, np.ones_like(dur_out_comparison) * -1))
        results_df['dur_out_amp_compar'] = -1.

        dur_index = results_df[results_df['dur_or_out'] == 0.0].index
        out_index = results_df[results_df['dur_or_out'] == 1.0].index

        results_df.loc[dur_index, 'dur_out_amp_compar'] = dur_out_comparison
        #results_df.loc[out_index, 'dur_out_amp_compar'] = np.ones_like(dur_out_comparison) * -1.

        results_df.to_pickle(
            "%s/synchexp1 acorr analysis results.pkl" % self.session_dir)


    def synchrony_experiment1_overlapping_rec_fields_OriPref_vs_OscAmp(self):
        # Prepare general variables and dirs
        exp_dir = os.path.join(self.session_dir,
                               'SynchronyExp1')
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)

        # batch_groups = [list(range(64)), list(range(64, 128))]
        bg_angles = self.just_angles_few_angles_batchgroupangles


        acorr_results_df = pd.read_pickle(
            "%s/synchexp1 acorr analysis results.pkl" % self.session_dir)
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

        # Get data for the difference in ori pref and osc amplitude
        not_self_cond = acorr_results_df['channel A'] != acorr_results_df['channel B']
        ori_pref_data = acorr_results_df['angle_diff'][not_self_cond]
        osc_amp_data = acorr_results_df['amplitude'][not_self_cond]

        ##Other secondarily important data
        num_peaks_data = acorr_results_df['num_peaks'][not_self_cond]

        # Fit a line and plot
        m, c = np.polyfit(ori_pref_data, osc_amp_data, 1)
        fig, ax = plt.subplots(1, figsize=(9, 5))
        ax.scatter(ori_pref_data, osc_amp_data, cmap='hsv',
                   c=acorr_results_df['channel B'][not_self_cond])
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
        match_0_A = (np.abs(acorr_results_df['ori_pref A'] - stim_angles) < thresh_ang*np.pi/180)
        match_180_A = (np.abs(acorr_results_df['ori_pref A'] + np.pi - stim_angles) < thresh_ang*np.pi/180)
        match_0_B = (np.abs(acorr_results_df['ori_pref B'] - stim_angles) < thresh_ang * np.pi / 180)
        match_180_B = (np.abs(acorr_results_df['ori_pref B'] + np.pi - stim_angles) < thresh_ang * np.pi / 180)
        orientation_match = \
            (match_0_A | match_180_A) & (match_0_B | match_180_B)

        ori_pref_data = \
            acorr_results_df['angle_diff'][not_self_cond & orientation_match]
        osc_amp_data = \
            acorr_results_df['amplitude'][not_self_cond & orientation_match]

        m, c = np.polyfit(ori_pref_data, osc_amp_data, 1)
        fig, ax = plt.subplots(1, figsize=(9, 5))
        ax.scatter(ori_pref_data, osc_amp_data, cmap='hsv',
                   c=acorr_results_df['channel A'][
                       not_self_cond & orientation_match])
        plt.plot(ori_pref_data, m * ori_pref_data + c, c='r')
        ax.set_xlabel('Angle difference')
        ax.set_ylabel('Oscillation amplitude')
        plt.savefig(os.path.join(exp_dir,
                                 "Angle_diff vs Oscillation amplitude w ori match.png"))
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
        #"/home/lee/Documents/AI_ML_neur_projects/deep_attractor_network/analysis_results/20200731-040415__rndidx_60805_loaded20200722-144054__rndidx_25624_experiment_at_8490its"
        # Add stim_orientation info to data
        stim_angles = [self.double_stim_angles[i] for i in
                       full_data['batch_idx']]
        full_data['stim_ori'] = stim_angles


        if patch_or_idx == 'patch': # probably remove this in future. unlikely to use patch but maybe in future.
            centre1 = self.central_patch_min
            centre2 = self.central_patch_max
            central_patch_size = self.central_patch_size
            save_name = 'patch'
        else:
            centre = 16
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
                [self.double_stim_locs[i % 8] for i in nrnact_ch['batch_idx']]
            mobile_locs_x = \
                [centre + mobile_locs[i][0] for i in range(len(mobile_locs))]
            mobile_locs_y = \
                [centre + mobile_locs[i][1] for i in range(len(mobile_locs))]

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
            # state_traces.columns = self.true_contrast_and_angle_contrasts
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
            mobile_loc = self.double_stim_locs[b % 8]
            mobile_loc_x = centre + mobile_loc[0]
            mobile_loc_y = centre + mobile_loc[1]

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
        """Git Gabor functions to xcorr plots as in Gray et al. (1989)."""
        # Prepare general variables and dirs
        exp_dir, acorr_dir = self.prepare_dirs(self.session_dir,
                                               'SynchronyExp2',
                                               'acorrs_gabor_fitted')

        batch_groups = self.double_stim_batchgroups

        plotting = False #True


        # Set initial gabor params (gaussian components) [sigma, amplitude]
        p01 = [7e1, 400.0]#[5e2, 5., 1.0, 0.] ######[7e2, 0.9, 500.0, 0.]
        p02 = [7e1, 500.0]#[5e2, -0.5, 100.0, 0.]
        p03 = [7e1, 600.0]#[7e2, 0.9, 500.0, 0.]
        init_params_gabor = [p02]#[p01, p02, p03]#, p04]


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
                            print("Channels %s %s" % (ch1, ch2))
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
                                    "%s/Xcorr between static ch %i and " + \
                                    "mobile ch %i at (x%i,y%i) in " + \
                                    "batch %i for %s" % (
                                    acorr_dir, ch1, ch2, x_loc,y_loc,b,label))
                                plt.close()

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

            if plotting:
                # Plot results
                fig, ax = plt.subplots(1, figsize=(9, 5))
                ax.plot(self.acorr_x, gabor_curve)  # /np.max(acorr_trace))
                ax.scatter(self.acorr_x[peaks[0]], gabor_curve[peaks[0]])
                ax.set_xlabel('lag')
                ax.set_ylabel('correlation coefficient')
                # ax.legend()

                fig.savefig(
                 "%s/diagnostic plot for fitted gabors_ch %i and %i in batch"+\
                 " %i for %s.jpg" % (
                     acorr_dir, int(row['channel A']), int(row['channel B']),
                     int(row['batch']), str(row['dur_or_out'])))
                plt.close()

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
        # dur_out_comparison = np.concatenate(
        #     (dur_out_comparison, np.ones_like(dur_out_comparison) * -1))
        results_df['dur_out_amp_compar'] = -1.

        dur_index = results_df[results_df['dur_or_out'] == 0.0].index
        out_index = results_df[results_df['dur_or_out'] == 1.0].index

        results_df.loc[dur_index, 'dur_out_amp_compar'] = dur_out_comparison
        #results_df.loc[out_index, 'dur_out_amp_compar'] = \
        #   np.ones_like(dur_out_comparison) * -1.

        results_df.to_pickle(
            "%s/synchexp2 acorr analysis results.pkl" % self.session_dir)


    def synchrony_experiment2_OriPref_OR_Distance_vs_OscAmp_OR_vs_Phase(self):

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

        acorr_results_df = pd.read_pickle(anlsys_res_title)
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
        mob_locs_x = [loc[0] for loc in locs]
        mob_locs_y = [loc[1] for loc in locs]
        acorr_results_df['mob_loc_x'] = mob_locs_x
        acorr_results_df['mob_loc_y'] = mob_locs_y

        ## Make mobile loc relative to static loc
        acorr_results_df['mob_loc_x'] = acorr_results_df['mob_loc_x'] - \
            self.double_stim_static_x_0centre
        acorr_results_df['mob_loc_y'] = acorr_results_df['mob_loc_y'] - \
            self.double_stim_static_y_0centre

        # Calculate (square) distance of mobile from static stim
        # as the max distance along an axis
        xbiggery = \
            acorr_results_df['mob_loc_x'] > acorr_results_df['mob_loc_y']
        acorr_results_df['mob_sq_dist'] = acorr_results_df['mob_loc_x'].where(
            xbiggery, other=acorr_results_df['mob_loc_y'])

        # Calculate euclidian distance of mobile from static stim
        centred_x_locs = [xy[0] - self.double_stim_static_x_0centre for xy in
         self.double_stim_batchgroup_locs]
        centred_y_locs = [xy[1] - self.double_stim_static_y_0centre for xy in
            self.double_stim_batchgroup_locs]
        eucl_dists = [np.sqrt(x**2 + y**2)
                 for (x,y) in zip(centred_x_locs, centred_y_locs)]

        acorr_results_df['mob_eucl_dist'] = \
            [eucl_dists[int(i)] for i in acorr_results_df['batch_group']]

        # Assign stim angles
        stim_angles = np.array([bg_angles[int(i)] for i in
                                acorr_results_df['batch_group']])
        acorr_results_df['mob_stim_angle'] = stim_angles

        # Define the conditions and their labels for plotting in a for-loop
        df = acorr_results_df  # for brevity

        #not_overlap_cond = df['mob_loc_x'] > -1 #for debugging
        not_overlap_cond = ~((df['mob_loc_x'] == 0) & (df['mob_loc_y']==0))

        ori_same_below_all = (df['mob_stim_angle'] == self.double_stim_static_angle) & (df['mob_loc_x'] == 0)
        ori_same_beside_all = (df['mob_stim_angle'] == self.double_stim_static_angle) & (df['mob_loc_x'] != 0)
        ori_different_below_all = (df['mob_stim_angle'] != self.double_stim_static_angle) & (df['mob_loc_x'] == 0)
        ori_different_beside_all = (df['mob_stim_angle'] != self.double_stim_static_angle) & (df['mob_loc_x'] != 0)
        ori_same_below_onlyhorizconnect = (df['mob_stim_angle'] == self.double_stim_static_angle) & (df['mob_loc_x'] == 0) & (df['mob_loc_y'] < self.double_stim_horizextent)
        ori_same_beside_onlyhorizconnect = (df['mob_stim_angle'] == self.double_stim_static_angle) & (df['mob_loc_x'] != 0) & (df['mob_loc_y'] < self.double_stim_horizextent)
        ori_different_below_onlyhorizconnect = (df['mob_stim_angle'] != self.double_stim_static_angle) & (df['mob_loc_x'] == 0) & (df['mob_loc_y'] < self.double_stim_horizextent)
        ori_different_beside_onlyhorizconnect = (df['mob_stim_angle'] != self.double_stim_static_angle) & (df['mob_loc_x'] != 0) & (df['mob_loc_y'] < self.double_stim_horizextent)
        ori_same_below_onlyNOThorizconnect = (df['mob_stim_angle'] == self.double_stim_static_angle) & (df['mob_loc_x'] == 0) & (df['mob_loc_y'] >= self.double_stim_horizextent)
        ori_same_beside_onlyNOThorizconnect = (df['mob_stim_angle'] == self.double_stim_static_angle) & (df['mob_loc_x'] != 0) & (df['mob_loc_y'] >= self.double_stim_horizextent)
        ori_different_below_onlyNOThorizconnect = (df['mob_stim_angle'] != self.double_stim_static_angle) & (df['mob_loc_x'] == 0) & (df['mob_loc_y'] >= self.double_stim_horizextent)
        ori_different_beside_onlyNOThorizconnect = (df['mob_stim_angle'] != self.double_stim_static_angle) & (df['mob_loc_x'] != 0) & (df['mob_loc_y'] >= self.double_stim_horizextent)

        ori_same_below_all_label = "Static and mobile stims with same orientation, mobile stim below static stim (i.e. colinear)"
        ori_same_beside_all_label = "Static and mobile stims with same orientation, mobile stim beside static stim (i.e. not colinear)"
        ori_different_below_all_label = "Static and mobile stims with different orientation, mobile stim below static stim"
        ori_different_beside_all_label = "Static and mobile stims with different orientation, mobile stim beside static stim"
        ori_same_below_onlyhorizconnect_label = "Static and mobile stims with same orientation, mobile stim below static stim and within horizontal connections"
        ori_same_beside_onlyhorizconnect_label = "Static and mobile stims with same orientation, mobile stim beside static stim and within horizontal connections"
        ori_different_below_onlyhorizconnect_label = "Static and mobile stims with different orientation, mobile stim below static stim and within horizontal connections"
        ori_different_beside_onlyhorizconnect_label = "Static and mobile stims with different orientation, mobile stim beside static stim and within horizontal connections"
        ori_same_below_onlyNOThorizconnect_label = "Static and mobile stims with same orientation, mobile stim below static stim and outside horizontal connections"
        ori_same_beside_onlyNOThorizconnect_label = "Static and mobile stims with same orientation, mobile stim beside static stim and outside horizontal connections"
        ori_different_below_onlyNOThorizconnect_label = "Static and mobile stims with different orientation, mobile stim below static stim and outside horizontal connections"
        ori_different_beside_onlyNOThorizconnect_label = "Static and mobile stims with different orientation, mobile stim beside static stim and outside horizontal connections"

        cond_and_labs = [[ori_same_below_all, ori_same_below_all_label],
                         [ori_same_beside_all, ori_same_beside_all_label],
                         [ori_different_below_all, ori_different_below_all_label],
                         [ori_different_beside_all, ori_different_beside_all_label],
                         [ori_same_below_onlyhorizconnect, ori_same_below_onlyhorizconnect_label],
                         [ori_same_beside_onlyhorizconnect, ori_same_beside_onlyhorizconnect_label],
                         [ori_different_below_onlyhorizconnect, ori_different_below_onlyhorizconnect_label],
                         [ori_different_beside_onlyhorizconnect, ori_different_beside_onlyhorizconnect_label],
                         [ori_same_below_onlyNOThorizconnect, ori_same_below_onlyNOThorizconnect_label],
                         [ori_same_beside_onlyNOThorizconnect, ori_same_beside_onlyNOThorizconnect_label],
                         [ori_different_below_onlyNOThorizconnect, ori_different_below_onlyNOThorizconnect_label],
                         [ori_different_beside_onlyNOThorizconnect, ori_different_beside_onlyNOThorizconnect_label]]

        # Go through the various conditions and data and make plots
        for indep_label in ['angle_diff', 'mob_eucl_dist', 'mob_sq_dist']:
            for dependent_label in ['amplitude', 'sine_phase', 'num_peaks']:

                # Prepare directory for saving plots
                plot_dir = os.path.join(exp_dir, indep_label+" vs "+dependent_label)
                if not os.path.isdir(plot_dir):
                    os.mkdir(plot_dir)

                for cond, label in cond_and_labs:
                    print(indep_label, dependent_label, label)
                    # Define data using conds
                    indep_data = acorr_results_df[indep_label][not_overlap_cond & cond]
                    dependent_data = acorr_results_df[dependent_label][not_overlap_cond & cond]
                    # num_peaks_data = acorr_results_df['num_peaks'][not_overlap_cond & cond]


                    # Plot basic plot
                    m, c = np.polyfit(indep_data, dependent_data, 1) # Calc line
                    fig, ax = plt.subplots(1, figsize=(9, 5))
                    ax.scatter(indep_data, dependent_data, cmap=cmap,
                               c=acorr_results_df['mob_sq_dist'][not_overlap_cond & cond])
                    plt.plot(indep_data, m * indep_data + c, c='r')  # Plot line
                    ax.set_xlabel('%s' % indep_label)
                    ax.set_ylabel('Oscillation %s' % dependent_label)
                    plot_title = "%s vs Oscillation %s for " % \
                                 (indep_label, dependent_label)
                    plot_title = plot_title + label
                    plt.savefig(os.path.join(plot_dir,
                                             "SynchExp2 " + plot_title + ".png"))
                    plt.close()


                    # Plots with ori match:
                    ## See whether there is a match between BOTH channels and the stim
                    thresh_ang = 10
                    match_0_A = (np.abs(acorr_results_df['ori_pref A'] - stim_angles) < thresh_ang*np.pi/180)
                    match_180_A = (np.abs(acorr_results_df['ori_pref A'] + np.pi - stim_angles) < thresh_ang*np.pi/180)
                    match_0_B = (np.abs(acorr_results_df['ori_pref B'] - stim_angles) < thresh_ang * np.pi / 180)
                    match_180_B = (np.abs(acorr_results_df['ori_pref B'] + np.pi - stim_angles) < thresh_ang * np.pi / 180)
                    orientation_match = \
                        (match_0_A | match_180_A) & (match_0_B | match_180_B)

                    ## Redefine data
                    indep_data = acorr_results_df[indep_label][not_overlap_cond & cond & orientation_match]
                    dependent_data = acorr_results_df[dependent_label][not_overlap_cond & cond & orientation_match]

                    ## plot
                    m, c = np.polyfit(indep_data, dependent_data, 1)
                    fig, ax = plt.subplots(1, figsize=(9, 5))
                    ax.scatter(indep_data, dependent_data, cmap=cmap,
                               c=acorr_results_df['mob_sq_dist'][not_overlap_cond & cond & orientation_match])
                    plt.plot(indep_data, m * indep_data + c, c='r')
                    ax.set_xlabel('%s' % indep_label)
                    ax.set_ylabel('Oscillation %s' % dependent_label)
                    plot_title = "%s vs Oscillation %s with ori match for " % (indep_label, dependent_label)
                    plot_title = plot_title + label
                    plt.savefig(os.path.join(plot_dir,
                                             "SynchExp2 " + plot_title + ".png"))
                    plt.close()

    # SYNCH EXP 3
    def synchrony_experiment3_xcorrs(self, patch_or_idx=None):
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
        #"/home/lee/Documents/AI_ML_neur_projects/deep_attractor_network/analysis_results/20200731-040415__rndidx_60805_loaded20200722-144054__rndidx_25624_experiment_at_8490its"
        # Add stim_orientation info to data
        stim_angles = [self.long_just_angle_angles[i] for i in
                       full_data['batch_idx']]
        full_data['stim_ori'] = stim_angles


        if patch_or_idx == 'patch': # probably remove this in future. unlikely to use patch but maybe in future.
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

            # centre_rc_cond = (nrnact_ch['height'] == centre) | \
            #               (nrnact_ch['width'] == centre)


            # Select a few locations along the centre row and column (r&c-->rc)
            bool_dfs_w = [(nrnact_ch['width'] == i) & (
                           nrnact_ch['height'] == centre) for i in
                          self.rc_locs]
            bool_dfs_h = [(nrnact_ch['height'] == i) & (
                           nrnact_ch['width'] == centre) for i in
                          self.rc_locs]
            bool_dfs = bool_dfs_w
            bool_dfs.extend(bool_dfs_h)
            for i, bdf in enumerate(bool_dfs):
                if i==0:
                    centre_rc_cond = bdf
                else:
                    centre_rc_cond = centre_rc_cond | bdf


            nrnact_ch = nrnact_ch.loc[centre_rc_cond]
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

            # if full_state_traces.empty:
            #     full_state_traces = state_traces
            # else:
            #     full_state_traces = full_state_traces.append(state_traces)

            # Now, having collected all the state traces of the neurons of
            # interest, compute cross correlation plots between neurons in THE SAME
            # channel in the same batch element:

            ## Define the locations of the central row and column for this batch
            bool_dfs_w = [(full_state_traces['width'] == i)&(full_state_traces['height']==centre) for i in self.rc_locs]
            bool_dfs_h = [(full_state_traces['height'] == i)&(full_state_traces['width']==centre) for i in self.rc_locs]
            bool_dfs = bool_dfs_w
            bool_dfs.extend(bool_dfs_h)
            for i, bdf in enumerate(bool_dfs):
                if i==0:
                    centre_rc_cond = bdf
                else:
                    centre_rc_cond = centre_rc_cond | bdf

            centre_only_cond = (full_state_traces['height']==centre) & \
                               (full_state_traces['width'] == centre)

            for b in range(self.num_batches):
                acorr_data_during = pd.DataFrame()
                acorr_data_outside = pd.DataFrame()
                acorr_data_dfs = [acorr_data_during, acorr_data_outside]

                print("b%s   ch%s  " % (b, ch))

                # Define conditions to get traces from full df
                cond_b = full_state_traces['batch'] == b
                cond_ch = full_state_traces['channel'] == ch
                cond1 = centre_only_cond & cond_b & cond_ch
                cond2 = centre_rc_cond & cond_b & cond_ch

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
        p01 = [7e1, 400.0]#[5e2, 5., 1.0, 0.] ######[7e2, 0.9, 500.0, 0.]
        p02 = [7e1, 500.0]#[5e2, -0.5, 100.0, 0.]
        p03 = [7e1, 600.0]#[7e2, 0.9, 500.0, 0.]
        init_params_gabor = [p02]#[p01, p02, p03]#, p04]

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
                        print("Channels %s" % (ch))

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
        batch_groups = self.just_angles_few_angles_batchgroups #longjustangles
        # batch groups are the same as short stim version.
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

            if plotting:
                # Plot results
                fig, ax = plt.subplots(1, figsize=(9, 5))
                ax.plot(self.acorr_x, gabor_curve)  # /np.max(acorr_trace))
                ax.scatter(self.acorr_x[peaks[0]], gabor_curve[peaks[0]])
                ax.set_xlabel('lag')
                ax.set_ylabel('correlation coefficient')
                # ax.legend()

                fig.savefig(
                    "%s/diagnostic plot for fitted gabors_ch%i in batch %i at (x%i, y%i) for %s.jpg" % (
                        acorr_dir, int(row['channel']),
                        int(row['batch']), int(x), int(y), str(row['dur_or_out'])))
                plt.close()

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
        # bg_locs = self.double_stim_batchgroup_locs

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

        # Calculate (square) distance of mobile from static stim
        # as the max distance along an axis
        xbiggery = \
            acorr_results_df['x'] > acorr_results_df['y']
        acorr_results_df['sq_dist'] = acorr_results_df['x'].where(
            xbiggery, other=acorr_results_df['y'])

        # Calculate euclidian distances
        eucl_dists = [np.sqrt(x**2 + y**2)
                 for (x,y) in zip(acorr_results_df['x'], acorr_results_df['y'])]
        acorr_results_df['eucl_dist'] = eucl_dists

        # Assign stim angles
        stim_angles = np.array([bg_angles[int(i)] for i in
                                acorr_results_df['batch_group']])
        acorr_results_df['stim_angle'] = stim_angles

        # Calculate the difference in ori pref of stim from channel
        difference = acorr_results_df['ori_pref'] - \
                     acorr_results_df['stim_angle']
        angle_diff = np.pi - np.abs(np.abs(difference) - np.pi)
        acorr_results_df['oripref_stim_angle_diff'] = angle_diff

        # Calculate the difference in ori pref of stim from vertical
        difference = acorr_results_df['ori_pref'] - vert_ang
        angle_diff = np.pi - np.abs(np.abs(difference) - np.pi)
        acorr_results_df['oripref_vert_angle_diff'] = angle_diff

        # Calculate the difference in ori pref of stim from horizontal
        difference = acorr_results_df['ori_pref'] - horiz_ang
        angle_diff = np.pi - np.abs(np.abs(difference) - np.pi)
        acorr_results_df['oripref_horiz_angle_diff'] = angle_diff

        # Calculate the distance in the direction of ori pref
        vert_aligned_dists = acorr_results_df['y'].loc[(acorr_results_df['oripref_vert_angle_diff'] < ten_degrees)]
        horiz_aligned_dists = acorr_results_df['x'].loc[(acorr_results_df['oripref_horiz_angle_diff'] < ten_degrees)]

        vert_aligned_dists = np.abs(vert_aligned_dists)
        horiz_aligned_dists = np.abs(horiz_aligned_dists)

        ## Add 'distance in the direction of ori pref' to main df
        combo_vert_horiz_dists = pd.concat(
            [vert_aligned_dists, horiz_aligned_dists], axis=1)
        combo_vert_horiz_dists = \
            combo_vert_horiz_dists['x'].fillna(combo_vert_horiz_dists['y'])
        combo_vert_horiz_dists = pd.DataFrame(combo_vert_horiz_dists)
        combo_vert_horiz_dists = \
            combo_vert_horiz_dists.rename(columns={'x':'aligned_dist'})
        acorr_results_df['aligned_dist'] = np.nan
        acorr_results_df.update(combo_vert_horiz_dists)

        # Define the conditions and their labels for plotting in a for-loop
        df = acorr_results_df  # for brevity

        """Conditions I want:
        stim_ori alignment conds:
        [stim and oripref are aligned; unaligned; either aligned or unaligned (all)]
        x
        loc_conds:
        [locations aligned with oripref; locations unaligned to oripref]
        x
        dur_out_conds:
        [during stim; outside stim]
        x
        distance_conds
        [sq distance; euclidean distancs; vertical distance; horizontal distance, distance in direction of aligned orientation preference]
        """

        stim_ori_align_conds = []
        loc_conds = []
        dur_out_conds = []
        distance_conds = []

        # Create labels based on above description
        stim_ori_align_labels = ['stim_oripref_aligned', 'stim_oripref_unaligned', 'stim_oripref_all']
        loc_labels = ['locs_vertical', 'locs_horiztonal', 'locs_oripref_aligned', 'locs_oripref_unaligned', 'locs_all']
        dur_out_labels = ['During experimental stimulus','Before experimental stimulus']
        distance_labels = ['square distance', 'Euclidean distance', 'vertical distance', 'horizontal distance', 'distance in direction of aligned orientation preference']

        # Create conditions that fit the labels, in the same order
        stim_ori_align_conds.append(df['oripref_stim_angle_diff']<ten_degrees)
        stim_ori_align_conds.append(df['oripref_stim_angle_diff']>=ten_degrees)
        stim_ori_align_conds.append(df['oripref_stim_angle_diff']>-1)

        loc_conds.append(df['x']==centr)
        loc_conds.append(df['y']==centr)
        loc_conds.append(((df['x']==centr)&(df['oripref_vert_angle_diff']<ten_degrees)) | ((df['y']==centr)&(df['oripref_horiz_angle_diff']<ten_degrees)))
        loc_conds.append(((df['x']==centr)&(df['oripref_vert_angle_diff']>=ten_degrees)) | ((df['y']==centr)&(df['oripref_horiz_angle_diff']>=ten_degrees)))
        loc_conds.append((df['x']==centr) | (df['y']==centr))


        dur_out_conds.append(df['dur_or_out']==0)
        dur_out_conds.append(df['dur_or_out']==1)

        distance_conds.append(df['sq_dist'])
        distance_conds.append(df['eucl_dist'])
        distance_conds.append(np.abs(df['y']))
        distance_conds.append(np.abs(df['x']))
        distance_conds.append(df['aligned_dist'])


        not_overlap_cond = ~((df['x'] == 0) & (df['y']==0))
        # not_overlap_cond = df['x'] > -8888 #for debugging

        # Go through the various conditions and data and make plots
        for indep_idx, (indep_label, indep_data_df) in \
                enumerate(zip(distance_labels, distance_conds)):
            for dependent_label in ['amplitude']:

                # Prepare directory for saving plots
                # plot_sub_dir = \
                #     os.path.join(plot_dir, indep_label+" vs "+dependent_label)
                # if not os.path.isdir(plot_sub_dir):
                #     os.mkdir(plot_sub_dir)

                for st_ori_idx, (st_ori_lab, st_ori_cond) in \
                        enumerate(zip(stim_ori_align_labels, stim_ori_align_conds)):
                    for loc_idx, (loc_label, loc_cond) in \
                            enumerate(zip(loc_labels, loc_conds)):

                        title = "%s vs %s for %s when %s" % (indep_label, dependent_label, loc_label, st_ori_lab)
                        print(title)



                        # Create figure
                        fig, ax = plt.subplots(2, figsize=(9, 9))

                        for dur_out_idx, (dur_out_lab, dur_out_cond) in \
                                enumerate(zip(dur_out_labels,
                                              dur_out_conds)):
                            print(dur_out_lab)
                            # # Combine conditions
                            # if indep_idx == 4:
                            #     full_cond = not_overlap_cond & loc_cond & dur_out_cond  # No st_ori_cond
                            # else:
                            #     full_cond = not_overlap_cond & st_ori_cond & loc_cond & dur_out_cond
                            full_cond = not_overlap_cond & st_ori_cond & loc_cond & dur_out_cond


                            # Define data using conds
                            indep_data = indep_data_df[full_cond]
                            dependent_data = acorr_results_df[dependent_label][full_cond]
                            # num_peaks_data = acorr_results_df['num_peaks'][not_overlap_cond & cond]

                            colors = acorr_results_df['ori_pref'][full_cond]

                            # Check for nans and all-zero data (both break np.polyfit)
                            if indep_data.hasnans:
                                wherena = indep_data.isna()
                                indep_data = indep_data.loc[~wherena]
                                dependent_data = dependent_data.loc[~wherena]
                                colors = colors.loc[~wherena]
                            if dependent_data.hasnans:
                                wherena = dependent_data.isna()
                                indep_data = indep_data.loc[~wherena]
                                dependent_data = dependent_data.loc[~wherena]
                                colors = colors.loc[~wherena]
                            if all(indep_data==0) or all(dependent_data==0):
                                print("data only zeros", )
                                plt.close()
                                break

                            # Plot basic plot
                            m, c = np.polyfit(indep_data, dependent_data, 1) # Calc line

                            ax[dur_out_idx].title.set_text(dur_out_lab)

                            ax[dur_out_idx].scatter(indep_data, dependent_data, cmap=cmap,
                                       c=colors)
                            ax[dur_out_idx].plot(indep_data, m * indep_data + c, c='r')  # Plot line
                            ax[dur_out_idx].set_xlabel('%s' % indep_label)
                            ax[dur_out_idx].set_ylabel('Oscillation %s' % dependent_label)

                        plt.savefig(os.path.join(plot_dir,
                                                 "SynchExp3 " + title + ".png"))
                        plt.close()

















































































































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
        reshaped_data = [rsd.reshape(len(dm.timesteps), self.num_batches)
                              for rsd in data] # make into columns
        summed_data = reshaped_data[0] + reshaped_data[1]  # Sum momenta and energy

        num_ts = len(dm.timesteps)
        timesteps = [np.arange(num_ts).reshape(num_ts, 1)] * self.num_batches #time idxs
        timesteps = np.concatenate(timesteps, axis=0)
        summed_data = summed_data.reshape(num_ts * self.num_batches, 1)
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
        reshaped_data = [rsd.reshape(len(dm.timesteps), self.num_batches)
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

    def plot_pixel_vs_activity(self):
        print("Plotting pixel vs activity map")
        nrnact = pd.read_pickle(
            os.path.join(self.session_dir,
                         'neuron_activity_results_primary.pkl'))

        im_start = self.top_left_pnt[0]

        # Reorganises activity dataframe so we can sum over pixels
        map_act = pd.melt(nrnact, id_vars=['batch_idx', 'height', 'width'],
                          value_vars=list(range(32)))
        map_act = map_act.rename(
            columns={'variable': 'channel', 'value': 'active'})
        pixel_inds = [(i, j) for i in list(range(self.extracted_im_size)) for j in
                      range(self.extracted_im_size)]

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
        exp_stims = torch.stack(
            exp_stims)  # should have generated only 128 images
        exp_stims = exp_stims.mean(dim=1)  # makes image black and white
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
        for ch in range(self.num_ch):
            # Group the df data by pixel index (i.e. get the probability of
            pch = pix_int_df['channel'] == ch
            values_for_channel = pix_int_df.loc[pch]
            values_for_channel = \
                values_for_channel.groupby(['Pixel intensity'])[
                    'active'].mean()
            values_for_channel = pd.DataFrame(values_for_channel)
            plt.scatter(values_for_channel.index,
                        values_for_channel['active'], )
            plt.xlabel('Pixel intensity')
            plt.savefig(os.path.join(
                self.session_dir,
                'activity proportion vs pixel intensity for ch %i.png' % ch))
            plt.close()

    def explore_oscillations_in_channel(self, patch_or_idx=None):
        # Make dir to save plots for this experiment
        exp_dir = os.path.join(self.session_dir, 'summed centre states')
        if not os.path.isdir(exp_dir): os.mkdir(exp_dir)

        if patch_or_idx == 'patch':
            centre1 = self.central_patch_min
            centre2 = self.central_patch_max
            save_name = 'patch'
        else:
            centre1 = self.top_left_pnt[0] + self.extracted_im_size / 2
            centre2 = centre1
            save_name = 'neuron'
        print("Exploring oscillations")


        # Load data
        full_data = pd.read_pickle(os.path.join(self.session_dir,
            "neuron_activity_results_alternativeformat_primary_w_ori_w_match.pkl"))

        # Load data
        # nrnact = pd.read_pickle(
        #     os.path.join(self.session_dir,
        #                  'neuron act ori.pkl'))

        # Prepare general variables
        model_exp_name = self.primary_model_exp_name
        # TODO this func is a bit of a mess, will return to it
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
            nrnact_ch = full_data.loc[full_data['channel'] == ch]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['matched_stim_ori_pref']]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['active']]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['height'] >= centre1]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['height'] <= centre2]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['width'] >= centre1]
            nrnact_ch = nrnact_ch.loc[nrnact_ch['width'] <= centre2]
            nrnact_ch = nrnact_ch.reset_index()

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

                lags1, acorrs1, plot11, plot12 = ax[2].acorr(df[name][self.primary_stim_start:3550],
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

            state_trace_during_trunc = state_trace_during[-300:]
            state_trace_before_trunc = state_trace_before[-300:]
            momentum_trace_during_trunc = momentum_trace_during[-300:]
            momentum_trace_before_trunc = momentum_trace_before[-300:]


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
            # fig.set_size_inches(4.5, 8.5)
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
            # fig.set_size_inches(4.5, 8.5)
            ## During
            stattr = ax[0].plot(np.array(list(range(len(state_trace_during_trunc)))),
                       state_trace_during_trunc)
            momtr = ax[0].plot(np.array(list(range(len(momentum_trace_during_trunc)))),
                       -momentum_trace_during_trunc, c='r')
            ax[0].set_ylabel('State or negative momentum [a.u.]')
            ax[0].legend(['State', 'Negative Momentum'])

            ##Before
            stattr = ax[1].plot(np.array(list(range(len(state_trace_before_trunc)))),
                       state_trace_before_trunc)
            momtr = ax[1].plot(np.array(list(range(len(momentum_trace_before_trunc)))),
                       -momentum_trace_before_trunc, c='r')
            ax[1].set_ylabel('State or negative momentum [a.u.]')
            ax[1].set_xlabel('Timesteps')
            ax[1].legend(['State', 'Negative Momentum'])

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
            fig, ax = plt.subplots(3, sharex=True)
            fig.set_size_inches(8.5, 8.5)

            ax[0].plot(range(-trc_mid_len,trc_mid_len),
                       state_trace_acorr_result_dur)
            ax[1].plot(range(-trc_mid_len,trc_mid_len),
                       state_trace_acorr_result_before)
            ax[2].plot(range(-trc_mid_len,trc_mid_len),
                       st_mom_xcorr_result)

            for i in range(3):
                ax[i].axvline(x=0, c='black', linewidth=1, linestyle='dashed')
                ax[i].set_ylabel('Correlation')

            ax[0].title.set_text('State autocorrelation during stimulation period')
            ax[1].title.set_text('State autocorrelation before stimulation period')
            ax[2].title.set_text('Cross correlation between state and negative momentum')

            ax[2].set_xlabel('Timestep lag')
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
        # model_exp_name = self.primary_model_exp_name #todo change to self.long_just_angles_exp_name
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




    def butter_lowpass_filter(self, data, cutoff, fs=1, order=2):
        normal_cutoff = cutoff / 0.5  # Nyquist freq = 0.5 = 1/2
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