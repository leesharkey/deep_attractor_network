"""To use this script, you'll need data. You can generate data from a trained
model in the script main.py by using the experiment functions found in the
experiment manager. For even a modest size of architecture, saving the values
of the states, their energies, their gradients, and their momenta takes up
a lot of storage space e.g. tens or hundreds of megabytes per timestep, and
you'll need hundreds or thousands of timesteps. Recommend using an EHD. """

import argparse
import lib.analysis.analysismanager as analysismanager
import lib.utils

shapes = lambda x: [y.shape for y in x]

def main():

    parser = argparse.ArgumentParser(description='Analysis of inference dynamics.')
    sgroup = parser.add_argument_group('Path')
    sgroup.add_argument('--root_path', type=str, default='/media/lee/DATA/DDocs/AI_neuro_work/DAN/exp_data/',
                        help='The path to the recorded experimental data.' +
                             'session that the data were generated by. ' +
                             'Default: %(default)s.')
    sgroup.add_argument('--ehd_results_storage', action='store_true',
                        help='Whether or not to store the results of the '
                             'analysis on the EHD.'
                             'Default: %(default)s.')
    parser.set_defaults(ehd_results_storage=False)

    args = parser.parse_args()


    # sgroup.add_argument('--modelexp_name', type=str, default='20200422-145311__rndidx_25238_loaded20200417-165337__rndidx_70768/observeCIFAR10',
    #                     help='The name of the model and the visualisation ' +
    #                          'session that the data were generated by. ' +
    #                          'Default: %(default)s.')
    # root_path = '/media/lee/DATA/DDocs/AI_neuro_work/DAN/exp_data/'
    # modelexp_path = '20200422-145311__rndidx_25238_loaded20200417-165337__rndidx_70768/observeCIFAR10'


    session_name = lib.utils.datetimenow()

    am = analysismanager.AnalysisManager(args, session_name)
    #am.print_stimuli()# you don't really use this
    #am.plot_pixel_vs_activity() # you don't really use this
    #am.print_activity_map() # you don't really use this

    # In each experiment, find the active neurons, the mean activities
    # during and outside of stimulation, and shapiro tests.
    # am.find_active_neurons('primary')
    # am.find_active_neurons('just_angles')
    am.find_active_neurons('just_angles_few_angles')
    am.find_active_neurons('double_stim')
    am.find_active_neurons('long_just_fewangles')

    am.print_activity_maps_by_batch_ch()
    am.print_activity_map_GIFs_by_batch_ch()

    am.print_activity_maps_by_batch_ch('long_just_fewangles')
    am.print_activity_map_GIFs_by_batch_ch('long_just_fewangles') #TODO fix the exp_name issue for getting state traces

    am.find_orientation_preferences()
    am.assign_ori_info()
    am.plot_state_traces_with_spec_and_acorr('patch')
    am.plot_state_traces_with_spec_and_acorr(None)
    am.plot_contrast_specgram_comparison_local('patch')
    am.plot_contrast_specgram_comparison_local(None)
    am.plot_contrast_specgram_comparison_LFP()
    am.plot_contrast_power_spectra_LFP()
    am.plot_contrast_dependent_transients_of_active_neurons('patch')
    am.plot_contrast_dependent_transients_of_active_neurons(None)

    am.plot_state_traces('patch')
    am.plot_state_traces()
    am.plot_single_neuron_state_traces()
    #TODO: am.plot_single_neuron_energy_traces()
    # TODO: am.plot_single_neuron_mom_traces()
    am.find_oscillating_neurons()  #memory error?
    am.plot_contrast_frequency_plots()


    # #am.single_neuron_dynamics_plot_Ham_case()
    # #am.single_neuron_autocorrelation()
    # am.plot_energy_timeseries()
    # # #am.single_neuron_trial_avg_EI_balance() #unused
    # am.timeseries_EI_balance()
    # am.xcorr_EI_lag() #hasn't been used in a long time
    # am.two_neuron_crosscorrelation() #hasn't been used in a long time
    # am.explore_oscillations_in_channel() #in a messy state right now but will maybe return to.


    am.plot_state_and_mom_trace()
    am.calculate_EI_lag()


    am.synchrony_experiment1_overlapping_rec_fields('single')
    am.synchrony_experiment1_overlapping_rec_fields_fit_Gabors() # TODO change name to remove 'overlapping rec fields' part
    am.synchrony_experiment1_overlapping_rec_fields_Plot_acorrs_individually() #not really used
    am.synchrony_experiment1_overlapping_rec_fields_Plot_acorrs_overlay() # not really used
    am.synchrony_experiment1_overlapping_rec_fields_Analyze_fitted_Gabors()
    am.synchrony_experiment1_overlapping_rec_fields_OriPref_vs_OscAmp()

    am.print_activity_maps_by_batch_ch_double_stim()

    am.synchrony_experiment2_xcorrs()
    am.synchrony_experiment2_fit_Gabors()
    am.synchrony_experiment2_Analyze_fitted_Gabors()
    am.synchrony_experiment2_OriPref_OR_Distance_vs_OscAmp_OR_vs_Phase()

    am.synchrony_experiment3_xcorrs()
    am.synchrony_experiment3_fit_Gabors()
    am.synchrony_experiment3_Analyze_fitted_Gabors()
    am.synchrony_experiment3_Distance_vs_OscAmp()




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
#
#
#     def one_d_gabor_function(self, p, x):
#         return (1 / (np.sqrt(2*np.pi) * p['sigma'])) \
#                * np.exp((-x**2)/(2*p['sigma'])) * p['amp'] \
#                * np.cos(2 * np.pi * p['omega'] * x + p['bias1']) \
#                + p['bias2'] * x
#
#     def gabor_fitting_func(self, p, x, y):
#         return self.one_d_gabor_function(p, x) - y












# The below copy of find_oscillations was before I started with the state+grad
# as pyr output hypothesis

    # def find_oscillating_neurons(self):
    #     """Looks at a target subset of the data and returns the indices of the
    #     neurons whose responses qualify as oscillating."""
    #     print("Finding oscillating neurons")
    #     model_exp_name = self.primary_model_exp_name
    #     # var_names = ['energy']
    #     # var_label_pairs = [('energy_1', 'Energy')]
    #     var_names = ['grad', 'state']
    #     var_label_pairs = [('grad_1', 'Grad'),
    #                        ('state_1', 'State')]
    #     dm = DataManager(root_path=self.args.root_path,
    #                      model_exp_name=model_exp_name,
    #                      var_names=var_names,
    #                      state_layers=[1],
    #                      batches=[0],
    #                      channels=None,
    #                      hw=[[13,13],[18,18]],
    #                      timesteps=range(0, 7050))
    #
    #     # Process data before putting in a dataframe
    #     reshaped_data = [np.arange(len(dm.timesteps)).reshape(
    #         len(dm.timesteps), 1)]
    #     reshaped_data.extend([
    #                 dm.data['energy_1'].reshape(len(dm.timesteps), -1)])
    #     num_energy_cols = reshaped_data[1].shape[1]
    #     arr_sh = dm.data['energy_1'].shape[2:]
    #     inds = np.meshgrid(np.arange(arr_sh[0]),
    #                        np.arange(arr_sh[1]),
    #                        np.arange(arr_sh[2]),
    #                        sparse=False, indexing='ij')
    #     inds = [i.reshape(-1) for i in inds]
    #     energy_names = []
    #     for i,j,k in zip(*inds):
    #         idx = str(i) + '_' + str(j) + '_' + str(k)
    #         energy_names.append('energy_1__' + idx)
    #
    #
    #     data = np.concatenate(reshaped_data, axis=1)
    #
    #
    #     colnames = ['Timestep']
    #     colnames.extend(energy_names)
    #     df = pd.DataFrame(data, columns=colnames)
    #     for name in colnames[1:]:
    #         fig, ax = plt.subplots(4)
    #         fig.set_size_inches(18.5, 10.5)
    #         spec_data = (df[name] - np.mean(df[name]))
    #         spec_data = spec_data / np.linalg.norm(spec_data)
    #
    #         #Denoise
    #         #spec_data = self.butter_lowpass_filter(spec_data,cutoff=0.1,fs=100)
    #
    #         ax[0].plot(df['Timestep'], spec_data)
    #         ax[0].set_ylabel('Energy [a.u.]')
    #         ax[0].set_xlabel('Timesteps')
    #
    #         ax[1].specgram(spec_data, NFFT=512, Fs=100, Fc=0, detrend=None,
    #                        noverlap=511, xextent=None, pad_to=None,
    #                        sides='default',
    #                        scale_by_freq=True, scale='dB', mode='default',
    #                        cmap=plt.get_cmap('hsv'))
    #         ax[1].set_ylabel('Frequency [a.u.]')
    #
    #
    #         lags1, acorrs1, plot11, plot12 = ax[2].acorr(df[name][3050:4050],
    #                                        detrend=plt.mlab.detrend_linear, #lambda x: x - np.mean(x)
    #                                        maxlags=999)
    #         lags2, acorrs2, plot21, plot22 = ax[3].acorr(df[name][4050:5050],
    #                                        detrend=plt.mlab.detrend_linear,
    #                                        maxlags=999)
    #         plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))  # Uses sci notation for units
    #         plt.tight_layout()  # Stops y label being cut off
    #
    #         # plt.title('STFT Magnitude')
    #         plt.savefig("spectrum__single_gabor%s.png" % name)
    #         plt.close()
    #
    #         # p0 = {'sigma': 1.0,
    #         #      'omega':  1.0,
    #         #      'amp':    1.0,
    #         #      'bias1':  0.0,
    #         #      'bias2':  0.0}
    #
    #         p0 = [555.0, 550.0, 1e20, 0.0, 0.0]
    #         p1, success = optimize.leastsq(self.gabor_fitting_func, p0,
    #                                        args=(lags1))
    #         #plt.plot(lags1, acorrs1)
    #         plt.plot(lags1, self.one_d_gabor_function(p1, lags1))
    #         plt.savefig("gabors.png")
    #         plt.close()
    #
    #
    #
    #     # Make one plot per variable
    #     # for j, colname in enumerate(colnames[1:]):
    #     #     sns.set(style="darkgrid")
    #     #     _, _, plot1, plot2 = plt.acorr(df[colname],
    #     #                                    detrend=lambda x: x - np.mean(x),
    #     #                                    maxlags=749)
    #     #     plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))  # Uses sci notation for units
    #     #     plt.tight_layout()  # Stops y label being cut off
    #     #     fig = plot1.get_figure()
    #     #     fig.savefig('plot_osc_search_%s.png' % colname)
    #     #     fig.clf()
