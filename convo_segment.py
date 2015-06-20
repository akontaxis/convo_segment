from scipy.io import wavfile
import features
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import mixture, cluster, decomposition
import math
import sys
from ggplot import *


class ClusteredAudioMFCC:
    
    '''
    Object containing the audio, its extracted MFCC features for each time
    window, and the resulting speaker labels. 

    Required positional arguments are a filename and and a number of 
    clusters (number of speakers in the conversation). 
    
    At this point, only .wav files are supported.
    '''

    
    def __init__(self,
                 filename, n_clusters, use_covariance = False,
                 feature_winlen = 0.020, feature_numcep =13 , feature_nfilt = 26, 
                 gauss_comp = 1, iters = 20, cov = 'full',
                 bin_width = 1.0, bin_step = 0.2):
        
        self.filename = filename
        self.feature_winlen = feature_winlen
        self.feature_numcep = feature_numcep
        self.feature_nfilt = feature_nfilt
        self.gauss_comp = gauss_comp
        self.iters = iters
        self.cov = cov
        self.bin_step = bin_step
        self.bin_width = bin_width
        self.use_covariance = use_covariance
        self.n_clusters = n_clusters

        try:
            [self.sample_rate, self.signal] = wavfile.read(self.filename)
        except:
            "Error: problem reading audio."


        self.binned_signals = self.get_binned_signals()
        self.binned_features = self.get_binned_features()
        self.binned_model_params = self.get_binned_model_params()
        self.cluster_labels = self.get_cluster_labels()
        self.labeled_timestamps = self.get_labeled_timestamps()
        self.binned_means = self.get_binned_means()
        

    def get_binned_signals(self):

    	if self.bin_step > self.bin_width:
    		print 'Error, bin step is larger than bin width, creating gaps.'
    	else:
    		n_bin_comp = int(float(self.sample_rate*self.bin_width))
    		bin_step_comp = math.floor(self.sample_rate*self.bin_step)
    		n_bins = int(math.floor(float(self.signal.shape[0])/bin_step_comp))
    		binned_signals = []
    		for i in range(n_bins):
    			try:
    				binned_signals.append(self.signal[i*bin_step_comp: i*bin_step_comp + n_bin_comp])
    			except: 
    				pass
    		return binned_signals


    def get_binned_features(self):
    	binned_features = {}
    	for i in range(len(self.binned_signals)):
    		binned_features[i] = features.mfcc(self.binned_signals[i], 
                                               self.sample_rate, 
                                               winlen = self.feature_winlen, 
                                               numcep = self.feature_numcep, 
                                               nfilt = self.feature_nfilt)
    	return binned_features



    def get_binned_model_params(self):
    	binned_models = {}
    	binned_model_params = {}
	
    	for key in self.binned_features.keys():
    		G = mixture.GMM(n_components = self.gauss_comp, 
                            n_iter = self.iters, 
                            covariance_type = self.cov) 
    		binned_models[key] = G.fit(self.binned_features[key])
    	for key in binned_models.keys():
    		binned_model_params[key] = [binned_models[key].means_, binned_models[key].covars_]
    	return binned_model_params
    
    
    
    def get_cluster_labels(self):

    	n_samples = len(self.binned_model_params)
	
    	if self.use_covariance == False:
    		data_mat = np.zeros((n_samples, self.feature_numcep))
    		for key in self.binned_model_params.keys():
    			data_mat[key:] = self.binned_model_params[key][0]

    	elif self.use_covariance == True:
    		data_mat = np.zeros((n_samples, 
                                self.feature_numcep + self.feature_numcep**2))

    		for key in self.binned_model_params.keys():
#    			print self.binned_model_params[key][0].shape
#    			print self.binned_model_params[key][1].reshape(1,-1).shape
   			data_mat[key:] = np.concatenate([self.binned_model_params[key][0],  
                                             self.binned_model_params[key][1].reshape(1,-1)], 
                                             axis = 1)

    	clust = cluster.KMeans(n_clusters = self.n_clusters, init = 'random')
    	clust_fit = clust.fit(data_mat)
    	return clust_fit.labels_
     


    def get_binned_means(self):
        binned_means = []
        for i in sorted(list(self.binned_model_params.keys())):
            binned_means.append(self.binned_model_params[i][0][0])
        return np.array(binned_means)

            
    
    def get_labeled_timestamps(self):
    	timestamps = []
    	for i in range(len(self.cluster_labels)):
    		timestamps.append((
            (i*self.bin_step, i*self.bin_step + self.bin_width), 
                             self.cluster_labels[i])) 
    	return timestamps
    
            
        





def main():

    '''
    The filename and number of speakers must be passed as command line 
    arguments to the script.
    
    Optional command line argument 'plot' will produce of plot of the cepstral
    means for each timestep. The x-axis is time and the y-axis is the first 
    principal component of the cepstral means (by default these sit in a 
    13-dimensional space). Points are colored by cluster label.
    
    Optional command line feature 'csv' will record the cepstral mean and 
    cluster label for each time window. 
    '''

    filename = sys.argv[1]
    n_speakers = int(sys.argv[2])
    args = sys.argv
    
    clusteredAudio = ClusteredAudioMFCC(filename, n_speakers)
    
    
    if 'csv' in args:
        '''
        
        '''
        numcep = clusteredAudio.feature_numcep
        mean_labels = ['cepstral_mean_' + str(i) for i in range(1, numcep + 1)]
        timestamps = clusteredAudio.labeled_timestamps
        timesteps_begin = np.array([item[0][0] for item in timestamps])
        timesteps_end = np.array([item[0][1] for item in timestamps])
        num_windows = timesteps_end.shape[0]
        csv_df = DataFrame(data = np.zeros((num_windows, 3 + numcep)),
                           columns = ['timestep_begin',
                                     'timestep_end', 'cluster_label'] +
                                     mean_labels)
        csv_df.ix[:, 'timestep_end'] = timesteps_end
        csv_df.ix[:, 'timestep_begin'] = timesteps_begin
        csv_df.ix[:, mean_labels] = clusteredAudio.binned_means
        csv_df.ix[:, 'cluster_label'] = clusteredAudio.cluster_labels
        csv_df.to_csv(path_or_buf = filename + '_MFCC_clustered.csv')

    if 'plot' in args:
        means_PCA = decomposition.PCA(n_components = 1)
        timestamps = clusteredAudio.get_labeled_timestamps()
        timesteps = np.array([item[0][0] for item in timestamps])
        means = np.array(clusteredAudio.get_binned_means())    
        mean_component1 = means_PCA.fit_transform(means)
        labels = clusteredAudio.get_cluster_labels()
        plot_df = DataFrame({'time':timesteps, 
                             'PC1': means_PCA, 
                             'label': labels})
        
        p = ggplot(aes(x = 'time', y = 'mean_component1', color = 'label'), data = plot_df) + geom_point()
        plot_filename = filename + '_plot.png'
        ggsave(p, plot_filename)
         

                 
            
if __name__ == '__main__':
    main()


    