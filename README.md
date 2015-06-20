# convo_segment
A script for unsupervised labeling of changes of speaker in a (polite) conversation or interview. Works by clustering speech features (means of MFCCs over 1 second time windows) . Uses the python_speech_features module. 


================
Details
================


Given a .wav recording of a conversation or interview between a small number of people, convo_segment tries to label when each participant is speaking. 

convo_segment works by extracting Mel-Frequency Cepstral Coefficients (MFCCs) from the full recording using the python speech features module. MFCCs are one of the most commonly used features of speech; they are useful not only for automated speech recognition (i.e. determining what words were said in a recording) but also in speaker recognition (determining who spoke).

In the context of speaker recognition, one usually models the aggregate distribution of MFCCs over the span of a speech recording (MFCCs are vectors -- usually 10-20 dimensional -- associated with short time windows of speech, typically around 0.02 second in length). The most common choice is Gaussian mixture models (GMMs). Given training recordings for a set of N possible speakers, here is simplified framework for identifying the speaker of a new recording:

1) Extract MFCCs from each of the training recordings. For each recording, model the distribution of MFCCs as a GMM.

2) Extract MFCCs from the new recording. 

3) Compute the likelihood of these MFCCs under each of the GMMs fit to the training recordings.

3) Find the training GMM which under which the MFCCs of the new recording have the highest likelihood. 

4) The corresponding speaker is our prediction for the speaker of the new recordings. 



convo_segment is a simplified adaptation of this idea to the unsupervised setting. 

1) We break a conversation between two speakers into 1 second time windows T(i) (with a step of 0.2 seconds). 

2) We then extract MFCCs from the recording over each 1s time window (there 50 samples for each window). 

3) Instead of fitting a GMM to the MFCCs of each window, we just compute the mean. For each window T(i), this gives is a 10-20 dimensional vector M(i). 

4) We cluster these vectors M(i). We just use k-means, but fancier choices are possible. Note that this does mean we need to know the number of speakers in advance. 

5) The predictions for who's speaking in the conversation during time window T(i) is just the cluster label of M(i).



================
How to use:
================

Put an audio file example.wav in the same directory as the convo_segment.py. If the number of speakers in example.wav is k, then run

python convo_segment.py example.wav 2

There are two optional command line arguments, 'plot' and 'csv':
The 'csv' option will write a .csv containing the cluster labels and the cepstral means M(i) for each time window.

The 'plot' option will plot the M(i) against time. The y-axis is the first principal component of the M(i) over all time windows, the x-axis is time, and the colorign represent the cluster labels (speaker predictions) produced by convo_segment. 


================
Dependencies:
================

scipy
python_speech_features
numpy 
pandas 
sklearn
ggplot (for use of the plotting feature)