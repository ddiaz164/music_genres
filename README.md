![](https://uidesign.gbtcdn.com/gb_blog/2776/What-is-audio-format-and-the-main-types-of-it-Z01.jpg)
# Audio Classification: Predicting Song Genres
Audio data is quickly becoming an important part of machine learning. We’re using it to interact with virtual assistants like Alexa or Siri and helping self-driving cars hear their surroundings rather than just seeing them.  Music has been a part of my life ever since I started playing my first instrument when I was 4 years old so I really wanted to use a neural network and see if it could learn to distinguish different types of music.
## Data Source
The data I used for this project was the free music archive. It contains up to 106,574 tracks with 161 unbalanced genres, but in the interest of time I decided to work with the small dataset which only has 8 genres and 8,000 30-second tracks evenly distributed for each genre. The genres in this set were Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, and Rock.
## Principal Component Analysis
When first looking at the data I did some PCA down to 2 dimensions in order to plot and get a sense of how these genres would compare to each other. 

<img src="https://github.com/ddiaz164/music_genres/blob/master/images/pca_ins_hip.png" width="440" height="300"><img src="https://github.com/ddiaz164/music_genres/blob/master/images/pca_rock_pop.png" width="440" height="300">

With Instrumental and Hip-Hop I could see the separation, but with Rock and Pop it would prove a lot tougher to predict one way or the other.

## Mel-Spectrogram
Since I was dealing with audio data I knew I wanted to look at spectrograms as a visual representation of frequencies over time. I used librosa to squash a regular spectrogram, which is the squared magnitude of the short term Fourier transform of the audio signal, into something a human can better understand using mel scale. 

<img src="https://github.com/ddiaz164/music_genres/blob/master/images/mel_spec_Folk.png" width="440" height="200"><img src="https://github.com/ddiaz164/music_genres/blob/master/images/mel_spec_Electronic.png" width="440" height="200">
<img src="https://github.com/ddiaz164/music_genres/blob/master/images/mel_spec_Experimental.png" width="440" height="200"><img src="https://github.com/ddiaz164/music_genres/blob/master/images/mel_spec_Rock.png" width="440" height="200">

Looking at the mel-spectrograms for each genre I could see they had noticeable differences.

<img src="https://github.com/ddiaz164/music_genres/blob/master/images/mel_spec_International.png" width="440" height="200"><img src="https://github.com/ddiaz164/music_genres/blob/master/images/mel_spec_Instrumental.png" width="440" height="200">
<img src="https://github.com/ddiaz164/music_genres/blob/master/images/mel_spec_Hip-Hop.png" width="440" height="200"><img src="https://github.com/ddiaz164/music_genres/blob/master/images/mel_spec_Pop.png" width="440" height="200">

This gave me the idea to use a CNN for image classification. But knowing that there was a time component to the mel-spectrograms I wanted to incorporate an RNN since they excel at understanding sequential data.

## Data Preparation
To prepare my feature matrix, I converted the mp3 files from the small dataset (8GB) to mel-spectrograms, converted the array from decibels to power, and then log scaled it. For my target values I simply turned them into a categorical matrix by one hot encoding the targets.

## Convolutional Recurrent Neural Network
![](https://github.com/ddiaz164/music_genres/blob/master/images/CRNN.png)
My first model was a convolutional recurrent neural network that used 1D CNNs and an LSTM. Each 1D convolution layer extracted features from a small slice of the spectrogram, it applied RELU activation, then batch normalization, and lastly 1D Max Pooling to reduce dimensions and prevent over fitting. This chain of operations was performed 3 times and then its output was fed into an LSTM. The output from the LSTM was passed into a Dense Layer and then the final output layer of the model was a dense layer with Softmax activation assigning probability to the 8 classes. 

<img src="https://github.com/ddiaz164/music_genres/blob/master/images/crnn_acc.png" width="440" height="350"><img src="https://github.com/ddiaz164/music_genres/blob/master/images/crnn_loss.png" width="440" height="350">

This model gave me a 52% validation accuracy with its best weights.

## CNN - RNN in Parallel
![](https://github.com/ddiaz164/music_genres/blob/master/images/CNN-RNN.png)

The second model I tried was one that used a CNN and RNN in parallel. The convolutional block consisted of a 2D convolution layer followed by a 2D Maxpooling layer, for a total of 5 blocks of convolution max pooling layers before flattening the final output. The recurrent block started with 2D max pooling to reduce image size before sending it to a bidirectional GRU with 64 units. The two resulting outputs were then concatenated before the final dense layer with Softmax activation. 

<img src="https://github.com/ddiaz164/music_genres/blob/master/images/cnn_rnn_acc.png" width="440" height="350"><img src="https://github.com/ddiaz164/music_genres/blob/master/images/cnn_rnn_loss.png" width="440" height="350">

This model did about the same as the first with a 53% validation accuracy using its best weights. 

## Model Comparison
<img src="https://github.com/ddiaz164/music_genres/blob/master/images/crnn_conf_mat.png" width="440" height="400"><img src="https://github.com/ddiaz164/music_genres/blob/master/images/cnn_rnn_conf_mat.png" width="440" height="400">

The two models performed very similarly as far as overall accuracy but they did differ in class wise performance. The parallel CNN-RNN model had better performance for Electronic, Instrumental, Pop, and Rock.

### Accuracy
All in all the accuracy was not very high, but when looking at top 2 or top 3 accuracy the results look much better with almost 80% for top 3 accuracy. 

![](https://github.com/ddiaz164/music_genres/blob/master/images/pareto_all.png)

This data set is challenging in that some of the genres could contain multiple genres within like say Pop. Additionally since the tracks were 30 second samples, most of what categorized the track as one genre could be left out. There is actually an FMA Genre Recognition challenge and the top leaderboard score only has an accuracy of around 63%.

### Pop Genre
Illustrating some of difficulties with the data, Pop songs could be very versatile and therefore the Pop genre performed the worst out of all the genres. 

![](https://github.com/ddiaz164/music_genres/blob/master/images/pop_pareto.png)

It takes the model a couple of guesses to get to a good accuracy with the top 3 accuracy being 71% all the way up from 21% on the first guess.

### Human vs. Model
To test my model’s ability I decided to use a human subject. Our resident Dylan, who is a trained audio engineer, kindly volunteered to guess on some of the tracks the model got incorrect.

![](https://github.com/ddiaz164/music_genres/blob/master/images/human_vs_model.png)

Dylan and the model agreed on one of the tracks, but still was unable to correctly identify the true genre of the tracks. Having labels that not even a human was able to identify would prove hard to predict for any model.

## Future Steps
Some future steps to take would be to ensemble both of the neural networks since they had different class wise performance. Also using more than just 6400 songs to train on or perhaps even having the model train on the entire song rather than a 30-second excerp might lead to better results.

## References
[Data](https://github.com/mdeff/fma)

[Mel-Spectrogram](https://medium.com/@priya.dwivedi)

[CRNN](https://arxiv.org/pdf/1609.04243.pdf)

[CNN RNN Parallel](https://arxiv.org/pdf/1712.08370.pdf)
