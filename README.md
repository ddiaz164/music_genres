![](https://uidesign.gbtcdn.com/gb_blog/2776/What-is-audio-format-and-the-main-types-of-it-Z01.jpg)
# Audio Classification: Predicting Song Genres
Audio data is quickly becoming an important part of machine learning. From using audio to recommend songs for radio channels, all the way to interacting with virtual assistants like Alexa or Siri and helping self-driving cars hear their surroundings rather than just seeing them. I wanted to explore audio files and see if I could find patterns within the data that I could use in order to build a model.
## Data Source
The data I used for this project was the free music archive. It contains up to 106,574 tracks with 161 unbalanced genres, but in the interest of time I decided to work with the small dataset which only has 8 genres and 8,000 30-second tracks evenly distributed for each genre. The genres in this set were Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, and Rock.
## Principal Component Analysis
When first looking at the data I did some PCA down to 2 dimensions in order to plot and get a sense of how these genres would compare to each other. 
![](https://github.com/ddiaz164/music_genres/blob/master/images/pca_ins_hip.png)
![](https://github.com/ddiaz164/music_genres/blob/master/images/pca_rock_pop.png)

With Instrumental and Hip-Hop I could see the separation, but with Rock and Pop it would prove a lot tougher to predict one way or the other.

