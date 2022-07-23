#Libraries
import os
import numpy as np
import pandas as pd
import random
'''
## Spotify Virtual User Environment
The objective of this class is to use the Spotify dataset available on Kaggle for (Song features) https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset?resource=download and (playlists) https://www.kaggle.com/datasets/andrewmvd/spotify-playlists/code  to create a 'virtual user'. A virtual user will mimic the real user, with a combination of static and dynamic features.
Static features
1. Gender
2. Age
3. Race
4. Ethinicity
5. Country
6. State 
7. Languages
* The above static features are just an example, the spotify playlist dataset, we are using doesn't have such static features 

Dynamic Features
1. Affinity towards a Genre
2. Dynamic playlists
3. Features of last $n$ played songs 
4. Time_of_the_day 
* We only have (3) available as the dynamic feature from the song features dataset. The other can be implemented in a similar fashion.
'''

class User:
    """
    Parameters
    ----------
    PL           : (dataframe) User playlist
    Songs_data   : (dataframe) Song Feature data
    seed         : (int) seed to reproduce the experiment
    n_songs      : (int) a positive number > 0 for minimum number of song in a plalist for a 
                    user and also for last 'n_songs' played by the user criterion for generating rewards.
    rate         : (float) rate parameter of the exponential distribution (1/scale)
    imp_features : (list of strings) having the features of the song data to be used for reward computation 
                    (i.e the context)
        
    Attributes
    ----------

    """
    
    def __init__(self,PL, Songs_data, seed, n_songs, threshold,rate,imp_features):
        self.PL =PL
        self.n_songs=n_songs
        print(self.PL.info())
        
        self.Songs_data=Songs_data
        print('\n== Filtering User Playlist===\n')
        
        self.Filtered_PL= self.Check_PL_song_D()
        print(self.Filtered_PL.info())
        
        self.playlist, self.User_dict= self.flattening_by_playlist()
        self.imp_features=imp_features
        self.threshold=threshold
        np.random.seed(seed)
        self.rate=rate
    
    
    def Check_PL_song_D(self):
        '''
        Checks if all the songs in the User playlist in the Song features data set and return
            filtered User_playlist with the one which are in the playlist Data
        '''
        s=0
        print('Checking for Songs in the playlists \n which are not in the Spotify Song list\n')
        for name in self.PL['trackname'].unique():
            temp=self.Songs_data['name']==name
            if sum(temp)==0:
                print(name,sum(temp))
                s+=1
                self.PL=self.PL.drop(self.PL.index[self.PL['trackname']==name]) 
        if s==0:
            print('Kudos! All songs are there in the Spotify Song list!')
        else:
            print(str(s)+' songs are not in the list with above names')
        return(self.PL)

    def flattening_by_playlist(self):
        '''
        Returns:
        1. playlist:  Updated playlist
        2. flattened_dict: Seperated User Playlist with number of songs greater than n_songs into 
                            seperate element of the dictionary with keys as playlist
        '''
        playlist=self.Filtered_PL['playlistname'].unique()
        flattened_dict=dict.fromkeys(playlist)

        popkeys=[]
        for key in flattened_dict.keys():
            flattened_dict[key]=self.Filtered_PL[['artistname','trackname']][self.Filtered_PL['playlistname']==key].reset_index()
            if len(flattened_dict[key])<=self.n_songs:
                popkeys.append(key)

        for key in popkeys:
            del flattened_dict[key]
            
        playlist=list(flattened_dict.keys())
        
        return playlist, flattened_dict
    
    def get_context(self,rn,pl=None):
        '''
        Randomly selects a playlist (if pl is none) and then randomly selects n_songs song from that playlist
        Input
        -----
        rn       :(Random) Random object to transfer seed for replication
        pl       :(str) playlist name if context is to generated from a fixed playlist otherwise default is none.
        
        Output
        -------
        context  :(str list) names of last `n_songs' songs played by the User (randomly selected)
        '''
        if pl is None:
            pl= self.playlist[rn.randint(0,len(self.playlist)-1)] ## randomly selecting a playlist
        context=rn.sample(self.User_dict[pl]['trackname'].to_list(),self.n_songs) 
        return context


    def get_reward(self,context,rec_song_name):
        '''
        Input  
        --------
        context      : (string array) names of last `n_songs songs' played by the User
                        Note: The actual context are song features but we only access these features by song names
        rec_song_name: (string) recommended song name (action)
                    
        Output   : Implement the following reward generation procedure:
        
            1. Randomly (uniformly) select a playlist and then uniformly select n_songs 
                (assuming there are at least n_songs song in that playlist) songs ($S_t$) from that playlist as the 
                last 5 songs played by the user.
            2. For each song, collect the features ($\{X_t^i\}_{i\in S_t}$) from the 'song 
                description dataset'.  
            3. To compute reward for a given song recommendation with feature $B_t$
                    a. compute Average_distance= $\frac{1}{5}\sum_{i\in S_t}\|X_t^i-B_t\|$ 
                        between features of the recommended song and the feature of 5 song selected above.
                    b. If  (Average_distance <= threshold)
                            T=1
                            ## Then the user listen to the recommended song for time T~exp(rate) 
                                (rate is the rate of the exponential distribution) ## For continous reward
                       Else T=0
                    c. T is the reward observed
        
        Output
        ------
        T.            : (int) reward (binary)
        '''
        X=[]
        for i in range(self.n_songs):
            X.append(self.Songs_data[self.Songs_data['name']==context[i]][self.imp_features].mean().to_numpy())
        
        B = self.Songs_data[self.Songs_data['name']==rec_song_name][self.imp_features].mean().to_numpy()

        Avg_dist = np.mean([np.linalg.norm(B-x) for x in X])

        if Avg_dist <= self.threshold:
            return(1)
            #return(np.random.exponential(1/self.rate))
        else:
            return(0)

