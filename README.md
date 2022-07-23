# CoMRec
A Contextual Bandit Music Recommender system for a Virtual Spotify User

The objective of this project are two-fold:
1. (User.py) To use the Spotify dataset available on Kaggle for (Song features) https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset?resource=download and (playlists) https://www.kaggle.com/datasets/andrewmvd/spotify-playlists/code  to create a 'virtual user'. A virtual user will mimic the real user, with a combination of static and dynamic features.
    Static features
    1. Gender
    2. Age
    3. Race
    4. Ethinicity
    5. Country
    6. State 
    7. Languages

    Dynamic Features
    1. Affinity towards a Genre
    2. Dynamic playlists
    3. Features of last $n$ played songs
    4. Time_of_the_day
2. (ContextualBandit.py) To build a recommender system using a contextual Bandit algorithm using a library called VowpalWabbit https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_Contextual_bandits_and_Vowpal_Wabbit.html


