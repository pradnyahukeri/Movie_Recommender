import pickle


movies={
        # movieId, rating
    4470:5, 
    48:5,
    594:5,
    27619:5,
    152081:5,
    595:5,
    616:5,
    1029:5
}

movie_1 = {
    'Zootopia (2016)',
    'Richie Rich (1994)',
    'Lion King, The (1994)',
    'Beauty and the Beast (1991)',
    'Aladdin (1992)',
    '1992'
  }

#loading Nmf model
with open('../nmf_recommender.pkl', 'rb') as file:  # This "with open" mimic is a so-called "context manager".
    nmf_model = pickle.load(file)


