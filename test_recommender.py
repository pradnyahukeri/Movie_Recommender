"""we will write program to test our program is working"""

from utils import movie_1
from recommenders import recommend_cosin
import requests
import pickle

my_movie_list = pickle.load(open('my_movie_list.pkl', 'rb'))


#check whether the result is a list of strings
def test_movie_as_str():
    query={'Zootopia (2016)':3,
    'Richie Rich (1994)':4,
    'Lion King, The (1994)':4,
    'Beauty and the Beast (1991)':4,
    'Aladdin (1992)':4
    }
    top3 =recommend_cosin(query,k=3)
    for movie in top3:
        assert type(movie) == str

#check whether the size of the output changes with n
def test_recomm_result_size():
    query={'Zootopia (2016)':3,
    'Richie Rich (1994)':4,
    'Lion King, The (1994)':4,
    'Beauty and the Beast (1991)':4,
    'Aladdin (1992)':4
    }
    top3 =recommend_cosin(query,k=3)
    assert len(top3) ==3

#check whether the first result is really a valid movie name
def test_check_movie_in_movielist():
    query={'Zootopia (2016)':3,
    'Richie Rich (1994)':4,
    'Lion King, The (1994)':4,
    'Beauty and the Beast (1991)':4,
    'Aladdin (1992)':4
    }
    top3 =recommend_cosin(query,k=3)
    assert top3[0] in (my_movie_list)


def test_webpp():
    page=requests.get('http://127.0.0.1:5000/')
    assert page.status_code==200