
from flask import Flask,render_template,request
from recommenders import recommend_cosin


app=Flask(__name__)

@app.route("/")
def landing_page():
    
    return render_template("landing_page.html")


@app.route("/recommendation")
def recommendation_page():
   user_query=request.args.to_dict()
   user_query={movie:float(rate) for movie, rate in user_query.items()}
   print(user_query)
   """query= {
           'Zootopia (2016)': 5,
            'Richie Rich (1994)': 4,
            'Lion King, The (1994)': 5,
            'Beauty and the Beast (1991)': 5,
            'Aladdin (1992)': 4
        }"""
   top4=recommend_cosin(user_query,4)
   return render_template("recommendation.html",movie_list=top4)

if __name__ == "__main__":
    app.run(debug=True)