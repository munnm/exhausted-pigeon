from flask import render_template, request
from exhaustedpigeon import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2

from sleep import sleep_rec
from recommendation import recommendation
from from_full_df import accuracy

@app.route('/index')
def app_index():
    return render_template("index.html")
@app.route('/')
def app_front():
    from requests_oauthlib import OAuth2, OAuth2Session
    AUTHORIZE_ENDPOINT = "https://www.fitbit.com"
    authorization_url = "%s/oauth2/authorize" % AUTHORIZE_ENDPOINT
    client_id = '227ZHG'
    redirect_uri = 'http://www.exhaustedpigeon.xyz/code/'
    oauth = OAuth2Session(client_id)
    oauth.redirect_uri = redirect_uri
    oauth.scope = [
                    "activity", "nutrition", "heartrate", "location", "nutrition",
                    "profile", "settings", "sleep", "social", "weight"
                ]

    url,_ = oauth.authorization_url(authorization_url)
    return render_template("input.html", step_one_url = url)


@app.route('/code')
def app_input():
    url_code = request.args.get('code')

#    """Step 2: Given the code from fitbit from step 1, call
#    fitbit again and returns an access token object. Extract the needed
#    information from that and save it to use in future API calls.
#    the token is internally saved
#    """
    AUTHORIZE_ENDPOINT = "https://www.fitbit.com"
    API_ENDPOINT = "https://api.fitbit.com"

    authorization_url = "%s/oauth2/authorize" % AUTHORIZE_ENDPOINT
    request_token_url = "%s/oauth2/token" % API_ENDPOINT

    refresh_token_url = request_token_url
    access_token_url = request_token_url

    client_id = '********'                      # This is your client_id for the fitbit app
    client_secret = '********'                  # This is your client_secret for the fitbit app

    redirect_uri = 'http://www.exhaustedpigeon.xyz/output'

    auth = OAuth2Session(client_id, redirect_uri=redirect_uri)
    token = auth.fetch_token(
          access_token_url,
          username=client_id,
          password=client_secret,
          code=url_code)

@app.route('/output')
def app_output():
    values = sleep_rec()
    vals = (round(values[0],1),round(values[1],1), round(values[2],1),  int(values[3]),str(int(values[4])).zfill(2),int(values[5]),str(int(values[6])).zfill(2),round(values[7],1))
    return render_template("output.html", accuracy = vals[0], margin = vals[1],
                                opt_num_hours = vals[2], opt_h = vals[3], opt_m = vals[4],
                                h = vals[5], m = vals[6], avg_twoWksSleep = vals[7])

@app.route('/slides')
def app_slides():
    return render_template("slides.html")


@app.route('/about')
def app_about():
    return render_template("about.html")

