from flask import render_template, request
from exhaustedpigeon import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2

from sleep import sleep_rec
from recommendation import recommendation
from from_full_df import accuracy

#user = 'michaelmunn' #add your username here (same as previous postgreSQL)
#host = 'localhost'
#dbname = 'birth_db'
#db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
#con = None
#con = psycopg2.connect(database = dbname, user = user)

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

    client_id = '227ZHG'
    client_secret = '685a6a8d94213e2f3e5542464be43622'

    redirect_uri = 'http://www.exhaustedpigeon.xyz/output'

    auth = OAuth2Session(client_id, redirect_uri=redirect_uri)
    token = auth.fetch_token(
          access_token_url,
          username=client_id,
          password=client_secret,
          code=url_code)


#    """Step 3: obtains a new access_token from the the refresh token
#    obtained in step 2.
#    the token is internally saved
#    """
#    import requests
#    token = oauth.refresh_token(
#            refresh_token_url,
#            refresh_token=token['refresh_token'],
#                auth=requests.auth.HTTPBasicAuth(client_id, client_secret)
#            )

#    matchedACCESS_TOKEN = token['access_token']
#    matchedREFRESH_TOKEN = token['refresh_token']

#    Client_ID = client_id
#    Client_Secret = '685a6a8d94213e2f3e5542464be43622'

#    """ THEN YOU CAN ACCESS THE API """

#    import fitbit
#    authd_client = fitbit.Fitbit(Client_ID, Client_Secret,
#        access_token = matchedACCESS_TOKEN, refresh_token = matchedREFRESH_TOKEN)

#    values = sleep_rec(authd_client = authd_client)





#@app.route('/thinking')
#def app_thinking():
#    render_template("thinking.html")
#    values = sleep_rec()
#    vals = (values[0],values[1], values[2], int(values[3]),str(int(values[4])).zfill(2),int(values[5]),str(int(values[6])).zfill(2),values[7])
#    vals = (0,1,2,3,4,5,6,7)
#    return render_template("output.html", accuracy = vals[0], margin = vals[1],
#                                opt_num_hours = vals[2], opt_h = vals[3], opt_m = vals[4],
#                                h = vals[5], m = vals[6], avg_twoWksSleep = vals[7])


@app.route('/output')
def app_output():
    values = sleep_rec()
    vals = (round(values[0],1),round(values[1],1), round(values[2],1),  int(values[3]),str(int(values[4])).zfill(2),int(values[5]),str(int(values[6])).zfill(2),round(values[7],1))
#    vals = (0,1,2,3,4,5,6,7)
    return render_template("output.html", accuracy = vals[0], margin = vals[1],
                                opt_num_hours = vals[2], opt_h = vals[3], opt_m = vals[4],
                                h = vals[5], m = vals[6], avg_twoWksSleep = vals[7])

@app.route('/slides')
def app_slides():
    return render_template("slides.html")


@app.route('/about')
def app_about():
    return render_template("about.html")


def __init__(self, client_id, client_secret,
             access_token=None, refresh_token=None,
             *args, **kwargs):
    """
    Create a FitbitOauth2Client object. Specify the first 7 parameters if
    you have them to access user data. Specify just the first 2 parameters
    to start the setup for user authorization (as an example see gather_key_oauth2.py)
        - client_id, client_secret are in the app configuration page
        https://dev.fitbit.com/apps
        - access_token, refresh_token are obtained after the user grants permission
    """

    self.session = requests.Session()
    self.client_id = "227ZHG"
    self.client_secret = "685a6a8d94213e2f3e5542464be43622"
    self.token = {
        'access_token': access_token,
        'refresh_token': refresh_token
    }
    self.oauth = OAuth2Session(client_id)

def _request(self, method, url, **kwargs):
    """
    A simple wrapper around requests.
    """
    return self.session.request(method, url, **kwargs)

def make_request(self, url, data={}, method=None, **kwargs):
    """
    Builds and makes the OAuth2 Request, catches errors

    https://wiki.fitbit.com/display/API/API+Response+Format+And+Errors
    """
    if not method:
        method = 'POST' if data else 'GET'

    try:
        auth = OAuth2(client_id=self.client_id, token=self.token)
        response = self._request(method, url, data=data, auth=auth, **kwargs)
    except (HTTPUnauthorized, TokenExpiredError) as e:
        self.refresh_token()
        auth = OAuth2(client_id=self.client_id, token=self.token)
        response = self._request(method, url, data=data, auth=auth, **kwargs)

    # yet another token expiration check
    # (the above try/except only applies if the expired token was obtained
    # using the current instance of the class this is a a general case)
    if response.status_code == 401:
        d = json.loads(response.content.decode('utf8'))
        try:
            if(d['errors'][0]['errorType'] == 'expired_token' and
                d['errors'][0]['message'].find('Access token expired:') == 0):
                    self.refresh_token()
                    auth = OAuth2(client_id=self.client_id, token=self.token)
                    response = self._request(method, url, data=data, auth=auth, **kwargs)
        except:
            pass

    if response.status_code == 401:
        raise HTTPUnauthorized(response)
    elif response.status_code == 403:
        raise HTTPForbidden(response)
    elif response.status_code == 404:
        raise HTTPNotFound(response)
    elif response.status_code == 409:
        raise HTTPConflict(response)
    elif response.status_code == 429:
        exc = HTTPTooManyRequests(response)
        exc.retry_after_secs = int(response.headers['Retry-After'])
        raise exc

    elif response.status_code >= 500:
        raise HTTPServerError(response)
    elif response.status_code >= 400:
        raise HTTPBadRequest(response)
    return response

def authorize_token_url(self, scope=None, redirect_uri=None, **kwargs):
    """Step 1: Return the URL the user needs to go to in order to grant us
    authorization to look at their data.  Then redirect the user to that
    URL, open their browser to it, or tell them to copy the URL into their
    browser.
        - scope: pemissions that that are being requested [default ask all]
        - redirect_uri: url to which the reponse will posted
                        required only if your app does not have one
        for more info see https://wiki.fitbit.com/display/API/OAuth+2.0
    """

    # the scope parameter is caussing some issues when refreshing tokens
    # so not saving it
    old_scope = self.oauth.scope
    old_redirect = self.oauth.redirect_uri
    if scope:
        self.oauth.scope = scope
    else:
        self.oauth.scope = [
            "activity", "nutrition", "heartrate", "location", "nutrition",
            "profile", "settings", "sleep", "social", "weight"
        ]

    if redirect_uri:
        self.oauth.redirect_uri = redirect_uri

    out = self.oauth.authorization_url(self.authorization_url, **kwargs)
    self.oauth.scope = old_scope
    self.oauth.redirect_uri = old_redirect
    return(out)

def fetch_access_token(self, code, redirect_uri):

    """Step 2: Given the code from fitbit from step 1, call
    fitbit again and returns an access token object. Extract the needed
    information from that and save it to use in future API calls.
    the token is internally saved
    """
    auth = OAuth2Session(self.client_id, redirect_uri=redirect_uri)
    self.token = auth.fetch_token(
        self.access_token_url,
        username=self.client_id,
        password=self.client_secret,
        code=code)

    return self.token

def refresh_token(self):
    """Step 3: obtains a new access_token from the the refresh token
    obtained in step 2.
    the token is internally saved
    """
    self.token = self.oauth.refresh_token(
        self.refresh_token_url,
        refresh_token=self.token['refresh_token'],
        auth=requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
    )

    return self.token
