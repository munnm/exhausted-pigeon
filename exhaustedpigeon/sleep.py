import pandas as pd
import numpy as np

#import matplotlib.pyplot as plt   #useful for plots
#import matplotlib.dates as md     #useful for plots

import datetime, dateutil.parser
import dateutil
import time

import fitbit
import os
import re
from IPython.utils import io
import datetime

import pandas as pd
import numpy as np

import fitbit
import os       #this was needed when using gather_keys on my machine
import re
from IPython.utils import io
import datetime

import statsmodels.api as sm

from patsy import dmatrices
from sklearn import linear_model, datasets
from sklearn import preprocessing
import statsmodels.api as sm

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

from sklearn import svm, grid_search

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV

from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

from sklearn.cross_validation import KFold


def sleep_rec(user):
    Client_ID = '****'
    Client_Secret = '****'
    
    ################################################################################################################
    #This code was used when connecting to the API on my machine. Made irrelevant when accessing Fitbit from AWS
    ################################################################################################################
    #captured_output = os.popen("python ~/Dropbox/DS/InsightHealthDS/fitbit/python-fitbit/gather_keys_oauth2.py 227ZHG 685a6a8d94213e2f3e5542464be43622").read()
    #gather_keys_output = str(captured_output)

    #find the ACCESS_TOKEN from the output above and make it a string
    #matchedACCESS_TOKEN = re.findall(r"ACCESS_TOKEN\s=\s(.+?)\n", gather_keys_output)
    #matchedACCESS_TOKEN = ''.join(matchedACCESS_TOKEN)
    #find the REFRESH_TOKEN from the output above and make it a string
    #matchedREFRESH_TOKEN = re.findall(r"REFRESH_TOKEN\s=\s(.+?)\n", gather_keys_output)
    #matchedREFRESH_TOKEN = ''.join(matchedREFRESH_TOKEN)

    #set up the authenticated user data
    #authd_client = fitbit.Fitbit(Client_ID, Client_Secret, access_token = matchedACCESS_TOKEN, refresh_token = matchedREFRESH_TOKEN)

    #collect user's data for the last 75 days
    ################################################################################################################
    #---------THIS ACCESSES THE FITBIT API--------------
    ################################################################################################################
    #start with an empty list
    user_sleep_data = []
    user_activities_data = []

    #create the date list
    numdays = 75
    base = datetime.datetime.today()
    date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]

    #collect sleep data
    for i in range(numdays):
        user_sleep_data.append(authd_client.sleep(date = date_list[i]))

    #collect activities data
    user_activities_data = []
    for i in range(0, numdays):
        user_activities_data.append(authd_client.activities(date = date_list[i]))

    #now clean the data
    #start with the sleep info
    user_sleep_startTime = []
    user_sleep_minutesAsleep = []
    user_sleep_awakeningsCount = []
    user_sleep_restlessCount = []
    user_sleep_restlessDuration = []

    for i in range(0,numdays):
        if not not user_sleep_data[i]['sleep']:
            user_sleep_startTime.append(user_sleep_data[i]['sleep'][0]['startTime'])
        else: user_sleep_startTime.append(np.NaN)
        if len(user_sleep_data[i]['sleep']) > 0:
            user_sleep_minutesAsleep.append(user_sleep_data[i]['sleep'][0]['minutesAsleep'])
        else: user_sleep_minutesAsleep.append(np.NaN)
        if len(user_sleep_data[i]['sleep']) > 0:
            user_sleep_awakeningsCount.append(user_sleep_data[i]['sleep'][0]['awakeningsCount'])
        else: user_sleep_awakeningsCount.append(np.NaN)
        if len(user_sleep_data[i]['sleep']) > 0:
            user_sleep_restlessCount.append(user_sleep_data[i]['sleep'][0]['restlessCount'])
        else: user_sleep_restlessCount.append(np.NaN)
        if len(user_sleep_data[i]['sleep']) > 0:
            user_sleep_restlessDuration.append(user_sleep_data[i]['sleep'][0]['restlessDuration'])
        else: user_sleep_restlessDuration.append(np.NaN)

    #make date just year-month-day
    user_sleep_date = []
    for i in range(numdays):
        user_sleep_date.append(date_list[i].date())


    #add a column of just the startTimes for sleep. Recorded in minute of the day
    #we'll want it to be in minute of the day so define
    def get_min(time_str):
        h, m = time_str.split(':')
        return float(h) * 60 + float(m)

    user_sleep_minuteStartTime = []
    for i in range(numdays):
        if not pd.isnull(user_sleep_startTime[i]):
            startTime = dateutil.parser.parse(user_sleep_startTime[i])
            hour_min = startTime.strftime('%H:%M')
            minuteStartTime = get_min(hour_min)
            user_sleep_minuteStartTime.append(minuteStartTime)
            user_sleep_minuteStartTime
        else: user_sleep_minuteStartTime.append(np.nan)

    #add day of the week to dataframe so we can focus on weekdays vs weekends
    user_sleep_weekday = []
    for i in range(numdays):
          user_sleep_weekday.append(user_sleep_date[i].weekday())

    #now create the sleep dataframe
    sleep_df = pd.DataFrame({'date' : user_sleep_date,
                        'weekday' : user_sleep_weekday,
                        'sleepStartTime' : user_sleep_startTime,
                        'minuteStartTime' : user_sleep_minuteStartTime,
                        'minutesAsleep' : user_sleep_minutesAsleep,
                        'awakeningsCount' : user_sleep_awakeningsCount,
                        'restlessCount' : user_sleep_restlessCount,
                        'restlessDuration' : user_sleep_restlessDuration})

    #set the date to index
    sleep_df = sleep_df.set_index(['date'])

    user_activities_restingHeartRate = []
    user_activities_steps = []
    user_activities_floors = []
    user_activities_activityCalories = []
    user_activities_caloriesOut = []
    user_activities_sedentaryMinutes = []
    user_activities_fairlyActiveMinutes = []
    user_activities_lightlyActiveMinutes = []
    user_activities_veryActiveMinutes = []


    for i in range(numdays):
        if 'restingHeartRate' in user_activities_data[i]['summary']:
            user_activities_restingHeartRate.append(user_activities_data[i]['summary']['restingHeartRate'])
        else: user_activities_restingHeartRate.append(np.NaN)
        if 'steps' in user_activities_data[i]['summary']:
            user_activities_steps.append(user_activities_data[i]['summary']['steps'])
        else: user_activities_steps.append(np.NaN)
        if 'floors' in user_activities_data[i]['summary']:
            user_activities_floors.append(user_activities_data[i]['summary']['floors'])
        else: user_activities_floors.append(np.NaN)
        if 'activityCalories' in user_activities_data[i]['summary']:
            user_activities_activityCalories.append(user_activities_data[i]['summary']['activityCalories'])
        else: user_activities_activityCalories.append(np.NaN)
        if 'caloriesOut' in user_activities_data[i]['summary']:
            user_activities_caloriesOut.append(user_activities_data[i]['summary']['caloriesOut'])
        else: user_activities_caloriesOut.append(np.NaN)
        if 'sedentaryMinutes' in user_activities_data[i]['summary']:
            user_activities_sedentaryMinutes.append(user_activities_data[i]['summary']['sedentaryMinutes'])
        else: user_activities_sedentaryMinutes.append(np.NaN)
        if 'lightlyActiveMinutes' in user_activities_data[i]['summary']:
            user_activities_lightlyActiveMinutes.append(user_activities_data[i]['summary']['lightlyActiveMinutes'])
        else: user_activities_sedentaryMinutes.append(np.NaN)
        if 'fairlyActiveMinutes' in user_activities_data[i]['summary']:
            user_activities_fairlyActiveMinutes.append(user_activities_data[i]['summary']['fairlyActiveMinutes'])
        else: user_activities_fairlyActiveMinutes.append(np.NaN)
        if 'veryActiveMinutes' in user_activities_data[i]['summary']:
            user_activities_veryActiveMinutes.append(user_activities_data[i]['summary']['veryActiveMinutes'])
        else: user_activities_veryActiveMinutes.append(np.NaN)


    #make date just year-month-day
    user_activities_date = []
    for i in range(numdays):
        user_activities_date.append(date_list[i].date())

    activities_df = pd.DataFrame({'date' : user_activities_date,
                                'restingHeartRate': user_activities_restingHeartRate,
                               'steps': user_activities_steps,
                                  'floors': user_activities_floors,
                                 'activityCalories' : user_activities_activityCalories,
                                  'caloriesOut' : user_activities_caloriesOut,
                                  'sedentaryMinutes' : user_activities_sedentaryMinutes,
                                  'lightlyActiveMinutes' : user_activities_lightlyActiveMinutes,
                                  'fairlyActiveMinutes' : user_activities_fairlyActiveMinutes,
                                  'veryActiveMinutes' : user_activities_veryActiveMinutes
                                 })

    activities_df = activities_df.set_index(['date'])


    ################################################################################################################
    #################         join sleep_df and activities_df to get full_df of measurements
    ################################################################################################################

    full_df = pd.concat([sleep_df, activities_df], axis=1)

    #and rearrange so weekdays is in the front, then sleep data, then activity data
    cols = ['weekday',
     'sleepStartTime',
     'minuteStartTime',
     'minutesAsleep',
     'awakeningsCount',
     'restlessCount',
     'restlessDuration',
     'steps',
     'floors',
     'activityCalories',
     'caloriesOut',
     'fairlyActiveMinutes',
     'lightlyActiveMinutes',
     'veryActiveMinutes']
    full_df = full_df[cols]


    df = full_df
    df = df[['weekday', 'minuteStartTime', 'minutesAsleep', 'steps', 'caloriesOut', 'activityCalories']]

    #can either drop NA values
    df = df.dropna()
    #or you can average them with their neighbors
    #full_df = full_df.fillna(full_df.mean())

    ######################################## caloriesOut as step_goal ########################################
    #add in a step goal
#    step_goal = 1.2*df.caloriesOut.mean()
#
#    user_step_goal = []
#    for i in range(len(df)):
#        if df.caloriesOut[i] > step_goal: user_step_goal.append(1)
#        else: user_step_goal.append(0)
#    df['step_goal'] = user_step_goal
    ######################################## caloriesOut as step_goal ########################################

   #add in a step goal
    step_goal = 1.2*df.steps.mean()

    user_step_goal = []
    for i in range(len(df)):
        if df.steps[i] > step_goal: user_step_goal.append(1)
        else: user_step_goal.append(0)
    df['step_goal'] = user_step_goal


    #add higher order features
    df['StartTime2'] = df['minuteStartTime']**2
    df['minutesAsleep2'] = df['minutesAsleep']**2
    df['StartTimeMinutesAsleep'] = df['minutesAsleep'] * df['minuteStartTime']


    ################################################################################################################
    ### One-hot encoding for weekday
    ################################################################################################################
    weekday_coded = pd.get_dummies(df.weekday)
    df = pd.concat([weekday_coded, df], axis = 1)

    df['weekend'] = df[5] + df[6]

    ################################################################################################################
    ### just look at weekdays
    ################################################################################################################
    df = df.loc[df['weekend']!=1]

    ################################################################################################################
    ### Normalize features for logistic regression
    ################################################################################################################
    scaler = preprocessing.StandardScaler()

    scaled_df = df
    scaled_df['minuteStartTime'] = scaler.fit_transform(scaled_df[['minuteStartTime']])
    scaled_df['activityCalories'] = scaler.fit_transform(scaled_df[['activityCalories']])
    scaled_df['minutesAsleep'] = scaler.fit_transform(scaled_df[['minutesAsleep']])
    scaled_df['caloriesOut'] = scaler.fit_transform(scaled_df[['caloriesOut']])
    scaled_df['steps'] = scaler.fit_transform(scaled_df[['steps']])

    scaled_df['minutesAsleep2'] = scaler.fit_transform(scaled_df[['minutesAsleep2']])
    scaled_df['StartTime2'] = scaler.fit_transform(scaled_df[['StartTime2']])
    scaled_df['StartTimeMinutesAsleep'] = scaler.fit_transform(scaled_df[['StartTimeMinutesAsleep']])

    ################################################################################################################
    #### include yesterday's sleepTime
    ################################################################################################################
    #yesterday's sleepTime
    yesterday_StartTime = []
    for i in range(len(df)-1):
        yesterday_StartTime.append(scaled_df['minuteStartTime'][i+1])
    yesterday_StartTime.append(scaled_df['minuteStartTime'][len(df)-1])
    scaled_df['yestday_StartTime'] = yesterday_StartTime

    #yesterday's minutesAsleep
    yesterday_minutesAsleep = []
    for i in range(len(df)-1):
        yesterday_minutesAsleep.append(scaled_df['minutesAsleep'][i+1])
    yesterday_minutesAsleep.append(scaled_df['minutesAsleep'][len(df)-1])
    scaled_df['yestday_minutesAsleep'] = yesterday_minutesAsleep

    X = scaled_df[['minuteStartTime', 'minutesAsleep',
                   'StartTime2', 'minutesAsleep2', 'StartTimeMinutesAsleep',
                   'yestday_minutesAsleep', 'yestday_StartTime']]

    y = scaled_df[['steps']]
    y = np.ravel(y)

    scaled_df.to_csv("full_scaled_df.csv")

    lin_reg = LinearRegression()
    selector = RFE(lin_reg, 5, step = 1)
    selector.fit(X,y)

    ################################################################################################################
    ###   Do grid search to determine optimal parameter alpha in Ridge Regression
    ################################################################################################################

    '''
    # prepare a range of alpha values to test
    parameters = {'alpha':[0.001, 0.01, 0.05, 0.1, 1.1, 1.2, 1.3, 1.5, 5, 10, 100, 1000]}
    # create and fit a ridge regression model, testing each alpha
    model = Ridge()
    grid = GridSearchCV(estimator=model, param_grid=parameters)
    grid.fit(X, y)
    #print(grid)
    # summarize the results of the grid search
    #print(grid.best_score_)
    grid_alpha = grid.best_estimator_.alpha
    '''

    ################################################################################################################
    ###   Use Reverse Feature Elimination to determine most relevant features
    ################################################################################################################
    # use linear regression as the model
    lin_reg = LinearRegression()
    #rank all features, i.e continue the elimination until the last one
    selector = RFE(lin_reg, 5, step = 1)
    #selector = RFECV(lin_reg, step = 1, cv = 10)
    selector.fit(X,y)

    names = X.columns

    features = names[selector.support_]

    ################################################################################################################
    ## Now compute the logistic regression using these features
    ################################################################################################################
    X = X[features]   ## Now make X contain just the most relevant features

    y = scaled_df[['step_goal']]
    y = np.ravel(y)

    model_log = LogisticRegression()
    
    ################################################################################################################
    ## use grid.search to find the best constant for regularization
    ################################################################################################################
    c_range = [.001, .01, .1, 1, 10, 100]
    dict(C=c_range)

    model_log = grid_search.GridSearchCV(estimator=model_log, param_grid=dict(C=c_range))
    model_log.fit(X,y)
    best_c = model_log.best_params_['C']

    best_model_log = LogisticRegression(C = best_c)

    ################################################################################################################
    ## Now compute the K-fold CV to determine accuracy
    ################################################################################################################
    kf_total = KFold(len(X), n_folds=5, shuffle=True, random_state=4)

    scores = cross_val_score(best_model_log, X, y, cv=kf_total, n_jobs = 1)

    accuracy = round(scores.mean()*100, 2)
    margin = round(1.96*scores.std()*10**(0.5),2)

    ################################################################################################################
    # find the optimal sleep duration for the sleep recommendation
    ################################################################################################################
    df = full_df
    df = df[['weekday', 'minuteStartTime', 'minutesAsleep', 'steps', 'caloriesOut', 'activityCalories']]


    #can either drop NA values
    df = df.dropna()
    #or you can average them with their neighbors
    #full_df = full_df.fillna(full_df.mean())


    #add in a step goal
    step_goal = 1.2*df.steps.mean()

    user_step_goal = []
    for i in range(len(df)):
        if df.steps[i] > step_goal: user_step_goal.append(1)
        else: user_step_goal.append(0)
    df['step_goal'] = user_step_goal

    good_df = df.loc[df['step_goal'] == 1]
    good_df = good_df.loc[good_df['minutesAsleep'] > 360]

    good_df.minutesAsleep.mean()
    opt_num_hours = round(good_df.minutesAsleep.mean()/60, 2)

    ################################################################################################################
    # find the optimal sleep time for the sleep recommendation
    ################################################################################################################
    good_df = good_df.loc[good_df['minuteStartTime'] > 480]
    opt_minuteStartTime = good_df.minuteStartTime.mean()

    opt_seconds = opt_minuteStartTime*60

    opt_m, opt_s = divmod(opt_seconds, 60)
    opt_h, opt_m = divmod(opt_m, 60)


    ################################################################################################################
    # Conclusion
    ################################################################################################################
    #this print statement isnt needed anymore. It is passed to the result.html page
    #print "\n\nWith %r%% accuracy, you should get at least %r hours of sleep a night" % (accuracy, opt_num_hours)
    #print "\t\t\t and you do best when you're in bed by %d:%02d:%02d." % (opt_h, opt_m, opt_s)

    ################################################################################################################
    # compare with past two weeks
    ################################################################################################################
    # collect last two weeks data
    two_wks_df = df[0:15]
    avg_twoWksSleep = round(two_wks_df.minutesAsleep.mean()/60,2)

    two_wks_df = two_wks_df[two_wks_df['minuteStartTime'] > 480]
    num_minuteStartTime = two_wks_df.minuteStartTime.mean()

    seconds = num_minuteStartTime*60

    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    
    #this print statement isnt needed anymore. It is passed to the result.html page
    #print "Over the last two weeks, you generally got about %r hours of sleep each night" % (avg_twoWksSleep)
    #print "\t and you were in bed by %d:%02d:%02d.\n" % (h,m,s)

    return accuracy, margin, opt_num_hours, opt_h, opt_m, h, m, avg_twoWksSleep
