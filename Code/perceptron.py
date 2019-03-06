import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
import json
import ast
import sys
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# X, y = load_digits(return_X_y=True)
# clf = Perceptron(tol=1e-3, random_state=0)
# clf.fit(X, y)
#
# clf.score(X, y)


def gather_QB_results(players , year):
    # print(year)
    data = open("point-results-"+year+".txt").read()
    data = ast.literal_eval(data)
    output = []
    for player in players:
        for points in data:
            # print(player)
            if player[0] in points:
                output.append([player[0] , points[player[0]]])

    # print(output)
    return output


def prep_data(stats):
    # The standard order each week of stats will be arranged
    structure = ["Passing Attempts","Passing Completions","Incomplete Passes","Passing Yards","Passing Touchdowns","Interceptions Thrown","Rushing Attempts","Rushing Yards","Rushing Touchdowns"]
    # Add name to output file
    output = [stats[0]]

    # Iterate through stats first so they are all in the same order
    for stat in structure:
        # Convert the 2 item list of [playername , {dictionary of stats}]
        # Into an iterable and skip the player name
        iterstats = iter(stats)
        next(iterstats)
        # Add each of the
        for week in iterstats:
            if stat in week:
                output.append(str(week[stat]))
            else:
                output.append(0)
    return output

def gather_QB_data(year):
    # Open the week 16 data file of specified year and convert to JSON
    # Since we are using week 16 as test data
    # We will only gather QB data from the other weeks if we have data from week 16
    data = open("./data/"+year+"-16.txt").read()
    data = ast.literal_eval(data)

    # Add each QB's name to a list
    players = []
    for player in data:
        if player["position"] == "QB":
            players.append([player["name"]])

    # Iterate over each week and open that weeks data
    for week in range(1,16):
        # Open the file associated with the week and year and convert to JSON
        data = open("./data/"+year+"-"+str(week)+".txt").read()
        data = ast.literal_eval(data)

        # Check each players data in the weekly data
        # If they are in the QB list append their data to that QB list
        for player_weekly in data:
            for player in players:
                if player[0] == player_weekly["name"]:
                    player.append(player_weekly)

    # Only accept QB data from people who played every week
    players = list(filter(lambda x: len(x) > 14 , players))

    # Convert the player from a name and a dictionary to in a list to just a list of data
    prepared_data = list(map(lambda x: prep_data(x) , players))

    return prepared_data


def test_2017():

    clf = Perceptron(n_iter=40 , tol=1e-3, random_state=0)

    # Iterate through each year of data
    for i in range(2010,2017):
        # Gather the data for each player in a list, The first value of each sub list will be the players name
        # each players data will be in a sublist orginized in a format specified in the function
        QB_data = gather_QB_data(str(i))

        # Gather all the values to be tested agianst
        # in the same order as the players data given by as an argument
        QB_results = gather_QB_results(QB_data , str(i))


        QB_data = list(map(lambda x: x[1:] , QB_data))
        QB_data = list(map(lambda x: list(map(lambda x: int(x) , x)) , QB_data))

        QB_results = list(map(lambda x: str(round(x[1])) , QB_results))

        # sc = StandardScaler()
        # sc.fit(QB_data)

        # QB_data = sc.transform(QB_data)

        clf.fit(QB_data, QB_results)

    QB_data = gather_QB_data("2017")
    # print(len(QB_data))
    QB_results = gather_QB_results(QB_data , "2017")
    # print(len(QB_results))

    QB_data = list(map(lambda x: x[1:] , QB_data))
    QB_data = list(map(lambda x: list(map(lambda x: int(x) , x)) , QB_data))
    QB_results = list(map(lambda x: str(round(x[1])) , QB_results))

    predictions = clf.predict(QB_data)

    y_pred = list(map(lambda x: int(x) , predictions))
    y_test = list(map(lambda x: int(x) , QB_results))

    return "mean squared error testing on 2017 data: %.2f" % (mean_squared_error(y_test , y_pred)/int(len(y_test)))

    # print('Accuracy for Week 2017 baised of 2010-2016 data: %.2f' % accuracy_score(QB_results , predictions))



def test_random():

    train_data = []
    data_label = []
    for i in range(2010,2018):
        QB_data = gather_QB_data(str(i))
        # print(len(QB_data))
        QB_results = gather_QB_results(QB_data , str(i))
        # print(len(QB_results))

        QB_data = list(map(lambda x: x[1:] , QB_data))

        QB_data = list(map(lambda x: list(map(lambda x: int(x) , x)) , QB_data))
        QB_results = list(map(lambda x: str(round(x[1])) , QB_results))
        for val in QB_data:
            train_data.append(val)
        for val in QB_results:
            data_label.append(val)

    x_train, x_test, y_train, y_test = train_test_split(train_data, data_label, test_size=0.3)

    sc = StandardScaler()
    sc.fit(x_train)

    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)

    # clf = Perceptron(n_iter=60, eta0=0.1, random_state=0)
    clf = Perceptron(n_iter=40 , tol=1e-3, random_state=0)

    clf.fit(x_train, y_train)

    #Apply the trained perceptron on the X data to make predicts for the y test data
    y_pred = clf.predict(x_test)

    # print('Accuracy of random train/test split: %.2f' % accuracy_score(y_test, y_pred))

    y_pred = list(map(lambda x: int(x) , y_pred))
    y_test = list(map(lambda x: int(x) , y_test))

    return "average mean squared error of random train/test split: %.2f" % (mean_squared_error(y_test , y_pred)/int(len(y_test)))


def main():
    # Training Perceptron on all the quarterback data from 2010-2016
    # Testing on the QB data from 2017
    mean_squared_error_2017 = test_2017()

    # Training Perceptron on a random 70% of the data availible
    # Testing Perceptron on the remaining 30% of data availible
    mean_squared_error_random = test_random()

    print("\n\nRESULTS:")
    print(mean_squared_error_2017)
    print(mean_squared_error_random)
main()
