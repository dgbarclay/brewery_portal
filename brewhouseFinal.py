import csv
import pandas as pd
import threading
import json
import logging
import sched
import time
import requests
import pyttsx3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template
from flask import request
from datetime import datetime

APP = Flask(__name__)
# create logging file.
logging.basicConfig(filename='brewhouse.log', level=logging.DEBUG)
# declare global lists and dictionaries.
SALES_DICTIONARY = {}
QUANTITY_REQUIRED = []
DATE_REQUIRED = []
DUNKEL_SALES = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
PILSNER_SALES = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
HELLES_SALES = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
NOTIFICATION = []
ACTIVE_BREWING = []
# declare global variables.
PILSNER_RESERVE = 0
HELLES_RESERVE = 0
DUNKEL_RESERVE = 0
PREDICTION_UPPER = 0
PREDICTION_LOWER = 0
PREDICTION = 0
# declare global lists and populate with data.
MONTH = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
FUTURE_MONTHS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
MONTH_STRING = ['N', 'D', 'J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O']
# declare dictionaries containing Tank information.
BREWHOUSE_TANKS = {
'Albert' : {'Name' : 'Albert', 'Volume' : 1000, 'Capability': 'Fermenter/conditioner'\
, 'Bottles' : 3030, 'Active' : False, 'Beer' : ''},
'Emily' : {'Name' : 'Emily', 'Volume' : 1000, 'Capability': 'Fermenter/conditioner'\
, 'Bottles' : 3030, 'Active' : False, 'Beer' : ''},
'Camilla' : {'Name' : 'Camilla', 'Volume' : 1000, 'Capability': 'Fermenter/conditioner'\
, 'Bottles' : 3030, 'Active' : False, 'Beer' : ''},
'Brigadier' : {'Name' : 'Brigadier', 'Volume' : 800, 'Capability': 'Fermenter/conditioner'\
, 'Bottles' : 2424, 'Active' : False, 'Beer' : ''},
'Dylon' : {'Name' : 'Dylon', 'Volume' : 800, 'Capability': 'Fermenter/conditioner',\
 'Bottles' : 2424, 'Active' : False, 'Beer' : ''},
'Florence' : {'Name' : 'Florence', 'Volume' : 800, 'Capability': 'Fermenter/conditioner',\
 'Bottles' : 2424, 'Active' : False, 'Beer' : ''},
'Gertrude' : {'Name' : 'Gertrude', 'Volume' : 680, 'Capability': 'conditioner', 'Bottles'\
 : 2060, 'Active' : False, 'Beer' : ''},
'Harry' : {'Name' : 'Harry', 'Volume' : 680, 'Capability': 'conditioner', \
'Bottles' : 2060, 'Active' : False, 'Beer' : ''},
'R2D2' : {'Name' : 'R2D2', 'Volume' : 680, 'Capability': 'Fermenter', 'Bottles' : 2060,\
 'Active' : False, 'Beer' : ''},
}
# declare dictionary holding the stock amounts.
BEER_IN_STOCK = {
'Organic Pilsner' : 0,
'Organic Dunkel' : 0,
'Organic Red Helles' : 0
}


def prediction_dunkel(months) -> str:
    """
    This function is passed the value 'months' as a string argument.

    The purpose of this function is the generate a prediction for the amount of
    bottles to be sold after a given amount of months.
    """
    # 35.81818182 linear increase per month.
    months = int(months)
    # creates function to generate regression line and predict sales for future months
    # based on previous data.
    dunkel_data = np.polyfit(MONTH,DUNKEL_SALES,1)
    poly1d_dunkel = np.poly1d(dunkel_data)
    DUNKEL_PREDICTION = poly1d_dunkel(months)
    # prediction is returned.
    return DUNKEL_PREDICTION

def prediction_pilsner(months) -> str:
    """
    This function is passed the value 'months' as a string argument.

    The purpose of this function is the generate a prediction for the amount of
    bottles to be sold after a given amount of months.
    """
    # 48.13286713 linear increase per month.
    months = int(months)
    # creates function to generate regression line and predict sales for future months
    # based on previous data.
    pilsner_data = np.polyfit(MONTH, PILSNER_SALES, 1)
    poly1d_pilsner = np.poly1d(pilsner_data)
    PILSNER_PREDICTION = poly1d_pilsner(months)
    # prediction is returned.
    return PILSNER_PREDICTION


def prediction_helles(months) -> str:
    """
    This function is passed the value 'months' as a string argument.

    The purpose of this function is the generate a prediction for the amount of
    bottles to be sold after a given amount of months.
    """
    # 41.32867133 linear increase per month.
    months = int(months)
    # creates function to generate regression line and predict sales for future months
    # based on previous data.
    helles_data = np.polyfit(MONTH, HELLES_SALES, 1)
    poly1d_helles = np.poly1d(helles_data)
    HELLES_PREDICTION = poly1d_helles(months)
    # prediction is returned.
    return HELLES_PREDICTION

#
def prediction_beer(PREDICTION) -> str:
    """
    This function is passed the value PREDICTION. It is a string list containing the date and beer type.

    The purpose of this function is to call the functions to generate the predictions for each beer type.
    """
    # exception handling for correct length of list. Ensures both values given.
    if len(PREDICTION) == 2:
        beer = PREDICTION[0]
        date = PREDICTION[1]
    else:
        return
    # initialise prediction variable.
    prediction = 0
    # last known data, all predictions from this point onwards.
    prediction_begin_date = 2019-10-30
    # slice year from full date.
    date_year = int(date[0:4])
    # calculate difference in year.
    year_difference = date_year - 2019
    # slice month from full date.
    date_month = int(date[5:7])
    # calculate difference in months between two dates.
    month_difference = abs(date_month - 10)
    month_difference = (year_difference * 12) - month_difference

    # use prediction functions to generate predicted value for that many months in future.
    if beer == 'Organic Pilsner':
        prediction = prediction_pilsner(int(month_difference))
    elif beer == 'Organic Dunkel':
        prediction = prediction_dunkel(int(month_difference))
    elif beer == 'Organic Red Helles':
        prediction = prediction_helles(int(month_difference))
    # prediction is returned.
    return prediction

def dictionary_creation():
    """
    This function is not passed an arguments.

    The purpose of this function is to populate the dictionary SALES_DICTIONARY by
    reading values from the CSV file.

    '"""
    # open CSV file and begin to read.
    with open('Barnabys_sales_fabriacted_data.csv', newline='') as csvfile:
        # remove first line, contains column headings.
        first_line = csvfile.readline()
        # initialise reader.
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        counter = 0
        # read each row of dictionary.
        for row in reader:
            # update dictionary with the relevant information for each seperate order.
            SALES_DICTIONARY.update( {counter : [('Invoice Number' , row[0]),\
             ('Customer' , row[1]),  ('Date Required' , row[2]), ('Recipe' , row[3]), \
             ('Gyle Number' , row[4]), ('Quantity Ordered' , row[5]) ]} )
            counter = counter + 1

def get_sales():
    """
    This function is not passed any arguments.

    The purpose of this function is to gather the total amount of sales for each
    beer type as a total and also as a monthly total. This data is later used
    to create the graph.
    """
    # intialise counters and total sales variables.
    INVOICE_LIST = []
    total_sales = 0
    counter = 0
    # loop through dictionary to calculate total sales.
    for key in SALES_DICTIONARY:
        quantity_tuple = SALES_DICTIONARY[counter][5]
        total_sales = total_sales + int(quantity_tuple[1])
        counter = counter + 1
    # reset counters
    counter = 0
    dunkel_counter = 0
    pilsner_counter = 0
    helles_counter = 0
    # for each entry in dictionary
    for key in SALES_DICTIONARY:
        # get beer type from tuple in dictionary
        beer_tuple = SALES_DICTIONARY[counter][3]
        beer_type = beer_tuple[1]
        # add the value for quantity to counter for each beer.
        if beer_type == 'Organic Dunkel':
            quantity_tuple = SALES_DICTIONARY[counter][5]
            quantity = quantity_tuple[1]
            dunkel_counter = dunkel_counter + int(quantity)
        elif beer_type == 'Organic Pilsner':
            quantity_tuple = SALES_DICTIONARY[counter][5]
            quantity = quantity_tuple[1]
            pilsner_counter = pilsner_counter + int(quantity)
        elif beer_type == 'Organic Red Helles':
            quantity_tuple = SALES_DICTIONARY[counter][5]
            quantity = quantity_tuple[1]
            helles_counter = helles_counter + int(quantity)
        # find date tuple
        date_tuple = SALES_DICTIONARY[counter][2]
        date = date_tuple[1]
        # convert date from string format to number format.
        date_fix = datetime.strptime(date,'%d-%b-%y')
        date_correct = date_fix.strftime('%Y-%m-%d')
        # slice string and month from date
        month = date_correct[5:7]
        year = date_correct[0:4]
        # add 12 months if 2019
        if year == '2019':
            month = int(month) + 12
        # allows november to be month 1.
        month = int(month) - 11
        # append each value to the sales counter.
        if beer_type == 'Organic Dunkel':
            DUNKEL_SALES[month] = DUNKEL_SALES[month] + int(quantity)
        elif beer_type == 'Organic Pilsner':
            PILSNER_SALES[month] = PILSNER_SALES[month] + int(quantity)
        elif beer_type == 'Organic Red Helles':
            HELLES_SALES[month] = int(HELLES_SALES[month]) + int(quantity)
        # increment
        counter = counter + 1


def generate_graph():
    """
    This function is not passed any arguments.

    The purpose of this function is to generate the graph using the data created by get_sales().
    The function draws one graph with all beer sales on and also a line of best fit (regression line) for each.
    """
    # Prediction functions are initialised for all types of beer.
    dunkel_data = np.polyfit(MONTH,DUNKEL_SALES,1)
    pilsner_data = np.polyfit(MONTH,PILSNER_SALES,1)
    helles_data = np.polyfit(MONTH,HELLES_SALES,1)
    # the poly1d function will prediction the values for future given the previous values
    poly1d_dunkel = np.poly1d(dunkel_data)
    poly1d_pilsner = np.poly1d(pilsner_data)
    poly1d_helles = np.poly1d(helles_data)
    # the regression line and all other points are plotted onto the same graph.
    plt.plot(MONTH,DUNKEL_SALES, 'yo', MONTH, poly1d_dunkel(MONTH), '--k', color='green')
    plt.plot(MONTH,DUNKEL_SALES, 'yo', FUTURE_MONTHS, poly1d_dunkel(FUTURE_MONTHS), '--k', color='green')
    plt.plot(MONTH,PILSNER_SALES, 'yo', MONTH, poly1d_pilsner(MONTH), '--k', color='red')
    plt.plot(MONTH,PILSNER_SALES, 'yo', FUTURE_MONTHS, poly1d_pilsner(FUTURE_MONTHS), '--k', color='red')
    plt.plot(MONTH,HELLES_SALES, 'yo', MONTH, poly1d_helles(MONTH), '--k',color='blue')
    plt.plot(MONTH,HELLES_SALES, 'yo', FUTURE_MONTHS, poly1d_helles(FUTURE_MONTHS), '--k',color='blue')
    # original lines are also plotted.
    plt.plot(MONTH, DUNKEL_SALES, label = 'Organic Dunkel Sales', color='green')
    plt.plot(MONTH, PILSNER_SALES, label = 'Organic Pilsner Sales', color='red')
    plt.plot(MONTH, HELLES_SALES, label = 'Organic Red Helles Sales', color='blue')
    # X and Y axis are labelled.
    plt.ylabel('Individual Beer Sale')
    plt.xlabel('Month')
    # Integer X axis values are masked by char values for each month.
    plt.xticks(MONTH, MONTH_STRING)
    plt.legend()
    # Graph is saved as an image, allowing it to be sent to flask to be displayed on webpage.
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    # Graph closed.
    plt.close()
    # Address of graph is returned to be used for flask.
    return 'data:image/png;base64,{}'.format(graph_url)

def complete_order(ORDER) -> str:
    """
    This function is passed the value ORDER as a string list.

    The purpose of this function is to produce a recomendation for the user as to
    where best to brew their beer. The algorithm works on the assumption that Gertrude,
    Harry and R2D2 are tanks that should only be used as a worst case scenario
    as a reserve backup. It attempts to find the most efficient solution by
    always filling the tanks that will produce the minimal excess bottles.

    """
    # brewing is initialised to false
    brewing = False
    # exception handling to ensure correct length of list containing all data.
    if len(ORDER) == 4:
        # variables assigned.
        beer_type = ORDER[0]
        quantity = ORDER[1]
        date_required = ORDER[2]
        process = ORDER[3]
    else:
        return

    # checks whether beer in stock is enough to fulfill order without having to produce any more.
    if beer_type == 'Organic Pilsner':
        # if beer in stock is greate than order, order will be completed instantly
        if quantity <= BEER_IN_STOCK['Organic Pilsner']:
            BEER_IN_STOCK['Organic Pilsner'] = BEER_IN_STOCK['Organic Pilsner'] - quantity
            NOTIFICATION.append('Enough Organic Pilsner in stock to satisy order.')
        else:
            # otherwise the beers in stock are included and the total to be made is reduced.
            quantity = quantity - BEER_IN_STOCK['Organic Pilsner']
            NOTIFICATION.append(str(quantity) + ' ' + beer_type + "'s to brew.")
    elif beer_type == 'Organic Dunkel':
        # if beer in stock is greate than order, order will be completed instantly
        if quantity <= BEER_IN_STOCK['Organic Dunkel']:
            BEER_IN_STOCK['Organic Dunkel'] = BEER_IN_STOCK['Organic Dunkel'] - quantity
            NOTIFICATION.append('Enough Organic Pilsner in stock to satisy order.')
        else:
            # otherwise the beers in stock are included and the total to be made is reduced.
            quantity = quantity - BEER_IN_STOCK['Organic Dunkel']
            NOTIFICATION.append(str(quantity) + ' ' + beer_type + "'s to brew.")
    elif beer_type == 'Organic Red Helles':
        # if beer in stock is greate than order, order will be completed instantly
        if quantity <= BEER_IN_STOCK['Organic Red Helles']:
            BEER_IN_STOCK['Organic Red Helles'] = BEER_IN_STOCK['Organic Red Helles'] - quantity
            NOTIFICATION.append('Enough Organic Pilsner in stock to satisy order.')
        else:
            # otherwise the beers in stock are included and the total to be made is reduced.
            quantity = quantity - BEER_IN_STOCK['Organic Red Helles']
            NOTIFICATION.append(str(quantity) + ' ' + beer_type + "'s to brew.")

    if quantity <= 2424: # 2424 330ml can be made using 800L
        # select a tank of 800L capacity thats not active and make active,
        # add any excess produced that is not needed to the reserve.
        for key in BREWHOUSE_TANKS:
            # if correct volume and not being used, fill.
            if BREWHOUSE_TANKS[key]['Volume'] == 800 and BREWHOUSE_TANKS[key]['Active'] \
             == False and process in BREWHOUSE_TANKS[key]['Capability']:
                # change status of tank, now it is use.
                BREWHOUSE_TANKS[key]['Active'] = True
                # tank status is changed to what beer the tank is holding.
                BREWHOUSE_TANKS[key]['Beer'] = beer_type
                # spare beer is calculated.
                spare_beer = 2424 - quantity
                # updates stock for each.
                if beer_type == 'Organic Pilsner':
                    BEER_IN_STOCK['Organic Pilsner'] = BEER_IN_STOCK['Organic Pilsner'] + spare_beer
                elif beer_type == 'Organic Dunkel':
                    BEER_IN_STOCK['Organic Dunkel'] = BEER_IN_STOCK['Organic Dunkel'] + spare_beer
                elif beer_type == 'Organic Red Helles':
                    BEER_IN_STOCK['Organic Red Helles'] = BEER_IN_STOCK['Organic Red Helles'] + spare_beer
                # update user by adding notification.
                NOTIFICATION.append(str(quantity) + ' bottles of ' + beer_type + '  \
                are being brewed in ' + key + ' (' + process + ').')
                # update logging.
                logging.info(str(quantity) + ' bottles of ' + beer_type + ' are being brewed in ' \
                + key + ' (' + process + ').')
                # brewing set to True.
                brewing = True
                break
                # If all 800L tanks are full
            if BREWHOUSE_TANKS[key]['Volume'] == 800 and BREWHOUSE_TANKS[key]['Active'] \
             == True and process in BREWHOUSE_TANKS[key]['Capability']:
                NOTIFICATION.append(key + ' is full')
        if brewing == False:
            for key in BREWHOUSE_TANKS:
                # if correct volume and not being used, fill.
                if BREWHOUSE_TANKS[key]['Volume'] == 1000 and BREWHOUSE_TANKS[key]['Active'] \
                == False and process in BREWHOUSE_TANKS[key]['Capability']:
                    BREWHOUSE_TANKS[key]['Active'] = True
                    BREWHOUSE_TANKS[key]['Beer'] = beer_type
                    # spare beer calculated.
                    spare_beer = 3030 - quantity
                    # updates stock for each.
                    if beer_type == 'Organic Pilsner':
                        BEER_IN_STOCK['Organic Pilsner'] = BEER_IN_STOCK['Organic Pilsner'] \
                        + spare_beer
                    elif beer_type == 'Organic Dunkel':
                        BEER_IN_STOCK['Organic Dunkel'] = BEER_IN_STOCK['Organic Dunkel'] \
                         + spare_beer
                    elif beer_type == 'Organic Red Helles':
                        BEER_IN_STOCK['Organic Red Helles'] = BEER_IN_STOCK['Organic Red Helles'] \
                         + spare_beer
                    # update user by adding notification.
                    NOTIFICATION.append(str(quantity) + ' bottles of ' + beer_type + \
                     ' are being brewed in ' + key + ' (' + process + ').')
                    # logging updated.
                    logging.info(str(quantity) + ' bottles of ' + beer_type +  \
                    ' are being brewed in ' + key + ' (' + process + ').')
                    brewing = True
                    break
                else:
                    if BREWHOUSE_TANKS[key]['Volume'] == 1000 and BREWHOUSE_TANKS[key]['Active'] \
                     == True and process in BREWHOUSE_TANKS[key]['Capability']:
                        NOTIFICATION.append(key + ' is full')
            # if a tank cannot be found, brewing will still be false.
        if brewing == False:
            NOTIFICATION.append('All tanks are full')

    # for orders greater than 2424 but less than 3030.
    if quantity <= 3030 and quantity > 2424:
        for key in BREWHOUSE_TANKS:
            # if correct volume and not being used, fill.
            if BREWHOUSE_TANKS[key]['Volume'] == 1000 and BREWHOUSE_TANKS[key]['Active'] == False \
            and process in BREWHOUSE_TANKS[key]['Capability']:
                BREWHOUSE_TANKS[key]['Active'] = True
                BREWHOUSE_TANKS[key]['Beer'] = beer_type
                # spare beer calculated.
                spare_beer = 3030 - quantity
                # spare volumes updated
                if beer_type == 'Organic Pilsner':
                    BEER_IN_STOCK['Organic Pilsner'] = BEER_IN_STOCK['Organic Pilsner'] + spare_beer
                elif beer_type == 'Organic Dunkel':
                    BEER_IN_STOCK['Organic Dunkel'] = BEER_IN_STOCK['Organic Dunkel'] + spare_beer
                elif beer_type == 'Organic Red Helles':
                    BEER_IN_STOCK['Organic Red Helles'] = BEER_IN_STOCK['Organic Red Helles'] + spare_beer
                # notification updated.
                NOTIFICATION.append(str(quantity) + ' bottles of ' + beer_type +  \
                ' are being brewed in ' + key + ' (' + process + ').')
                # logging updated
                logging.info(str(quantity) + ' bottles of ' + beer_type + ' are being brewed in '  \
                + key + ' (' + process + ').')
                brewing = True
                break
            else:
                # if tank is active, it is full.
                if BREWHOUSE_TANKS[key]['Volume'] == 1000 and BREWHOUSE_TANKS[key]['Active'] \
                 == True and process in BREWHOUSE_TANKS[key]['Capability']:
                    NOTIFICATION.append(key + ' is full')
        # if cannot find larger tank, move to smaller tanks.
        if brewing == False:
            for key in BREWHOUSE_TANKS:
                # if correct volume and not being used, fill.
                if BREWHOUSE_TANKS[key]['Volume'] == 800 and BREWHOUSE_TANKS[key]['Active'] \
                == False and process in BREWHOUSE_TANKS[key]['Capability']:
                    BREWHOUSE_TANKS[key]['Active'] = True
                    BREWHOUSE_TANKS[key]['Beer'] = beer_type
                    # notification and logging updated.
                    NOTIFICATION.append(str(quantity) + ' bottles of ' + beer_type + \
                    ' are being brewed in ' + key + ' (' + process + ').')
                    logging.info(str(quantity) + ' bottles of ' + beer_type + \
                    ' are being brewed in ' + key + ' (' + process + ').')
                    quantity = quantity - 2424
                    # 2424 taken from larger value, repeat needed.
                    if quantity <= 0:
                        spare_beer = -quantity
                        if beer_type == 'Organic Pilsner':
                            BEER_IN_STOCK['Organic Pilsner'] = BEER_IN_STOCK['Organic Pilsner'] + spare_beer
                        elif beer_type == 'Organic Dunkel':
                            BEER_IN_STOCK['Organic Dunkel'] = BEER_IN_STOCK['Organic Dunkel'] + spare_beer
                        elif beer_type == 'Organic Red Helles':
                            BEER_IN_STOCK['Organic Red Helles'] = BEER_IN_STOCK['Organic Red Helles'] + spare_beer
                        break
                else:   #if active is True, tank is full
                    if BREWHOUSE_TANKS[key]['Volume'] == 1000 and BREWHOUSE_TANKS[key]['Active'] \
                     == True and process in BREWHOUSE_TANKS[key]['Capability']:
                        NOTIFICATION.append(key + ' is full')
    # for all orders larger than biggest tank
    if quantity > 3030:
        for key in BREWHOUSE_TANKS:
            # if correct volume and not being used, fill.
            if BREWHOUSE_TANKS[key]['Volume'] == 1000 and BREWHOUSE_TANKS[key]['Active'] \
             == False and process in BREWHOUSE_TANKS[key]['Capability']:
                BREWHOUSE_TANKS[key]['Active'] = True
                BREWHOUSE_TANKS[key]['Beer'] = beer_type
                if quantity > 3030:
                    # full large tank
                    NOTIFICATION.append('3030 bottles of ' + beer_type + ' are being brewed in '  \
                    + key + ' (' + process + ').')
                else:
                    # update notifications and logging.
                    NOTIFICATION.append(str(quantity) + ' bottles of ' + beer_type + \
                    ' are being brewed in ' + key + ' (' + process + ').')
                    logging.info(str(quantity) + ' bottles of ' + beer_type + ' are being brewed in ' \
                     + key + ' (' + process + ').')
                quantity = quantity - 3030
                # will continue until quantity is fullfilled.
                if quantity <= 0:
                    spare_beer = -quantity
                    # spare beer volumes are updated.
                    if beer_type == 'Organic Pilsner':
                        BEER_IN_STOCK['Organic Pilsner'] = BEER_IN_STOCK['Organic Pilsner'] + spare_beer
                    elif beer_type == 'Organic Dunkel':
                        BEER_IN_STOCK['Organic Dunkel'] = BEER_IN_STOCK['Organic Dunkel'] + spare_beer
                    elif beer_type == 'Organic Red Helles':
                        BEER_IN_STOCK['Organic Red Helles'] = BEER_IN_STOCK['Organic Red Helles'] + spare_beer
                    break
            else:
                # if nothing else, tank is full
                if BREWHOUSE_TANKS[key]['Volume'] == 1000 and BREWHOUSE_TANKS[key]['Active'] \
                 == True and process in BREWHOUSE_TANKS[key]['Capability']:
                    NOTIFICATION.append(key + ' is full')
        # if all larger tanks checked and quantity still remians, search smaller tanks.
        if quantity > 0:
            for key in BREWHOUSE_TANKS:
                # if correct volume and not being used, fill.
                if BREWHOUSE_TANKS[key]['Volume'] == 800 and BREWHOUSE_TANKS[key]['Active'] \
                == False and process in BREWHOUSE_TANKS[key]['Capability']:
                    BREWHOUSE_TANKS[key]['Active'] = True
                    BREWHOUSE_TANKS[key]['Beer'] = beer_type
                    if quantity > 2424:
                        NOTIFICATION.append('2424 bottles of ' + beer_type + ' are being brewed in ' \
                         + key + ' (' + process + ').')
                    else:
                        NOTIFICATION.append(str(quantity) + ' bottles of ' + beer_type + \
                        ' are being brewed in ' + key + ' (' + process + ').')
                        logging.info(str(quantity) + ' bottles of ' + beer_type + \
                        ' are being brewed in ' + key + ' (' + process + ').')
                    # will continue until quantity is fullfilled.
                    quantity = quantity - 2424
                    if quantity <= 0:
                        spare_beer = -quantity
                        # spare beer volumes are updated.
                        if beer_type == 'Organic Pilsner':
                            BEER_IN_STOCK['Organic Pilsner'] = BEER_IN_STOCK['Organic Pilsner'] + spare_beer
                        elif beer_type == 'Organic Dunkel':
                            BEER_IN_STOCK['Organic Dunkel'] = BEER_IN_STOCK['Organic Dunkel'] + spare_beer
                        elif beer_type == 'Organic Red Helles':
                            BEER_IN_STOCK['Organic Red Helles'] = BEER_IN_STOCK['Organic Red Helles'] + spare_beer
                        break
                else:
                    if BREWHOUSE_TANKS[key]['Volume'] == 800 and BREWHOUSE_TANKS[key]['Active'] \
                    == True and process in BREWHOUSE_TANKS[key]['Capability']:
                        NOTIFICATION.append(key + ' is full')
        # if quantity still above 0, order not possible to complete.
        if quantity > 0:
            # update notifications and logging.
            NOTIFICATION.append('Unable to fully complete order, there are ' + str(quantity) + ' left to brew.')
            logging.info('Unable to fully complete order, there are ' + str(quantity) + ' left to brew.')
    # start time of brewing.
    order_begin = datetime.now()
    NOTIFICATION.append('Time of process start: ' + str(order_begin))
    NOTIFICATION.append('-----------------------')


# R2D2 and 2 conditioners are left as a back up tanks and so is not included in the recomendation algorithm.


def active_brewing(beer_type, quantity, date_required, process) -> str:
    """
    This function is passed four string arguments.

    The purpose of this function is create a small text update for the user on
    the UI allowing them to see brewing activity that is currently underway.

    """
    # values passed in are appended to active brew list which will be displayed to user.
    # uses string concatination to joing the variable with the original string.
    # puts date into readible format.
    current_time = datetime.today().strftime('%Y-%m-%d')
    ACTIVE_BREWING.append('-----------------------')
    ACTIVE_BREWING.append('BEER TYPE: ' + beer_type)
    ACTIVE_BREWING.append('QUANTITY: ' + quantity)
    ACTIVE_BREWING.append('DATE REQUIRED: ' + date_required)
    ACTIVE_BREWING.append('DATE STARTED: ' + str(current_time))
    ACTIVE_BREWING.append('PROCESS: ' + process)
    ACTIVE_BREWING.append('-----------------------')

# populate the SALES_DICTIONARY
dictionary_creation()
# gather sales information
get_sales()

@APP.route('/')
@APP.route('/home')
def home():
    # populate the SALES_DICTIONARY after each refresh, checking for any changes.
    dictionary_creation()
    # graph is generated and updated with each refresh.
    graph = generate_graph()
    # variables are generated by taking them directly from URL after being submitted by a form.
    beer_type = request.args.get('beer')
    date_required = request.args.get('date_required')
    quantity = request.args.get('quantity')
    process = request.args.get('process')
    notification_clear = request.args.get('clear')
    active_clear = request.args.get('clear_active')
    reset_tanks = request.args.get('reset_tanks')
    beer_prediction = request.args.get('beer_prediction')
    date_prediction = request.args.get('date_prediction')
    beer_stock = request.args.get('beer_stock')
    quantity_stock = request.args.get('quantity_stock')
    beer_csv = request.args.get('beer_csv')
    date_required_csv = request.args.get('date_required_csv')
    invoice_csv = request.args.get('invoice_csv')
    quantity_csv = request.args.get('quantity_csv')
    customer_name_csv = request.args.get('customer_name_csv')
    gyle_number_csv = request.args.get('gyle_number_csv')
    # declaration of variables.
    global BREWHOUSE_TANKS
    PREDICTION_UPPER = 0
    PREDICTION_LOWER = 0
    PREDICTION = 0
    # if all variables populated, then continue.
    if beer_type and date_required and quantity and process:
        # begin new order, trigger algorithm.
        NOTIFICATION.append('Order for ' + quantity + ' ' + beer_type + "'s for "\
         + date_required + ' has began brewing.')
        # store variables in list.
        ORDER = [beer_type, int(quantity), date_required, process]
        complete_order(ORDER)
        active_brewing(beer_type, quantity, date_required, process)
    if notification_clear:
        # clear notification list.
        NOTIFICATION.clear()
        # logging update
        logging.info('Notifications have been cleared.')
    if active_clear:
        # clear active brewing list.
        ACTIVE_BREWING.clear()
        # logging update
        logging.info('Active list has been cleared.')
    if beer_prediction and date_prediction:
        # gather prediction after passing variables into list.
        get_prediction = [beer_prediction, date_prediction]
        PREDICTION = prediction_beer(get_prediction)
        # round all answers
        PREDICTION_UPPER = round(int(PREDICTION) * 1.25)
        PREDICTION_LOWER = round(int(PREDICTION) * 0.75)
        # round answer after calculations. More accurate.
        PREDICTION = round(PREDICTION)
        # Prediction is based on the linear regression line generated from the test data.
        # Two additional predictions are provided in order to account for potential inaccurasies in the data.
    if reset_tanks:
        # Tank is reset to original value.
        BREWHOUSE_TANKS = {
        'Albert' : {'Name' : 'Albert', 'Volume' : 1000, 'Capability': 'Fermenter/conditioner'\
        , 'Bottles' : 3030, 'Active' : False, 'Beer' : ''},
        'Emily' : {'Name' : 'Emily', 'Volume' : 1000, 'Capability': 'Fermenter/conditioner'\
        , 'Bottles' : 3030, 'Active' : False, 'Beer' : ''},
        'Camilla' : {'Name' : 'Camilla', 'Volume' : 1000, 'Capability': 'Fermenter/conditioner'\
        , 'Bottles' : 3030, 'Active' : False, 'Beer' : ''},
        'Brigadier' : {'Name' : 'Brigadier', 'Volume' : 800, 'Capability': 'Fermenter/conditioner'\
        , 'Bottles' : 2424, 'Active' : False, 'Beer' : ''},
        'Dylon' : {'Name' : 'Dylon', 'Volume' : 800, 'Capability': 'Fermenter/conditioner',\
         'Bottles' : 2424, 'Active' : False, 'Beer' : ''},
        'Florence' : {'Name' : 'Florence', 'Volume' : 800, 'Capability': 'Fermenter/conditioner',\
         'Bottles' : 2424, 'Active' : False, 'Beer' : ''},
        'Gertrude' : {'Name' : 'Gertrude', 'Volume' : 680, 'Capability': 'conditioner', 'Bottles'\
         : 2060, 'Active' : False, 'Beer' : ''},
        'Harry' : {'Name' : 'Harry', 'Volume' : 680, 'Capability': 'conditioner', \
        'Bottles' : 2060, 'Active' : False, 'Beer' : ''},
        'R2D2' : {'Name' : 'R2D2', 'Volume' : 680, 'Capability': 'Fermenter', 'Bottles' : 2060,\
         'Active' : False, 'Beer' : ''},
        }
        # logged.
        logging.info('Tanks have been reset.')
    if beer_stock and quantity_stock:
        # user entered stock is added to dictionary of all stock.
        BEER_IN_STOCK[beer_stock] = BEER_IN_STOCK[beer_stock] + int(quantity_stock)
    # if all variables populated, then continue.
    if beer_csv and date_required_csv and invoice_csv and quantity_csv and customer_name_csv and gyle_number_csv:
        # convert month from number to string equivalent eg. 01 to Jan.
        date_time_fix = datetime.strptime(date_required_csv, '%Y-%m-%d')
        date_correct = date_time_fix.strftime('%d-%b-%y')
        # open CSV file.
        with open('Barnabys_sales_fabriacted_data.csv', 'a', newline='') as file:
            # write data to CSV file.
            writer = csv.writer(file)
            writer.writerow([invoice_csv, customer_name_csv, date_correct, beer_csv, gyle_number_csv, quantity_csv])
        # logged.
        logging.info('CSV file updated.')
    # template rendered with variables passed in also.
    return render_template('breweryMain.html', graph = generate_graph(), \
    NOTIFICATION = NOTIFICATION, BEER_IN_STOCK = BEER_IN_STOCK, PREDICTION = PREDICTION,\
     PREDICTION_UPPER = PREDICTION_UPPER, PREDICTION_LOWER = PREDICTION_LOWER, \
     ACTIVE_BREWING = ACTIVE_BREWING, date_prediction = date_prediction, BREWHOUSE_TANKS = BREWHOUSE_TANKS)
# Allows program to be executed.
if __name__ == '__main__':
    APP.run()
