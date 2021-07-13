'''
Classes for the objects used in the simulation
'''
import math
from sklearn import neighbors
import numpy as np
import random

class Map:
    '''Holds all the data points in the area (The area is rectangle)'''
    def __init__(self,x_bounds,y_bounds, points = []):
        '''
        :param x_bounds: a tuple (+float,+float) stating the bounds of the x coordinates
        :param y_bounds:  a tuple (+float,+float) stating the bounds of the y coordinates
        :param points: list of points on the map
        '''
        assert (x_bounds[0]<x_bounds[1] and x_bounds[0]>0 and x_bounds[1]>0)
        assert (y_bounds[0]<y_bounds[1] and y_bounds[0]>0 and y_bounds[1]>0)
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.points = points


def create_random_map(n_points,size, only_data_points = True):
    '''
    Creates a random Map object
    :param n_points: how many data points should the map have
    :param size: a tuple (a,b) stating the size of the rectangle.
    :param only_data_points: If true, all points are data-points (can be reported), else some of them can be obstacles
    :return: A map n-points and bounds fitting the size
    '''

    #create data points
    points = []
    for i in range(n_points):
        y_cor =  round(random.uniform(0,size[0]),2)
        x_cor = round(random.uniform(0,size[1]),2)

        if only_data_points:
            new_point = DataPoint(x_cor,y_cor)
        #add other points when the model supports it
        points.append(new_point)
    res_map = Map((0,size[1]),(0,size[0]),points)
    return res_map

class Point:
    '''General point in the Map'''
    def __init__(self, x, y):
        '''
        :param x: x coordinate
        :param y: y coordinate
        '''
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Point: (x:{self.x},y:{self.y})'

    def calc_dist(self, other):
        ''' calculates euclidean distance between two points'''
        return math.sqrt((self.x-other.x)**2+(self.y-other.y)**2)

class DataPoint(Point):
    '''A data point of interest in the map'''
    def __init__(self, x, y, s = 0):
        super().__init__(self, x, y, t)
        self.x = x
        self.y = y
        self.s = s
        self.history = []

    def __repr__(self):
        return f'Data Point: (x:{self.x},y:{self.y},s:{self.s})'

    def update_s(self, new_s):
        '''Changes s value and saves the previous one in history'''
        self.history.append(new_s)
        self.s = new_s

    def clear_history(self):
        '''clears the points history'''
        self.history = []


class Observation(DataPoint):
    #TODO work on this
    def __init__(self, x, y, t,reported_s,v = None, s = None):
        super().__init__(self, x, y, t)
        self.v = v
        self.reported_s = reported_s

def make_kernel(alpha,beta):
    '''
    Creates the function that will be used as the kernel for points effecting each other
    :param alpha: Decay in distance parameter
    :param beta: Decay in time parameter
    :return: Function  that takes in observation and a point and calculates observations weight in affecting the point
    '''
    def calc_weight(source, target, dist_decay = alpha, time_decay = beta):
        '''
        :param source: source observation (FieldObservation)
        :param target: target point (FieldPoint)
        :return: source's weight in predicting target's s
        '''
        assert source.t < target.t
        time_diff = target.t-source.t
        dist = source.calc_dist(target)
        return math.exp(-time_decay*time_diff-dist_decay*dist)
    return calc_weight



def make_neighbors_list(data_points,dist_radius,time_range, limit = 8):
    '''
    Calculates the neighbors for each data_point
    :param data_points: a list of FieldPoints
    :param dist_radius:  How far neighbors can be
    :param time_range: How long in the past is relevant
    :param limit = limit number of neighbors
    :return: a list where for the item in index i contains all the neighbors of data_points[i]
    '''
    location_arr = np.array([[point.x,point.y] for point in data_points])
    time_arr = np.array([point.t] for point in data_points)


