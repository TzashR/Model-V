'''
Classes for data points of infected fields to be used in the data maker
'''
import math
from sklearn import neighbors
import numpy as np

class FieldPoint:
    '''A data point representing a field in the data maker'''
    def __init__(self, x, y, t,  s = None, is_ob = False):
        '''
        :param x_rnd_gen: A generator for x value
        :param y_rnd_gen: A generator for y value
        :param t_rnd_gen: A generator for t value
        '''
        self.x = x
        self.y = y
        self.t = t
        self.s = s
        self.is_ob = is_ob
    def __repr__(self):
        return f'FieldPoint: (x:{self.x},y:{self.y},t:{self.t},s:{self.s})'

    def calc_dist(self, other):
        ''' gets euclidean distance between two points'''
        return math.sqrt((self.x-other.x)**2+(self.y-other.y)**2)


class FieldObservation(FieldPoint):

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



def make_neighbors_list(data_points,dist_radius,time_range, limit = -1):
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


