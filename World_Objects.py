'''
Classes for the objects used in the simulation
'''
from Generic_Calcs import calc_adj_mat
import math
from sklearn import neighbors
import numpy as np
import random

class Map:
    '''Holds all the data points in the area (The area is rectangle)'''
    def __init__(self,x_bounds,y_bounds, data_points = [], obstacles = []):
        '''
        :param x_bounds: a tuple (+float,+float) stating the bounds of the x coordinates
        :param y_bounds:  a tuple (+float,+float) stating the bounds of the y coordinates
        :param data_points: list of DataPoints on the map
        :param obstacles: list of Obstacles on the map

        '''
        assert (x_bounds[0]<x_bounds[1] and x_bounds[0]>0 and x_bounds[1]>0)
        assert (y_bounds[0]<y_bounds[1] and y_bounds[0]>0 and y_bounds[1]>0)
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.data_points = data_points
        self.obstacles = obstacles
        self.neighbors = {}



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
        self.neighbors = []
        self.history = []

    def __repr__(self):
        return f'Data Point: (x:{self.x},y:{self.y},s:{self.s})'

    def update_s(self, new_s, intervention = False):
        '''
        Changes s value and saves the previous one in history
        :param new_s:
        :param intervention: If this is due to intervention, it will be recorded at hisotry
        :return:
        '''
        self.hisotry.append((new_s, new_s-self.s, intervention))
        self.s = new_s

    def clear_history(self):
        '''clears the points history'''
        self.history = []


class Observation(DataPoint):
    '''An observation produced by reporter, representing perceived s in point p at time t '''
    #TODO work on this
    def __init__(self, x, y, t,reported_s,v = None, s = None):
        super().__init__(self, x, y, t)
        self.v = v
        self.reported_s = reported_s


class Reporter:
    '''Representing a reporter'''
    def __init__(self, id,data_points):
        self.id = id
        self.data_points = data_points
        self.reputation = None #Reputation will be generated based on first report




def make_neighbors_list_geo(data_points, dist_radius, limit=8):
    # TODO finish this
    '''
    Calculates the neighbors for each data_point based on distance only
    :param data_points: a list of FieldPoints
    :param dist_radius:  How far neighbors can be
    :param limit = limit number of neighbors
    :return: a dictionary {point: list of point's neighbors}
    '''
    location_arr = np.array([[point.x, point.y] for point in data_points])
    adj_mat = calc_adj_mat(location_arr)
    n = len(data_points)

    res_dic = {}

    for i in range(n):
        point_distances = adj_mat[i]

        #Sorts the neighbors, second row is the index in the original array
        point_distances = np.vstack((adj_mat[i], np.array([x for x in range(n)])))
        point_distances = point_distances[:, point_distances[0].argsort()]
        optional_neighbors = point_distances[:limit+1]
        neighbors_mask = optional_neighbors[0]<dist_radius
        neighbors_indices = optional_neighbors[1][neighbors_mask]
        point_neighbors = []
        for index in neighbors_indices[1:]: #starting from 1 because point is always closest to itself
            point_neighbors.append(data_points[int(index)])
        res_dic[data_points[i]] = point_neighbors
    return res_dic




