'''
Classes for the objects used in the simulation
'''
import math
import random

import numpy as np

from Generic_Calcs import calc_adj_mat, plot_dist


class Map:
    '''Holds all the data points in the area (The area is rectangle)'''

    def __init__(self, x_bounds, y_bounds, data_points=[], obstacles=[]):
        '''
        :param x_bounds: a tuple (+float,+float) stating the bounds of the x coordinates
        :param y_bounds:  a tuple (+float,+float) stating the bounds of the y coordinates
        :param data_points: list of DataPoints on the map
        :param obstacles: list of Obstacles on the map

        '''
        assert (x_bounds[0] < x_bounds[1] and x_bounds[0] >= 0 and x_bounds[1] >= 0)
        assert (y_bounds[0] < y_bounds[1] and y_bounds[0] >= 0 and y_bounds[1] >= 0)
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.data_points = data_points
        self.obstacles = obstacles
        self.neighbors = {}

    def __repr__(self):
        return f'Map bounds {self.x_bounds, self.y_bounds}\n' \
               f'Number of DataPoints {len(self.data_points)}\n' \
               f'Number of obstcales {len(self.obstacles)}'

    def make_neighbors_list_geo(self, dist_radius=float('inf'), limit=8):
        # TODO finish this
        '''
        Calculates the neighbors for each data_point based on distance only
        :param data_points: a list of FieldPoints
        :param dist_radius:  How far neighbors can be
        :param limit = limit number of neighbors
        :return: a dictionary {point: list of point's neighbors}
        '''
        data_points = self.data_points
        location_arr = np.array([[point.x, point.y] for point in data_points])
        adj_mat = calc_adj_mat(location_arr)
        n = len(data_points)

        res_dic = {}

        for i in range(n):
            # Sorts the neighbors, second row is the index in the original array
            point_distances = np.vstack((adj_mat[i], np.array([x for x in range(n)])))
            point_distances = point_distances[:, point_distances[0].argsort()]
            optional_neighbors = point_distances[:,0:limit + 1]
            neighbors_mask = optional_neighbors[0] < dist_radius
            neighbors_indices = optional_neighbors[1][neighbors_mask]
            point_neighbors = []
            for index in neighbors_indices[1:]:  # starting from 1 because point is always closest to itself
                point_neighbors.append(data_points[int(index)])
            res_dic[data_points[i].id] = point_neighbors
        self.neighbors = res_dic


class Point:
    '''General point in the Map'''

    def __init__(self, x, y, point_id):
        '''
        :param x: x coordinate
        :param y: y coordinate
        :param id: id for the point
        '''
        self.x = x
        self.y = y
        self.id = point_id

    def __repr__(self):
        return f'Point: (id:{self.id}, x:{self.x},y:{self.y})'

    def calc_dist(self, other):
        ''' calculates euclidean distance between two points'''
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class DataPoint(Point):
    '''A data point of interest in the map'''

    def __init__(self, x, y, point_id,hyper_prior, s=0):
        super().__init__(x, y, point_id)
        self.s = s
        self.neighbors = []
        self.history = []
        self.prior = hyper_prior #The current distribution of the point's s. should be a dictionary {"a":2, "scale":1}

    def __repr__(self):
        return f'Data Point {self.id}: (x:{self.x},y:{self.y},s:{self.s})'

    def update_s(self, new_s, intervention=False):
        '''
        Changes s value and saves the previous one in history
        :param new_s:
        :param intervention: If this is due to intervention, it will be recorded at hisotry
        :return:
        '''
        if new_s > 1: new_s = 1
        if new_s <0 : new_s = 0
        self.history.append((new_s, new_s - self.s, intervention))
        self.s = new_s

    def clear_history(self):
        '''clears the points history'''
        self.history = []

    def update_report(self, reported_s, veracity):
        assert 0<=reported_s<=1 and 0<=veracity<=1
        self.last_report = (reported_s,veracity)



class Reporter:
    '''Representing a reporter'''

    def __init__(self, id, data_points, veracity):
        self.id = id
        self.veracity = veracity  # The higher this is, the more likely the reporter to give reliable reports
        self.reputation = None  # Reputation will be generated based on first report

        self.data_points = []
        for p in data_points:
            self.add_DataPoint(p)

    def __repr__(self):
        return f'Reporter id:{self.id}, veracity: {self.veracity}, reputation: {self.reputation} \n' \
               f'number of points: {len(self.data_points)}'

    def report_points(self, T):
        '''
        Generate reports on the reporter's data points. The higher the s of the point, the more likely
        it will be reported.
        :param: Report date (or T)
        :return: A list of reports. Each report is a 5 tuple: (point_id, reporter_id T, reported s, true s)
        '''
        reports = []
        for point in self.data_points:
            does_report = random.uniform(0, 1) < (point.s+0.15)  # chance to report goes up the more infected the point is
            if not does_report: continue
            reported_s = random.gauss(point.s, (1 - self.veracity) / 3.5)  # This is not scientific
            reports.append([T, point.id, self.id, reported_s, self.veracity, point.s])
        return reports

    def add_DataPoint(self, p):
        if not isinstance(p, DataPoint):
            raise TypeError(f"Must append DataPoints. You appended {type(p)}")
        self.data_points.append(p)


def create_random_map(n_points, size,hyper_prior, only_data_points=True):
    '''
    Creates a random Map object
    :param n_points: how many data points should the map have
    :param size: a tuple (a,b) stating the size of the rectangle.
    :param hyper_prior: the default prior distribution every datapoint will have
    :param only_data_points: If true, all points are data-points (can be reported), else some of them can be obstacles
    :return: A map n-points and bounds fitting the size
    '''

    # create data points
    points = []
    cur_id = 0
    for i in range(n_points):
        y_cor = round(random.uniform(0, size[0]), 2)
        x_cor = round(random.uniform(0, size[1]), 2)

        if only_data_points:
            new_point = DataPoint(x_cor, y_cor, f"p{cur_id}", hyper_prior)
            points.append(new_point)
        # add other points when the model supports it
        cur_id += 1
    res_map = Map((0, size[1]), (0, size[0]), points)
    return res_map


def generate_random_reporters(num_of_reporters, data_points, veracity_dist=lambda: random.gauss(0.6, 0.2)):
    '''
    Creates a random list of reporters based on data_points
    :param num_of_reporters: number of wanted reporters. Can't be larger than the number of data_points
    :param data_points: A list of dataPoints to be distributed between the reporters
    :param veracity_dist: A distribution to choose the reporter's veracity from
    :return: A list of reporters
    '''
    assert num_of_reporters <= len(data_points)

    reps = []

    for i in range(num_of_reporters):
        v = veracity_dist()
        if v > 1: v = 1
        if v < 0: v = 0
        new_rep = Reporter(f"r{i}", [data_points[i]], v)
        reps.append(new_rep)

    remaining_points = data_points[num_of_reporters:]
    while len(remaining_points) > 0:
        p = remaining_points.pop(0)
        rep = random.choice(reps)
        rep.data_points.append(p)
    return reps
