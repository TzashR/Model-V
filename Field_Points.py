'''
Classes for data points of infected fields to be used in the data maker
'''

class FieldPoint:
    '''A data point representing a field in the data maker'''
    def __init__(self, x_rnd_gen, y_rnd_gen, t_rnd_gen):
        '''
        :param x_rnd_gen: A generator for x value
        :param y_rnd_gen: A generator for y value
        :param t_rnd_gen: A generator for t value
        '''
        self.x = x_rnd_gen()
        self.y = y_rnd_gen()
        self.t = t_rnd_gen()
        self.s = None
    def __repr__(self):
        return f'FieldPoint: (x:{self.x},y:{self.y},t:{self.t},s:{self.s})'

def get_x(): return 1

point = FieldPoint(get_x,get_x,get_x)
print(point)
