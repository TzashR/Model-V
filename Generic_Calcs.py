'''
Functions used on data points. Useful when generating data,training the model and when predicting
'''

def predict_point(target, sources,weight_func):
    '''
    Calculates the target's value based on the set of sources.
    :param target: point where value should be predicted
    :param sources: A list of tuples, every tuple is (observation, value)
    :param weight_func: A weight function for the type of the points. The
    function should recieve (source, target)
    :return: Predicted value (depends on the type of data points)
    '''
    weights_sum = 0
    weighted_values_sum = 0

    for observation, val in sources:
        weight = weight_func(observation, target)
        weights_sum+=weight
        weighted_values_sum +=weight*val
    return weighted_values_sum/weights_sum
