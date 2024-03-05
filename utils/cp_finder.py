import numpy as np


def find_local_maximum_indices(
        values: np.ndarray,
        warning, alpha = 1, warning_time = 3, change_time = 1):
    '''
        Finds points with local maximum in the values array such that the point is
        over a certain level (warning * 2).

        If alpha 0<alpha<1, a exponentially moving average is computed on the values

        Warning time is the minimum time steps on warning level to consider being
        on change state

        Change time is the minimum time on change state to report a change point

        If a change is reported, waits until the 
    '''

    cp = []

    running_mean = values[0]
    in_warning = 0
    in_change_point = 0
    waiting = False

    for i in range(values.shape[0]):
        running_mean = (1-alpha) * running_mean + alpha * values[i]
        compare_value = running_mean # / (1-alpha)

        if running_mean >= warning:
            in_warning += 1

        if not waiting:
            if in_warning > warning_time and compare_value >= 2*warning:
                in_change_point +=1

            if in_change_point > change_time and compare_value >= 2*warning:
                cp.append(i)
                waiting = True
                in_change_point = 0
                in_warning = 0

        if running_mean < warning:
            in_warning = max(in_warning - 1, 0)
            waiting = False

    return cp
