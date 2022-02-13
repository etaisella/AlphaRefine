from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone

def parameters():
    params = TrackerParams()

    params.num_particles = 100
    params.x_rand_scale = 3
    params.y_rand_scale = 3
    params.x_box_rand_scale = 0
    params.y_box_rand_scale = 0
    params.min_box_width = 5
    params.min_box_height = 5

    return params
