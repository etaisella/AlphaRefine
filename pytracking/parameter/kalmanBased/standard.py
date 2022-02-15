from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone

def parameters():
    params = TrackerParams()

    params.num_particles = 140
    params.x_rand_scale = 1.25
    params.y_rand_scale = 1.25
    params.x_box_rand_scale = 0.8
    params.y_box_rand_scale = 1.25
    params.min_box_width = 5
    params.min_box_height = 5
    params.gt_update_interval = 100

    return params
