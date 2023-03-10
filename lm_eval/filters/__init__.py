from lm_eval.api.filter import Filter
from . import *


FILTER_REGISTRY = {
    "none": Filter,
}


def get_filter(filter_name):
    return FILTER_REGISTRY[filter_name]