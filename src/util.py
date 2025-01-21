# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

from datetime import datetime
from dateutil.parser import isoparse
from functools import wraps
from time import time
from logging import getLogger
import inspect
import config
import cv2 as cv


def add_img_mask(img, mask, transparency):
    new_img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    if transparency < 1.0:
        return cv.addWeighted(new_img, 1.0, mask, transparency, 0.0)
    else:
        return mask


def date_diff(a, b):
    diff = datetime.strptime(a, '%Y-%m-%d') - isoparse(b).replace(tzinfo=None)
    return divmod(diff.total_seconds(), 3600 * 24)[0]


def get_metric_str(metric, percentage, actual, higher_is_better=True):
    if higher_is_better:
        color = "green" if percentage > 0.0 else ("red" if percentage < 0.0 else "grey")
    else:
        color = "red" if percentage > 0.0 else ("green" if percentage < 0.0 else "grey")
    pre = "+" if percentage > 0.0 else ""
    return f"### {metric}\n  ### :{color}[{pre}{percentage:.0%}]"


def get_recommendation_str(key, before, after, higher_is_better=True, unit=""):
    if higher_is_better:
        color = "green" if after > before else ("red" if after < before else "blue")
        change = "improved" if after > before else ("worsened" if after < before else "unchanged")
    else:
        color = "green" if after < before else ("red" if after > before else "blue")
        change = "improved" if after < before else ("worsened" if after > before else "unchanged")
    from_to = f"from {before:.2f} to {after:.2f} {unit}" if change != "unchanged" else f"from {before:.2f} {unit}"
    return f"{key} :{color}[{change}] {from_to}."


def timing(f):
    """Simple timing decorator."""
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    filename = module.__name__

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        if config.LoggingConfig.timing:
            getLogger(filename).info('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result
    return wrap
