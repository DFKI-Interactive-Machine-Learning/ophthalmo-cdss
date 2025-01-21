# This file is part of Ophthalmo-CDSS.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator, LinearNDInterpolator
from logging import getLogger


logger = getLogger(__name__)


class Interpolator:
    """
    Abstract class for interpolation
    """
    def __init__(self, fit_dates=None, fit_values=None, min_val=None, max_val=None):
        """
        :param fit_dates: dates to fit
        :param fit_values: values to fit
        :param min_val: minimum value for interpolation.
        :param max_val: maximum value for interpolation.
        """
        self.fit_dates = fit_dates
        self.fit_values = fit_values
        self.min_val = min_val
        self.max_val = max_val
        self.interpolator = None

    def __name__(self):
        return ""

    def set_fit_data(self, fit_dates, fit_values):
        """
        Set the fit data
        :param fit_dates: dates to fit
        :param fit_values: values to fit
        """

        if np.any(fit_dates.duplicated()):
            duplicated_index = fit_dates.duplicated()
            fit_dates = fit_dates[~duplicated_index]
            fit_values = fit_values.loc[~duplicated_index]
        self.fit_dates = fit_dates.values
        self.fit_values = fit_values.values

    def set_value_range(self, min_val, max_val):
        """
        Set the value range
        :param min_val: minimum value for interpolation.
        :param max_val: maximum value for interpolation.
        """
        self.min_val = min_val
        self.max_val = max_val

    def new(self):
        """ Create a new instance of the interpolator. """
        return self.__class__()

    def fit(self):
        pass

    def __call__(self, x_new):
        """
        Interpolate the data
        :param x_new: x values to interpolate
        :return: y values interpolated
        """
        if self.interpolator is None:
            self.fit()
        if self.min_val is not None or self.max_val is not None:
            y_new = np.clip(self.interpolator(x_new), self.min_val, self.max_val)
        else:
            y_new = self.interpolator(x_new)
        return pd.DataFrame(y_new).interpolate(method='bfill').interpolate(method="ffill").values.flatten()


class ConstantInterpolator(Interpolator):
    def __name__(self):
        return "ConstantInterpolator"

    def __init__(self, fit_dates, fit_values):
        super().__init__(fit_dates, fit_values)
        self.constant = fit_values[0]

    def __call__(self, x_new):
        return np.full(x_new.shape, self.constant)


class LinearInterpolator(Interpolator):
    def __name__(self):
        return "LinearInterpolator"

    def __init__(self, fit_dates=None, fit_values=None, min_val=None, max_val=None):
        super().__init__(fit_dates, fit_values, min_val, max_val)

    def __call__(self, x_new):
        x_new_daily = pd.date_range(start=x_new[0], end=x_new[-1], freq="1d", normalize=True)  # Assert daily frequency
        y_new = np.empty(x_new_daily.shape)
        y_new[:] = np.nan
        new_df = pd.DataFrame(y_new, index=x_new_daily)
        for date, value in zip(self.fit_dates, self.fit_values):
            try:
                new_df.loc[date] = value
            except KeyError:
                pass
        if new_df.isna().values.all():
            if x_new[0] > self.fit_dates[-1]:
                return np.full(x_new.shape, self.fit_values[-1])
            elif x_new[-1] < self.fit_dates[0]:
                return np.full(x_new.shape, self.fit_values[0])
        new_df["mean"] = (new_df.iloc[:, 0].interpolate(method="time", limit_area="inside")
                          .interpolate(method="bfill").interpolate(method="ffill"))
        return new_df["mean"].loc[x_new].values


class AkimaInterpolator(Interpolator):
    def __name__(self):
        return "AkimaInterpolator"

    def __init__(self, fit_dates=None, fit_values=None, min_val=None, max_val=None):
        super().__init__(fit_dates, fit_values)

    def fit(self):
        previous_value = 1.0
        previous_date = pd.to_datetime(self.fit_dates[0] - pd.DateOffset(months=24), unit="D")
        previous_date_2 = pd.to_datetime(self.fit_dates[0] - pd.DateOffset(months=12), unit="D")
        later_date = pd.to_datetime(self.fit_dates[-1] + pd.DateOffset(months=12), unit="D")
        later_date_2 = pd.to_datetime(self.fit_dates[-1] + pd.DateOffset(months=24), unit="D")
        last_values_index = int(len(self.fit_values) * 0.1)
        later_value = self.fit_values[-last_values_index:].mean()
        fit_dates = pd.to_datetime(
            [previous_date, previous_date_2] + list(self.fit_dates) + [later_date, later_date_2]
        ).values
        fit_values = np.array([previous_value, previous_value] + list(self.fit_values) + [later_value, later_value]
                              , dtype='float32')
        self.interpolator = Akima1DInterpolator(fit_dates, fit_values)


class PChipInterpolator(Interpolator):
    def __name__(self):
        return "PChipInterpolator"

    def __init__(self, fit_dates=None, fit_values=None, min_val=None, max_val=None):
        super().__init__(fit_dates, fit_values, min_val, max_val)

    def fit(self):
        previous_value = 1.0
        previous_date = pd.to_datetime(self.fit_dates[0] - pd.DateOffset(months=24), unit="D")
        previous_date_2 = pd.to_datetime(self.fit_dates[0] - pd.DateOffset(months=12), unit="D")
        later_date = pd.to_datetime(self.fit_dates[-1] + pd.DateOffset(months=12), unit="D")
        later_date_2 = pd.to_datetime(self.fit_dates[-1] + pd.DateOffset(months=24), unit="D")
        last_values_index = int(len(self.fit_values) * 0.1)
        later_value = self.fit_values[-last_values_index:].mean()
        fit_dates = pd.to_datetime(
            [previous_date, previous_date_2] + list(self.fit_dates) + [later_date, later_date_2]
        ).values
        fit_values = np.array([previous_value, previous_value] + list(self.fit_values) + [later_value, later_value]
                              , dtype='float32')
        try:
            self.interpolator = PchipInterpolator(fit_dates, fit_values)
        except ValueError as e:
            logger.critical(f"ValueError in PChipInterpolator for {self.fit_dates} and {self.fit_values}")
            raise e


class SmoothedInterpolator(Interpolator):
    def __init__(self, fit_dates=None, fit_values=None, min_val=None, max_val=None, moving_average_window=90):
        """ Smoothed interpolator
        :param fit_dates: dates to fit
        :param fit_values: values to fit
        :param min_val: minimum value for interpolation.
        :param max_val: maximum value for interpolation.
        :param moving_average_window: window size for moving average in days.
        """
        super().__init__(fit_dates, fit_values, min_val, max_val)
        self.moving_average_window = moving_average_window
        self.df = None
        if fit_dates is not None and fit_values is not None:
            self.set_fit_data(fit_dates, fit_values)

    def __name__(self):
        return f"SmoothedInterpolator_{self.moving_average_window}"

    def set_fit_data(self, fit_dates, fit_values):
        super().set_fit_data(fit_dates, fit_values)
        x_new_daily = pd.date_range(start=self.fit_dates.min(), end=self.fit_dates.max(), freq="1d",
                                    normalize=True)  # Assert daily frequency
        y_new = np.empty(x_new_daily.shape)
        y_new[:] = np.nan
        new_df = pd.DataFrame(y_new, index=x_new_daily)
        for date, value in zip(self.fit_dates, self.fit_values):
            new_df.loc[date] = value
        new_df = (new_df.interpolate(method="linear", limit_area="inside"))
        new_df["mean"] = (new_df.iloc[:, 0].rolling(window=self.moving_average_window, center=True,
                                                    win_type='triang', min_periods=1)
                          .mean())
        self.df = new_df

    def __call__(self, x_new):
        x_new = np.array(x_new)
        before = x_new < pd.to_datetime(self.fit_dates.min())
        after = x_new > pd.to_datetime(self.fit_dates.max())
        valid_range = ~before & ~after
        ret_array = np.zeros_like(x_new, dtype=float)
        ret_array[before] = self.fit_values[0]
        ret_array[after] = self.fit_values[-1]
        ret_array[valid_range] = self.df["mean"].loc[x_new[valid_range]].values
        return ret_array


class LowPassFilter(Interpolator):
    def __init__(self, fit_dates=None, fit_values=None, min_val=None, max_val=None, alpha=0.01):
        """ Low pass filter
        :param fit_dates: dates to fit
        :param fit_values: values to fit
        :param min_val: minimum value for interpolation.
        :param max_val: maximum value for interpolation.
        :param alpha: alpha value for the filter.
        """
        super().__init__(fit_dates, fit_values, min_val, max_val)
        self.alpha = alpha
        self.df = None
        if fit_dates is not None and fit_values is not None:
            self.set_fit_data(fit_dates, fit_values)

    def __name__(self):
        return f"LowPassFilter_{self.alpha}"

    def set_fit_data(self, fit_dates, fit_values):
        super().set_fit_data(fit_dates, fit_values)
        x_new_daily = pd.date_range(start=self.fit_dates.min(), end=self.fit_dates.max(), freq="1d",
                                    normalize=True)  # Assert daily frequency
        y_new = np.empty(x_new_daily.shape)
        y_new[:] = np.nan
        new_df = pd.DataFrame(y_new, index=x_new_daily)
        for date, value in zip(self.fit_dates, self.fit_values):
            new_df.loc[date] = value
        new_df = (new_df.interpolate(method="time", limit_area="inside")
                  .interpolate(method="bfill").interpolate(method="ffill"))
        new_df["mean"] = (new_df.iloc[:, 0].interpolate(method="time", limit_area="inside")
                          .interpolate(method="bfill").interpolate(method="ffill")
                          .ewm(alpha=self.alpha, adjust=False).mean())
        self.df = new_df

    def __call__(self, x_new):
        x_new = np.array(x_new)
        before = x_new < pd.to_datetime(self.fit_dates.min())
        after = x_new > pd.to_datetime(self.fit_dates.max())
        valid_range = ~before & ~after
        ret_array = np.zeros_like(x_new, dtype=float)
        ret_array[before] = self.fit_values[0]
        ret_array[after] = self.fit_values[-1]
        ret_array[valid_range] = self.df["mean"].loc[x_new[valid_range]].values
        return ret_array
