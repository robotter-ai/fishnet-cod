from peewee import Model

from .common import db


class TimeseriesDBModel(Model):
    """
    A simple database model for storing raw timeseries data.
    """



    class Meta:
        database = db