from peewee import SqliteDatabase

from ...core.conf import settings

db = SqliteDatabase(settings.DATABASE_PATH)
