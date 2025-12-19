from sqlalchemy import create_engine

from hikerverseuniverse import config
from hikerverseuniverse.database import Base

DB_URL = f"mysql+pymysql://{config.POSTGRES_USER}:{config.POSTGRES_PASSWORD}@{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"

engine = create_engine(str(DB_URL), future=True, echo=True)

from models import Celestial

Base.metadata.drop_all(engine, checkfirst=True)
Base.metadata.create_all(engine)
