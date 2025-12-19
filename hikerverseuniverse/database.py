from sqlalchemy import select, create_engine

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session, scoped_session
from hikerverseuniverse import config

DB_URL = f"mysql+pymysql://{config.POSTGRES_USER}:{config.POSTGRES_PASSWORD}@{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"

engine = create_engine(str(DB_URL), future=True, echo=False)

SessionFactory = scoped_session(sessionmaker(engine, autoflush=False, expire_on_commit=False))

class Base(DeclarativeBase):

    def save(self, db: Session):
        """
        :param db:
        :return:
        """
        try:
            db.add(self)
            return db.commit()
        except SQLAlchemyError as ex:
            print(ex)
            raise Exception() from ex

    @classmethod
    def find_by_id(cls, db: Session, id: str):
        query = select(cls).where(cls.id == id)
        result = db.execute(query)
        return result.scalars().first()
