
from sqlalchemy import select, String, DECIMAL, Integer
from sqlalchemy.orm import Mapped, mapped_column, Session

from hikerverseuniverse.database import Base


class Celestial(Base):
    __tablename__ = "celestials"
    id: Mapped[int] = mapped_column(Integer,
                                    primary_key=True, index=True, autoincrement=True)

    celestial_id: Mapped[int] = mapped_column()
    x: Mapped[float] = mapped_column(DECIMAL(25, 10))
    y: Mapped[float] = mapped_column(DECIMAL(25, 10))
    z: Mapped[float] = mapped_column(DECIMAL(25, 10))
    radius: Mapped[float] = mapped_column(DECIMAL(50, 10))
    temperature: Mapped[float] = mapped_column(DECIMAL(25, 10))
    mass: Mapped[float] = mapped_column(DECIMAL(25, 10))
    abs_mag: Mapped[float] = mapped_column(DECIMAL(5, 2))
    luminosity: Mapped[float] = mapped_column(DECIMAL(60, 0))
    spec: Mapped[str] = mapped_column(String(15))
    lum: Mapped[float] = mapped_column(DECIMAL(25, 10))

    @classmethod
    def find_by_celestial_id(cls, db: Session, celestial_id: int):
        query = select(cls).where(cls.celestial_id == celestial_id)
        result = db.execute(query)
        return result.scalars().first()


