from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        String, Table, create_engine, select)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Target(Base):
    __tablename__ = "target"
    id = Column(Integer, primary_key=True)

    name = Column(String(50))
    pmra = Column(Float)
    pmdec = Column(Float)
    parallax = Column(Float)

    @property
    def params(self):
        return {"pmra": self.pmra, "pmdec": self.pmdec, "parallax": self.parallax}


class GDAS_profile(Base):
    __tablename__ = "GDAS"
    id = Column(Integer, primary_key=True)

    gdas_filename = Column(String(60), unique=True)
    instrument = Column(String(50))
    download_date = Column(DateTime)

    def __repr__(self):
        return f"Target({self.name=}"

    def __str__(self):
        string = f"GDAS profile from {self.instrument}"
        return string
