import datetime
import os
from pathlib import Path
from pdb import pm
from typing import List, NoReturn, Optional, Union

from loguru import logger
from scipy.datasets import download_all
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import create_database, database_exists, drop_database

from SBART import __version__ as SBART_version

resource_path = Path(__file__).parent.parent / "resources"
import numpy as np

from SBART.internals.db_tables import Base, GDAS_profile, Target


class DB_connection:
    def __init__(self, debug_mode=False):
        logger.debug("Launching new DB connection")

        """Setup the database engine and session maker. If the database does not exist: create it, alongside the tables"""
        url = "sqlite:///" + (resource_path / "internalSBART.db").as_posix()

        self.engine = create_engine(url, echo=False)
        # drop_database(url)

        # Base.metadata.drop_all(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

        if not database_exists(url):
            logger.info("Creating database")
            create_database(url)
        self.sessionmaker = sessionmaker(bind=self.engine)

    ###########################
    #      search data        #
    ###########################

    def get_GDAS_profile(self, gdas_filename: str):
        with self.sessionmaker() as session:
            chosen_target = (
                session.query(GDAS_profile)
                .filter_by(gdas_filename=gdas_filename)
                .first()
            )
        if chosen_target is None:
            raise FileNotFoundError("GDAS profile does not exist")

        data_path = resource_path / "atmosphere_profiles" / gdas_filename
        return np.loadtxt(data_path)

    def get_star_params(self, star_name: str):
        with self.sessionmaker() as session:
            chosen_target = session.query(Target).filter_by(name=star_name).first()

        if chosen_target is None:
            raise FileNotFoundError(
                f"Target {star_name} does not have cached information"
            )

        return chosen_target.params

    def add_new_star(self, star_name, pmra, pmdec, parallax):
        with self.sessionmaker() as session:
            try:
                new_prof = Target(
                    name=star_name,
                    pmra=pmra,
                    pmdec=pmdec,
                    parallax=parallax,
                )
                session.add(new_prof)
                session.commit()
            except Exception as e:
                session.rollback()
                raise e

    def add_new_profile(self, gdas_filename, instrument, data):
        with self.sessionmaker() as session:
            try:
                data_path = resource_path / "atmosphere_profiles" / gdas_filename
                np.savetxt(fname=data_path, X=data)

                new_prof = GDAS_profile(
                    gdas_filename=gdas_filename,
                    instrument=instrument,
                    download_date=datetime.datetime.now(),
                )

                logger.info(f"Added new GDAS profile {instrument}")
                session.add(new_prof)
                session.commit()
            except Exception as e:
                session.rollback()
                raise e

    def delete_all(self):
        logger.info("Deleting data from all Tables")
        Base.metadata.drop_all(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)


if __name__ == "__main__":
    conn = DB_connection()

    conn.add_new_star("testes", 0, 0, 0)
    conn.delete_all()
    print(conn.get_star_params("testes"))
