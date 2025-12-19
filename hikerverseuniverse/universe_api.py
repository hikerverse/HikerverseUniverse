import math
from decimal import Decimal
from typing import List, Dict, Any

import pymysql
from sqlalchemy import text

from hikerverseuniverse.dependencies import get_db
from hikerverseuniverse.library.constants import pc

Ly2m = 9.4607e+15  # meters in a light year
Pc2m = 3.086e+16  # meters in a parsec


def get_all_celestials() -> Dict[str, Any]:
    try:
        with get_db() as db:
            q = text("SELECT * FROM celestials")
            result = db.execute(q)
            res_ = result.fetchall()
            return {"success": True, "data": res_}
    except pymysql.Error as err:
        print(f"Error: {err}")
        return {"success": False, "data": []}


def all_celestials_within_distance(coordinates: List[float], distance: float):
    if len(coordinates) < 3:
        raise ValueError("Coordinate list should contain at least three elements")
    try:
        with get_db() as db:
            q = text("SELECT * FROM celestials t WHERE SQRT(POWER(t.x - :x0,2)+POWER(t.y - :y0,2)+POWER(t.z - :z0,2)) < :dist")
            result = db.execute(q, {"x0": coordinates[0], "y0": coordinates[1], "z0": coordinates[2], "dist": distance})
            res = result.fetchall()
            return res
    except pymysql.Error as err:
        print(f"Error: {err}")


def query(self, coordinates: List[float], distance: float):
    try:
        cur = self.db.cursor()
        query = "SELECT * FROM celestials t WHERE " \
                "t.x between (%s - %s) AND (%s + %s) AND " \
                "t.y between (%s - %s) AND (%s + %s) AND " \
                "t.z between (%s - %s) AND (%s + %s)"

        result = cur.execute(query, (
            coordinates[0], distance, coordinates[0], distance,
            coordinates[1], distance, coordinates[1], distance,
            coordinates[2], distance, coordinates[2], distance))

        res = result.fetchall()

        return res

    except pymysql.Error as err:
        print(f"Error: {err}")

    finally:
        self.db.close()


def query_by_luminosity(self, coordinates: List[Decimal], detection_threshold_watts: float):
    pc_dec = Decimal(pc)
    try:
        with get_db() as db:

            q = text(
                "SELECT * FROM celestials t WHERE :detection_threshold_watts < "
                "(t.luminosity / (4 * PI() * "
                "POWER(SQRT( (:Pc2m * t.x - :Pc2m * :x0)*(:Pc2m * t.x - :Pc2m * :x0) + "
                "(:Pc2m * t.y - :Pc2m * :y0)*(:Pc2m * t.y - :Pc2m * :y0) + "
                "(:Pc2m * t.z - :Pc2m * :z0)*(:Pc2m * t.z - :Pc2m * :z0)), 2 )))"
            )

            result = db.execute(q, {
                "Pc2m": Pc2m,
                "detection_threshold_watts": detection_threshold_watts,
                "x0": coordinates[0],
                "y0": coordinates[1],
                "z0": coordinates[2]
            })

            res = result.fetchall()

            counter = 0
            for r in res:
                counter += 1
                if counter < 10:
                    x_diff = pc_dec * (coordinates[0] - r[2])
                    y_diff = pc_dec * (coordinates[1] - r[3])
                    z_diff = pc_dec * (coordinates[2] - r[4])
                    dd = math.sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff)

                    rrr = r[9] / Decimal((4 * math.pi * math.pow(dd, 2)))
                    print(f"{dd / pc}, {rrr}, {r[9]}")

                else:
                    break

            return res

    except pymysql.Error as err:
        print(f"Error: {err}")


if __name__ == '__main__':
    coords = [Decimal(0.0), Decimal(0.0), Decimal(0.0)]
    detection_threshold = 1e-12

    res = query_by_luminosity(None, coords, detection_threshold)
    print(len(res))

    sl = 3.86e26  # Watts
    sun_distance_pc = math.sqrt(sl / (4 * math.pi * detection_threshold)) / Pc2m
    print(f"Sun would be detectable out to {sun_distance_pc} parsecs")

