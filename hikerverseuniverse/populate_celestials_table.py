import csv
import os

from sqlalchemy import text

from hikerservespacecraft.library.sensor_physics import abs_mag_2_luminosity_in_w, bv_to_temp_kelvin, star_radius_in_m
from hikerverseuniverse.dependencies import get_db


def populate():
    add_celestial = text("INSERT INTO celestials " \
                         "(celestial_id, x,  y,  z,  radius, temperature, mass, abs_mag, luminosity, spec, lum) " \
                         "VALUES (:celestial_id, :x, :y, :z, :radius, :temperature, :mass, :abs_mag, :luminosity, :spec, :lum)")

    dir_path = os.path.dirname(os.path.realpath(__file__))

    with open(dir_path + os.sep + '../data/hygdata_v3_mod.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        with get_db() as db:
            row_counter = 0
            for row in csv_reader:
                if row_counter != 0:
                    id = int(row[0])
                    app_mag = float(row[10])
                    abs_mag = float(row[11])
                    spec = row[12]
                    ci = row[13]

                    x = float(row[14])
                    y = float(row[15])
                    z = float(row[16])
                    luminosity_in_watts = abs_mag_2_luminosity_in_w(float(abs_mag))
                    spec = spec.replace(" ", "")
                    lum = float(row[24])

                    try:
                        ci_float = float(ci)
                        temp_in_kelvin = bv_to_temp_kelvin(ci_float)
                    except ValueError:
                        temp_in_kelvin = -1

                    radius = star_radius_in_m(temp_in_kelvin, luminosity_in_watts)

                    data_row = {
                        "celestial_id": id,
                        "x": x,
                        "y": y,
                        "z": z,
                        "radius": radius,
                        "temperature": temp_in_kelvin,
                        "mass": "-1",
                        "abs_mag": abs_mag,
                        "luminosity": luminosity_in_watts,
                        "spec": spec,
                        "lum": lum
                    }
                    db.execute(add_celestial, data_row)

                row_counter += 1


            db.commit()


if __name__ == '__main__':
    populate()
