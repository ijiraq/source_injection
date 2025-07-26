# This file is part of source_injection.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

__all__ = ["generate_sso_injection_catalog"]

import itertools
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from astropy.table import Table, hstack, vstack
from astropy.time import Time
from astropy.coordinates import SkyCoord, GCRS
from astropy import units
from scipy.stats import qmc

from lsst.afw.geom import SkyWcs

from . import sso

# to get area of ecliptic we take a sphere less two caps of 70 degrees
AREA_OF_ECLIPTIC = (180/np.pi)**2*(4*np.pi -
                                   2*(4*np.pi*(np.sin(np.radians(70)/2)**2)))


DEFAULT_ELEMENT_LIMITS = {
    'a_limits': (30, 250),
    'e_limits': (0, 1E-5),
    'inc_limits': (0, np.pi),
    'Omega_limits': (0, 2 * np.pi),
    'omega_limits': (0, 2 * np.pi),
    'M_limits': (0, 2 * np.pi),
}

DEFAULT_PARAMETERS = {
    'mjd': Time("J2000.0").mjd,
    'density': 2000,  # /sq.deg which is about 50 sources per HSC detector
    'fov': 180,  # catalog FOV in degrees 
    'mag_lim': (21, 28.0),  # Appropriate for 4 hours on Subaru HSC
}


def generate_sso_injection_catalog(
        ra_centre: float,
        dec_centre: float,
        day_obs: int,
        mag_lim: Sequence[float] | None = None,
        wcs: SkyWcs = None,
        fov: float | None = None,
        density: int | None = None,
        number: int = 1,
        seed: Any = 123456789,
        log_level: int = logging.INFO,
        **kwargs: Any,
) -> Table:
    """Generate a synthetic SSO source injection catalog.

    This function generates synthetic orbits.  The orbits are within the given
    keplerian orbit bounds and within the given ra/dec limits at the
    given observational epoch.  The catalog is returned as an astropy Table.

    This source injection catalog is orbit based and is designed to be used
    with the `sso_injection` plugin.

    Orbits are generated using the quasi-random Halton sequence.

    **kwargs are used to specify the bounds of the keplerian orbital elements.
    Other arbitrary parameters that should be included in the output catalog
    are also passed in as kwargs.  The output catalog will contain a row for
    each unique combination of input parameters that are not orbital elements.

    H magnitudes are also generated using the same sequence.

    Use the Halton sequence default seed if not given.

    Generates a unique injection ID for each source. The injection ID
    encodes two pieces of information: the unique source identification number
    and the version number of the source as specified by the ``number``
    parameter. To achieve this, the unique source ID number is multiplied by
    `10**n` such that the sum of the multiplied source ID number with the
    unique repeated version number will always be unique. For example, an
    injection catalog with `number = 3` versions of each source will have
    injection IDs: 0, 1, 2, 10, 11, 12, 20, 21, 22, etc. If `number = 20`, then
    the injection IDs will be: 0, 1, 2, ..., 17, 18, 19, 100, 101, 102, etc.
    If `number = 1` (default) then the injection ID will be a simple sequential
    list of integers.

    Parameters
    ----------
    ra_centre : float
        The right ascension of the centre of the catalog in degrees.
    dec_centre : float
        The declination of the centre of the catalog in degrees.
    fov: float
        The diameter of the fov of the catalog in degrees.
    day_obs : int 
        The day_obs (YYYYMMDD) that this orbit files epoch is set to. 
        (M at this epoch puts the source in the heliocentric RA/DEC box set 
        by ra_lim,dec_lim at that epoch and this is added as a dimension)
    mag_lim : `Sequence` [`float`], optional
        The magnitude limits of the catalog in magnitudes.  The catalog has H
        magnitudes assigned based on these magnitude limits being met at day_obs
    number : `int`, optional
        The number of times to generate each unique combination of input
        parameters. The default is 1 (i.e., no repeats). This will be ignored
        if ``density`` is specified.
    density : `int` | 2000,
        The desired source density in sources per square degree. If given, the
        ``number`` parameter will be ignored. Instead, the number of unique
        parameter combination generations will be calculated to achieve the
        desired density. The default is `None` (i.e., no density calculation).
    seed : `Any`, optional | None
        The seed to use for the Halton sequence. If not given or ``None``
        (default), the seed will be set using the product of the right
        ascension and declination limit ranges.
    log_level : `int`, optional
        The log level to use for logging.
    **kwargs : `Any`
        The input parameters used to generate the orbital elements are popped
        off kqargs, any parameters remaining are added to the output catalog.
        Each remaining parameter key will be a column name in the catalog.
        The values are the unique values for that parameter. The output catalog
        will contain a row for each unique combination of input parameters and
        be generated the number of times specified by ``number``.

    Returns
    -------
    table : `astropy.table.Table`
        The fully populated synthetic source orbit catalog. The catalog
        will contain an automatically generated ``injection_id`` column that
        is unique for each source. The injection ID encodes two pieces of
        information: the unique source identification number and the repeated
        version number of the source as defined by the ``number`` parameter.
    """
    # Instantiate logger.
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    mag_lim = mag_lim is None and DEFAULT_PARAMETERS['mag_lim'] or mag_lim
    density = density is None and DEFAULT_PARAMETERS['density'] or density
    fov = fov is None and DEFAULT_PARAMETERS['fov'] or fov


    elements = ['a', 'e', 'inc', 'Omega', 'omega', 'M']
    limits = [kwargs.pop(f"{f}_limits",
                         DEFAULT_ELEMENT_LIMITS[f"{f}_limits"])
              for f in elements]

    limits.append(mag_lim)
    elements.append('mag')
    lower_limits, upper_limits = np.array(limits).T

    # Parse optional keyword input parameters, after having used up any
    # element based ones.
    values: list[Any] = [np.atleast_1d(x) for x in kwargs.values()]

    # Automatically calculate the number of generations if density is given.
    if density:
        area = 4*np.pi * np.sin(np.radians(fov)/4)**2
        area = area*(180/np.pi)**2
        rows = list(itertools.product(*values))
        number = np.round(density * area / len(rows)).astype(int)
        if number > 0:
            logger.info(
                (f"Setting number of generations to {number}, "
                 f"equivalent to {number/area:.1f} sources per square degree.")
            )
        else:
            logger.warning(
                ("Requested source density would require number < 1; "
                 "setting number = 1.")
            )
            number = 1

    # Generate the fully expanded parameter table.
    values.append(range(number))
    keys = list(kwargs.keys())
    keys.append("version_id")
    param_table = Table(rows=list(itertools.product(*values)), names=keys)
    logger.info(f"Area of ecliptic {AREA_OF_ECLIPTIC}")
    logger.info(f"Aera of patch {area}")
    number_per_loop = len(param_table)*AREA_OF_ECLIPTIC/area
    number_per_loop = int(min(number_per_loop, 1E8))
    logger.info(f"Generating {number_per_loop} orbits per iteration.")
    _seed = np.random.default_rng(int(seed))
    sampler = qmc.Halton(d=len(elements), seed=_seed)
    num_of_workers = number_per_loop > 1E3 and -1 or 1
    source_table = None
    while True:
        logger.info((f"Creating {number_per_loop} orbits using "
                     f"{num_of_workers} workers"))
        sample = sampler.random(n=number_per_loop, workers=num_of_workers)
        # generate a distribution of orbital elements
        orbits = Table(qmc.scale(sample, lower_limits, upper_limits),
                       names=elements, meta={"day_obs": Time.strptime(str(day_obs),
                                                                      '%Y%m%d').isot,
                                             "SSO": True})
        coords, vxyz = sso.kepToHelioCartSkyCoord(orbits)
        coords = coords.transform_to('icrs')
        orbits['r'] = coords.distance.to('au')
        reference_frame = GCRS(obstime=Time(orbits.meta["day_obs"]))
        coordinates = coords.transform_to(reference_frame)
        orbits['delta'] = coordinates.distance.to('au')
        orbits['ra'] = coordinates.ra.deg
        orbits['dec'] = coordinates.dec.deg
        orbits['H'] = orbits['mag'] - sso.apparent_magnitude(orbits['r'],
                                                             orbits['delta'],
                                                             1,
                                                             0)
        orbits['source_type'] = 'Star'
        # select orbits whose current position is within the desired FOV
        field_centre = SkyCoord(ra=ra_centre, dec=dec_centre, unit='deg')
        orbits = orbits[coords.separation(field_centre) < (fov/2)*units.degree]

        # build the full table of orbits and loop if we need more orbits
        # after having limited to those in the fov to reach the desired number
        if source_table is None:
            source_table = orbits
        else:
            source_table = vstack([source_table, orbits])
        if len(source_table) >= len(param_table):
            break

    # truncate the source table so length matches param_table
    source_table = source_table[:len(param_table)]

    # Generate the unique injection ID and construct the final table.
    source_id = np.concatenate([([i] * number) for i in range(int(len(param_table) / number))])
    injection_id = param_table["version_id"] + source_id * int(10 ** np.ceil(np.log10(number)))
    injection_id.name = "injection_id"
    source_table = hstack([injection_id, source_table, param_table])
    source_table.remove_column("version_id")
    
    # Final logger report and return.
    if number == 1:
        extra_info = f"{len(source_table)} unique sources."
    else:
        num_combinations = int(len(source_table) / number)
        grammar = "combination" if num_combinations == 1 else "combinations"
        extra_info = f"{len(source_table)} sources: {num_combinations} {grammar} repeated {number} times."
    logger.info("Generated an injection catalog containing %s", extra_info)

    source_table.meta['day_obs'] = Time.strptime(str(day_obs), '%Y%m%d').isot
    source_table.meta['fov'] = fov
    source_table.meta['ra_centre'] = ra_centre
    source_table.meta['dec_centre'] = dec_centre
    source_table.meta['density'] = density
    source_table.meta['mag_lim'] = mag_lim
    source_table.meta['number'] = number
    source_table.meta['seed'] = seed

    return source_table

