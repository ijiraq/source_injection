from astropy import units
from astropy.coordinates import SkyCoord, GCRS
from astropy.coordinates import EarthLocation
from astropy.coordinates import solar_system_ephemeris
from astropy.table import Table
from astropy.time import Time
from lsst.afw.image import ExposureF
import numpy as np

BOWELL_G = -0.12
NEWTON_ITERS = 20


def apparent_magnitude(r: np.ndarray[float] | float,
                       delta: np.ndarray[float] | float,
                       robs: np.ndarray[float] | float, 
                       h: np.ndarray[float] | float, 
                       g: float = BOWELL_G) -> np.ndarray[float] | float:
    """
    Compute the apparent magnitude given the observing circumstances and h

    Args:
        r (float): Sun-object distance
        delta (float): Observer-object distance
        robs (float): Sun-Observer distance
        h (float): absolute magnitude
        g (float): phase angle coefficient (following Bowell et al. 1989)
    """
    denom = 2 * r * delta
    cos_alpha = (-robs ** 2 + delta ** 2 + r ** 2) / denom
    if isinstance(cos_alpha, np.ndarray):
        cos_alpha[cos_alpha > 1] = 1
    elif cos_alpha > 1:
        cos_alpha = 1
    alpha = np.arccos(cos_alpha)
    phi1 = np.exp(-3.33 * (np.tan(alpha / 2.0)) ** 0.63)
    phi2 = np.exp(-1.87 * (np.tan(alpha / 2.0)) ** 1.22)
    mag = 5.0 * np.log10(r * delta) + h - 2.5 * np.log10((1.0 - g) * phi1 + g * phi2)
    if isinstance(r, float) and isinstance(mag, np.ndarray):
        mag = mag[0]
    return mag


def keplerian_to_cartesian(**orbits: dict[str: np.ndarray]) -> [np.ndarray, np.ndarray]:
    """
    Compute the cartesian position and velocity elements given
    a set of keplerian elements

    Parameters
    ----------
    orbits: dict The dictionary orbits contains the following keys:
        a: semi-major axis (AU)
        e: eccentricity
        inc: inclination (radians)
        Omega: longitude of the ascending node (radians)
        omega: argument of the pericentre (radians)
        M: mean anomaly (radians)

    """
    M = orbits["M"] % (2 * np.pi)
    signe = np.sign(orbits['a'])
    cos_i = np.cos(orbits['inc'])
    sin_i = np.sqrt(1 - cos_i ** 2)
    delau1 = M
    delau2 = np.cos(orbits['omega'])
    delau3 = np.sin(orbits['omega'])
    delau4 = np.cos(orbits['Omega'])
    delau5 = np.sin(orbits['Omega'])
    delau6 = signe * np.sqrt(orbits['a'] * signe)
    delau7 = np.abs(delau6) * np.sqrt((1 - orbits['e'] ** 2) * signe)

    E = compute_E(orbits['e'], delau1)

    cos_E = np.cos(E)
    sin_E = np.sin(E)
    q_vec = np.zeros_like(cos_E, shape=(2, cos_E.shape[0]))
    q_vec[0, :] = (cos_E - orbits['e']) * (delau6 ** 2)
    q_vec[1, :] = delau7 * delau6 * sin_E

    p_vec = np.zeros_like(cos_E, shape=(2, cos_E.shape[0]))

    tmp = 1/(delau6*(1 - orbits['e']*cos_E))
    p_vec[0, :] = -sin_E*tmp
    p_vec[1, :] = delau7*cos_E*tmp/delau6

    mat = np.zeros_like(cos_i, shape=(3, 2, cos_i.shape[0]))
    d53 = delau5 * delau3
    d42 = delau4 * delau2
    d52 = delau5 * delau2
    d43 = delau4 * delau3
    mat[0, 0, :] = d42 - cos_i * d53
    mat[0, 1, :] = -d43 - cos_i * d52
    mat[1, 0, :] = d52 + cos_i * d43
    mat[1, 1, :] = -d53 + cos_i * d42
    mat[2, 0, :] = sin_i * delau3
    mat[2, 1, :] = sin_i * delau2

    # Cartesian coordinates
    p = (mat * q_vec).sum(axis=1)
    v = (mat * p_vec).sum(axis=1)

    return [p, v]


def compute_E(e: np.array, M: np.array) -> np.array:
    """
    Compute the eccentric anomaly E from mean anomaly M and eccentricity.

    e: eccentricity
    M: mean anomaly (radians)

    Uses Newton-Raphson method to solve Kepler's equation.
    """
    E = M + 0.85 * np.sign(np.sin(M)) * e
    i = 0
    while True:
        sin_e = e * np.sin(E)
        f = E - sin_e - M
        iterate = np.fabs(f) > 1e-14
        if iterate.sum() > 0:
            cos_e = e[iterate] * np.cos(E[iterate])
            fp = 1 - cos_e
            fpp = sin_e[iterate]
            fppp = cos_e
            de = -f[iterate] / fp
            de = -f[iterate] / (fp + de * fpp / 2)
            de = -f[iterate] / (fp + de * fpp / 2 + de * de * fppp / 6)
            E[iterate] = E[iterate] + de
            i += 1
            if i < NEWTON_ITERS:
                continue
            raise ValueError('No convergence after {i} iterations')
        break
    return E


def kepToHelioCartSkyCoord(orbits: Table) -> [SkyCoord, np.ndarray]:
    """
    Provide SkyCoord object for a Table of keplarian orbital elements.
    """
    xyz, v_xyz = keplerian_to_cartesian(**orbits['a', 'e', 'inc', 'Omega', 'omega', 'M'])
    obstime = Time(str(orbits.meta['day_obs']))
    pos = SkyCoord(x=xyz[0], y=xyz[1], z=xyz[2],
                   unit='au', obstime=obstime,
                   frame='heliocentrictrueecliptic',
                   representation_type='cartesian')
    return pos, v_xyz * units.au/units.year


def propagate_orbits(orbits, mjd) -> Table:
    """
    Propagate the given table of orbits from orbit.meta['day_obs'] is the isot

    The orbits database must have orbits.meta['day_obs'] of the 'M' given in the table.
    """
    orbtime = Time(str(orbits.meta['day_obs']))
    obstime = Time(mjd, format='mjd')
    delta_time = obstime - orbtime
    orbits['M'] = orbits['M'] + delta_time * 2 * np.pi / (orbits['a'] ** (3 / 2) * units.year)
    orbits.meta['day_obs'] = obstime.isot
    return orbits


def get_reference_frame(visitInfo) -> GCRS:
    """
    Compute a GCRS reference frame at location and time of Visit info.
    """
    obstime = visitInfo.date.toAstropy()
    observatory = visitInfo.getObservatory()
    location = EarthLocation.from_geodetic(lon=observatory.getLongitude().asDegrees(),
                                           lat=observatory.getLatitude().asDegrees(),
                                           height=observatory.getElevation())
    obsgeoloc, obsgeovel = location.get_gcrs_posvel(obstime)
    with solar_system_ephemeris.set('builtin'):
        reference_frame = GCRS(obstime=obstime,
                               obsgeoloc=obsgeoloc,
                               obsgeovel=obsgeovel)
    return reference_frame


def propagate_injection_catalog(orbits: Table,
                                inputExposure: ExposureF) -> Table:
    """
    Compute the heliocentric ecliptic coordinates of the orbits at the time
    of the visit described by visitInfo

    if number_of_steps is greater than 1 we must vstack a set of rows for
    each additional step
    """
    reference_frame = get_reference_frame(inputExposure.visitInfo)
    pixel_scale = inputExposure.wcs.getPixelScale().asArcseconds() * units.arcsec
    exposure_time = inputExposure.visitInfo.exposureTime * units.second
    orbits = propagate_orbits(orbits, reference_frame.obstime.mjd)
    coordinates, v_xyz = kepToHelioCartSkyCoord(orbits)
    dt = 24 * units.hour
    orbits = propagate_orbits(orbits,
                              reference_frame.obstime.mjd+dt.to('day').value)
    coordinates2, v_xyz = kepToHelioCartSkyCoord(orbits)
    # print(coordinates.transform_to('icrs')[0])
    # print(v_xyz[:,0])
    orbits['r'] = np.sqrt(coordinates.x * coordinates.x +
                          coordinates.y * coordinates.y +
                          coordinates.z * coordinates.z)
    
    # coordinates2 = SkyCoord(x=coordinates.x + v_xyz[0] * dt,
    #                        y=coordinates.y + v_xyz[1] * dt,
    #                        z=coordinates.z + v_xyz[2] * dt,
    #                        obstime=reference_frame.obstime,
    #                        frame='heliocentrictrueecliptic',
    #                        representation_type='cartesian')
    # print(coordinates2.transform_to('icrs')[0])
    # print(coordinates.separation(coordinates2)[0].to('arcsec')/dt)
    # raise IOError("JUNK")
    coordinates2 = coordinates2.transform_to(reference_frame)
    coordinates = coordinates.transform_to(reference_frame)
    rate = coordinates.separation(coordinates2)/dt
    position_angle = coordinates.position_angle(coordinates2)
    orbits['delta'] = coordinates.distance.to('au')
    orbits['rate'] = rate.to('arcsec/hour')
    orbits['angle'] = position_angle.to('degree')

    orbits['ra'] = coordinates.ra.to('degree')
    orbits['dec'] = coordinates.dec.to('degree')
    orbits['mag'] = apparent_magnitude(orbits['r'],
                                       orbits['delta'],
                                       1,
                                       orbits['H'])
    orbits.meta['day_obs'] = reference_frame.obstime.isot

    return orbits

def tailing():
    """
    Coded this up but decided to not use.
    """
    number_of_steps = np.ceil((coordinates.separation(coordinates2)/pixel_scale).value).astype('int')
    
    new_orbits = dict([(k, []) for k in orbits.colnames])
    for idx, steps in enumerate(number_of_steps):
        orbit = orbits[idx]
        coordinate = coordinates[idx]
        rate = orbit['rate']*orbits['rate'].unit
        angle = orbit['angle']*orbits['angle'].unit
        motion_per_step = rate * exposure_time / steps
        for step in range(steps):
            for col in orbits.colnames:
                new_orbits[col].append(orbit[col])
            coordinate = coordinate.directional_offset_by(angle, motion_per_step*(step+0.5))
            coordinate = coordinate.transform_to(reference_frame)
            new_orbits['ra'][-1] = coordinate.ra.to('degree')
            new_orbits['dec'][-1] = coordinate.dec.to('degree')
            new_orbits['mag'][-1] = apparent_magnitude(orbit['r'],
                                                       orbit['delta'],
                                                       1,
                                                       orbit['H']) + 2.5 * np.log10(steps)
    new_orbits = Table(new_orbits, meta=orbits.meta)
    new_orbits.meta['day_obs'] = reference_frame.obstime.isot

    return new_orbits
