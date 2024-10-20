from astropy import units
from astropy.coordinates import SkyCoord, GCRS
from astropy.coordinates import EarthLocation
from astropy.coordinates import solar_system_ephemeris
from astropy.table import Table
from lsst.afw.image import VisitInfo
import numpy as np

BOWELL_G = -0.12
NEWTON_ITERS = 20


def apparent_magnitude(r: np.ndarray[float],
                       delta: np.ndarray[float],
                       robs: np.ndarray[float] | float, h: float, g: float = BOWELL_G):
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
    cos_alpha[cos_alpha > 1] = 1
    alpha = np.arccos(cos_alpha)
    phi1 = np.exp(-3.33 * (np.tan(alpha / 2.0)) ** 0.63)
    phi2 = np.exp(-1.87 * (np.tan(alpha / 2.0)) ** 1.22)
    mag = 5.0 * np.log10(r * delta) + h - 2.5 * np.log10((1.0 - g) * phi1 + g * phi2)
    return mag


def keplerian_to_cartesian(**orbits: dict[str: np.ndarray]) -> np.ndarray:
    """
    This routine transforms keplerian elements into cartesian elements
    (only positions are computed).

    Parameters
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

    return p


def compute_E(e: np.array, M: np.array) -> np.array:
    """
    Compute the eccentric anomaly E from the mean anomaly M and the eccentricity.

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


def get_particle_coordinates(orbits) -> SkyCoord:
    """
    Given a set of keplarian orbital elements with M propagated to the epoch
    """
    xyz = keplerian_to_cartesian(**orbits['a', 'e', 'inc', 'Omega', 'omega', 'M'])

    coords = SkyCoord(x=xyz[0], y=xyz[1], z=xyz[2],
                      unit='au', obstime=orbits.meta['obstime'],
                      frame='heliocentrictrueecliptic',
                      representation_type='cartesian')
    return coords


def propagate_orbits(orbits, obstime) -> Table:
    """
    Propogate the given table of orbits from orbit.meta['epoch'] to 'obstime'
    """
    delta_time = obstime - orbits.meta['obstime']

    orbits['M'] = orbits['M'] + delta_time * 2 * np.pi / (orbits['a'] ** (3 / 2) * units.year)
    orbits.meta['obstime'] = obstime
    return orbits


def get_reference_frame(visitInfo) -> GCRS:
    """
    Compute a GCRS reference frame at location and time of Visit info.
    """
    obstime = visitInfo.date.toAstropy()
    location = EarthLocation.from_geodetic(lon=visitInfo.getLongitude().asDegrees(),
                                           lat=visitInfo.getLatitude().asDegrees(),
                                           height=visitInfo.getHeight().asDegrees())
    obsgeoloc, obsgeovel = location.get_gcrs_posvel(obstime)
    with solar_system_ephemeris.set('builtin'):
        reference_frame = GCRS(obstime=obstime,
                               obsgeoloc=obsgeoloc,
                               obsgeovel=obsgeovel)
    return reference_frame


def propagate_injection_catalog(orbits: Table, visitInfo: VisitInfo) -> Table:
    """
    Compute the heliocentric ecliptic coordinates of the orbits at the time
    of the visit described by visitInfo
    """
    reference_frame = get_reference_frame(visitInfo)
    orbits = propagate_orbits(orbits, reference_frame.obstime)
    coordinates = get_particle_coordinates(orbits)
    orbits['r'] = np.sqrt(coordinates.x * coordinates.x +
                          coordinates.y * coordinates.y +
                          coordinates.z * coordinates.z)
    coordinates = coordinates.transform_to(reference_frame)
    orbits['delta'] = coordinates.distance.to('au')
    orbits['RA2000'] = coordinates.ra.to('degree')
    orbits['DEC2000'] = coordinates.dec.to('degree')
    orbits['mag'] = apparent_magnitude(orbits['r'],
                                       orbits['delta'],
                                       1,
                                       orbits['H'])
    orbits.meta['MJD'] = reference_frame.obstime.mjd
    return orbits