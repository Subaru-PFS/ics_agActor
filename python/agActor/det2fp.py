import numpy


# sines & cosines of 0, 60, 120, 180, 240, & 300 deg
_SIN60 = numpy.array([0, 0.8660254037844386, 0.8660254037844386, 0, -0.8660254037844386, -0.8660254037844386])
_COS60 = numpy.array([1, 0.5, -0.5, -1, -0.5, 0.5])

# camera 1 at 6 o'clock position (0, 241.292) mm
# camera 2 at 4 o'clock position (-121, 209) mm
# camera 3 at 2 o'clock position (-121, -209) mm
# camera 4 at 12 o'clock position (0, -241.292) mm
# camera 5 at 10 o'clock position (121, -209) mm
# camera 6 at 8 o'clock position (121, 209) mm
# (ref. kawanomoto-san's autoguider.pdf)


def det2fp(icam, x_det, y_det):
    """
    Convert detector coordinates to focal plane coordinates.

    Convert Cartesian coordinates of points on one of the detectors in the
    detector coordinate system to those on the focal plane in the focal plane
    coordinate system.

    Parameters
    ----------
    icam : array_like
        The detector identifiers ([0, 5])
    x_det : array_like
        The Cartesian coordinates x's of points on the specified detector in
        the detector coordinate system (pix)
    y_det : array_like
        The Cartesian coordinates y's of points on the specified detector in
        the detector coordinate system (pix)

    Returns
    -------
    2-tuple of array_likes
        The Cartesian coordinates x's and y's of points on the focal plane in
        the focal plane coordinate system (mm)
    """

    p = 0.013  # mm
    r = 241.292  # mm
    a = numpy.mod(3 - icam, 6).astype(numpy.intc)
    sin_a = _SIN60[a]
    cos_a = _COS60[a]
    x = (x_det - 511.5) * p
    y = (y_det - 511.5) * p - r
    x_fp = cos_a * x + sin_a * y
    y_fp = - sin_a * x + cos_a * y
    return x_fp, y_fp
