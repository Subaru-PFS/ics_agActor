import numpy


# sines & cosines of 30, 90, 150, 210, 270, & 330 deg
_SIN60 = numpy.array([0.5, 1, 0.5, -0.5, -1, -0.5])
_COS60 = numpy.array([0.8660254037844386, 0, -0.8660254037844386, -0.8660254037844386, 0, 0.8660254037844386])

# camera 1 at 9 o'clock position (241.292, 0) mm
# camera 2 at 7 o'clock position (120.646, 208.965) mm
# camera 3 at 5 o'clock position (-120.646, 208.965) mm
# camera 4 at 3 o'clock position (-241.292, 0) mm
# camera 5 at 1 o'clock position (-120.646, -208.965) mm
# camera 6 at 11 o'clock position (120.646, -208.965) mm
# (ref. morimoto-san's memo_AGcamera_20171023.pdf)


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
    a = numpy.mod(1 - icam, 6).astype(numpy.intc)
    sin_a = _SIN60[a]
    cos_a = _COS60[a]
    x = (x_det - 511.5) * p
    y = - (y_det - 511.5) * p + r
    x_fp = cos_a * x + sin_a * y
    y_fp = - sin_a * x + cos_a * y
    return x_fp, y_fp
