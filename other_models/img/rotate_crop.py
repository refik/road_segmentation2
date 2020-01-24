import collections
import numpy as np

_Point = collections.namedtuple('Point', ['x', 'y'])


class Point(_Point):
    def __add__(self, p):
        return Point(self.x + p.x, self.y + p.y)

    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def recenter(self, old_center, new_center):
        return self + (new_center - old_center)

    def rotate(self, center, angle):
        # angle should be in radians
        x = np.cos(angle) * (self.x - center.x) - np.sin(angle) * (self.y - center.y) + center.x
        y = np.sin(angle) * (self.x - center.x) + np.cos(angle) * (self.y - center.y) + center.y
        return Point(x, y)


def getCenter(im):
    return Point(*(d / 2 for d in im.size))


Bound = collections.namedtuple('Bound', ('left', 'upper', 'right', 'lower'))


def getBounds(points):
    xs, ys = zip(*points)
    # left, upper, right, lower using the usual image coordinate system
    # where top-left of the image is 0, 0
    return Bound(min(xs), min(ys), max(xs), max(ys))


def getBoundsCenter(bounds):
    return Point(
        (bounds.right - bounds.left) / 2 + bounds.left,
        (bounds.lower - bounds.upper) / 2 + bounds.upper
    )


def roundint(values):
    return tuple(int(round(v)) for v in values)


def getRotatedRectanglePoints(angle, base_point, height, width):
    # base_point is the upper left (ul)
    ur = Point(
        width * np.cos(angle),
        -width * np.sin(angle)
    )
    lr = Point(
        ur.x + height * np.sin(angle),
        ur.y + height * np.cos(angle)
    )
    ll = Point(
        height * np.cos(np.pi / 2 - angle),
        height * np.sin(np.pi / 2 - angle)
    )
    return tuple(base_point + pt for pt in (Point(0, 0), ur, lr, ll))
