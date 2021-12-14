
import cv2
import numpy as np
from functools import partial


def to_list(func):
    '''Converts the return value of the generator to a list.
    '''
    def __wrapper(*args, **kwargs):
        return list(func(*args, **kwargs))
    return __wrapper


class Particle():
    '''Represents an individual particle.

    This class can also be used to represent the change in coordinates
    of particles over time as a linear list.

    Attributes:
        x (int): x-coordinate of the particle.
        y (int): y-coordinate of the particle.
    '''

    def __init__(self, contour, size_coeffs):
        '''Initializes a new instance of the `Particle` class.

        Args:
            contour: Coordinates of the particle contour.
            One of the return values of `cv2.findContours` is used as is.
            size_coeffs: Coefficients for converting the coordinates
                of a particle to its actual distance.
        '''
        X = [pt[0][0] for pt in contour]
        Y = [pt[0][1] for pt in contour]
        self.x, self.y = round(np.average(X)), round(np.average(Y))
        self.__next_particles = None
        self.__prev = None
        self.__next = None
        self.size_coeffs = size_coeffs

    def __repr__(self):
        return f'({self.x}, {self.y})'

    def __sub__(self, other):
        return self.__class__(
            [[[self.x - other.x, self.y - other.y]]], self.size_coeffs
        )

    def __eq__(self, other):
        if not other:
          return False
        return self.x == other.x and self.y == other.y

    def distance_to(self, other):
        '''Calculates the distance to another particle.

        Args:
            other: The other particle.

        Returns:
            The distance.
        '''
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    @property
    def coordinate(self):
        '''Tuple of x- and y-coordinates of the particle.
        '''
        return self.x, self.y

    def find_nearest(self, particles, distance):
        '''Finds the particle that is closest to the current particle.

        Args:
            particles: A `Particles` object
                from which the nearest one is looked for.
            distance: The longest distance to search for a particle.

        Returns:
            If there are particles within `distance`, the closest particle;
            otherwise, `None`.
        '''
        p = min(particles, key=lambda p: self.distance_to(p))
        return p if self.distance_to(p) < distance else None

    def set_next_particles(self, particles):
        self.__next_particles = sorted(particles, key=lambda p: self.distance_to(p))

    @property
    def prev(self):
        '''A particle that represents the position of the current particle
        at a previous time.'''
        return self.__prev

    @prev.setter
    def prev(self, value):
        self.__prev = value
        if value:
            value.__next = self

    @property
    def next(self):
        '''A particle that represents the position of the current particle
        at a next time.'''
        return self.__next

    @next.setter
    def next(self, value):
        self.__next = value
        if value:
            value.__prev = self

    def correct_next(self, distance):
        index = self.__next_particles.index(self.__next)
        while index + 1 < len(self.__next_particles):
            index += 1
            p = self.__next_particles[index]
            d = self.distance_to(p)
            if d >= distance:
                return
            if p.prev:
                d0 = p.prev.distance_to(p)
                if d < d0:
                    p.prev.correct_next()
                    self.next = p
                    return
        self.__next = None

    @property
    def displacements(self):
        '''Future displacement vector starting from the current particle.

        Note:
            Refering this property may destroy linear list struct.
        '''
        O = self
        _x, _y = self.size_coeffs
        ret = [(0, 0, self.coordinate)]
        p = self
        while p.next:
            p = p.next
            p.prev.next = None
            x, y = (p - self).coordinate
            ret.append((x * _x, y * _y, p.coordinate))
        return ret


class Particles(list):
    '''Represents a group of particles at a specific time.
    '''

    def __init__(self, contours, size_coeffs):
        '''Initializes a new instance of the `Particles` class.

        Args:
            contours: Coordinates of the contour of each particle.
            size_coeffs: Coefficients for converting the coordinates
                of a particle to its actual distance.
        '''
        super().__init__(
            map(partial(Particle, size_coeffs=size_coeffs), contours)
        )

    def link_next(self, particles, distance):
        '''Links the group of particles at the next time with the current one.

        Note:
            This method is used to track changes in particles over time.
        '''
        for particle in self:
            particle.set_next_particles(particles)
            for p in particles:
                if (d := particle.distance_to(p)) > distance:
                    return
                if p.prev:
                    d0 = p.prev.distance_to(p)
                    if d < d0:
                        p.prev.correct_next(distance)
                        break
                    else:
                        continue
                else:
                    break
            else:
                return
            particle.next = p

    @to_list
    def displacements(self, lower_limit):
        '''Detects displacement vectors
        starting at the current time for each particle.

        Note:
            Calling this method may destroy linear list struct.

        Args:
            lower_limit: Lower limit of the number of tracks over time.

        Yields:
            list: A list of displacement vectors.
        '''
        for particle in self:
            d = particle.displacements
            if len(d) < lower_limit:
                continue
            yield d


class ParticlesData(list):
    '''Represents the information for each particle at all times.

    Attributes:
        path (str): Path of the gif image.
        gif (list): List of frames corresponding to each time.
        size_coeffs: Coefficients for converting the coordinates
            of a particle to its actual distance.
    '''

    def __init__(self, path, threshold, width, height, distance):
        '''Initializes a new instance of the `ParticlesData` class.

        Args:
            path: Path of the gif image from which data is loaded.
            threshold: Threshold value used to binarize an image.
            width: The actual size of the image's width.
            height: The actual size of the image's height.
            distance: The longest distance to search for a particle.
        '''
        self.gif = self.load_gif(path, threshold)
        w, h = self.gif[0].shape
        if not width:
            width = w
        if not height:
            height = h
        self.size_coeffs = width / w, height / h

        particles = map(self.get_particles, self.gif)
        super().__init__(particles)

        for i, particles in enumerate(self[:-1]):
            particles.link_next(self[i+1], distance)

    def load_gif(self, path, threshold):
        '''Loads a gif image.

        Args:
            path: Path of the gif image from which data is loaded.
            threshold: Threshold value used to binarize an image.

        Returns:
            Loaded gif image.
            The image is a black and white image binarized
            by a `threshold` value.
        '''
        self.path = path
        cap = cv2.VideoCapture(path)
        gif = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                org = cv2.bitwise_not(frame)
                gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
                th, im = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                gif.append(im)
            else:
                cap.release()

        return gif

    def get_particles(self, frame):
        '''Detects particle contours from a frame.

        Args:
            frame: The frame from which particles are detected.

        Returns:
            A `Particles` object containing detected particles.
        '''
        contours, hierarchy = cv2.findContours(
            frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        # exclude the end of the list
        # because the entire frame is also detected as a contour
        return Particles(contours[:-1], self.size_coeffs)

    def mark_particles(self):
        '''Marks the positions of the particles on the gif image.
        '''
        for frame, coordinates in zip(self.gif, self):
            for particle in coordinates:
                frame = cv2.circle(
                    frame, particle.coordinate, radius=10, color=(0, 0, 0)
                )

    def show(self, interval=100):
        '''Shows a gif image.

        Args:
            interval (int, optional): The interval between each frame
                in milliseconds.
        '''
        for frame in self.gif:
            cv2.imshow(self.path, frame)
            cv2.waitKey(interval)

    def displacements(self, lower_limit):
        '''Detects all detectable displacement vectors.

        Note:
            Calling this method may destroy linear list struct.

        Args:
            lower_limit: Lower limit of the number of tracks over time.

        Returns:
            A list of displacement vectors.
        '''
        return sum((l.displacements(lower_limit) for l in self), [])
