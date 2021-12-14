
import click
import pandas as pd
from cv2 import destroyAllWindows
from os.path import splitext
from particle_utils import ParticlesData


class Displacement():
    '''Represents a displacement vector.
    '''

    def __init__(self, datum):
        '''Initializes a new instance of the `Displacement` class.

        Args:
            datum: A datum, which contains x- and y-coordinates
                in actual size, and the coordinate.

        Attributes:
            x (int): x-coordinate in actual size.
            y (int): y-coordinate in actual size.
            coordinate: The coordinate on an image.
        '''
        self.x, self.y, self.coordinate = datum


class Displacements(list):
    '''Represents the change in displacement vector over time.
    '''

    def __init__(self, data):
        '''Initializes a new instance of the `Displacements` class.

        Args:
            data: a list of tuples of x- and y-coordinates
                in actual size, and coordinates.
        '''
        super().__init__(map(Displacement, data))

    @property
    def X(self):
        '''Change in x-coordinates over time.
        '''
        t = -10
        return {(t:= t+10): vec.x for vec in self}

    @property
    def Y(self):
        '''Change in y-coordinates over time.
        '''
        t = -10
        return {(t:= t+10): vec.y for vec in self}

    @property
    def Xsq(self):
        '''Variation of squared displacement in x-direction with time.
        '''
        return {t: x**2 for t, x in self.X.items()}

    @property
    def Ysq(self):
        '''Variation of squared displacement in y-direction with time.
        '''
        return {t: y**2 for t, y in self.Y.items()}


@click.command()
@click.argument('source')
@click.option('--destination', '-o', default=None,
              help='Path of the CSV file to output the data.')
@click.option('--mode', '-m', default='w',
              help='CSV writing mode.')
@click.option('--threshold', '-t', default=150,
              help='Threshold for binarizing an image.')
@click.option('--width', '-w', default=150,
              help='Actual width of the image.')
@click.option('--height', '-h', default=150,
              help='Actual height of the image.')
@click.option('--distance', '-d', default=15,
              help='Maximum distance a particle can travel '
              'in one unit time.')
@click.option('--lower', '-l', default=2,
              help='Minimum length of change over time '
              'used in the analysis.')
def main(source, destination, mode, threshold,
         width, height, distance, lower
         ):
    fname = splitext(source)[0]
    if destination is None:
        destination = fname + ".csv"

    data = ParticlesData(source, threshold, width, height, distance)
    displacements = map(Displacements, data.displacements(lower))
    dfm = pd.DataFrame()
    for displacement in displacements:
        dfm = dfm.append(
            [displacement.Xsq, displacement.Ysq], ignore_index=True
        )
    dfm.to_csv(destination, mode=mode, header=False, index=False)
    #data.mark_particles()
    #data.show()

if __name__ == '__main__':
    try:
        main()
    finally:
        destroyAllWindows()
