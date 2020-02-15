import numpy as np
from math import floor
# Sourced from https://github.com/mmchugh/pynoise
from helpers.perlin_noise.perlin import Perlin
from domain.GradientMap import GradientMap


class GradientMapPerlinNoise(GradientMap):
	"""
	Subclass of Gradient Map with a perlin noise surface.

	Attributes:
	----------
	map_values      : numpy.ndarray
		shape - (space_size, space_size)
		matrix of function values to visualize the height map
	highest_point   : numpy.ndarray
		shape - (2,)
		position of the point with the highest function value
	"""
	def __init__(self, space_size: float):
		"""
		Parameters
		----------
		space_size : float
			dimension of the mission plane
		"""
		self.map = Perlin(frequency=13, seed=int(100 * np.random.random((1,))))
		super(GradientMapPerlinNoise, self).__init__(space_size)

	def get_value(self, position: np.ndarray((2,))) -> float:
		"""
		Calculates the field's height value for the given position.

		Parameters
		----------
		position : numpy.ndarray
			shape - (2,)
			position of which to calculate the height

		Returns
		-------
		float
			height value
		"""
		x = position[0] / 400
		y = position[1] / 400
		return self.map.value(x, y, 0)

	def get_gradient_values(self, position: np.ndarray((2,))) -> np.ndarray((2,)):
		"""
		Calculates the gradient of the field in the given position.
		Because the perlin noise is not defined by a mathematical function, no gradient can be calculated.
		(Unless a function was approximated from all the points, but that was deemed unnecessary work.)

		Parameters
		----------
		position : numpy.ndarray
			shape - (2,)
			position of which to calculate the gradient

		Returns
		-------
		numpy.ndarray
			shape - (2,)
			gradient vector
		"""
		return np.zeros((2,))
