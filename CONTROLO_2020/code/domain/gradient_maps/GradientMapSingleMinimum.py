import numpy as np
from domain.GradientMap import GradientMap
from domain.MissionPlane import MissionPlane


class GradientMapSingleMinimum(GradientMap):
	"""
	Subclass of Gradient Map with a single minimum point.

	Attributes:
	----------
	map_values      : numpy.ndarray
		shape - (space_size, space_size)
		matrix of function values to visualize the height map
	highest_point   : numpy.ndarray
		shape - (2,)
		position of the point with the highest function value
	minimum_point   : numpy.ndarray
		shape - (2,)
		position of the single minimum point
	minimum_point_offset    : numpy.ndarray
		shape - (2,)
		minimum point divided by the PARABOLOID_MULTIPLIER to improve efficiency when offsetting x, y value in order
		to move the paraboloid's center to minimum_point's position
	"""

	PARABOLOID_MULTIPLIER = 2
	"""
	float: Multiplier constant that defines paraboloid.
	"""

	def __init__(self, mission_plane: MissionPlane, space_size: float, minimum_point: np.ndarray((2,)) = None):
		"""
		Parameters
		----------
		mission_plane   : MissionPlane
			mission plane instance, that contains all the invalid areas contained in the plane
		space_size      : float
			dimension of the mission plane
		minimum_point   : numpy.ndarray
			shape - (2,)
			position of the single minimum point
		"""
		if minimum_point is None:
			minimum_point = np.random.random((2,)) * space_size
		self.minimum_point = minimum_point
		self.minimum_point_offset = minimum_point / GradientMapSingleMinimum.PARABOLOID_MULTIPLIER
		super(GradientMapSingleMinimum, self).__init__(mission_plane, space_size)

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
		position = position / GradientMapSingleMinimum.PARABOLOID_MULTIPLIER - self.minimum_point_offset
		x = position[0]
		y = position[1]
		return -1000 / (x * x + y * y + 15)

	def get_gradient_values(self, position: np.ndarray((2,))) -> np.ndarray((2,)):
		"""
		Calculates the gradient of the field in the given position.

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
		position = position / GradientMapSingleMinimum.PARABOLOID_MULTIPLIER - self.minimum_point_offset
		x = position[0]
		y = position[1]
		x_derivative = (1000 * (2 * x)) / ((x * x + y * y + 15) * (x * x + y * y + 15))
		y_derivative = (1000 * (2 * y)) / ((x * x + y * y + 15) * (x * x + y * y + 15))
		return np.array([x_derivative, y_derivative])

	def check_if_agents_explored_areas(self, get_random_position_in_boundaries, positions: np.ndarray) -> None:
		"""
		Verifies if the agents are near any rendezvous area. If they are, the utility value decreases.
		For each rendezvous area still unexplored, its utility value is increased.

		Parameters
		----------
		get_random_position_in_boundaries   : function
			function to obtain a random valid position inside the mission plane
		positions : numpy.ndarray
			positions of the agents - given by the centroid of their polytopes
		"""
		raise NotImplementedError("TODO - GradientMapSingleMinimum")

	def update_utility_value(self, is_update_increasing: bool = True, is_update_maximum: bool = True,
								area_index: int = -1) -> None:
		"""
		Updates the utility value of a given rendezvous area, increasing or decreasing.

		Parameters
		----------
		is_update_increasing    : bool
			True if the utility value is to grow, False if is to decrease.
		is_update_maximum       : bool
			True if the area of the value to update is a maximum, False if minimum
		area_index              : int
			identifier of the area whose value to update
		"""
		raise NotImplementedError("TODO - GradientMapSingleMinimum")

	def create_rendezvous_area(self, get_random_position_in_boundaries, is_create_maximum: bool = True) -> None:
		"""
		Creates a new rendezvous are, in a random position inside the map.

		Parameters
		----------
		get_random_position_in_boundaries   : function
			function to obtain a random valid position inside the mission plane
		is_create_maximum                   : bool
			True if the area to create is a maximum, False if minimum
		"""
		# This method is only defined for GradientMapMultipleMaximumsMinimums
		pass

	def translate_rendezvous_area(self, is_in_boundaries_function, is_move_maximum: bool = None,
									area_index: int = None) -> None:
		"""
		Moves the gradient map's single minimum to a close random position.

		Parameters
		----------
		is_in_boundaries_function   : function
			function to callback to check if point is in boundaries
		is_move_maximum     : bool
			boolean to represent if the area to move is a gradient map's maximum
		area_index          : int
			in case of multiple areas, represents the index of the maximum (if is_move_maximum is True) or the minimum
		"""
		new_position = self.minimum_point + (np.random.rand(2, ) * 2 - 1)
		if is_in_boundaries_function(new_position):
			self.minimum_point = new_position
			self.minimum_point_offset = self.minimum_point / GradientMapSingleMinimum.PARABOLOID_MULTIPLIER
			self.update_map_values()

	def remove_rendezvous_area(self, is_remove_maximum: bool = True, area_index: int = -1) -> None:
		"""
		Removes a rendezvous area from the gradient map.

		Parameters
		----------
		is_remove_maximum   : bool
			True if the area to remove is a maximum, False if minimum
		area_index          : int
			identifier of the area to remove
		"""
		# This method is only defined for GradientMapMultipleMaximumsMinimums
		pass

	def encode(self) -> dict:
		"""
		Encodes the map to a savable and reproducible format.

		Returns
		-------
		dict
			data
		"""
		return {
			'gradient_map_type': GradientMap.GradientMapType.SINGLE_MIN.value,
			'minimum_point': self.minimum_point.tolist()}
