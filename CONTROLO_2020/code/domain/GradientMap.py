from math import floor
import numpy as np
from enum import Enum

from domain.MissionPlane import MissionPlane


class GradientMap:
	class GradientMapType(Enum):
		"""
		Represents the type of gradient map in use:
			- SINGLE_MAX        : map with a single maximum.
			- SINGLE_MIN        : map with a single minimum.
			- MULTIPLE_MAX_MIN  : map with multiple maximums and minimums.
		"""
		SINGLE_MAX = 'SingleMax'
		SINGLE_MIN = 'SingleMin'
		MULTIPLE_MAX_MIN = 'MultipleMaxMin'

	"""
	Responsible for:
		- calculating the gradient attraction value of a given position to use as movement component;
		- storing discrete height map values, to be able to visualize.

	Attributes:
	----------
	mission_plane   : MissionPlane
		mission plane instance, that contains all the invalid areas contained in the plane
	space_size      : float
		size of the mission plane
	map_values      : numpy.ndarray
		shape - (space_size, space_size)
		matrix of function values to visualize the height map
	highest_point   : numpy.ndarray
		shape - (2,)
		position of the point with the highest function value
	"""
	def __init__(self, mission_plane: MissionPlane, space_size: float):
		"""
		Parameters
		----------
		mission_plane   : MissionPlane
			mission plane instance, that contains all the invalid areas contained in the plane
		space_size      : float
			dimension of the mission plane
		"""

		# TODO - Use space_size from mission_plane

		self.mission_plane = mission_plane
		self.space_size = space_size
		self.map_values = np.zeros((space_size, space_size))
		# TODO - Remove
		self.highest_point = np.zeros((2,))
		highest = 0
		for i in range(floor(space_size)):
			for j in range(floor(space_size)):
				position = np.array([i, j])
				if mission_plane.is_subdivision_available(position):
					value = self.get_value(position)
				else:
					# TODO - Make class constant.
					value = -2000
				self.map_values[i, j] = value
				if value > highest:
					highest = value
					self.highest_point = np.array([i, j])
		self.map_values = np.transpose(self.map_values)

	def update_map_values(self) -> None:
		"""
		Updates the map values matrix.
		"""
		for i in range(floor(self.space_size)):
			for j in range(floor(self.space_size)):
				position = np.array([i, j])
				if self.mission_plane.is_subdivision_available(position):
					value = self.get_value(position)
				else:
					# TODO - Make class constant.
					value = -2000
				self.map_values[i, j] = value
		self.map_values = np.transpose(self.map_values)

	@staticmethod
	def get_value(position: np.ndarray((2,))) -> float:
		"""
		Calculates the gradient attraction value for the given position.

		Parameters
		----------
		position : numpy.ndarray
			shape - (2,)
			position of which to calculate the gradient attraction value

		Returns
		-------
		float
			gradient attraction value
		"""
		raise NotImplementedError("GradientMap is an abstract class")

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
		raise NotImplementedError("GradientMap is an abstract class")

	def update_utility_value(self, is_update_increasing: bool = True,
								is_update_maximum: bool = True, area_index: int = -1) -> None:
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
		raise NotImplementedError("GradientMap is an abstract class")

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
		raise NotImplementedError("This method is only defined for GradientMapMultipleMaximumsMinimums")

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
		raise NotImplementedError("This method is only defined for GradientMapMultipleMaximumsMinimums")

	def translate_rendezvous_area(self, is_in_boundaries_function, is_move_maximum: bool = None,
									area_index: int = None) -> None:
		"""
		Moves one of the gradient map's maximums / minimums to a close random position.

		Parameters
		----------
		is_in_boundaries_function   : function
			function to callback to check if point is in boundaries
		is_move_maximum             : bool
			boolean to represent if the area to move is a gradient map's maximum
		area_index                  : int
			in case of multiple areas, represents the index of the maximum (if is_move_maximum is True) or the minimum
		"""
		raise NotImplementedError("GradientMap is an abstract class")

	def encode(self) -> dict:
		"""
		Encodes the map to a savable and reproducible format.

		Returns
		-------
		dict
			data
		"""
		raise NotImplementedError("GradientMap is an abstract class")
