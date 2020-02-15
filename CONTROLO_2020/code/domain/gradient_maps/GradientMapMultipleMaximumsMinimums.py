import numpy as np
from domain.GradientMap import GradientMap
from domain.MissionPlane import MissionPlane


class GradientMapMultipleMaximumsMinimums(GradientMap):
	"""
	Subclass of Gradient Map with a single maximum point.

	Attributes:
	----------
	map_values              : numpy.ndarray
		shape - (space_size, space_size)
		matrix of function values to visualize the height map
	highest_point           : numpy.ndarray
		shape - (2,)
		position of the point with the highest function value
	maximum_points_number   : int
		number of maximum points
	maximum_points          : numpy.ndarray
		shape - (maximum_points_number, 2)
		positions of the maximum points
	maximum_points_offsets  : numpy.ndarray
		shape - (maximum_points_number, 2)
		maximum points divided by the PARABOLOID_MULTIPLIER to improve efficiency when offsetting x, y value in order
		to move the paraboloids' centers to its respective maximum_point's position
	maximum_points_utilities    : numpy.ndarray
		shape - (maximum_points_number,)
		utility values of the maximum points
	minimum_points_number   : int
		number of minimum points
	minimum_points          : numpy.ndarray
		shape - (minimum_points_number, 2)
		positions of the minimum points
	minimum_points_offsets  : numpy.ndarray
		shape - (minimum_points_number, 2)
		minimum points divided by the PARABOLOID_MULTIPLIER to improve efficiency when offsetting x, y value in order
		to move the paraboloids' centers to its respective minimum_point's position
	minimum_points_utilities    : numpy.ndarray
		shape - (minimum_points_number,)
		utility values of the minimum points
	"""

	PARABOLOID_MULTIPLIER = 2
	"""
	float: Multiplier constant that defines paraboloids.
	"""

	DEFAULT_UTILITY_VALUE = 1000.0
	"""
	float: Default utility value for the rendezvous points. Negative for the minimums.
	"""

	MAXIMUM_ABSOLUTE_UTILITY_VALUE = 2000.0
	"""
	float: Maximum absolute utility value for any rendezvous point. 
	"""

	MINIMUM_ABSOLUTE_UTILITY_VALUE = 50.0
	"""
	float: Minimum absolute utility value for any rendezvous point. If after updating, the value of a rendezvous area
	is smaller, the rendezvous area is removed from the gradient map.
	"""

	UPDATE_INCREASE_MULTIPLIER = 1.05
	"""
	float: Multiplier to apply to the current utility value, when the update is to increase.
	"""

	UPDATE_DECREASE_MULTIPLIER = 0.5
	"""
	float: Multiplier to apply to the current utility value, when the update is to decrease.
	"""

	MINIMUM_DISTANCE_TO_CONSIDER_EXPLORED = 8
	"""
	float: Minimum distance between an agent and an area, for it to be considered explored, and to decrease its utility
	value.
	"""

	def __init__(self, mission_plane: MissionPlane, space_size: float, maximum_points_number: int = 2, maximum_points: np.ndarray = None,
					maximum_points_utility_values: np.ndarray = None, minimum_points_number: int = 2,
					minimum_points: np.ndarray = None, minimum_points_utility_values: np.ndarray = None):
		"""
		Parameters
		----------
		mission_plane           : MissionPlane
			mission plane instance, that contains all the invalid areas contained in the plane
		space_size              : float
			dimension of the mission plane
		maximum_points_number   : int
			number of maximum points
		maximum_points          : numpy.ndarray
			shape - (maximum_points_number, 2)
			positions of the maximum points
		minimum_points_number   : int
			number of minimum points
		minimum_points          : numpy.ndarray
			shape - (minimum_points_number, 2)
			positions of the minimum points
		"""
		if maximum_points is None:
			maximum_points = np.random.random((maximum_points_number, 2)) * space_size
		else:
			maximum_points_number = np.size(maximum_points, 0)
		self.maximum_points_number = maximum_points_number
		self.maximum_points = maximum_points
		self.maximum_points_offset = maximum_points / GradientMapMultipleMaximumsMinimums.PARABOLOID_MULTIPLIER
		self.maximum_points_utilities = np.ones((maximum_points_number,)) * GradientMapMultipleMaximumsMinimums.DEFAULT_UTILITY_VALUE if maximum_points_utility_values is None else maximum_points_utility_values

		if minimum_points is None:
			minimum_points = np.random.random((minimum_points_number, 2)) * space_size
		else:
			minimum_points_number = np.size(minimum_points, 0)
		self.minimum_points_number = minimum_points_number
		self.minimum_points = minimum_points
		self.minimum_points_offset = minimum_points / GradientMapMultipleMaximumsMinimums.PARABOLOID_MULTIPLIER
		self.minimum_points_utilities = np.ones((minimum_points_number,)) * -1 * GradientMapMultipleMaximumsMinimums.DEFAULT_UTILITY_VALUE if minimum_points_utility_values is None else minimum_points_utility_values

		super(GradientMapMultipleMaximumsMinimums, self).__init__(mission_plane, space_size)

	def update_map_values(self, is_update_maximums: bool = True) -> None:
		if is_update_maximums:
			self.maximum_points_offset = self.maximum_points / GradientMapMultipleMaximumsMinimums.PARABOLOID_MULTIPLIER
		else:
			self.minimum_points_offset = self.minimum_points / GradientMapMultipleMaximumsMinimums.PARABOLOID_MULTIPLIER
		super().update_map_values()

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
		value = 0
		for i in range(self.maximum_points_number):
			p = position / GradientMapMultipleMaximumsMinimums.PARABOLOID_MULTIPLIER - self.maximum_points_offset[i, :]
			x = p[0]
			y = p[1]
			value += self.maximum_points_utilities[i] / (x * x + y * y + 15)
		for i in range(self.minimum_points_number):
			p = position / GradientMapMultipleMaximumsMinimums.PARABOLOID_MULTIPLIER - self.minimum_points_offset[i, :]
			x = p[0]
			y = p[1]
			value += self.minimum_points_utilities[i] / (x * x + y * y + 15)
		return value

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
		value = np.zeros((2,))
		for i in range(self.maximum_points_number):
			p = position / GradientMapMultipleMaximumsMinimums.PARABOLOID_MULTIPLIER - self.maximum_points_offset[i, :]
			x = p[0]
			y = p[1]
			x_derivative = (-self.maximum_points_utilities[i] * (2 * x)) / ((x * x + y * y + 15) * (x * x + y * y + 15))
			y_derivative = (-self.maximum_points_utilities[i] * (2 * y)) / ((x * x + y * y + 15) * (x * x + y * y + 15))
			value += np.array([x_derivative, y_derivative])
		for i in range(self.minimum_points_number):
			p = position / GradientMapMultipleMaximumsMinimums.PARABOLOID_MULTIPLIER - self.minimum_points_offset[i, :]
			x = p[0]
			y = p[1]
			x_derivative = (-self.minimum_points_utilities[i] * (2 * x)) / ((x * x + y * y + 15) * (x * x + y * y + 15))
			y_derivative = (-self.minimum_points_utilities[i] * (2 * y)) / ((x * x + y * y + 15) * (x * x + y * y + 15))
			value += np.array([x_derivative, y_derivative])
		return value

	# TODO - Pass get_random_position_in_boundaries in __init__
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
		# TODO - maximum_points_number was raising errors, changed it to maximum_points.shape[0]
		# TODO - maximum_points.shape[0] is also raising errors sometimes
		for area_index in range(self.maximum_points.shape[0]):
			area_position = self.maximum_points[area_index, :]
			distances = np.linalg.norm(positions - area_position, axis=1)
			if np.any(distances < GradientMapMultipleMaximumsMinimums.MINIMUM_DISTANCE_TO_CONSIDER_EXPLORED):
				self.update_utility_value(is_update_increasing=False, area_index=area_index)
				continue

			self.update_utility_value(is_update_increasing=True, area_index=area_index)

		if self.maximum_points_number == 0:
			self.create_rendezvous_area(get_random_position_in_boundaries, is_create_maximum=True)

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
		value_multiplier = GradientMapMultipleMaximumsMinimums.UPDATE_INCREASE_MULTIPLIER if is_update_increasing \
							else GradientMapMultipleMaximumsMinimums.UPDATE_DECREASE_MULTIPLIER
		if is_update_maximum and self.maximum_points_number > 0:
			maximums_number = self.maximum_points.shape[0]
			if not (0 <= area_index < maximums_number):
				area_index = np.random.randint(0, maximums_number)
			new_value = self.maximum_points_utilities[area_index] * value_multiplier
			if new_value > GradientMapMultipleMaximumsMinimums.MAXIMUM_ABSOLUTE_UTILITY_VALUE:
				new_value = GradientMapMultipleMaximumsMinimums.MAXIMUM_ABSOLUTE_UTILITY_VALUE
			elif new_value < GradientMapMultipleMaximumsMinimums.MINIMUM_ABSOLUTE_UTILITY_VALUE:
				self.remove_rendezvous_area(is_remove_maximum=is_update_maximum, area_index=area_index)
				return
			self.maximum_points_utilities[area_index] = new_value
			self.update_map_values(True)
		elif not is_update_maximum and self.minimum_points_number > 0:
			minimums_number = self.minimum_points_number
			if not (0 <= area_index < minimums_number):
				area_index = np.random.randint(0, minimums_number)
			new_value = self.minimum_points_utilities[area_index] * value_multiplier
			if new_value > GradientMapMultipleMaximumsMinimums.MAXIMUM_ABSOLUTE_UTILITY_VALUE:
				new_value = GradientMapMultipleMaximumsMinimums.MAXIMUM_ABSOLUTE_UTILITY_VALUE
			elif new_value < GradientMapMultipleMaximumsMinimums.MINIMUM_ABSOLUTE_UTILITY_VALUE:
				self.remove_rendezvous_area(is_remove_maximum=is_update_maximum, area_index=area_index)
				return
			self.minimum_points_utilities[area_index] = new_value
			self.update_map_values(False)

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
		random_position = get_random_position_in_boundaries()
		if is_create_maximum:
			self.maximum_points = np.append(self.maximum_points, random_position, 0)
			self.maximum_points_utilities = np.append(self.maximum_points_utilities,
														GradientMapMultipleMaximumsMinimums.DEFAULT_UTILITY_VALUE)
			self.maximum_points_number += 1
			self.maximum_points_offset = self.maximum_points / GradientMapMultipleMaximumsMinimums.PARABOLOID_MULTIPLIER
			self.update_map_values(True)
		else:
			self.minimum_points = np.append(self.minimum_points, random_position, 0)
			self.minimum_points_utilities = np.append(self.minimum_points_utilities,
														-1 * GradientMapMultipleMaximumsMinimums.DEFAULT_UTILITY_VALUE)
			self.minimum_points_number += 1
			self.minimum_points_offset = self.minimum_points / GradientMapMultipleMaximumsMinimums.PARABOLOID_MULTIPLIER
			self.update_map_values(False)

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
		if is_remove_maximum and self.maximum_points_number > 0:
			maximums_number = self.maximum_points.shape[0]
			if not (0 <= area_index < maximums_number):
				area_index = np.random.randint(0, maximums_number)
			self.maximum_points = np.delete(self.maximum_points, area_index, 0)
			self.maximum_points_utilities = np.delete(self.maximum_points_utilities, area_index)
			self.maximum_points_number -= 1
			self.update_map_values(True)
		elif not is_remove_maximum and self.minimum_points_number > 0:
			minimums_number = self.minimum_points.shape[0]
			if not (0 <= area_index < minimums_number):
				area_index = np.random.randint(0, minimums_number)
			self.minimum_points = np.delete(self.minimum_points, area_index, 0)
			self.minimum_points_utilities = np.delete(self.minimum_points_utilities, area_index)
			self.minimum_points_number -= 1
			self.update_map_values(False)

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

		if is_move_maximum is None:
			is_move_maximum = True if np.random.rand() > 0.5 else False

		# TODO - Refactor
		if is_move_maximum is True and self.maximum_points_number > 0:
			maximums_number = self.maximum_points.shape[0]
			if area_index is None or not (0 <= area_index < maximums_number):
				area_index = np.random.randint(0, maximums_number)
			current_position = self.maximum_points[area_index, :]
			new_position = current_position + (np.random.rand(2, ) * 2 - 1)
			if is_in_boundaries_function(new_position):
				self.maximum_points[area_index, :] = new_position
				self.update_map_values(True)
		elif not is_move_maximum and self.minimum_points_number > 0:
			minimums_number = self.minimum_points.shape[0]
			if area_index is None or not (0 <= area_index < minimums_number):
				area_index = np.random.randint(0, minimums_number)
			current_position = self.minimum_points[area_index, :]
			new_position = current_position + (np.random.rand(2, ) * 2 - 1)
			if is_in_boundaries_function(new_position):
				self.minimum_points[area_index, :] = new_position
				self.update_map_values(False)

	def encode(self) -> dict:
		"""
		Encodes the map to a savable and reproducible format.

		Returns
		-------
		dict
			data
		"""
		return {
			'gradient_map_type': GradientMap.GradientMapType.MULTIPLE_MAX_MIN.value,
			'maximum_points': self.maximum_points.tolist(),
			'maximum_points_utility_values': self.maximum_points_utilities.tolist(),
			'minimum_points': self.minimum_points.tolist(),
			'minimum_points_utility_values': self.minimum_points_utilities.tolist()
		}
