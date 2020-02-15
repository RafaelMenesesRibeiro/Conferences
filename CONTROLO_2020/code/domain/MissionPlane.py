import numpy as np


class MissionPlane:
	"""
	Represents the mission plane.

	Attributes:
	----------
	plane_subdivisions_availability : np.ndarray
		shape - (horizontal_subdivisions_number, vertical_subdivisions_number)
		matrix that represents if a subdivision of the mission plane is a legal area - if any agent can be there
	square_subdivision_length
		size of the square that represents a subdivision of the mission plane
	"""
	def __init__(self, space_size: float, square_subdivision_length: float = 10, availability_matrix: np.ndarray = None):
		"""
		Parameters
		----------
		space_size                  : float
			size of the mission plane
		square_subdivision_length   : float
			size of the square that represents a subdivision of the mission plane
		availability_matrix         : numpy.ndarray
			shape - (horizontal_subdivisions_number, vertical_subdivisions_number)
			built matrix that represents if a subdivision of the mission plane is a legal area
		"""
		self.square_subdivision_length = square_subdivision_length
		if availability_matrix is None:
			# So far, the mission plane is considered to be a square, so the number of subdivisions in both axis is equal.
			squares_number = int(space_size // square_subdivision_length)
			availability_matrix = np.ones((squares_number, squares_number))

			# TODO - Remove. This is test.
			# availability_matrix[7, 5] = 0
			# availability_matrix[5, 3] = 0

		self.plane_subdivisions_availability = availability_matrix

	@staticmethod
	def create_full_availability_matrix(space_size: float, square_subdivision_length: float = 10) -> np.ndarray:
		"""
		Creates an availability matrix with all legal positions.

		Parameters
		----------
		space_size                  : float
			size of the mission plane
		square_subdivision_length   : float
			size of the square that represents a subdivision of the mission plane

		Returns
		-------
		numpy.ndarray
			shape - (horizontal_subdivisions_number, vertical_subdivisions_number)
			built matrix that represents if a subdivision of the mission plane is a legal area
		"""
		squares_number = int(space_size // square_subdivision_length)
		return np.ones((squares_number, squares_number))

	def is_subdivision_available(self, position: np.ndarray) -> bool:
		"""
		Maps the given position to one of the mission plane's subdivisions, and verifies if it is a legal area.

		Parameters
		----------
		position : numpy.ndarray
			position to check

		Returns
		-------
		bool
			True if the subdivision containing the position is legal, False otherwise.
		"""
		subdivision_x_index = int(position[0]) // self.square_subdivision_length
		subdivision_y_index = int(position[1]) // self.square_subdivision_length
		return self.plane_subdivisions_availability[subdivision_x_index, subdivision_y_index] == 1

	def encode(self) -> dict:
		"""
		Encode the execution data to a dictionary.
		"""
		return {
			'Subdivision length': self.square_subdivision_length,
			'Availability matrix': self.plane_subdivisions_availability.tolist()
		}

