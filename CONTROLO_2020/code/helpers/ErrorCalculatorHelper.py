import numpy as np


class ErrorCalculator:
	"""
	Responsible for calculating the error, at every time step, based on the agents' positions.
	"""

	@staticmethod
	def calculate_error(positions: np.ndarray) -> float:
		"""
		Calculates the error based on the positions.

		Parameters
		----------
		positions : numpy.ndarray
			shape - (agents_number, 2)
			positions of the agents
		Returns
		-------
		float
			calculated error value
		"""
		average_positions = np.mean(positions, axis=0)
		distances_to_average = abs(positions - average_positions)
		mean_distance_components = np.mean(distances_to_average, axis=0)
		return np.linalg.norm(mean_distance_components)
