from math import sin, cos, radians
import numpy as np
from domain import CommsTower
from domain.Polytope import Polytope
from helpers.MathHelper import MathHelper


class StripHelper:
	"""
	Responsible for:
		- calculating the comms strip boundaries;
		- checking if a polyshape is inside the strip.

	Attributes:
	----------
	strip_half_rads     : float
		comms tower's strip half angle converted to radians
	strip_half_rads_sin : float
		sine of the strips's half radians
	strip_half_rads_cos : float
		cosine of the strips's half radians
	"""
	def __init__(self, strip_half_angle: float):
		"""
		Parameters
		----------
		strip_half_angle : float
			comms tower's strip half angle
		"""
		self.strip_half_rads = radians(strip_half_angle)
		self.strip_half_rads_sin = sin(self.strip_half_rads)
		self.strip_half_rads_cos = cos(self.strip_half_rads)

	@staticmethod
	def is_in_strip_boundaries(polytope: Polytope, lower_boundary: np.ndarray((2, 2)),
								upper_boundary: np.ndarray((2, 2))) -> bool:
		"""
		Calculates if the polytope given by the polytope parameter is inside the strip defined by the boundaries given
		by the lower_boundary and upper_boundary parameters.

		Parameters
		----------
		polytope        : domain.Polytope.Polytope
		lower_boundary  : numpy.ndarray
			shape - (2, 2)
			start and end points of the comms tower's strip's lower boundary
		upper_boundary  : numpy.ndarray
			shape - (2, 2)
			start and end points of the comms tower's strip's upper boundary

		Returns
		-------
		bool
			True if polytope is inside the strip, False otherwise
		"""
		position = polytope.get_center_position()

		lower_boundary_vector = lower_boundary[0, :] - lower_boundary[1, :]
		lower_boundary_gradient = lower_boundary_vector[1] / lower_boundary_vector[0]
		lower_boundary_b = lower_boundary[0, 1]
		lower_boundary_y = lower_boundary_gradient * position[0] + lower_boundary_b

		upper_boundary_vector = upper_boundary[0, :] - upper_boundary[1, :]
		upper_boundary_gradient = upper_boundary_vector[1] / upper_boundary_vector[0]
		upper_boundary_b = upper_boundary[0, 1]
		upper_boundary_y = upper_boundary_gradient * position[0] + upper_boundary_b

		# Checks if the center of the polytope is inside the boundaries
		if not MathHelper.is_point_between_points(position[1], lower_boundary_y, upper_boundary_y):
			return False
		# If the center is inside the boundaries, checks the polytope
		if MathHelper.is_line_intersecting_polytope(polytope, lower_boundary[0, :], lower_boundary[1, :]):
			return False
		if MathHelper.is_line_intersecting_polytope(polytope, upper_boundary[0, :], upper_boundary[1, :]):
			return False
		return True

	def get_strip_boundaries(self, comms_tower: CommsTower, target_position: np.ndarray((2,))) \
								-> (np.ndarray((2, 2)), np.ndarray((2, 2))):
		"""
		Calculates the lower and upper boundaries for the strip starting at the comms tower given by the comms_tower
		parameter, centered on the position given by the target_position parameter.

		Parameters
		----------
		comms_tower     : domain.CommsTower.CommsTower
			comms tower that emits the strip
		target_position : numpy.ndarray
			shape - (2,)
			position of the target

		Returns
		-------
		(numpy.ndarray, numpy.ndarray)
			shape - (2, 2) and (2, 2)
			start and end points of lower and upper boundaries of the strip
		"""
		vector = target_position - comms_tower.get_position()
		lower_boundary = self.get_boundary(vector, comms_tower, is_lower=True)
		upper_boundary = self.get_boundary(vector, comms_tower)

		if lower_boundary[0, 1] != upper_boundary[0, 1]:
			if lower_boundary[0, 1] < upper_boundary[0, 1]:
				return lower_boundary, upper_boundary
			return upper_boundary, lower_boundary

		if lower_boundary[1, 1] < upper_boundary[1, 1]:
			return lower_boundary, upper_boundary
		return upper_boundary, lower_boundary

	def get_boundary(self, vector: np.ndarray((2,)), tower: CommsTower, is_lower: bool = False) -> np.ndarray((2, 2)):
		"""
		Calculates one of the boundaries of the comms tower's strip.

		Parameters
		----------
		vector      : numpy.ndarray
			shape - (2,)
			vector from the tower to the target
		tower       : domain.CommsTower.CommsTower
			comms tower instance that emits the strip
		is_lower    : bool
			True if calculating the lower boundary, False if calculating the upper one

		Returns
		-------
		numpy.ndarray
			shape - (2, 2)
			start and end points of the boundary
		"""
		if is_lower:
			rotated_vector = MathHelper.rotate_vector(vector, -1 * self.strip_half_rads_sin, self.strip_half_rads_cos)
		else:
			rotated_vector = MathHelper.rotate_vector(vector, self.strip_half_rads_sin, self.strip_half_rads_cos)
		tower_position = tower.get_position()
		space_size = tower.get_space_size()
		return MathHelper.get_strip_start_end_points_from_vector(rotated_vector, tower_position, space_size)
