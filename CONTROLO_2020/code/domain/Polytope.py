import numpy as np
from domain.AgentProperties import AgentProperties


class Polytope:
	"""
	Represents an agent's physical form.
	Contains the position, velocity and polyshape.

	Attributes:
	----------
	center_position : np.ndarray
		shape - (2,)
		position of the agent
	velocity        : np.ndarray
		shape - (2,)
		velocity of the agent
	vertices        : np.ndarray
		shape - (8, 2)
		vertices of the polyshape
	"""
	def __init__(self, center_x: float, center_y: float, radius: float):
		"""
		Parameters
		----------
		center_x : float
			x coordinate of the agent's initial position
		center_y : float
			y coordinate of the agent's initial position
		radius : float
			radius bounding the size of the polyshape
		"""
		self.center_position = np.array([center_x, center_y])
		self.velocity = np.random.random((2,)) * 2 - 1
		self.vertices = Polytope.create_octagonal_polytope_around_circle(center_x, center_y, radius)

	@staticmethod
	def create_polytopes_from_positions(agents_number: int, positions: np.ndarray, radius: float = None) -> list:
		"""
		Creates a list of polytopes, one in each given position, with the given radius.

		Parameters
		----------
		agents_number   : int
			total number of agents in the mission plane
		positions       : numpy.ndarray
			shape - (agents_number, 2)
			initial center positions of the polytopes
		radius          : float
			radius bounding the size of the polyshape

		Returns
		-------
		list
			list of polytopes, each in a given center position
		"""
		if radius is None:
			radius = AgentProperties.MEASUREMENT_ERROR_RADIUS * 0.8

		polytopes = []
		for agentIndex in range(agents_number):
			polytope = Polytope(positions[agentIndex, 0], positions[agentIndex, 1], radius)
			polytopes.append(polytope)
		return polytopes

	@staticmethod
	def create_octagonal_polytope_around_circle(center_x: float, center_y: float, radius: float) -> np.ndarray((8, 2)):
		"""
		Creates the initial polyshape for an agent, an octagon bounded by the radius.

		Parameters
		----------
		center_x : float
			x coordinate of the agent's initial position
		center_y : float
			y coordinate of the agent's initial position
		radius : float
			radius bounding the size of the polyshape

		Returns
		-------
		np.ndarray
			shape - (8, 2)
			set of vertices that represent the polyshape
		"""
		points = np.zeros((8, 2))
		points[0, :] = [center_x + radius/2, center_y - radius]
		points[1, :] = [center_x + radius, center_y - radius / 2]
		points[2, :] = [center_x + radius, center_y + radius / 2]
		points[3, :] = [center_x + radius / 2, center_y + radius]
		points[4, :] = [center_x - radius / 2, center_y + radius]
		points[5, :] = [center_x - radius, center_y + radius / 2]
		points[6, :] = [center_x - radius, center_y - radius / 2]
		points[7, :] = [center_x - radius / 2, center_y - radius]
		return points

	def translate(self, heading: np.ndarray((2,))):
		"""
		Moves the polytope in the heading's direction, with the distance equal to the heading's magnitude.

		Parameters
		----------
		heading : numpy.ndarray
			shape - (2,)
			heading to move

		Returns
		-------
		None
		"""
		self.velocity = heading
		self.center_position += heading
		self.vertices += heading

	def peek_translate(self, direction: np.ndarray((2,))) -> (np.ndarray((2,)), np.ndarray((8, 2))):
		"""
		Returns the center position and the vertices as if they were translated by the given direction.

		Parameters
		----------
		direction : numpy.ndarray
			shape - (2,)
			heading to move

		Returns
		-------
		(numpy.ndarray, numpy.ndarray)
			shapes - (8, 2) and (8, 2).
			translated center position and translated vertices
		"""
		return self.center_position + direction, self.vertices + direction

	def peek_extended(self, new_radius: float) -> np.ndarray((8, 2)):
		"""
		Returns the vertices of a new polytope with the same center position but with the given radius.

		Parameters
		----------
		new_radius : float
			radius bounding the size of the polyshape

		Returns
		-------
		np.ndarray
			shape - (8, 2)
			set of vertices that represent the polyshape
		"""
		return self.create_octagonal_polytope_around_circle(self.center_position[0], self.center_position[1], new_radius)

	def get_center_position(self) -> np.ndarray((2,)):
		"""
		Returns
		-------
		numpy.ndarray
			shape - (2,)
			center position of the polyshape
		"""
		return self.center_position

	def get_velocity(self) -> np.ndarray((2,)):
		"""
		Returns
		-------
		numpy.ndarray
			shape - (2,)
			velocity of the agent
		"""
		return self.velocity

	def get_vertices(self) -> np.ndarray((8, 2)):
		"""
		Returns
		-------
		numpy.ndarray
			shape - (8, 2)
			set of vertices that represent the polyshape
		"""
		return self.vertices
