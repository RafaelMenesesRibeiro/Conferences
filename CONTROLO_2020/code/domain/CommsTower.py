import numpy as np
import random
from domain.Polytope import Polytope
from helpers.StripHelper import StripHelper
from helpers.NeighborsHelper import NeighborsHelper


class CommsTower:
	"""
	Handles the agents represented as polytopes.
	Responsible for:
		- storing the data regarding all the agents (polytopes);
		- intermediating with its strip helper class;
		- intermediating with its neighbors helper class.

	Attributes:
	----------
	agents_number       : int
		total number of agents in the mission plane
	space_size          : float
		dimension of the mission plane
	position            : numpy.ndarray
		shape - (2,)
		position of the tower
	strip_helper        : helpers.StripHelper.StripHelper
		strip helper instance
	neighbors_helper    : helpers.NeighborsHelper.NeighborsHelper
		neighbors helper instance
	polytopes           : list<domain.Polytope.Polytope>
		list of polytope instances (one for each agent)
	"""
	def __init__(self, initial_position: np.ndarray((2,)), space_size: float, agents_number: int, strip_half_angle: int):
		"""
		Parameters
		----------
		initial_position    : numpy.ndarray
			shape - (2,)
			initial position of the tower
		space_size          : float
			dimension of the mission plane
		agents_number       : float
			total number of agents in the mission plane
		strip_half_angle    : float
			half of the angle of the strip, used to calculate the boundaries by rotating the center
			vector +/- the half angle
		"""
		self.agents_number = agents_number
		self.space_size = space_size
		self.position = initial_position
		self.strip_helper = StripHelper(strip_half_angle)
		self.neighbors_helper = NeighborsHelper(agents_number)
		self.polytopes = []

	@staticmethod
	def get_random_initial_position(space_size: float) -> np.ndarray((2,)):
		"""
		Calculates a random initial position for the tower. In the x axis, it will have the value of either 0 or
		space_size, to be on one of the sides. In the y axis, it will have any value from 0 to space_size.

		Parameters
		----------
		space_size : float
			dimension of the mission plane

		Returns
		-------
		numpy.ndarray
			shape - (2,)
			calculated random initial position for the tower
		"""
		is_left_side = random.random() >= 0.5
		tower_x = 0 if is_left_side else space_size
		tower_y = random.random() * space_size
		return np.array([tower_x, tower_y])

	def create_polytopes(self, positions: np.ndarray, radius: float) -> None:
		"""
		Initializes the polytopes list with new instances of Polytope, each in the respective position given by the
		parameter position.

		Parameters
		----------
		positions   : numpy.ndarray
			shape - (agents_number, 2)
			positions of the agents, where to create each of the polytope instances
		radius      : float
			radius bounding the size of the polyshapes

		Returns
		-------
		None
		"""
		self.polytopes = Polytope.create_polytopes_from_positions(self.agents_number, positions, radius)

	def move(self, direction: np.ndarray((2,))) -> None:
		"""
		Moves the tower in the given direction, with a distance magnitude of 1.

		Parameters
		----------
		direction : numpy.ndarray
			shape: (2,).
			direction in which to move the tower

		Returns
		-------
		None
		"""
		norm = np.linalg.norm(direction)
		if norm != 0:
			self.position += (direction / norm)

	def get_strip_boundaries(self, target_position: np.ndarray((2,))) -> (np.ndarray((2, 2)), np.ndarray((2, 2))):
		"""
		Returns the two boundaries (top and bottom) for the angular strip, given by the StripHelper instance.

		Parameters
		----------
		target_position : numpy.ndarray
			shape - (2,)
			position of the agent in which the tower is going to center the strip

		Returns
		-------
		(numpy.ndarray, numpy.ndarray)
			shapes - (2, 2) and (2, 2).
			lower strip boundary and upper strip boundary
		"""
		return self.strip_helper.get_strip_boundaries(self, target_position)

	def get_current_neighbors_for_agent(self, agent_index: int, current_positions: np.ndarray) -> np.ndarray:
		"""
		Returns the current neighbors for an agent, given by the NeighborsHelper instance.

		Parameters
		----------
		agent_index : int
			index of the agent for which to calculate the neighbors, between 0 and agents_number - 1
		current_positions : numpy.ndarray
			shape - (agents_number, 2)
			positions of the agents' polytopes

		Returns
		-------
		numpy.ndarray
			shape - (agents_number,)
			boolean array that represents if each agent is a neighbor of the agent_index agent
		"""
		return self.neighbors_helper.get_current_neighbors_for_agent(agent_index, current_positions, self)

	def get_position(self) -> np.ndarray((2,)):
		"""
		Returns
		-------
		numpy.ndarray
			shape - (2,)
			current position of the tower
		"""
		return self.position

	def get_space_size(self) -> float:
		"""
		Returns
		-------
		float
			dimension of the mission plane
		"""
		return self.space_size

	def get_polytopes(self) -> list:
		"""
		Returns
		-------
		List[domain.Polytope.Polytope]
			list of the agents' polytopes
		"""
		return self.polytopes

	def set_polytopes(self, polytopes: list):
		"""
		Parameters
		----------
		polytopes : List[domain.Polytope.Polytope]
			list of agent's polytopes to update with

		Returns
		-------
		None
		"""
		self.polytopes = polytopes

	def get_polytopes_center_positions(self) -> np.ndarray:
		"""
		Builds a numpy array with all the agents' center positions.

		Returns
		-------
		numpy.ndarray
			shape - (agents_number, 2)
			center positions
		"""
		positions = np.zeros((self.agents_number, 2))
		for agent_index in range(self.agents_number):
			positions[agent_index, :] = self.polytopes[agent_index].get_center_position()
		return positions

	def get_polytopes_velocities(self) -> np.ndarray:
		"""
		Builds a numpy array with all the agents' velocities.

		Returns
		-------
		numpy.ndarray
			shape - (agents_number, 2)
			velocities
		"""
		velocities = np.zeros((self.agents_number, 2))
		for agent_index in range(self.agents_number):
			velocities[agent_index, :] = self.polytopes[agent_index].get_velocity()
		return velocities

	def get_polytope(self, agent_index: int) -> Polytope:
		"""
		Gets an agent's polytope.

		Parameters
		----------
		agent_index : int
			index of the agent for which to get the polytope

		Returns
		-------
		domain.Polytope.Polytope
			agent's polytope
		"""
		return self.polytopes[agent_index]

	def get_polytope_center_position(self, agent_index: int) -> np.ndarray((2,)):
		"""
		Gets an agent's polytope's center position.

		Parameters
		----------
		agent_index : int
			index of the agent for which to get the polytope's center position

		Returns
		-------
		numpy.ndarray
			shape - (2,)
			agent's polytope's center position
		"""
		return self.polytopes[agent_index].get_center_position()

	def get_polytope_vertices(self, agent_index: int) -> np.ndarray((8, 2)):
		"""
		Gets an agent's polytope's vertices.

		Parameters
		----------
		agent_index : int
			index of the agent for which to get the polytope's center position

		Returns
		-------
		numpy.ndarray
			shape - (8,2)
			agent's polytope's vertices
		"""
		return self.polytopes[agent_index].get_vertices()
