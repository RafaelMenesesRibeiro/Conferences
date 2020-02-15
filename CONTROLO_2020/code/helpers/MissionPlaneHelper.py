import numpy as np
from domain.AgentProperties import AgentProperties
from helpers.CollisionAvoidanceHelper import CollisionAvoidanceHelper


class MissionPlaneHelper:
	"""
	Responsible for:
		- creating the initial clusters;
		- checking if a position is too close to the mission plane boundaries.

	Attributes:
	----------
	agents_number       : int
		total number of agents in the mission plane
	agents_per_cluster  : int
		maximum number of agent for each cluster
	space_size          : float
		dimension of the mission plane
	"""
	def __init__(self, agents_number: int, agents_per_cluster: int, space_size: float):
		"""
		Parameters
		----------
		agents_number       : int
			total number of agents in the mission plane
		agents_per_cluster  : int
			maximum number of agent for each cluster
		space_size          : float
			dimension of the mission plane
		"""
		self.agents_number = agents_number
		self.agents_per_cluster = agents_per_cluster
		self.space_size = space_size

	@staticmethod
	def get_current_from_history(step: int, history: np.ndarray) -> np.ndarray:
		"""
		Gets the positions at a given time step, from the positions history.

		Parameters
		----------
		step    : int
			time step for which to get the positions
		history : numpy.ndarray
			shape - (agents_number, 2, steps_so_far)
			3D matrix with all the agents' positions for each time step so far
		Returns
		-------
		numpy.ndarray
			shape - (agents_number, 2)
			positions at a given time step
		"""
		return history[:, :, step-1]

	def get_random_position_in_boundaries(self) -> np.ndarray:
		"""
		Returns a random valid position inside the mission plane.

		Returns
		-------
		np.ndarray
			shape - (1, 2)
			position
		"""
		return np.random.randint(0, int(self.space_size + 1), size=(1, 2))

	def is_in_boundaries(self, position: np.ndarray((2,))) -> bool:
		"""
		Checks if a position is inside the mission plane's boundaries.

		Parameters
		----------
		position : numpy.ndarray
			shape - (2,)
			position to check

		Returns
		-------
		bool
			True if it is, False otherwise
		"""
		return 0 < position[0] < self.space_size and 0 < position[1] < self.space_size

	def is_close_to_boundaries(self, position: np.ndarray((2,))) -> bool:
		"""
		Checks if a position is too close to the mission plane boundaries.

		Parameters
		----------
		position : numpy.ndarray
			shape - (2,)
			position to check

		Returns
		-------
		bool
			True if it is too close, False otherwise
		"""
		radius = AgentProperties.MEASUREMENT_ERROR_RADIUS
		return position[0] + radius > self.space_size - 2 or \
				position[1] + radius > self.space_size - 2 or \
				position[0] - radius < 2 or \
				position[1] - radius < 2

	def create_initial_clusters(self) -> (bool, np.ndarray):
		"""
		Creates the initial clusters.

		Returns
		-------
		(bool, numpy.ndarray)
			True if all clusters were successfully created, False otherwise.
			If True, set of initial positions for the agents.
		"""
		positions = np.ones((self.agents_number, 2)) * -1
		agent_index = 0
		complete_clusters_number = self.agents_number // self.agents_per_cluster
		for i in range(complete_clusters_number):
			agent_index, positions = self.create_cluster(self.agents_per_cluster, agent_index, positions)
			if agent_index == -2:
				return False, positions

		number_of_remaining_agents = self.agents_number % self.agents_per_cluster
		agent_index, positions = self.create_cluster(number_of_remaining_agents, agent_index, positions)
		if agent_index == -2:
			return False, positions

		return True, positions

	def create_cluster(self, agents_in_cluster: int, agent_index: int, positions: np.ndarray) -> (int, np.ndarray):
		"""
		Creates a single cluster with the given number of agents.

		Parameters
		----------
		agents_in_cluster : int
			number of agent for this cluster
		agent_index : int
			index of the next agent to be created
		positions : numpy.ndarray
			numpy array where to put the positions

		Returns
		-------
		(int, numpy.ndarray)
			next agent index and positions so far
		"""
		radius = AgentProperties.MEASUREMENT_ERROR_RADIUS + AgentProperties.MAX_MOVEMENT_LENGTH
		spawning_area_radius = agents_in_cluster * 2 * radius
		cluster_center = np.random.random((2,)) * (self.space_size - 2 * (spawning_area_radius + 10)) + (spawning_area_radius + 10)
		spawning_range = spawning_area_radius - radius
		max_number_of_tries = 1000
		for j in range(agents_in_cluster):
			for try_number in range(max_number_of_tries):
				random_position = np.random.random((2,)) * (2 * spawning_range) - spawning_range
				new_position = cluster_center + random_position
				if not CollisionAvoidanceHelper.is_initial_position_colliding(agent_index, new_position, positions):
					positions[agent_index, :] = new_position
					break
				if try_number == max_number_of_tries - 1:
					return -2, positions
			agent_index += 1
		return agent_index, positions
