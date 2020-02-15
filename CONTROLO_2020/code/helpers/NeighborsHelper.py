import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from domain import CommsTower
from helpers.StripHelper import StripHelper


class NeighborsHelper:
	"""
	Responsible for calculating an agent's current neighbors, based on the agents' polytopes' positions and the comms
	tower strip.

	Attributes:
	----------
	agents_number       : int
		total number of agents in the mission plane
	"""
	def __init__(self, agents_number: int):
		"""
		Parameters
		----------
		agents_number : int
			total number of agents in the mission plane
		"""
		self.agents_number = agents_number

	@staticmethod
	def get_number_of_clusters(agents_number: int, positions: np.ndarray, towers_array: list) -> int:
		"""
		Calculates how many clusters of agents exist, at the final time step, based on the agents' positions.
		Each tower calculates the neighbors of an agent, for every agent.
		Then, an adjacency matrix is filled based on the neighbors found. This matrix is transformed into a graph.
		The number of clusters equals the number of connected components of the adjacency graph.

		Parameters
		----------
		agents_number   : int
			total number of agents in the mission plane
		positions       : numpy.ndarray
			shape - (agents_number, 2)
			positions of the agents
		towers_array    : list<domain.CommsTower.CommsTower>
			list of all the comms tower in the mission plane

		Returns
		-------
		int
			number of final clusters
		"""
		adjacency_matrix = np.zeros((agents_number, agents_number))
		for tower_index in range(len(towers_array)):
			comms_tower = towers_array[tower_index]
			for agent_index in range(agents_number):
				agent_neighbors = comms_tower.neighbors_helper.get_current_neighbors_for_agent(agent_index, positions,
																								comms_tower)
				for other_agent_index in range(agent_index, agents_number):
					if agent_neighbors[other_agent_index]:
						adjacency_matrix[agent_index, other_agent_index] = 1
						adjacency_matrix[other_agent_index, agent_index] = 1
		graph = csr_matrix(adjacency_matrix)
		number_components, _ = connected_components(csgraph=graph, directed=False)
		return number_components

	def get_current_neighbors_for_agent(self, agent_index: int, positions: np.ndarray, comms_tower: CommsTower) \
										-> np.ndarray:
		"""
		Returns the current neighbors for an agent, given by the NeighborsHelper instance.

		Parameters
		----------
		agent_index : int
			index of the agent for which to calculate the neighbors, between 0 and agents_number - 1
		positions : numpy.ndarray
			shape - (agents_number, 2)
			positions of the agents' polytopes
		comms_tower : domain.CommsTower.CommsTower
			active comms tower instance that casts the strip

		Returns
		-------
		numpy.ndarray
			shape - (agents_number,)
			boolean array that represents if each agent is a neighbor of the agent_index agent
		"""
		lower_boundary, upper_boundary = comms_tower.get_strip_boundaries(positions[agent_index, :])
		neighbors = np.zeros(self.agents_number)
		for agent_index in range(self.agents_number):
			other_agent_polytope = comms_tower.get_polytope(agent_index)
			neighbors[agent_index] = StripHelper.is_in_strip_boundaries(other_agent_polytope, lower_boundary, upper_boundary)
		return neighbors
