import numpy as np
from RendezvousAlgorithm import RendezvousAlgorithm
from domain.AgentProperties import AgentProperties
from domain.TimeStepData import TimeStepData
from helpers.MathHelper import MathHelper
from helpers.ErrorCalculatorHelper import ErrorCalculator


class Algorithm5Flocking(RendezvousAlgorithm):
	"""
	Rendezvous concretion with improved Flocking algorithm
	"""

	ALGORITHM_NAME = "Algorithm5_Flocking"
	MAX_EXPLORATION_TIME = 500
	"""
	int: Maximum time step allowed before quitting
	"""

	def __init__(self, agents_number: int, space_size: float):
		"""
		Parameters
		----------
		agents_number   : int
			total number of agents in the mission plane
		space_size      : float
			dimension of the mission plane
		"""
		super(Algorithm5Flocking, self).__init__(agents_number, space_size)

	def run(self):
		"""
		Runs the algorithm.

		Returns
		-------
		None
		"""
		current_positions = self.mission_plane_helper.get_current_from_history(1, self.coords_history)
		current_polytopes = self.towers_array[0].get_polytopes()
		step = 2
		stop = np.zeros(self.agents_number)

		while step <= Algorithm5Flocking.MAX_EXPLORATION_TIME:
			for tower_index in range(RendezvousAlgorithm.COMMS_TOWERS_NUMBER):
				comms_tower = self.towers_array[tower_index]
				comms_tower.set_polytopes(current_polytopes)

				self.coords_history = np.append(self.coords_history, np.zeros((self.agents_number, 2, 1)), 2)
				target_index = (step-1) % self.agents_number
				current_neighbors = comms_tower.get_current_neighbors_for_agent(target_index, current_positions)
				snapshot_scan_polytopes = comms_tower.get_polytopes()
				snapshot_scan_positions = comms_tower.get_polytopes_center_positions()
				snapshot_scan_velocities = comms_tower.get_polytopes_velocities()

				for agent_index in range(self.agents_number):
					# If the agent already stopped, continue
					if stop[agent_index]:
						continue

					# If the agent is not in the current strip, continue
					if not current_neighbors[agent_index]:
						continue

					heading = self.update_movement(agent_index, snapshot_scan_positions, snapshot_scan_velocities, current_neighbors)
					new_position = current_positions[agent_index, :] + heading

					if self.mission_plane_helper.is_close_to_boundaries(new_position):
						continue

					heading = self.collision_avoidance_helper.get_maximum_heading_without_collision(
																		agent_index, heading, snapshot_scan_polytopes)

					# TODO - Make this not a hammer - consider steer-to-avoid.
					new_position = current_positions[agent_index, :] + heading
					if not self.mission_plane.is_subdivision_available(new_position):
						heading = np.zeros((2,))

					agent_polytope = snapshot_scan_polytopes[agent_index]
					agent_polytope.translate(heading)
					current_positions[agent_index, :] = agent_polytope.get_center_position()
					stop[agent_index] = 0

				error = ErrorCalculator.calculate_error(current_positions)
				current_polytopes = comms_tower.get_polytopes()

				time_step_data = TimeStepData(step, self.gradient_map, tower_index + 1, target_index, current_polytopes, error)
				self.simulation_data.add_time_step(time_step_data)
				self.distance_errors = np.append(self.distance_errors, np.array([error]), 0)
				self.coords_history[:, :, step - 1] = current_positions
				step += 1

				'''
				self.gradient_map.translate_rendezvous_area(self.mission_plane_helper.is_in_boundaries)
				'''
				'''
				if step % 20 == 0:
					random_is_create = True if np.random.rand() < 0.5 else False
					random_is_maximum = True if np.random.rand() < 0.5 else False
					if random_is_create:
						self.gradient_map.create_rendezvous_area(self.mission_plane_helper.get_random_position_in_boundaries, is_create_maximum=random_is_maximum)
					else:
						self.gradient_map.remove_rendezvous_area(is_remove_maximum=random_is_maximum)
				'''

				# Only checking this every 10 discrete time steps for better performance.
				if step % 10 == 0:
					self.gradient_map.check_if_agents_explored_areas(self.mission_plane_helper.get_random_position_in_boundaries,
																		current_positions)

				if not self.SIMULATION_MODE:
					self.draw(target_index, comms_tower)

	def update_movement(self, agent_index: int, current_positions: np.ndarray, current_velocities: np.ndarray,
						distribution) -> np.ndarray:
		"""
		Calculates the heading of an agent, based on the algorithm rules and the other agent's positions and velocities.

		Parameters
		----------
		agent_index         : int
			index of the agent that is moving
		current_positions   : numpy.ndarray
			shape - (agents_number, 2)
			positions of the agents
		current_velocities  : numpy.ndarray
			shape - (agents_number, 2)
			velocities of the agents
		distribution        : numpy.ndarray
			shape - (agents_number,)
			1D array that represents if each agent is a neighbor of the moving agent

		Returns
		-------
		numpy.ndarray
		"""
		temp_error_current_positions = current_positions + (np.random.random((self.agents_number, 2)) * 2 - 1)
		temp_error_current_velocities = current_velocities + (np.random.random((self.agents_number, 2)) * 2 - 1)
		neighbors_number_to_consider = 7
		neighbors_distance_to_consider = 5

		separation_vector = self.get_separation_vector(agent_index, temp_error_current_positions, distribution,
														neighbors_distance_to_consider)
		cohesion_vector = Algorithm5Flocking.get_cohesion_vector(agent_index, temp_error_current_positions,
																	distribution, neighbors_number_to_consider)
		alignment_vector = Algorithm5Flocking.get_alignment_vector(agent_index, temp_error_current_velocities,
																	distribution, neighbors_number_to_consider)
		attraction_vector = self.get_attraction_vector(agent_index, temp_error_current_positions, distribution,
														neighbors_number_to_consider)
		randomness_vector = np.random.random((2,)) * 2 - 1

		gradient_vector = self.get_gradient_vector(agent_index, temp_error_current_positions)

		acceleration = separation_vector + alignment_vector + cohesion_vector + attraction_vector \
						+ gradient_vector + randomness_vector

		norm = np.linalg.norm(acceleration)
		if norm != 0:
			acceleration /= norm

		heading = current_velocities[agent_index, :] + acceleration
		norm = np.linalg.norm(heading)
		if norm != 0:
			heading /= norm

		return heading

	def get_separation_vector(self, agent_index: int, positions: np.ndarray, distribution: np.ndarray,
								neighbors_distance_to_consider: float) -> np.ndarray((2,)):
		"""
		Calculates the separation vector.

		Parameters
		----------
		agent_index                     : int
			index of the agent that is moving
		positions                       : numpy.ndarray
			shape - (agents_number, 2)
			positions of the agents
		distribution                    : numpy.ndarray
			shape - (agents_number,)
			1D array that represents if each agent is a neighbor of the moving agent
		neighbors_distance_to_consider  : int
			maximum distance to agent to consider it relevant neighbor

		Returns
		-------
		numpy.ndarray
			shape - (2,)
			separation vector
		"""
		agent_cluster = distribution[agent_index]
		agents_in_cluster = np.array(distribution == agent_cluster)
		agents_in_cluster[agent_index] = 0
		positions_in_cluster = positions[agents_in_cluster, :]
		agent_position = positions[agent_index, :]
		vector = np.zeros((2,))
		upper_limit = np.size(positions_in_cluster, 0)
		count = 0
		for i in range(upper_limit):
			neighbor_position = positions_in_cluster[i, :]
			avoidance = agent_position - neighbor_position
			distance = np.linalg.norm(avoidance)
			if 0 < distance < neighbors_distance_to_consider:
				vector += avoidance / distance
				count += 1
		vector, count = self.avoid_boundaries(agent_position, vector, count)
		norm = np.linalg.norm(vector)
		if norm != 0 and count != 0:
			vector /= count
			vector /= norm
		return vector

	@staticmethod
	def get_cohesion_vector(agent_index: int, positions: np.ndarray, distribution: np.ndarray,
							neighbors_number_to_consider: int) -> np.ndarray((2,)):
		"""
		Calculates the cohesion vector.
		Parameters
		----------
		agent_index                     : int
			index of the agent that is moving
		positions                       : numpy.ndarray
			shape - (agents_number, 2)
			positions of the agents
		distribution                    : numpy.ndarray
			shape - (agents_number,)
			1D array that represents if each agent is a neighbor of the moving agent
		neighbors_number_to_consider    : int
			maximum number of agents to consider as relevant neighbors

		Returns
		-------
		numpy.ndarray
			shape - (2,)
			cohesion vector
		"""
		agent_cluster = distribution[agent_index]
		agents_in_cluster = np.array(distribution == agent_cluster)
		agents_in_cluster[agent_index] = 0
		positions_in_cluster = positions[agents_in_cluster, :]
		res, considered_positions_in_cluster = MathHelper.get_relevant_elements_by_number(
															positions_in_cluster, neighbors_number_to_consider)

		if not res:
			return np.zeros((2,))

		vector = np.mean(considered_positions_in_cluster, axis=0)
		vector -= positions[agent_index, :]

		norm = np.linalg.norm(vector)
		if norm != 0:
			vector /= norm
		return vector

	@staticmethod
	def get_alignment_vector(agent_index: int, velocities: np.ndarray, distribution: np.ndarray,
								neighbors_number_to_consider: int) -> np.ndarray((2,)):
		"""
		Calculates the alignment vector.

		Parameters
		----------
		agent_index                     : int
			index of the agent that is moving
		velocities                      : numpy.ndarray
			shape - (agents_number, 2)
			velocities of the agents
		distribution                    : numpy.ndarray
			shape - (agents_number,)
			1D array that represents if each agent is a neighbor of the moving agent
		neighbors_number_to_consider  : int
			maximum number of agents to consider relevant agents

		Returns
		-------
		numpy.ndarray
			shape - (2,)
			alignment vector
		"""
		agent_cluster = distribution[agent_index]
		agents_in_cluster = np.array(distribution == agent_cluster)
		agents_in_cluster[agent_index] = 0
		velocities_in_cluster = velocities[agents_in_cluster, :]
		res, considered_velocities_in_cluster = MathHelper.get_relevant_elements_by_number(
															velocities_in_cluster, neighbors_number_to_consider)

		if not res:
			return np.zeros((2,))

		vector = np.mean(considered_velocities_in_cluster, axis=0)
		vector -= velocities[agent_index, :]

		norm = np.linalg.norm(vector)
		if norm != 0:
			vector /= norm
		return vector

	def get_attraction_vector(self, agent_index: int, positions: np.ndarray, distribution: np.ndarray,
								neighbors_number_to_consider: int) -> np.ndarray((2,)):
		"""
		Calculates the attraction vector based on the height map values of the agents.

		Parameters
		----------
		agent_index                     : int
			index of the agent that is moving
		positions                       : numpy.ndarray
			shape - (agents_number, 2)
			positions of the agents
		distribution                    : numpy.ndarray
			shape - (agents_number,)
			1D array that represents if each agent is a neighbor of the moving agent
		neighbors_number_to_consider  : int
			maximum number of agents to consider relevant neighbors

		Returns
		-------
		numpy.ndarray
			shape - (2,)
			attraction vector
		"""
		agent_cluster = distribution[agent_index]
		agents_in_cluster = np.array(distribution == agent_cluster)
		agents_in_cluster[agent_index] = 0
		positions_in_cluster = positions[agents_in_cluster, :]
		res, considered_positions_in_cluster = MathHelper.get_relevant_elements_by_number(
															positions_in_cluster, neighbors_number_to_consider)

		if not res:
			return np.zeros((2,))

		vector = np.zeros((2,))
		agent_position = positions[agent_index]
		agent_attraction_factor = self.gradient_map.get_value(agent_position)
		for i in range(np.size(considered_positions_in_cluster, 0)):
			other_agent_position = considered_positions_in_cluster[i]
			other_agent_attraction_factor = self.gradient_map.get_value(other_agent_position)
			attraction_vector = (other_agent_attraction_factor - agent_attraction_factor) * \
								(other_agent_position - agent_position)
			vector += attraction_vector

		norm = np.linalg.norm(vector)
		if norm != 0:
			vector /= norm

		return vector

	def get_gradient_vector(self, agent_index: int, positions: np.ndarray) -> np.ndarray((2,)):
		"""
		Calculates the gradient vector.

		Parameters
		----------
		agent_index                     : int
			index of the agent that is moving
		positions                       : numpy.ndarray
			shape - (agents_number, 2)
			positions of the agents

		Returns
		-------
		numpy.ndarray
			shape - (2,)
			gradient vector
		"""
		agent_position = positions[agent_index, :]
		return self.gradient_map.get_gradient_values(agent_position)

	def avoid_boundaries(self, position: np.ndarray, vector: np.ndarray((2,)), count: int) -> (np.ndarray((2,)), float):
		"""
		Calculates a vector that goes away from the mission plane boundaries, if the agent is close enough.

		Parameters
		----------
		position    : numpy.ndarray
			shape - (2,)
			position of the agent
		vector      : numpy.ndarray
			shape - (2,)
			movement vector
		count       : int
			number of components that were added to the vector so far

		Returns
		-------
		(numpy.ndarray, int)
			vector of separation from the mission plane boundaries, and the new number of vector components
		"""
		radius = AgentProperties.MEASUREMENT_ERROR_RADIUS
		x = position[0]
		y = position[1]

		# Top boundary
		if y + radius > self.space_size - 5:
			vector += np.array([0, -10])
			count += 1
		# Bottom boundary
		if y - radius < 5:
			vector += np.array([0, 10])
			count += 1
		# Right boundary
		if x + radius > self.space_size - 5:
			vector += np.array([-10, 0])
			count += 1
		# Left boundary
		if x - radius < 5:
			vector += np.array([10, 0])
			count += 1
		return vector, count
