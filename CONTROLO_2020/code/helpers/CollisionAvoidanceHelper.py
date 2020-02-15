import math
import numpy as np
from domain.AgentProperties import AgentProperties
from helpers.MathHelper import MathHelper


class CollisionAvoidanceHelper:
	"""
	Responsible for:
		- checking if initial position is colliding, for the clusters creation;
		- calculating the maximum distance an agent can move in a given direction without colliding with the others.

	Attributes:
	----------
	agents_number       : int
		total number of agents in the mission plane
	"""

	IGNORE_DISTANCE = 2 * AgentProperties.MEASUREMENT_ERROR_RADIUS + 2 * AgentProperties.MAX_MOVEMENT_LENGTH
	"""
	float: If the distance of two agents is bigger than this, they will not collide even if both move the maximum movement
	length towards each other, so no calculations need to be made and the performance is improved 
	"""

	SAFE_DISTANCE = 2 * AgentProperties.MEASUREMENT_ERROR_RADIUS + AgentProperties.MAX_MOVEMENT_LENGTH
	"""
	float: If the distance of two agents is bigger than this (and smaller than IGNORE_DISTANCE), they will collide,
	so calculations need to be made in order for the currently moving agent move up to the extended radius of the other
	agent
	"""

	EXTENDED_RADIUS = AgentProperties.MEASUREMENT_ERROR_RADIUS + AgentProperties.MAX_MOVEMENT_LENGTH
	"""
	float: Radius of the polytope that represents every position another agent could have moved to since the current agent 
	received the towers' polytopes measurements
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
	def is_initial_position_colliding(agent_index: int, position: np.ndarray((2,)), positions: np.ndarray) -> bool:
		"""
		Checks if a circle in a given position is colliding with any other circle, whose center is given by the
		positions parameter. This uses basic circle intersection, instead of polytope intersection, because it is
		simpler, faster, and in the initial positions it is best if the agents are spread apart, while being close
		enough to be in the same area.

		Parameters
		----------
		agent_index : int
			index of the circle for which to check collisions
		position    : numpy.ndarray
			shape - (2,)
			position of the circle
		positions   : numpy.ndarray
			shape - (agents_number, 2)
			positions of the other circles

		Returns
		-------
		bool
			True is it is colliding with any of the other, False otherwise
		"""
		radius = AgentProperties.MEASUREMENT_ERROR_RADIUS + AgentProperties.MAX_MOVEMENT_LENGTH
		for other_position_index in range(agent_index):
			other_position = positions[other_position_index, :]
			if MathHelper.is_intersecting_circle_circle(position, other_position, radius):
				return True
		return False

	@staticmethod
	def get_distance_to_collision_with_polyshape(poly_1: np.ndarray, poly_2: np.ndarray,
													heading_offset: np.ndarray((2,))) -> (bool, float):
		"""
		Returns the distance to collision between poly_1 and poly_2 if the polyshape 1 moves in the direction
		and magnitude given by the heading_offset parameter.

		Parameters
		----------
		poly_1          : numpy.ndarray
			shape - (8, 2)
			vertices of the moving polyshape
		poly_2          : numpy.ndarray
			shape - (8, 2)
			vertices of the stationary polyshape
		heading_offset  : numpy.ndarray
			shape - (2,)
			direction and magnitude of the desired movement

		Returns
		-------
		(bool, float)
			distance to collision or maximum movement length. True if collision, False otherwise
		"""
		is_colliding = False
		earliest = math.inf

		poly_1_lines, poly_1_lines_origins = MathHelper.get_lines(poly_1, heading_offset)
		poly_1_edges = MathHelper.get_edges(poly_1)
		for line_index in range(np.size(poly_1_lines, 0)):
			line = poly_1_lines[line_index, :]
			line_start = poly_1_lines_origins[line_index, :]
			result, distance = MathHelper.get_earliest_intersection_between_line_polyshape(poly_2, line, line_start)
			if result and distance < earliest:
				if distance == 0:
					return True, 0
				earliest = distance
				is_colliding = True

		poly_2_lines, poly_2_lines_origins = MathHelper.get_lines(poly_2, -1 * heading_offset)
		for line_index in range(np.size(poly_2_lines, 0)):
			line = poly_2_lines[line_index, :]
			line_start = poly_2_lines_origins[line_index, :]
			result, distance = MathHelper.get_earliest_intersection_between_line_polyshape(poly_1, line, line_start,
																							edges=poly_1_edges)
			if result and distance < earliest:
				if distance == 0:
					return True, 0
				earliest = distance
				is_colliding = True

		return is_colliding, earliest

	def get_maximum_heading_without_collision(self, agent_index: int, heading: np.ndarray((2,)), agent_polytopes: list) \
												-> np.ndarray:
		"""
		Calculates the maximum distance an agent can move in the direction given by the heading parameter without
		colliding with the other polytopes.

		Parameters
		----------
		agent_index     : int
			index of the agent moving
		heading         : numpy.ndarray
			shape - (2,)
			direction of the desired movement
		agent_polytopes : List[domain.Polytope.Polytope]
			polytopes for all the other agents in the mission plane

		Returns
		-------
		numpy.ndarray
			shape - (2,)
			unit heading vector multiplied by the maximum distance without collision
		"""
		heading = heading / np.linalg.norm(heading) * AgentProperties.MAX_MOVEMENT_LENGTH

		agent_polytope = agent_polytopes[agent_index]
		agent_position = agent_polytope.get_center_position()
		expected_agent_position = agent_position + heading
		agent_polyshape = agent_polytope.get_vertices()
		earliest = AgentProperties.MAX_MOVEMENT_LENGTH
		for other_agent_index in range(self.agents_number):
			if other_agent_index == agent_index:
				continue

			other_agent_polytope = agent_polytopes[other_agent_index]
			other_agent_position = other_agent_polytope.get_center_position()
			distance_between_agents = np.linalg.norm(other_agent_position - expected_agent_position)

			if distance_between_agents > CollisionAvoidanceHelper.IGNORE_DISTANCE:
				continue

			if distance_between_agents < CollisionAvoidanceHelper.SAFE_DISTANCE:
				other_agent_polyshape = other_agent_polytope.get_vertices()
				res, distance = self.get_distance_to_collision_with_polyshape(agent_polyshape,
																				other_agent_polyshape, heading)
				distance /= 10
			else:
				extended_other_polyshape = other_agent_polytope.peek_extended(CollisionAvoidanceHelper.EXTENDED_RADIUS)
				res, distance = self.get_distance_to_collision_with_polyshape(agent_polyshape, extended_other_polyshape,
																				heading)

			if res and distance < earliest:
				if distance == 0:
					return np.zeros((2,))
				earliest = distance

		heading = heading / np.linalg.norm(heading) * earliest
		return heading

	def get_circles_maximum_heading_without_collision(self, agent_index: int, heading: np.ndarray((2,)),
														agent_polytopes: list) -> np.ndarray:
		"""
		Similar to get_maximum_heading_without_collision, but assumes the agents' polytopes are circles.
		Used when polytope collision wasn't fully working, and is left here in for that case.
		Calculates the maximum distance an agent can move in the direction given by the heading parameter without
		colliding with the other circles.

		Parameters
		----------
		agent_index     : int
			index of the agent moving
		heading         : numpy.ndarray
			shape - (2,)
			direction of the desired movement
		agent_polytopes : List[domain.Polytope.Polytope]
			polytopes for all the other agents in the mission plane

		Returns
		-------
		numpy.ndarray
			shape - (2,)
			unit heading vector multiplied by the maximum distance without collision
		"""
		agent_polytope = agent_polytopes[agent_index]
		agent_position = agent_polytope.get_center_position()
		heading = heading / np.linalg.norm(heading) * AgentProperties.MAX_MOVEMENT_LENGTH
		expected_agent_position = agent_position + heading
		earliest = math.inf
		overlapping_distance = 2 * AgentProperties.MEASUREMENT_ERROR_RADIUS + AgentProperties.MAX_MOVEMENT_LENGTH

		for other_agent_index in range(self.agents_number):
			if other_agent_index == agent_index:
				continue

			other_agent_polytope = agent_polytopes[other_agent_index]
			other_agent_position = other_agent_polytope.get_center_position()
			distance = np.linalg.norm(expected_agent_position - other_agent_position)
			if distance > overlapping_distance:
				continue

			if distance >= earliest:
				continue

			earliest = distance
			d = overlapping_distance - earliest
			heading = heading / np.linalg.norm(heading) * (1 - d)
		return heading
