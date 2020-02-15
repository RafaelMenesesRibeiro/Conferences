import numpy as np

from domain.GradientMap import GradientMap


class TimeStepData:
	"""
	Represents all the data for a discrete time step.
	Will be used to replicate simulations, if new
	statistics or graphs are needed.

	Attributes:
	----------
	time_step                   : int
		identifier of time step
	gradient_map_encoding       : dict
		gradient map encoding
	tower_identifier            : int
		identifier of the active tower
	strip_centered_agent_index  : int
		index of the agent in which the strip is centered
	polytopes                   : list<domain.Polytope.Polytope>
		list of polytope instances (one for each agent)
	distance_error              : float
		calculated error for this time step
	"""

	def __init__(self, time_step: int, gradient_map: GradientMap, tower_identifier: int, agent_index: int, polytopes: list, error: float):
		"""
		Parameters
		----------
		time_step           : int
			identifier of time step
		gradient_map        : domain.GradientMap.GradientMap
			gradient map instance
		tower_identifier    : int
			identifier of the active tower
		agent_index         : int
			index of the agent in which the strip is centered
		polytopes           : list<domain:Polytope.Polytope>
			list of polytope instances (one for each agent)
		error               : float
			calculated error for this time step
		"""
		self.time_step = time_step
		# This encoding is done here, because the values change over time.
		self.gradient_map_encoding = gradient_map.encode()
		self.tower_identifier = tower_identifier
		self.strip_centered_agent_index = agent_index
		self.polytopes_number = len(polytopes)
		self.polytopes_center_positions = [[polytope.get_center_position()[0], polytope.get_center_position()[1]] for polytope in polytopes]
		self.polytopes_velocities = [[polytope.get_velocity()[0], polytope.get_velocity()[1]] for polytope in polytopes]
		self.polytopes_vertices = [polytope.get_vertices().tolist() for polytope in polytopes]
		self.distance_error = error

	def get_time_step(self):
		return self.time_step

	def encode(self) -> dict:
		"""
		Encode the time step data to a dictionary.
		"""
		result = {}
		result["Time step"] = self.time_step
		result['Gradient Map'] = self.gradient_map_encoding
		result["Active tower"] = self.tower_identifier
		result["Target agent"] = self.strip_centered_agent_index
		result["Error"] = self.distance_error

		agents = {}
		for i in range(self.polytopes_number):
			key = "Agent " + str(i + 1)
			agent = {}
			agent["Center Position"] = self.polytopes_center_positions[i]
			agent["Velocity"] = self.polytopes_velocities[i]
			agent["Vertices"] = self.polytopes_vertices[i]
			agents[key] = agent

		result["Agents"] = agents
		return result
