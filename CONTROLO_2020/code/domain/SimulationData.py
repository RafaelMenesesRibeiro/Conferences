import numpy as np

from domain.MissionPlane import MissionPlane
from domain.TimeStepData import TimeStepData


class SimulationData:
	"""
	Represents the static data throughout a simulation.
	Will be used to replicate simulations, if new
	statistics or graphs are needed.

	Attributes:
	----------
	agents_number       : int
		number of agents
	space_size          : float
		size of the mission plane
	towers_positions    : np.array
		shape - (2, towers_number)
		positions of the towers
	time_step_data_list : list<domain.TimeStepData.TimeStepData>
		list of time step data - data that changes with time
	"""

	def __init__(self, agents_number: int, mission_plane: MissionPlane, space_size: float, towers_positions: np.ndarray,
					strip_half_angle: int, polytope_radius: float):
		"""
		Parameters
		----------
		agents_number       : int
			number of agents
		mission_plane       : MissionPlane
			instance of MissionPlane
		space_size          : float
			size of the mission plane
		towers_positions    : numpy.ndarray
			shape - (towers number, 2)
			positions of the towers
		strip_half_angle    : float
			half of the angle of the strip, used to calculate the boundaries by rotating the center
			vector +/- the half angle
		polytope_radius     : float
			radius bounding the size of the polyshapes
		"""
		self.agents_number = agents_number
		self.mission_plane = mission_plane
		self.space_size = space_size
		self.towers_positions = towers_positions
		self.strip_half_angle = strip_half_angle
		self.polytope_radius = polytope_radius
		self.time_step_data_list = []

	def add_time_step(self, time_step_data):
		"""
		Adds a new time step data instance to the list.

		Parameters
		----------
		time_step_data : domain.TimeStepData.TimeStepData
			data to add
		"""
		self.time_step_data_list.append(time_step_data)

	def encode(self) -> dict:
		"""
		Encode the execution data to a dictionary.
		"""

		towers_data = {}
		towers_data["Towers number"] = self.towers_positions.shape[0]
		towers_data["Strip half angle"] = self.strip_half_angle
		towers_positions = {}
		for index in range(self.towers_positions.shape[0]):
			position = self.towers_positions[index, :]
			towers_positions["Tower " + str(index + 1)] = [position[0], position[1]]
		towers_data["Towers positions"] = towers_positions

		result = {}
		result["Agents number"] = self.agents_number
		result['Illegal zones'] = self.mission_plane.encode()
		result["Polytopes radius"] = self.polytope_radius
		result["Space size"] = [self.space_size, self.space_size]
		result["Towers data"] = towers_data
		time_steps_dict = {}
		for time_step in self.time_step_data_list:
			key = "Time step " + str(time_step.get_time_step())
			time_steps_dict[key] = time_step.encode()
		result["Execution length"] = len(self.time_step_data_list)
		result["Time steps"] = time_steps_dict
		return result
