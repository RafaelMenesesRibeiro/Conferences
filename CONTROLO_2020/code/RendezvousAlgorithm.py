import numpy as np
from domain.AgentProperties import AgentProperties
from domain.CommsTower import CommsTower
from domain.GradientMap import GradientMap
from domain.MissionPlane import MissionPlane
from domain.SimulationData import SimulationData
from domain.gradient_maps.GradientMapMultipleMaximumsMinimums import GradientMapMultipleMaximumsMinimums
from domain.gradient_maps.GradientMapSingleMaximum import GradientMapSingleMaximum
from domain.gradient_maps.GradientMapSingleMinimum import GradientMapSingleMinimum
from helpers.MissionPlaneHelper import MissionPlaneHelper
from helpers.CollisionAvoidanceHelper import CollisionAvoidanceHelper
from helpers.DrawerHelper import DrawerHelper
from helpers.ErrorCalculatorHelper import ErrorCalculator


class RendezvousAlgorithm:
	"""
	Main logic class.

	Attributes:
	----------
	agents_number               : int
		total number of agents in the mission plane
	space_size                  : float
		dimension of the mission plane
	coords_history              : numpy.ndarray
		shape - (agents_number, 2, time_steps_so_far)
		3D matrix with all the agents' positions for each time step so far
	distance_errors             : numpy.ndarray
		shape - (time_steps_so_far,)
		array that stores the calculated error for each time step
	towers_array                : list<domain.CommsTower.CommsTower>
		array with all the mission plane comms tower
	gradient_map                : domain.GradientMap.GradientMap
		gradient map instance
	mission_plane_helper        : helpers.MissionPlaneHelper.MissionPlaneHelper
		mission plane helper instance
	collision_avoidance_helper  : helpers.CollisionAvoidanceHelper.CollisionAvoidanceHelper
		collision avoidance helper instance
	drawer_helper               : helpers.DrawerHelper.DrawerHelper
		drawer helper instance
	simulation_data             : domain.SimulationData.SimulationData
		data structure to replicate simulations (in case new statistics or graphs are needed)
	"""

	AGENTS_PER_CLUSTER = 1
	"""
	int: Maximum number of agents per cluster 
	"""

	COMMS_TOWERS_NUMBER = 1
	"""
	int: Number of comms towers on the mission plane
	"""

	STRIP_HALF_ANGLE = 8
	"""
	float: Half of the angle of a comms tower's strip
	"""

	GRADIENT_MAP_TYPE = GradientMap.GradientMapType.SINGLE_MAX
	"""
	Enum: Type of gradient map in use.
	"""

	SIMULATION_MODE = False
	"""
	bool: True if running simulations, False otherwise
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
		self.agents_number = agents_number
		self.space_size = space_size
		self.mission_plane_helper = MissionPlaneHelper(
										agents_number,
										RendezvousAlgorithm.AGENTS_PER_CLUSTER,
										space_size)
		result, current_positions = self.mission_plane_helper.create_initial_clusters()

		# TODO - Refactor MissionPlaneHelper to be inside MissionPlane.
		self.mission_plane = MissionPlane(space_size=space_size, square_subdivision_length=10)

		if not result:
			msg = "Could not find non-overlapping positions for the agents."
			raise Exception(msg)

		self.coords_history = np.zeros((agents_number, 2, 1))
		self.coords_history[:, :, 0] = current_positions
		self.distance_errors = np.zeros((1,))
		self.distance_errors[0] = ErrorCalculator.calculate_error(current_positions)

		radius = AgentProperties.MEASUREMENT_ERROR_RADIUS
		polytope_radius = radius * 0.8
		self.towers_array = []
		for tower_index in range(RendezvousAlgorithm.COMMS_TOWERS_NUMBER):
			tower_position = CommsTower.get_random_initial_position(space_size)
			tower = CommsTower(tower_position, space_size, agents_number, RendezvousAlgorithm.STRIP_HALF_ANGLE)
			tower.create_polytopes(current_positions, polytope_radius)
			self.towers_array.append(tower)

		self.collision_avoidance_helper = CollisionAvoidanceHelper(agents_number)

		self.gradient_map = GradientMapSingleMaximum(space_size=space_size, mission_plane=self.mission_plane)
		gradient_map_type = RendezvousAlgorithm.GRADIENT_MAP_TYPE
		params = {'space_size': space_size}
		if gradient_map_type == GradientMap.GradientMapType.SINGLE_MAX:
			params['maximum_point'] = np.array([50, 50]).astype(np.float64)
			self.gradient_map = RendezvousAlgorithm.create_type_map(self.mission_plane, gradient_map_type, parameters=params)
		elif gradient_map_type == GradientMap.GradientMapType.SINGLE_MIN:
			params['minimum_point'] = np.array([50, 50]).astype(np.float64)
			self.gradient_map = RendezvousAlgorithm.create_type_map(self.mission_plane, gradient_map_type, parameters=params)
		elif gradient_map_type == GradientMap.GradientMapType.MULTIPLE_MAX_MIN:
			max_points = (np.random.randint(0, int(self.space_size), size=(2, 2))).astype(np.float64)
			# max_points = np.ones((2, 2))
			# max_points[0, :] = np.array([10, 90])
			# max_points[1, :] = np.array([80, 80])
			# min_points = np.ones((2, 2))
			# min_points[0, :] = np.array([10, 80])
			# min_points[1, :] = np.array([50, 50])
			min_points = (np.random.randint(0, int(self.space_size), size=(2, 2))).astype(np.float64)
			params['maximum_points'] = max_points
			params["maximum_points_utility_values"] = None
			params['minimum_points'] = min_points
			params["minimum_points_utility_values"] = None
			self.gradient_map = RendezvousAlgorithm.create_type_map(self.mission_plane, gradient_map_type, parameters=params)

		self.drawer_helper = DrawerHelper(agents_number, space_size, radius, self)

		towers_positions = np.zeros((RendezvousAlgorithm.COMMS_TOWERS_NUMBER, 2))
		for tower_index in range(RendezvousAlgorithm.COMMS_TOWERS_NUMBER):
			towers_positions[tower_index, :] = self.towers_array[tower_index].get_position()
		self.simulation_data = SimulationData(agents_number, self.mission_plane, space_size, towers_positions,
												RendezvousAlgorithm.STRIP_HALF_ANGLE, polytope_radius)

	def run(self) -> None:
		"""
		Runs the algorithm.

		Returns
		-------
		None
		"""
		raise NotImplementedError("RendezvousAlgorithm is an abstract class")
		pass

	def get_current_from_history(self, step: int, history: np.ndarray) -> np.ndarray:
		"""
		Intermediates with the mission plane helper instance to get the current positions from the coordinates history.

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
		return self.mission_plane_helper.get_current_from_history(step, history)

	@staticmethod
	def create_type_map(mission_plane: MissionPlane, map_type: GradientMap.GradientMapType, parameters: dict):
		"""
		Similar to Factory design pattern. Receives the type of map to create, and the necessary parameters,
		and returns a map instance of the given type.

		Parameters
		----------
		mission_plane   : MissionPlane
			mission plane instance, that contains all the invalid areas contained in the plane
		map_type        : GradientMap.GradientMapType
			type of the gradient map
		parameters      : dict
			parameters necessary for the type of Gradient Map
		"""

		# TODO - Add error handling for missing keys

		space_size = parameters["space_size"]
		if map_type == GradientMap.GradientMapType.SINGLE_MAX:
			return GradientMapSingleMaximum(mission_plane=mission_plane, space_size=space_size, maximum_point=parameters["maximum_point"])
		elif map_type == GradientMap.GradientMapType.SINGLE_MIN:
			return GradientMapSingleMinimum(mission_plane=mission_plane, space_size=space_size, minimum_point=parameters["minimum_point"])
		elif map_type == GradientMap.GradientMapType.MULTIPLE_MAX_MIN:
			return GradientMapMultipleMaximumsMinimums(
														mission_plane=mission_plane,
														space_size=space_size,
														maximum_points=parameters["maximum_points"],
														maximum_points_utility_values=parameters["maximum_points_utility_values"],
														minimum_points=parameters["minimum_points"],
														minimum_points_utility_values=parameters["minimum_points_utility_values"])

	def draw(self, target_index: int, comms_tower: CommsTower) -> None:
		"""
		Intermediates with the drawer helper instance to draw the gradient map, agents and current comms tower's strip.

		Parameters
		----------
		target_index    : int
			index of the target on which to center the strip
		comms_tower     : domain.CommsTower.CommsTower
			CommsTower instance

		Returns
		-------
		None
		"""
		target_position = comms_tower.get_polytope_center_position(target_index)
		lower_boundary, upper_boundary = comms_tower.get_strip_boundaries(target_position)
		self.drawer_helper.draw(target_position, lower_boundary, upper_boundary, comms_tower)

	def encode_data(self) -> dict:
		"""
		Saves all the data gathered during the execution of the algorithm, stored in self.simulation_data .
		"""
		return self.simulation_data.encode()
