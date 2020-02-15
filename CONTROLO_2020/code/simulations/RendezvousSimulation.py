import os
import time
import json
from enum import Enum
from math import floor
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Algorithm5_Flocking import Algorithm5Flocking
from RendezvousAlgorithm import RendezvousAlgorithm
from domain.AgentProperties import AgentProperties
from domain.GradientMap import GradientMap
from domain.MissionPlane import MissionPlane
from domain.Polytope import Polytope
from domain.CommsTower import CommsTower
from helpers.NeighborsHelper import NeighborsHelper
from helpers.DrawerHelper import DrawerHelper


class RendezvousSimulation:
	class SimulationType(Enum):
		"""
		Represents the type of simulation running:
			- SIMPLE                : no parameter changes.
			- BATCH                 : no parameter changes, but executes multiple simulations.
			- PARAM_AGENTS_NUMBER   : changes the agents number, uses simulations.AgentsNumberSimulation
			- PARAM_SPACE_SIZE      : changes the space size, uses simulations.SpaceSizeSimulation
		"""
		SIMPLE = 'Simple'
		BATCH = 'Batch'
		PARAM_AGENTS_NUMBER = 'Param_Agents_Number'
		PARAM_SPACE_SIZE = 'Param_Space_Size'

	"""
	Base class for simulations.
	"""

	SIMULATIONS_TYPE = SimulationType.SIMPLE
	"""
	Enum: Type of simulation running. 
	"""

	SIMULATIONS_NAME = 'UNDEFINED'
	"""
	str: Name of the batch of simulations. Example: '10_agents_100_space_size'
	"""

	DRAW_SIMULATION = True
	"""
	bool: Flag to draw distance errors and agents' positions in certain time steps at the end of each simulation.
	"""

	SAVE_SIMULATION_FIGURES = True
	"""
	bool: Flag to save distance errors and agents' positions in certain time steps at the end of each simulation.
	"""

	@staticmethod
	def run_simulations(algorithm: RendezvousAlgorithm, simulations_number: int) -> np.ndarray:
		"""
		Runs the algorithm for statistical analysis.

		Parameters
		----------
		algorithm           : RendezvousAlgorithm.RendezvousAlgorithm
			algorithm instance to run several times
		simulations_number  : int
			number of times to run the refreshed algorithm

		Returns
		-------
		numpy.ndarray
			shape - (simulations_number,)
			final number of clusters for each simulation
		"""
		RendezvousAlgorithm.SIMULATION_MODE = True
		RendezvousSimulation.DRAW_SIMULATION = False

		agents_number = algorithm.agents_number
		simulations_errors = np.zeros((simulations_number,))
		simulations_final_number_clusters = np.zeros((simulations_number,))
		for simulation_index in range(simulations_number):
			iteration_final_error, iteration_final_positions = RendezvousSimulation.run_single(algorithm)
			iteration_final_number_clusters = \
				NeighborsHelper.get_number_of_clusters(agents_number, iteration_final_positions, algorithm.towers_array)
			simulations_errors[simulation_index] = iteration_final_error
			simulations_final_number_clusters[simulation_index] = iteration_final_number_clusters

		print('Finished simulations with {} agents and size {}'.format(algorithm.agents_number, algorithm.space_size))

		average_simulation_error = np.mean(simulations_errors)
		average_final_number_clusters = np.mean(simulations_final_number_clusters)
		print('Average simulation error: {}'.format(average_simulation_error))
		print('Average simulation final number of clusters: {}'.format(average_final_number_clusters))

		return simulations_final_number_clusters

	@staticmethod
	def run_single(algorithm: RendezvousAlgorithm, dir_name: str = None) -> (float, np.ndarray):
		"""
		Runs the algorithm one time.

		Parameters
		----------
		algorithm : RendezvousAlgorithm
			algorithm to run
		dir_name  : str
			directory in which to save the figures

		Returns
		-------
		(float, numpy.ndarray)
			Final distance error and final positions
		"""
		algorithm.run()
		history = algorithm.coords_history
		distance_errors = algorithm.distance_errors
		final_step = np.size(history, 2)
		final_positions = algorithm.get_current_from_history(final_step, history)

		if RendezvousSimulation.SAVE_SIMULATION_FIGURES:
			RendezvousSimulation.draw_simulation(algorithm, history, final_step, final_positions, distance_errors, dir_name)

		final_error = distance_errors[final_step - 1]
		return final_error, final_positions

	@staticmethod
	def run_simulations_and_save(simulations_number: int, agents_number: int, space_size: float) -> np.ndarray:
		"""
		Runs the algorithm for statistical analysis and saves all the data gathered during each execution for
		replication purposes.

		Parameters
		----------
		simulations_number  : int
			number of times to run the refreshed algorithm
		agents_number       : int
			number of agents in the simulation
		space_size          : float
			size of the mission plane

		Returns
		-------
		numpy.ndarray
			shape - (simulations_number,)
			final number of clusters for each simulation
		"""
		RendezvousAlgorithm.SIMULATION_MODE = True
		RendezvousSimulation.DRAW_SIMULATION = False
		dir_name = RendezvousSimulation.create_execution_dir()

		simulations_errors = np.zeros((simulations_number,))
		simulations_final_number_clusters = np.zeros((simulations_number,))
		for simulation_index in range(simulations_number):
			try:
				simulation_name = 'simulation_{}'.format(simulation_index + 1)
				iteration_dir_name = os.path.join(dir_name, simulation_name)
				os.makedirs(iteration_dir_name)

				algorithm = Algorithm5Flocking(agents_number=agents_number, space_size=space_size)

				iteration_final_error, iteration_final_positions = RendezvousSimulation.run_single(algorithm, iteration_dir_name)
				iteration_final_number_clusters = NeighborsHelper.get_number_of_clusters(agents_number,
																iteration_final_positions, algorithm.towers_array)
				simulations_errors[simulation_index] = iteration_final_error
				simulations_final_number_clusters[simulation_index] = iteration_final_number_clusters

				data = algorithm.encode_data()
				data['simulation_error'] = iteration_final_error
				data['simulation_final_number_clusters'] = iteration_final_number_clusters

				file_name = os.path.join(iteration_dir_name, 'simulation_{}.json'.format(simulation_index+1))
				with open(file_name, "w") as outfile:
					json.dump(data, outfile, indent=4)

			except Exception as exc:
				print(str(exc))
				continue

		average_simulation_error = np.mean(simulations_errors)
		average_final_number_clusters = np.mean(simulations_final_number_clusters)

		data = {
				'average_simulation_error': average_simulation_error,
				'average_final_number_clusters': average_final_number_clusters,
				'simulations_errors': simulations_errors.tolist(),
				'simulations_final_number_clusters': simulations_final_number_clusters.tolist()
		}
		file_name = os.path.join(dir_name, 'general_statistics.json')
		with open(file_name, "w") as outfile:
			json.dump(data, outfile, indent=4)

		print('Average simulation error: {}'.format(average_simulation_error))
		print('Average simulation final number of clusters: {}'.format(average_final_number_clusters))
		return simulations_final_number_clusters

	@staticmethod
	def run_single_and_save(algorithm: RendezvousAlgorithm) -> None:
		"""
		Runs the algorithm one time and saves the execution data.

		Parameters
		----------
		algorithm : RendezvousAlgorithm
			algorithm to run
		"""
		algorithm.run()
		history = algorithm.coords_history
		distance_errors = algorithm.distance_errors
		final_step = np.size(history, 2)
		final_positions = algorithm.get_current_from_history(final_step, history)

		dir_name = RendezvousSimulation.create_execution_dir()
		if RendezvousSimulation.SAVE_SIMULATION_FIGURES:
			RendezvousSimulation.draw_simulation(algorithm, history, final_step, final_positions, distance_errors, dir_name)

		file_name = os.path.join(dir_name, 'data.json')
		data = algorithm.encode_data()
		with open(file_name, "w") as outfile:
			json.dump(data, outfile, indent=4)

	@staticmethod
	def replicate_simulation(file_path: str) -> None:
		"""
		Replicates the algorithm by drawing the execution saved in the data parameter.

		Parameters
		----------
		file_path : str
			file path of data representing a simulation:
				- size of the mission plane
				- gradient map
				- number of agents
				- position of the towers
				- positions of the agents over time
				- target agents for each of the towers over time
		"""
		with open(file_path, "r") as data_file:
			data = json.load(data_file)

		space_size = data["Space size"][0]
		agents_number = data["Agents number"]
		polytopes_radius = data["Polytopes radius"]
		comms_towers_data = data["Towers data"]
		towers_number = comms_towers_data["Towers number"]
		towers_strip_half_angle = comms_towers_data["Strip half angle"]
		towers_positions = comms_towers_data["Towers positions"]

		time_steps_number = data["Execution length"]
		time_steps = data["Time steps"]

		t = Algorithm5Flocking(agents_number, space_size)

		if 'Illegal zones' in data:
			mission_plane_data = data["Illegal zones"]
			square_subdivision_length = mission_plane_data["Subdivision length"]
			plane_subdivisions_availability = np.array(mission_plane_data["Availability matrix"])
			mission_plane = MissionPlane(space_size, square_subdivision_length, plane_subdivisions_availability)
		else:
			mission_plane = MissionPlane(space_size, availability_matrix=MissionPlane.create_full_availability_matrix(space_size))
		t.mission_plane = mission_plane

		comms_towers = []
		for i in range(towers_number):
			key = "Tower " + str(i+1)
			tower_position = np.array([towers_positions[key][0], towers_positions[key][1]])
			tower = CommsTower(tower_position, space_size, agents_number, towers_strip_half_angle)
			comms_towers.append(tower)
		t.towers_array = comms_towers

		# The algorithm only starts saving data at 3
		initial_time_step_number = 3

		plt.figure(figsize=(8, 6), dpi=120)

		for i in range(time_steps_number):
			key = "Time step " + str(initial_time_step_number + i)
			if key not in time_steps:
				break
			current_time_step = time_steps[key]

			active_comms_tower_index = current_time_step["Active tower"] - 1
			active_comms_tower = comms_towers[active_comms_tower_index]
			target_index = current_time_step["Target agent"]

			t.gradient_map = RendezvousSimulation.generate_gradient_map(mission_plane, space_size, current_time_step)

			agents_data = current_time_step["Agents"]
			current_positions = np.zeros((agents_number, 2))
			for agent_index in range(agents_number):
				key = "Agent " + str(agent_index + 1)
				agent_data = agents_data[key]
				agent_position_array = agent_data["Center Position"]
				agent_position = np.array([agent_position_array[0], agent_position_array[1]])
				current_positions[agent_index, :] = agent_position
			polytopes = Polytope.create_polytopes_from_positions(agents_number, current_positions, polytopes_radius)

			target_position = current_positions[target_index, :]
			DrawerHelper.draw_replica(t, polytopes, active_comms_tower, target_position)

			plt.savefig(os.path.join(os.getcwd(), 'videos', 'positions_t{:04d}.jpg'.format(i)))

			if i % 100 == 0:
				print('time ' + str(i))

		image_folder = os.path.join(os.getcwd(), 'videos')
		images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
		images.sort()
		fourcc = cv2.VideoWriter_fourcc(*'MPEG')
		video = cv2.VideoWriter('video.mp4', fourcc, 10.0, (960, 720))

		for i in range(len(images)):
			image = images[i]
			image = cv2.imread(os.path.join(image_folder, image))
			video.write(image)

		cv2.destroyAllWindows()
		video.release()

	@staticmethod
	def draw_simulation(algorithm: RendezvousAlgorithm, history: np.ndarray, final_step: int,
						final_positions: np.ndarray, distance_errors: np.ndarray, dir_name: str = None) -> None:
		"""
		Draws and saves the execution figures.

		Parameters
		----------
		algorithm       : RendezvousAlgorithm
			algorithm to run
		history         : numpy.ndarray
			shape - (agents_number, 2, time_steps_so_far)
			3D matrix with all the agents' positions for each time step so far
		final_step      : int
			last discrete time step of the simulation
		final_positions : numpy.ndarray
			shape - (agents_number, 2)
			positions of agents at the final step of the simulation
		distance_errors : numpy.ndarray
			shape - (time_steps_so_far,)
			array that stores the calculated error for each time step
		dir_name        : str
			name of the directory in which to save
		"""
		agents_number = algorithm.agents_number
		if dir_name is None:
			dir_name = RendezvousSimulation.create_execution_dir()

		plt.close('all')

		mission_plane = algorithm.mission_plane
		space_size = algorithm.space_size
		algorithm_data = algorithm.simulation_data.encode()

		# Plots the agents' positions at time step = 1
		plt.figure('Initial positions')
		plt.title('Initial positions')
		plt.xlabel('X coordinate')
		plt.ylabel('Y coordinate')
		positions = algorithm.get_current_from_history(1, history)
		polytopes = Polytope.create_polytopes_from_positions(agents_number, positions)
		key = 'Time step 2'
		algorithm.gradient_map = RendezvousSimulation.generate_gradient_map(mission_plane, space_size, algorithm_data['Time steps'][key])
		DrawerHelper.draw_simulation(algorithm, polytopes)
		plt.savefig(os.path.join(dir_name, 'positions_t0.pdf'))

		# Plots the agents' positions at 75% completion
		plt.figure('Positions at 75% completion')
		plt.title('Positions at 75% completion')
		plt.xlabel('X coordinate')
		plt.ylabel('Y coordinate')
		time_step = floor(final_step * 0.75)
		positions = algorithm.get_current_from_history(time_step, history)
		polytopes = Polytope.create_polytopes_from_positions(agents_number, positions)
		key = 'Time step ' + str(time_step)
		algorithm.gradient_map = RendezvousSimulation.generate_gradient_map(mission_plane, space_size, algorithm_data['Time steps'][key])
		DrawerHelper.draw_simulation(algorithm, polytopes)
		plt.savefig(os.path.join(dir_name, 'positions_t0.75.pdf'))

		# Plots the agents' positions at the final time step
		plt.figure('Final positions')
		plt.title('Final positions')
		plt.xlabel('X coordinate')
		plt.ylabel('Y coordinate')
		polytopes = Polytope.create_polytopes_from_positions(agents_number, final_positions)
		time_step = len(algorithm_data['Time steps'])
		key = 'Time step ' + str(time_step)
		algorithm.gradient_map = RendezvousSimulation.generate_gradient_map(mission_plane, space_size, algorithm_data['Time steps'][key])
		DrawerHelper.draw_simulation(algorithm, polytopes)
		plt.savefig(os.path.join(dir_name, 'positions_tf.pdf'))

		# Plots the maximum distance over time
		plt.figure('Distance error over time')
		plt.title('Distance error over time')
		plt.xlabel('Discrete time step')
		plt.ylabel('Error')
		plt.plot(distance_errors)
		plt.savefig(os.path.join(dir_name, 'errors.pdf'))

		if RendezvousSimulation.DRAW_SIMULATION:
			plt.show()

	@staticmethod
	def create_execution_dir() -> str:
		"""
		Creates a directory in which to save execution data / figures.

		Returns
		-------
		str
			Name of the created directory
		"""
		timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())
		sim_dir_name = str(timestamp) + '_' + RendezvousSimulation.SIMULATIONS_NAME
		dir_name = os.path.join(os.getcwd(), 'data', 'simulations', str(RendezvousSimulation.SIMULATIONS_TYPE.value),
								str(RendezvousAlgorithm.GRADIENT_MAP_TYPE.value), sim_dir_name)
		os.makedirs(dir_name)
		return dir_name

	@staticmethod
	def generate_gradient_map(mission_plane, space_size, current_time_step):
		# TODO - docstring.
		gradient_map_data = current_time_step['Gradient Map']
		gradient_map_type = gradient_map_data['gradient_map_type']
		params = {'space_size': space_size}
		if gradient_map_type == GradientMap.GradientMapType.SINGLE_MAX.value:
			params['maximum_point'] = np.array(gradient_map_data['maximum_point'])
			return RendezvousAlgorithm.create_type_map(mission_plane, GradientMap.GradientMapType.SINGLE_MAX, params)
		elif gradient_map_type == GradientMap.GradientMapType.SINGLE_MIN.value:
			params['minimum_point'] = np.array(gradient_map_data['minimum_point'])
			return RendezvousAlgorithm.create_type_map(mission_plane, GradientMap.GradientMapType.SINGLE_MIN, params)
		elif gradient_map_type == GradientMap.GradientMapType.MULTIPLE_MAX_MIN.value:
			params['maximum_points'] = np.array(gradient_map_data['maximum_points'])
			params['maximum_points_utility_values'] = np.array(gradient_map_data['maximum_points_utility_values'])
			params['minimum_points'] = np.array(gradient_map_data['minimum_points'])
			params['minimum_points_utility_values'] = np.array(gradient_map_data['minimum_points_utility_values'])
			return RendezvousAlgorithm.create_type_map(mission_plane, GradientMap.GradientMapType.MULTIPLE_MAX_MIN, params)
