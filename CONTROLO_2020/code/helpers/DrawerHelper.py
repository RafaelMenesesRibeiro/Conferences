import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.lines as mlines
from domain.CommsTower import CommsTower


class DrawerHelper:
	"""
	Visualizer.
	Responsible for:
		- drawing the gradient map;
		- drawing the agent's polytopes;
		- drawing the comms tower's strip.

	Attributes:
	----------
	agents_number       : int
		total number of agents in the mission plane
	space_size          : float
		dimension of the mission plane
	circle_radius       : float
		radius bounding the agent's polytopes
	algorithm           : domain.RendezvousAlgorithm.RendezvousAlgorithm
		RendezvousAlgorithm instance
	"""
	def __init__(self, agents_number: int, space_size: float, circle_radius: float, algorithm_instance):
		"""
		Parameters
		----------
		agents_number       : int
			total number of agents in the mission plane
		space_size          : float
			dimension of the mission plane
		circle_radius       : float
			radius bounding the agent's polytopes
		algorithm_instance  : RendezvousAlgorithm.RendezvousAlgorithm
			RendezvousAlgorithm instance
		"""
		self.agents_number = agents_number
		self.space_size = space_size
		# TODO - Remove
		self.circle_radius = circle_radius
		self.algorithm = algorithm_instance

	@staticmethod
	def draw_strip(target: np.ndarray((2,)), lower_boundary: np.ndarray((2, 2)), upper_boundary: np.ndarray((2, 2)),
					tower_position: np.ndarray((2,))) -> None:
		"""
		Draws the comms tower's strip, by drawing a line from the tower to the target (agent in which the strip is
		centered), and the top and bottom boundaries.

		Parameters
		----------
		target          : numpy.ndarray
			shape - (2,)
			position of the target in which the comms tower is centered
		lower_boundary  : numpy.ndarray
			shape - (2, 2)
			start and end points of strip's the lower boundary
		upper_boundary  : numpy.ndarray
			shape - (2, 2)
			start and end points of strip's the upper boundary
		tower_position  : numpy.ndarray
			shape - (2,)
			position of the tower
		Returns
		-------
		None
		"""
		ax = plt.gca()
		tower_x = tower_position[0]
		tower_y = tower_position[1]
		line_1 = mlines.Line2D([lower_boundary[0, 0], lower_boundary[1, 0]], [lower_boundary[0, 1], lower_boundary[1, 1]])
		line_2 = mlines.Line2D([upper_boundary[0, 0], upper_boundary[1, 0]], [upper_boundary[0, 1], upper_boundary[1, 1]])
		line_3 = mlines.Line2D([tower_x, target[0]], [tower_y, target[1]])
		ax.add_line(line_1)
		ax.add_line(line_2)
		ax.add_line(line_3)

	@staticmethod
	def draw_agents(agents_number: int, polytopes: list, space_size: float, draw_circles: bool = True) -> None:
		"""
		Draws the agents' positions as circles.
		Used to display simulation results.

		Parameters
		----------
		agents_number   : int
			total number of agents in the mission plane
		polytopes       : list
			shape - (agents_number, 2)
			positions of the agents
		space_size      : float
			dimension of the mission plane
		draw_circles    : bool
			true if the agent circles are to be drawn, false otherwise

		Returns
		-------
		None
		"""
		ax = plt.gca()
		plt.axis([0, space_size, 0, space_size])
		for agent_index in range(agents_number):
			polytope = polytopes[agent_index]
			position = polytope.get_center_position()
			position_x = position[0]
			position_y = position[1]
			agent_vertices = polytope.get_vertices()
			polygon = Polygon(agent_vertices)
			if draw_circles:
				circle = plt.Circle((position_x, position_y), 2, fill=False)
				ax.add_artist(circle)
			else:
				polygon.set_edgecolor("black")
			ax.add_artist(polygon)

	@staticmethod
	def draw_height_map(height_map) -> None:
		"""
		Draws the height map.

		Parameters
		----------
		height_map : domain.GradientMap.GradientMap

		Returns
		-------
		None
		"""
		# ax = plt.gca()
		# hp = height_map.highest_point
		# circle = plt.Circle((hp[0], hp[1]), 10, color='red', fill=False)
		# ax.add_artist(circle)

		# TODO - Use class constant instead of 2000
		# TODO - Save this map for next iteration, only update if GradientMap has flag of values_updated = True
		masked_array = np.ma.masked_where(height_map.map_values == -2000, height_map.map_values)
		color_map = plt.get_cmap('viridis')
		color_map.set_bad(color='black')
		plt.imshow(masked_array, cmap=color_map)

	@staticmethod
	def draw_comms_towers(towers: list) -> None:
		"""
		Draws the comms towers.

		Parameters
		----------
		towers : list
			list of towers

		Returns
		-------
		None
		"""
		for i in range(len(towers)):
			position = towers[i].get_position()
			plt.plot(position[0], position[1], 'x', mew=5, ms=10, color='red')

	def draw(self, target: np.ndarray((2,)), lower_boundary: np.ndarray((2, 2)), upper_boundary: np.ndarray((2, 2)),
				comms_tower: CommsTower) -> None:
		"""
		Main draw function, call the other ones to draw gradient map, agents and strip.

		Parameters
		----------
		target              : numpy.ndarray
			shape - (2,)
			position of the target in which the comms tower is centered
		lower_boundary  : numpy.ndarray
			shape - (2, 2)
			start and end points of strip's the lower boundary
		upper_boundary  : numpy.ndarray
			shape - (2, 2)
			start and end points of strip's the upper boundary
		comms_tower         : domain.CommsTower.CommsTower
			CommsTower instance
		Returns
		-------
		None
		"""
		plt.clf()
		plt.axis([0, self.space_size, 0, self.space_size])
		DrawerHelper.draw_height_map(self.algorithm.gradient_map)
		polytopes = comms_tower.get_polytopes()
		DrawerHelper.draw_agents(self.agents_number, polytopes, self.space_size)
		DrawerHelper.draw_strip(target, lower_boundary, upper_boundary, comms_tower.get_position())
		plt.pause(0.05)

	@staticmethod
	def draw_simulation(algorithm, polytopes: iter) -> None:
		"""
		Draw function used in simulations. Calls the other ones to draw the gradient map and the agents.

		Parameters
		----------
		algorithm   : domain.RendezvousAlgorithm.RendezvousAlgorithm
			RendezvousAlgorithm instance that is being simulated
		polytopes   : iter
			agents' polytopes

		Returns
		-------
		None
		"""
		space_size = algorithm.space_size
		plt.axis([0, space_size, 0, space_size])
		DrawerHelper.draw_agents(algorithm.agents_number, polytopes, space_size, False)
		DrawerHelper.draw_comms_towers(algorithm.towers_array)
		DrawerHelper.draw_height_map(algorithm.gradient_map)

	@staticmethod
	def draw_replica(algorithm, polytopes: list, comms_tower: CommsTower, target_position: np.ndarray) -> None:
		"""
		Draw function used to draw a saved simulation, replicating it.
		Calls the other ones to draw the gradient map and the agents.

		Parameters
		----------
		algorithm   : domain.RendezvousAlgorithm.RendezvousAlgorithm
			RendezvousAlgorithm instance that is being simulated
		polytopes   : iter
			agents' polytopes
		comms_tower     : domain.CommsTower.CommsTower
			CommsTower instance
		target_position : numpy.ndarray
			shape - (2,)
			position of the target on which to center the strip

		Returns
		-------
		None
		"""
		plt.clf()
		space_size = algorithm.space_size
		plt.axis([0, space_size, 0, space_size])

		lower_boundary, upper_boundary = comms_tower.get_strip_boundaries(target_position)

		DrawerHelper.draw_height_map(algorithm.gradient_map)
		DrawerHelper.draw_comms_towers(algorithm.towers_array)
		DrawerHelper.draw_agents(algorithm.agents_number, polytopes, space_size, False)
		DrawerHelper.draw_strip(target_position, lower_boundary, upper_boundary, comms_tower.get_position())
		# plt.pause(0.05)
