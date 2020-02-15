import math
import numpy as np
from domain.Polytope import Polytope


class MathHelper:
	@staticmethod
	def get_relevant_elements_by_number(array: np.ndarray, number: int) -> (bool, np.array):
		"""
		Builds a numpy array whose size is at most the number given by the parameter number.

		Parameters
		----------
		array   : numpy.ndarray
			array to cut
		number  : int
			maximum size of the array to return

		Returns
		-------
		(bool, numpy.ndarray)
			True if it was possible to cut the array, False otherwise.
			Cur array.
		"""
		size = np.size(array, 0)
		if size == 0:
			return False, array
		elif size <= number:
			return True, array
		return True, array[:number, :]

	@staticmethod
	def is_point_between_points(position_y: float, lower_point_y: float, upper_point_y: float) -> bool:
		"""
		Calculates if a point's y coordinate, given by the position_y parameter, is between two other y coordinates,
		given by the lower_point_y and upper_point_y parameters.

		Parameters
		----------
		position_y      : float
			y position coordinate to check
		lower_point_y   : float
			lower y limit
		upper_point_y   : float
			upper y limit

		Returns
		-------
		bool
			True if between the limits, False otherwise
		"""
		return lower_point_y < position_y < upper_point_y

	@staticmethod
	def rotate_vector(vector: np.ndarray((2,)), sin: float, cos: float) -> np.ndarray((2,)):
		"""
		Rotates the vector according to the given sine and cosine given as parameters.

		Parameters
		----------
		vector  : numpy.ndarray
			shape - (2,)
			vector to rotate
		sin     : float
			sine of the angle of rotation
		cos     : float
			cosine of the angle of rotation

		Returns
		-------
		numpy.ndarray
			rotated vector
		"""
		x = vector[0] * cos - vector[1] * sin
		y = vector[0] * sin + vector[1] * cos
		return np.array([x, y])

	@staticmethod
	def get_strip_start_end_points_from_vector(vector: np.ndarray((2,)), tower_position: np.ndarray((2,)),
												space_size: float) -> np.ndarray((2, 2)):
		"""
		Calculates the start and end points of a comms tower's strip, based on its orientation vector and tower's
		position. Regardless of the side of the mission plane the tower is on, the start point is always at x=0
		and the end at x=space_size.

		Parameters
		----------
		vector          : numpy.ndarray
			shape - (2,)
			orientation vector of the comms tower's strip
		tower_position  : numpy.ndarray
			shape - (2,)
			position of the tower
		space_size      : float
			dimension of the mission plane

		Returns
		-------
		numpy.ndarray
			shape - (2, 2)
			start and end points of the strip
		"""
		m = vector[1] / vector[0]
		b = tower_position[1] - m * tower_position[0]
		points = np.zeros((2, 2))
		# Point at x = 0:
		points[0, :] = np.array([0, b])
		# Point at x = space_size
		points[1, :] = np.array([space_size, m * space_size + b])
		return points

	@staticmethod
	def get_line_from_start_and_end_points(start_position: np.ndarray((2,)), end_position: np.ndarray((2,))) \
											-> np.ndarray((6,)):
		"""
		Calculates line properties based on the line's start and end points:
			- slope (m);
			- y coordinate at x=0 (b);
			- minimum x;
			- maximum x;
			- minimum y;
			- maximum y;

		Parameters
		----------
		start_position  : numpy.ndarray
			shape - (2,)
		end_position    : numpy.ndarray
			shape - (2,)

		Returns
		-------
		numpy.ndarray
			shape - (6,)
			line properties
		"""
		vertex_1 = start_position
		vertex_2 = end_position
		vector = vertex_2 - vertex_1
		if vector[0] != 0:
			m = vector[1] / vector[0]
		else:
			m = math.inf
		b = vertex_1[1] - m * vertex_1[0]
		vertex_1_x = vertex_1[0]
		vertex_2_x = vertex_2[0]
		if vertex_1_x < vertex_2_x:
			line = np.array([m, b, vertex_1_x, vertex_2_x, 0, 0])
		else:
			line = np.array([m, b, vertex_2_x, vertex_1_x, 0, 0])

		vertex_1_y = vertex_1[1]
		vertex_2_y = vertex_2[1]
		if vertex_1_y < vertex_2_y:
			line[4] = vertex_1_y
			line[5] = vertex_2_y
		else:
			line[4] = vertex_2_y
			line[5] = vertex_1_y
		return line

	@staticmethod
	def get_lines(vertices: np.ndarray, heading_offset: np.ndarray((2,))) -> (np.ndarray, np.ndarray):
		"""
		Calculates the lines originating from each of the polyshapes with the direction and magnitude given by the
		heading_ofsset parameter.

		Parameters
		----------
		vertices        : numpy.ndarray
			shape - (8, 2)
			polyshape's vertices
		heading_offset  : numpy.ndarray
			shape (2,)
			direction and magnitude of the lines

		Returns
		-------
		(numpy.ndarray, numpy.ndarray)
			shapes - (vertices_number, 6) and (vertices_number, 2)
			lines and line's origins
		"""
		vertices_number = np.size(vertices, 0)
		lines = np.zeros((vertices_number, 6))
		origins = np.zeros((vertices_number, 2))
		for vertex_index in range(vertices_number):
			vertex_1 = vertices[vertex_index, :]
			vertex_2 = vertex_1 + heading_offset
			vector = vertex_2 - vertex_1
			if vector[0] != 0:
				m = vector[1] / vector[0]
			else:
				m = math.inf
			b = vertex_1[1] - m * vertex_1[0]
			vertex_1_x = vertex_1[0]
			vertex_2_x = vertex_2[0]
			if vertex_1_x < vertex_2_x:
				lines[vertex_index, :] = np.array([m, b, vertex_1_x, vertex_2_x, 0, 0])
			else:
				lines[vertex_index, :] = np.array([m, b, vertex_2_x, vertex_1_x, 0, 0])

			vertex_1_y = vertex_1[1]
			vertex_2_y = vertex_2[1]

			if vertex_1_y < vertex_2_y:
				lines[vertex_index, 4] = vertex_1_y
				lines[vertex_index, 5] = vertex_2_y
			else:
				lines[vertex_index, 4] = vertex_2_y
				lines[vertex_index, 5] = vertex_1_y
			origins[vertex_index, :] = vertex_1
		return lines, origins

	@staticmethod
	def get_edges(vertices: np.ndarray) -> np.ndarray:
		"""
		Calculates the polyshape's edges based on its vertices.

		Parameters
		----------
		vertices : numpy.ndarray
			shape - (8, 2)
			polyshape's vertices

		Returns
		-------
		numpy.ndarray
			shape - (vertices_number, 6)
			set of edges
		"""
		vertices_number = np.size(vertices, 0)
		edges = np.zeros((vertices_number, 6))
		for vertex_index in range(vertices_number - 1):
			vertex_1 = vertices[vertex_index, :]
			vertex_2 = vertices[vertex_index + 1, :]
			vector = vertex_2 - vertex_1
			if vector[0] != 0:
				m = vector[1] / vector[0]
			else:
				m = math.inf
			b = vertex_1[1] - m * vertex_1[0]
			vertex_1_x = vertex_1[0]
			vertex_2_x = vertex_2[0]
			if vertex_1_x < vertex_2_x:
				edges[vertex_index, :] = np.array([m, b, vertex_1_x, vertex_2_x, 0, 0])
			else:
				edges[vertex_index, :] = np.array([m, b, vertex_2_x, vertex_1_x, 0, 0])

			vertex_1_y = vertex_1[1]
			vertex_2_y = vertex_2[1]

			if vertex_1_y < vertex_2_y:
				edges[vertex_index, 4] = vertex_1_y
				edges[vertex_index, 5] = vertex_2_y
			else:
				edges[vertex_index, 4] = vertex_2_y
				edges[vertex_index, 5] = vertex_1_y

		vertex_1 = vertices[vertices_number - 1, :]
		vertex_2 = vertices[0, :]
		vector = vertex_2 - vertex_1
		if vector[0] != 0:
			m = vector[1] / vector[0]
		else:
			m = math.inf
		b = vertex_1[1] - m * vertex_1[0]
		vertex_1_x = vertex_1[0]
		vertex_2_x = vertex_2[0]
		if vertex_1_x < vertex_2_x:
			edges[vertices_number - 1, :] = np.array([m, b, vertex_1_x, vertex_2_x, 0, 0])
		else:
			edges[vertices_number - 1, :] = np.array([m, b, vertex_2_x, vertex_1_x, 0, 0])

		vertex_1_y = vertex_1[1]
		vertex_2_y = vertex_2[1]

		if vertex_1_y < vertex_2_y:
			edges[vertices_number - 1, 4] = vertex_1_y
			edges[vertices_number - 1, 5] = vertex_2_y
		else:
			edges[vertices_number - 1, 4] = vertex_2_y
			edges[vertices_number - 1, 5] = vertex_1_y
		return edges

	@staticmethod
	def is_line_intersecting_polytope(polytope: Polytope, line_start: np.ndarray((2,)), line_end: np.ndarray((2,))) \
										-> bool:
		"""
		Calculates if a line is colliding with a polyshape.

		Parameters
		----------
		polytope    : domain.Polytope.Polytope
			Polytope to check collision
		line_start  : numpy.ndarray
			shape - (2,)
			start of the line to check intersection
		line_end    : numpy.ndarray
			shape - (2,)
			end of the line to check intersection

		Returns
		-------
		bool
			True if line is colliding with the polyshape, False otherwise
		"""
		line = MathHelper.get_line_from_start_and_end_points(line_start, line_end)
		edges = MathHelper.get_edges(polytope.get_vertices())
		edges_number = np.size(edges, 0)
		for edge_index in range(edges_number):
			result, _ = MathHelper.get_earliest_intersection_between_line_segments(line, edges[edge_index, :], line_start)
			if result:
				return True
		return False

	@staticmethod
	def get_earliest_intersection_between_line_polyshape(polyshape: np.ndarray((8, 2)), line: np.ndarray((6,)),
													line_start: np.ndarray((2,)), edges: np.ndarray = None) \
													-> (bool, float):
		"""
		Calculates the closest intersection point between line and polyshape.

		Parameters
		----------
		polyshape   : numpy.ndarray
			shape - (8, 2)
			polyshape's vertices
		line        : numpy.ndarray
			shape - (6,)
			line properties
		line_start  : numpy.ndarray
			shape - (2,)
			start of the line
		edges       : numpy.ndarray
			polyshape's edges

		Returns
		-------
		(bool, float)
			True if colliding, False otherwise. Distance to earliest collision, from start of the line
		"""
		is_colliding = False
		earliest = math.inf
		if edges is None:
			edges = MathHelper.get_edges(polyshape)
		for edge_index in range(np.size(edges, 0)):
			result, distance = MathHelper.get_earliest_intersection_between_line_segments(line, edges[edge_index, :], line_start)
			if result and distance < earliest:
				if distance == 0:
					return True, 0
				earliest = distance
				is_colliding = True
		return is_colliding, earliest

	@staticmethod
	def get_earliest_intersection_between_line_segments(line_1: np.ndarray((6,)), line_2: np.ndarray((6,)),
														origin: np.ndarray((2,))) -> (bool, float):
		"""
		Calculates the closest intersection between two lines.

		Parameters
		----------
		line_1  : numpy.ndarray
			shape - (6,)
			line 1 properties
		line_2  : numpy.ndarray
			shape - (6,)
			line 2 properties
		origin  : numpy.ndarray
			shape - (2,)
			origin point of line 1

		Returns
		-------
		(bool, float)
			True if collision, False otherwise. Distance to earliest collision, from start of the line 1
		"""
		m_1 = line_1[0]
		b_1 = line_1[1]
		m_2 = line_2[0]
		b_2 = line_2[1]
		if m_1 == math.inf and m_2 == math.inf:
			# This code represents collisions between vertical line and vertical edge.
			# It's commented because it is currently not necessary, as the results are equivalent to the current ones.
			# The results are equivalent because even if not comparing a vertical line and vertical edge, in the event
			# there would be one such collision there is also a collision between a point of a vertical line and the
			# vertex of a slanted edge.
			'''
			if line_1[2] == line_1[3] == line_2[2] == line_2[3]:
				if line_1[4] <= line_2[4] <= line_1[5] <= line_2[5]:
					if origin[1] == line_1[4]:
						return True, np.linalg.norm(np.array([line_2[2], line_2[4]]) - origin)
					elif origin[1] == line_1[5]:
						return True, 0
				if line_1[5] >= line_2[5] >= line_1[4] > line_2[4]:
					if origin[1] == line_1[4]:
						return True, 0
					elif origin[1] == line_1[5]:
						return True, np.linalg.norm(np.array([line_2[2], line_2[5]]) - origin)
			'''
			return False, math.inf

		if m_1 == math.inf:
			x = line_1[2]
		elif m_2 == math.inf:
			x = line_2[2]
		else:
			x = (b_2 - b_1) / (m_1 - m_2)

		if line_1[2] <= x <= line_1[3] and line_2[2] <= x <= line_2[3]:
			if m_2 == math.inf:
				y = m_1 * x + b_1
			else:
				y = m_2 * x + b_2

			if not (line_1[4] <= y <= line_1[5] and line_2[4] <= y <= line_2[5]):
				return False, math.inf

			return True, np.linalg.norm(np.array([x, y]) - origin)

		return False, math.inf

	@staticmethod
	def is_intersecting_circle_circle(center_1: np.ndarray((2,)), center_2: np.ndarray((2,)), radius: float) -> bool:
		"""
		Checks if two circles are intersecting.

		Parameters
		----------
		center_1    : numpy.ndarray
			shape - (2,)
			center of circle 1
		center_2    : numpy.ndarray
			shape - (2,)
			center of circle 2
		radius      : float
			radius of circles

		Returns
		-------
		bool
			True if the two circles are colliding, False otherwise
		"""
		return np.linalg.norm(center_1 - center_2) <= (radius * 2)
