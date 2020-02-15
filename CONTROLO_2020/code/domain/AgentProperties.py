class AgentProperties:
	"""
	Contains constants related with the Agent.
	"""

	MEASUREMENT_ERROR_RADIUS = 2
	""" 
	int: Represents the tower's measurement error of the agent's position. Used as the radius
	of the sphere bounding the agent's polytope.
	"""

	MAX_MOVEMENT_LENGTH = 1
	""" 
	int: Represents the maximum distance an agent can move each movement.
	"""

	ALLOWED_NEIGHBORS_LOSS = 3
	""" 
	int: Represents how many neighbors the agent can lose.  
	"""
