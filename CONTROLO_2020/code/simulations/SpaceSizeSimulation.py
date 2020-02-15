import matplotlib.pyplot as plt
from Algorithm5_Flocking import Algorithm5Flocking
from simulations.RendezvousSimulation import RendezvousSimulation


class SpaceSizeSimulation(RendezvousSimulation):
	"""
	Simulates the same algorithm with different mission plane sizes
	"""

	MINIMUM_SIZE = 100
	"""
	int: Minimum size of the mission plane to simulate.
	"""

	MAXIMUM_SIZE = 200
	"""
	int: Maximum size of the mission plane to simulate.
	"""

	SIZE_INCREMENT = 20
	"""
	int: Increment for the size change in each simulation.
	"""

	NUMBER_SIMULATIONS = 1
	"""
	int: Number of times to simulate.
	"""

	AGENTS_NUMBER = 10
	"""
	int: Total number of agents in the mission plane.
	"""

	@staticmethod
	def run() -> None:
		"""
		Runs the simulations, changing the space size on each iteration.

		Returns
		-------
		None
		"""
		plt.figure('Final number of clusters per space size')
		plt.title('Final number of clusters per space size')
		plt.xlabel("Space Size")
		plt.ylabel("Number of final clusters")
		data = []
		labels = []
		space_size = SpaceSizeSimulation.MINIMUM_SIZE
		while space_size <= SpaceSizeSimulation.MAXIMUM_SIZE:
			algorithm = Algorithm5Flocking(SpaceSizeSimulation.AGENTS_NUMBER, space_size)
			simulations_final_number_clusters = RendezvousSimulation.run_simulations(algorithm,
																				SpaceSizeSimulation.NUMBER_SIMULATIONS)
			space_size += SpaceSizeSimulation.SIZE_INCREMENT
			data.append(simulations_final_number_clusters)
			labels.append(str(space_size))

		ax = plt.gca()
		ax.boxplot(data, labels=labels)
		plt.show()
		# TODO - Save results

