import matplotlib.pyplot as plt
from Algorithm5_Flocking import Algorithm5Flocking
from simulations.RendezvousSimulation import RendezvousSimulation


class AgentsNumberSimulation(RendezvousSimulation):
	"""
	Simulates the same algorithm with a different number of agents
	"""

	MINIMUM_AGENTS_NUMBER = 10
	"""
	int: Minimum number of agents to simulate.
	"""

	MAXIMUM_AGENTS_NUMBER = 11
	"""
	int: Maximum number of agents to simulate.
	"""

	NUMBER_SIMULATIONS = 2
	"""
	int: Number of times to simulate.
	"""

	SPACE_SIZE = 100
	"""
	float: Dimension of the mission plane
	"""

	@staticmethod
	def run() -> None:
		"""
		Runs the simulations, changing the number of agents on each iteration.

		Returns
		-------
		None
		"""
		# Plots the final number of clusters per number of agents, using box plots to represent the multiple
		# simulations with the same number of agents.
		plt.figure('Final number of clusters per agent number')
		plt.title('Final number of clusters per agent number')
		plt.xlabel("Number of agents")
		plt.ylabel("Number of final clusters")
		data = []
		labels = []
		agents_number = AgentsNumberSimulation.MINIMUM_AGENTS_NUMBER
		while agents_number <= AgentsNumberSimulation.MAXIMUM_AGENTS_NUMBER:
			algorithm = Algorithm5Flocking(agents_number, AgentsNumberSimulation.SPACE_SIZE)
			simulations_final_number_clusters = RendezvousSimulation.run_simulations(algorithm,
																			AgentsNumberSimulation.NUMBER_SIMULATIONS)
			agents_number += 1
			data.append(simulations_final_number_clusters)
			labels.append(str(agents_number))

		ax = plt.gca()
		ax.boxplot(data, labels=labels)
		plt.show()
		# TODO - Save results

