import os
import traceback
import time
from Algorithm5_Flocking import Algorithm5Flocking
from RendezvousAlgorithm import RendezvousAlgorithm
from domain.GradientMap import GradientMap
from simulations.RendezvousSimulation import RendezvousSimulation


try:
	timestamp = time.strftime('%d-%m-%Y_%H-%M-%S', time.gmtime())
	print(timestamp)

	agents_number = 10
	space_size = 100
	# TODO - Load these variables from the simulation files.
	RendezvousAlgorithm.GRADIENT_MAP_TYPE = GradientMap.GradientMapType.MULTIPLE_MAX_MIN
	RendezvousSimulation.SIMULATIONS_TYPE = RendezvousSimulation.SimulationType.BATCH
	RendezvousSimulation.SIMULATIONS_NAME = '{}_agents_{}_space_size'.format(agents_number, space_size)
	RendezvousSimulation.DRAW_SIMULATION = False
	RendezvousSimulation.SAVE_SIMULATION_FIGURES = True
	t = Algorithm5Flocking(agents_number=agents_number, space_size=space_size)

	file_path = os.path.join(os.getcwd(), 'data', 'simulations', 'makevideos',
					'simulation_9', 'simulation_9.json')

	# RendezvousSimulation.replicate_simulation(file_path)
	# RendezvousSimulation.run_single_and_save(t)
	# RendezvousSimulation.run_single(t)
	# RendezvousSimulation.run_simulations_and_save(10, agents_number=agents_number, space_size=space_size)

	timestamp = time.strftime('%d-%m-%Y_%H-%M-%S', time.gmtime())
	print(timestamp)

except Exception as exc:
	print(str(exc))
	traceback.print_tb(exc.__traceback__)
