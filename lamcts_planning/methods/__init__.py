from .lamcts_plan import plan as lamcts_planner
from .lamcts_parameter import plan as lamcts_parameter_planner
from .random import plan as random_planner
from .random_shooting import plan as random_shooting_planner
from .cmaes import plan as cmaes_planner
from .mppi import plan as mppi_planner
from .cem import plan as cem_planner
from .ilqr import plan as ilqr_planner
from .voo import plan as voo_planner

PLANNING_METHODS = {
    'lamcts-planning': lamcts_planner,
    'lamcts-parameter': lamcts_parameter_planner,
    'random': random_planner,
    'random-shooting': random_shooting_planner,
    'cmaes': cmaes_planner,
    'mppi': mppi_planner,
    'cem': cem_planner,
    'ilqr': ilqr_planner,
    'voo': voo_planner,
}