# spatial_tool/optimizer.py
"""Module containing the Optimizer class definition"""


# pylint: disable=protected-access


import random
import multiprocessing
import copy
import warnings
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Literal, Annotated

import yaml
import rasterio
import deap.algorithms
import deap.base
import deap.creator
import deap.tools
import numpy as np
import inspect
import pydantic
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from deap.tools import Logbook
from deap.tools.emo import sortNondominated
from tqdm import tqdm

try:
    import cupy as cp
except ImportError:
    cp = None

from .layer import Layer
from .function import LayerFunction, DecisionFunction, ObjectiveFunction
from .criterion import ObjectiveCriterion, DecisionCriterion
from .algorithm import Algorithm
from .support import _doc_class_fields, _doc_function_fields, _get_backend


validate_all_types = pydantic.validate_arguments(
    config=ConfigDict(arbitrary_types_allowed=True)
)


def _update_progress_bar(_, progress_bar):

    progress_bar.update(1)


def _gpu_worker(fun, solution, device):

    cp.cuda.Device(device).use()
    return fun(solution)


def _mp_evaluate_solutions(optim, solutions):

    desc = 'Simulating solutions...'

    if optim.num_gpu > 1:
        multiprocessing.set_start_method("spawn", force=True)
        pool = multiprocessing.Pool(processes=optim.num_gpu)

        with tqdm(
            total=len(solutions),
            desc=desc,
            leave=False
        ) as progress_bar:

            fitnesses = [pool.apply_async(
                _gpu_worker,
                args=[optim._toolbox.evaluate, sol, s % optim.num_gpu],
                callback=lambda _: _update_progress_bar(_, progress_bar)
            ) for s, sol in enumerate(solutions)]
            fitnesses = [f.get() for f in fitnesses]

            pool.close()
            pool.join()

    elif optim.num_cpu > 1:
        pool = multiprocessing.Pool(processes=optim.num_cpu)

        with tqdm(
            total=len(solutions),
            desc=desc,
            leave=False
        ) as progress_bar:

            fitnesses = [pool.apply_async(
                optim._toolbox.evaluate,
                args=[sol],
                callback=lambda _: _update_progress_bar(_, progress_bar)
            ) for sol in solutions]
            fitnesses = [f.get() for f in fitnesses]

            pool.close()
            pool.join()

    else:
        iterator = tqdm(solutions,
            desc=desc,
            leave=False)
        fitnesses = []

        for solution in iterator:
            fitnesses.append(optim._toolbox.evaluate(solution))

    return fitnesses


@_doc_class_fields
class Optimizer(BaseModel):

    """
    _summary_
    """  

    layers: Annotated[List[Layer], Field(
        description="""The layers used by the optimizer, including at least the model space. Must be
            an iterable of Layer objects."""
    )]
    decision_criteria: Annotated[List[DecisionCriterion], Field(
        description="""The decision criteria used by the optimizer. Must be an iterable of
            DecisionCriterion objects.""",
    )]
    objective_criteria: Annotated[List[ObjectiveCriterion], Field(
        description="""The objective criteria used by the optimizer. Must be an iterable of
            ObjectiveCriterion objects."""
    )]
    demand: Annotated[Dict[str, int], Field(
        description="""A dictionary of format {cover type label: demand} containing the number of
            cells to be allocated during simulation per specified cover type. Each cover type label
            included in this dictionary must also be present in the label map of the model space."""
    )]
    algorithm: Annotated[Algorithm, Field(
        description="""An Algorithm object containing the parameters needed to specify the
            evolutionary algorithm used to perform optimization"""
    )]
    optimizer_name: Annotated[Optional[str], Field(
        description="""Name to be given to the optimizer"""
    )] = None
    check_inputs: Annotated[Optional[bool], Field(
        description="""Whether to check the input arguments for internal consistency and mandatory
            elements (recommended).""",
    )] = True
    check_dimensions: Annotated[Optional[bool], Field(
        description="""Whether to check that the dimensions of each layer and the decision values
            equal those of the model space (recommended). Only has effect if check_inputs is
            True."""
    )] = False
    update_distance: Annotated[Optional[Union[int, None]], Field(
        ge=1,
        description="""Distance, expressed in number of cells, over which dynamic layers and
            decision values will be updated after a cover type has been allocated to a cell. Specify
            if possible to speed up simulations and optimization."""
    )] = None
    update_buffer: Annotated[Optional[int], Field(
        ge=0,
        description="""Additional distance, expressed in number of cells, over which cell values
            will be taken in account when updating dynamic layers and decision values after a cover
            type allocation. Only has effect if update_distance is specified. Specify if the used
            dynamic variables and/or decision criteria are context-sensitive."""
    )] = 0
    num_cpu: Annotated[Optional[int], Field(
        ge=1,
        description="""Number of CPU cores to use for multiprocessing of simulations. Set to 2 or
            higher if possible to speed up simulations and optimization."""
    )] = 1
    num_gpu: Annotated[Optional[int], Field(
        ge=0,
        description="""Number of GPU to use for (multi)processing of simulations. Set to 1 or higher
            if possible to speed up simulations and optimization. If set to 1 or higher, make sure
            that CUDA is available and loaded on the used device and that a version of the spatial
            tool is installed that supports the corresponding CUDA version. Setting the number of
            GPU to 1 or more will make the number of CPU fall back to 1."""
    )] = 0
    seed: Annotated[Optional[int], Field(
        description="""Seed number that will be set prior to generating random numbers during each
            optimization. Makes optimization results reproduceable over repeated runs with identical
            settings."""
    )] = None
    log_delta_fitness: Annotated[Optional[bool], Field(
        description="""Whether to log fitness values as changes relative to the initial fitness
            values obtained at the start of simulation."""
    )] = False
    simulation_backend: Annotated[Optional[Literal['default', 'cupy']], Field(
        description="""Computational backend to use for simulation. If 'cupy' is chosen, make sure
            that CUDA is available and loaded on the used device and that a version of the spatial
            tool is installed that supports the corresponding CUDA version. Choosing 'cupy' only has
            effect if num_gpu >= 1. Will fall back to 'default' if num_gpu = 0. For greater effect,
            also use cupy in the layer functions, objective functions and decision functions."""
    )] = 'default'
    _created_at: Optional[str] = PrivateAttr(default=None)
    _optimization_started_at: Optional[str] = PrivateAttr(default=None)
    _optimization_finished_at: Optional[str] = PrivateAttr(default=None)
    _num_coef: Optional[int] = PrivateAttr(default=None)
    _toolbox: Optional[Any] = PrivateAttr(default=deap.base.Toolbox())
    _logbook: Optional[Any] = PrivateAttr(default=None)
    _population: Optional[List] = PrivateAttr(default=None)
    _mutation_strength: Optional[float] = PrivateAttr(default=None)

    def model_post_init(self, __context=None) -> None:  # pylint: disable=arguments-differ

        """
        Included only for pydantic compatibility. Safe for Binder and non-GPU environments.
        """

        # Call parent hook if it exists
        parent_hook = getattr(super(), "model_post_init", None)
        if callable(parent_hook):
            parent_hook(__context)

        self._created_at = datetime.now().isoformat()

        self.layers = list(self.layers)
        self.decision_criteria = list(self.decision_criteria)
        self.objective_criteria = list(self.objective_criteria)

        self._num_coef = sum(
            getattr(criterion.decision_function, "num_coef", 0)
            for criterion in self.decision_criteria
        )

        try:
            n_gpu = cp.cuda.runtime.getDeviceCount()
        except Exception:
            n_gpu = 0

        if getattr(self, "num_gpu", 0) > 0:
            self.num_gpu = min(self.num_gpu, n_gpu)
            self.num_cpu = 1
        else:
            self.num_gpu = 0
            self.simulation_backend = "default"

        if self.check_inputs:
            self._check()

        self._setup_algorithm()

    @_doc_function_fields
    def get_model_space(self) -> Union[None, Layer]:

        """       
        _summary_
        """
        model_space = None

        for layer in self.layers:

            if layer.model_space:

                model_space = layer

        return model_space

    @_doc_function_fields
    @validate_all_types
    def set_model_space_values(self,
        values: Annotated[np.ndarray, Field(
            description="""Array with which the model space is to be set."""
        )]
    ) -> None:

        """        
        _summary_
        """

        for layer in self.layers:

            if layer.model_space:

                if layer._values.shape != values.shape:

                    msg = """Dimensions of passed values do not match the dimensions of the model
                        space"""
                    raise ValueError(msg)

                layer.set(values)

    @_doc_function_fields
    def reset_model_space_values(self,
        from_cache: Annotated[Optional[bool], Field(
            description="""Whether to reset the model space with values stored in cache."""
        )] = False
    ) -> None:

        """       
        _summary_
        """    

        for layer in self.layers:

            if layer.model_space:

                layer.load(from_cache=from_cache)

    @_doc_function_fields
    def get_layer(self,
        layer_id: Annotated[str, Field(
            description="""ID of the requested layer."""
        )]
    ) -> Union[Layer, None]:

        """
         _summary_
        """    

        out_layer = None

        for layer in self.layers:

            if layer.layer_id == layer_id:

                out_layer = layer

        return out_layer

    @_doc_function_fields
    def get_input_layers(self,
        fun: Annotated[Union[LayerFunction, DecisionFunction, ObjectiveFunction], Field(
            description="""Function for which the input layers are to be fetched. Must either be a
                LayerFunction, DecisionFunction or ObjectiveFunction instance."""
        )]
    ) -> List[np.ndarray]:

        """
        _summary_
        """

        input_layers = []

        for layer_id in fun.input_layer_ids:

            input_layers.append([l.get() for l in self.layers if l.layer_id == layer_id][0])
    
        return input_layers

    def _check_model_space(self) -> None:

        # check that there is exactly one layer specified as the model space
        model_space_found = 0
        for layer in self.layers:
            model_space_found += int(layer.model_space)
        if model_space_found != 1:
            msg = """There must be exactly 1 model space layer in layers"""
            raise ValueError(msg)

        # check that the source of the model space is a file path
        model_space = self.get_model_space()
        if not (isinstance(model_space.source, str) or isinstance(model_space.source, Path)):
            msg = """The source of the model space layer must be a file path"""
            raise ValueError(msg)
    
    def _check_cover_types(self) -> None:

        model_space = self.get_model_space()

        # check that each cover type in demand is included in the label map of the model space
        for cover_type in self.demand.keys():
            if not cover_type in model_space.label_map.keys():
                msg = f"""Cover type "{cover_type}" not found in the label map of the model space
                    layer"""
                raise ValueError(msg)

        # check that there is at least one decision criterion for each cover type in demand
        cover_types = []
        for criterion in self.decision_criteria:
            cover_types.append(criterion.cover_type)

        cover_types = set(cover_types)

        for cover_type in self.demand.keys():
            if not cover_type in cover_types:
                msg = f"""Cover type {cover_type} not found in decision criteria"""
                raise ValueError(msg)

    def _check_criteria(self) -> None:

        # check that there are no duplicate criterion IDs
        crit_ids = [c.criterion_id for c in self.decision_criteria + self.objective_criteria]
        if len(crit_ids) != len(set(crit_ids)):
            msg = """Criterion IDs must be unique"""
            raise ValueError(msg)

        # check that the functions of the decision and objective criteria refer to existing layers
        layer_ids = [l.layer_id for l in self.layers]
        for crit in self.decision_criteria:
            for layer_id in crit.decision_function.input_layer_ids:
                if not layer_id in layer_ids:
                    msg = f"""Input layer "{layer_id}" of decision criteria "{crit.criterion_id} not
                        found in layers"""
                    raise ValueError(msg)

        for crit in self.objective_criteria:
            for layer_id in crit.objective_function.input_layer_ids:
                if not layer_id in layer_ids:
                    msg = f"""Input layer "{layer_id}" of objective criteria "{crit.criterion_id}
                        not found in layers"""
                    raise ValueError(msg)

    def _check_layers(self) -> None:

        # check that there are no duplicate layer IDs
        layer_ids = [l.layer_id for l in self.layers]
        if len(layer_ids) != len(set(layer_ids)):
            msg = """Layer IDs must be unique"""
            raise ValueError(msg)
        
        # check that layers whose source is a function only use previous entries of layers as input
        prev_layer_ids = []
        for layer in self.layers:
            if isinstance(layer.source, LayerFunction):
                for layer_id in layer.source.input_layer_ids:
                    if not layer_id in prev_layer_ids:
                        msg = f"""The function of "{layer.layer_id}" uses the layer "{layer_id}" as
                            input, but this layer is positioned after "{layer.layer_id}" in the
                            sequence of layers, which may cause a circular reference. Only use 
                            preceding layers to feed a layer function."""
                        raise ValueError(msg)
            prev_layer_ids.append(layer.layer_id)

    def _check_dimensions(self) -> None:

        # check that the dimenions of each layer are equal to the dimensions of the model space
        model_space = self.get_model_space()
        model_space.load()
        model_space_array = model_space.get()

        iterator = tqdm(self.layers,
            leave=True,
            desc='Checking layer dimensions...'
        )
        for layer in iterator:
            if isinstance(layer.source, LayerFunction):
                input_layers = self.get_input_layers(layer.source)
                layer.update(input_layers,
                    to_cache=True
                )
            else:
                layer.load(to_cache=True)
            layer_array = layer.get()
            if layer_array.shape != model_space_array.shape:
                msg = f"""The dimensions of layer "{layer.layer_id}" do not match the dimensions
                    of the model space"""
                raise ValueError(msg)

        # check that the dimensions of each yielded decision layer are equal to the dimensions
        # of the model space
        print('Checking decision criteria dimensions...')
        solution = np.random.uniform(
            low=self.algorithm.coef_ll,
            high=self.algorithm.coef_ul,
            size=self._num_coef
            )
        decision_layers = self.get_decision_layers(list(solution))
        dl = decision_layers[:, : , 0].squeeze()
        if dl.shape != model_space_array.shape:
            msg = """The dimensions of the layers yielded by decision criteria do not match the
                dimensions of the model space"""
            raise ValueError(msg)

        del decision_layers

        # check that the objective criteria values are either float, int or bool
        fitness = self.evaluate()
        iterator = tqdm(list(enumerate(fitness)),
            desc='Checking objective criterion dimensions...',
            leave=True
        )
        for i, val in iterator:
            con1 = isinstance(val, float)
            con2 = isinstance(val, int)
            con3 = isinstance(val, bool)
            if not (con1 or con2 or con3):
                msg = f"""The fitness value yielded by objective criterion
                "{self.objective_criteria[i].criterion_id}" is neither float, int or bool"""
                raise ValueError(msg)

    def _check(self) -> None:

        print('Checking optimizer inputs...')

        # perform basic input checks
        self._check_model_space()
        self._check_cover_types()
        self._check_criteria()
        self._check_layers()

        # check the dimensions of the layers, decision criteria and objective criteria
        if self.check_dimensions:
            self._check_dimensions()

        print('Optimizer input checks have been passed successfully')

    def _setup_algorithm(self) -> None:

        # specify the directions of the criteria
        weights = [1.0 if c.maximize else -1.0 for c in self.objective_criteria]
        weights = tuple(weights)

        # get the sample parameters
        if self.algorithm.sample_scheme == 'uniform':
            sample_scheme = random.uniform
            sample_arg1 = self.algorithm.coef_ll
            sample_arg2 = self.algorithm.coef_ul
        else:
            sample_scheme = random.gauss
            sample_arg1 = self.algorithm.sample_mean
            sample_arg2 = self.algorithm.sample_std

        # create base GA elements
        if hasattr(deap.creator, "fitness"):
            del deap.creator.fitness
        deap.creator.create("fitness", deap.base.Fitness,
            weights=weights)
        if hasattr(deap.creator, "individual"):
            del deap.creator.individual
        deap.creator.create("individual", list,
            fitness=getattr(deap.creator, 'fitness'))

        # register specific GA elements
        self._toolbox.register(
            "coefficient",
            sample_scheme,
            sample_arg1,
            sample_arg2
        )
        self._toolbox.register(
            "solution",
            deap.tools.initRepeat,
            getattr(deap.creator, 'individual'),
            getattr(self._toolbox, 'coefficient'),
            n=self._num_coef
        )
        self._toolbox.register(
            "population",
            deap.tools.initRepeat,
            list,
            getattr(self._toolbox, 'solution'),
            n=self.algorithm.num_parent
        )
        self._toolbox.register("evaluate", self.simulate_evaluate)
        self._toolbox.register("mate", self._mate)
        self._toolbox.register("mutate", self._mutate)
        self._toolbox.register("select", deap.tools.selNSGA2)

    def _mate(self, solution1: Any, solution2: Any) -> Tuple:

        # solution1 and solution2 arguments are ignored, only included for DEAP compatibility
        # mate must yield two solutions for DEAP compatibility

        del solution1
        del solution2

        children = []

        for _ in range(2):

            # collect pareto ranks and crowding distances
            rank = np.array([solution.fitness.rank for solution in self._population])
            rank += 1  # so values start at 1 instead of 0
            crowding = np.array([solution.fitness.crowding_dist for solution in self._population])

            # address possibly infinite or undefined crowding distances
            con1 = np.isinf(crowding)
            con2 = np.isnan(crowding)
            crowding[np.logical_or(con1, con2)] = 1.
    
            # use z-scores of crowding distance
            # address possibly non-existing variance
            crowding_mean = crowding.mean()
            crowding_std = crowding.std()
            if np.isnan(crowding_std) or crowding_std == 0:
                crowding_std = 1.
            crowding = (crowding - crowding_mean) / crowding_std

            # calculate sample weights
            weights_base = -1 * self.algorithm.rank_pressure * rank
            weights_base += self.algorithm.crowding_pressure * crowding
            prob_base = np.exp(weights_base)
            prob_base /= prob_base.sum()

            weights_pair = self.algorithm.rank_pressure * rank
            weights_pair -= self.algorithm.crowding_pressure * crowding
            prob_pair = np.exp(weights_pair)
            prob_pair /= prob_pair.sum()

            # sample the base donor vector
            indices = range(len(self._population))
            sample = np.random.choice(indices,
                size=1,
                replace=False,
                p=prob_base)
            base = np.array(self._population[sample[0]])

            # sample the pair donor vectors
            prob_pair[sample[0]] = 0
            prob_pair /= prob_pair.sum()
            sample = np.random.choice(indices,
                size=2,
                replace=False,
                p=prob_pair)
            pair1 = np.array(self._population[sample[0]])
            pair2 = np.array(self._population[sample[1]])

            # generate the trial vector
            cx = np.random.uniform(low=0, high=1, size=len(base)) <= self.algorithm.crossover_rate
            child_values = base + self.algorithm.crossover_strength * (pair1 - pair2) * cx

            # insert the trial vector values into a solution
            solgen = getattr(self._toolbox, 'solution')
            child = solgen()
            for v, value in enumerate(child_values):
                child[v] = value

            # mutate the solution
            child = self._mutate(child)[0]

            # add solution to output
            children.append(child)

        return tuple(children) # tuple output format required for DEAP compatibility

    def _mutate(self, solution) -> Tuple:

        mutated_solution = copy.deepcopy(solution)

        for i in range(len(solution)):

            # apply mutation on coefficient value
            if random.random() <= self.algorithm.mutation_rate:

                mutated_solution[i] += random.gauss(0, self._mutation_strength)
            
                # enforce coefficient lower and upper bounds
                mutated_solution[i] = np.clip(
                    mutated_solution[i],
                    self.algorithm.coef_ll,
                    self.algorithm.coef_ul
                )
    
        return mutated_solution,  # tuple output format required for DEAP compatibility

    def _adapt_mutation_strength(self) -> None:

        # collect success rates
        success_rates = []

        for objcrit in self.objective_criteria:

            if objcrit.maximize:
                statistic = 'max'
            else:
                statistic = 'min'

            perf = self.get_performance_statistics(
                objective_criterion=objcrit.criterion_id,
                statistic=statistic
            )
            
            if len(perf) > 1:
                if objcrit.maximize:
                    success_rate = (perf[-1] - perf[-2]) / perf[-2]
                else:
                    success_rate = (perf[-2] - perf[-1]) / perf[-2]
                success_rates.append(success_rate)
            else:
                success_rates.append(self.algorithm.target_success_rate)

        # scale mutation strength
        sr = max(success_rates)
        lr = self.algorithm.learning_rate
        tsr = self.algorithm.target_success_rate
        self._mutation_strength *= np.exp(lr * (tsr - sr))

        # enforce mutation strength lower and upper bounds
        self._mutation_strength = np.clip(
            self._mutation_strength,
            self.algorithm.min_mutation_strength,
            self.algorithm.max_mutation_strength
        )

    def _add_fitness(self, solutions, fitness) -> List[List[float]]:

        adj_solutions = []

        for solution, fit in zip(solutions, fitness):
            solution.fitness.values = fit
            adj_solutions.append(solution)

        fronts = deap.tools.emo.sortNondominated(
            adj_solutions,
            k=len(adj_solutions),
            first_front_only=False
        )

        for front in fronts:
            deap.tools.emo.assignCrowdingDist(front)

        for r, front in enumerate(fronts):
            for ind in front:
                ind.fitness.rank = r

        return adj_solutions

    def optimize(self) -> None:

        """
         _summary_
        """

        self._optimization_started_at = datetime.now().isoformat()

        # load and update layers
        self.load_layers(
            from_cache=True,
            to_cache=True)
        self.update_layers(
            from_cache=True,
            to_cache=True
        )
        fitness_start = self.evaluate()

        # initiate logbook and statistics to be included in the logbook
        mstats = {}

        for c, objcrit in enumerate(self.objective_criteria):

            get_fitness = lambda ind, i=c: ind.fitness.values[i]
            stats = deap.tools.Statistics(get_fitness)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)

            mstats[objcrit.criterion_id] = stats

        mstats = deap.tools.MultiStatistics(**mstats)

        logbook = Logbook()
        logbook.header = ["gen", "nevals"] + mstats.fields
        self._logbook = logbook

        # register the initial mutation strength value 
        self._mutation_strength = self.algorithm.mutation_strength

        # set the seed number
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        # initiate the population and evaluate it
        popgen = getattr(self._toolbox, 'population')
        self._population = popgen()

        fitnesses = _mp_evaluate_solutions(self, self._population)
        if self.log_delta_fitness:
            fitnesses = [tuple(np.array(fit) - np.array(fitness_start)) for fit in fitnesses]
        parents = self._add_fitness(self._population, fitnesses)
        self._population = parents
        
        # iterate over the specified number of generations
        generations = list(range(self.algorithm.generations))
        iterator = tqdm(generations,
            leave=True,
            desc='Iterating over generations...')

        for gen in iterator:

            # generate offspring (crossover + mutation)
            offspring = deap.algorithms.varOr(
                self._population,
                self._toolbox,
                self.algorithm.num_offspring,
                cxpb=1.0,
                mutpb=0.0
            )

            # evaluate the offspring
            fitnesses = _mp_evaluate_solutions(self, offspring)
            if self.log_delta_fitness:
                fitnesses = [tuple(np.array(fit) - np.array(fitness_start)) for fit in fitnesses]
            offspring_eval = self._add_fitness(offspring, fitnesses)

            # select the new parents
            combined = self._population + offspring_eval
            self._population = combined
            selector = getattr(self._toolbox, "select")
            self._population = selector(self._population, self.algorithm.num_parent)

            # log the current generation's performance
            record = mstats.compile(self._population)
            self._logbook.record(
                gen=gen + 1,
                nevals=len(offspring),
                **record)

            # adapt the mutation strength based on obtained performance
            self._adapt_mutation_strength()

        # only retain the first non-dominated front of solutions
        self._population = sortNondominated(
            self._population,
            len(self._population),
            first_front_only=True
        )[0]

        # calculate and assign the crowding_dist attribute in each solution of the Pareto front
        deap.tools.emo.assignCrowdingDist(self._population)

        self._optimization_finished_at = datetime.now().isoformat()

    @_doc_function_fields
    def simulate(self,
        solution: Annotated[List[Union[float, int, bool]], Field(...,
            description="""Solution for which spatial simulation is to be performed. The passed
                solution must be an iterable of numerical values. The length of solution must match
                the total number of coefficients specified in the decision criteria."""
        )]
    ) -> None:

        """
        _summary_
        """

        solution = list(solution)

        msg = """The length of solution must match the total number of coefficients specified in the
            decision criteria"""
        assert len(solution) == self._num_coef, msg

        # enforce coefficient lower and upper bounds on passed solution
        solution = np.clip(
            solution,
            self.algorithm.coef_ll,
            self.algorithm.coef_ul
        )

        # create a seeded random generator
        rng = np.random.default_rng(seed=self.seed)

        # reset the model space and function-based layers to their initial states
        self.reset_model_space_values(from_cache=True)
        self.update_layers(from_cache=True)

        # get the initial decision layers
        dl = self.get_decision_layers(solution)

        # create a tracker for demand
        label_map = self.get_model_space().label_map
        track_demand = copy.deepcopy(self.demand)

        # get the model space
        model_space = self.get_model_space()

        # create a change mask to avoid repeated changes in a cell
        change = np.zeros(model_space._values.shape, dtype=bool)

        # get the cover type labels corresponding to each decision criterion
        cover_types = [crit.cover_type for crit in self.decision_criteria]

        # get the computation backend and perform array conversions
        xp = _get_backend(self.simulation_backend)
        dl = xp.asarray(dl)
        change = xp.asarray(change)

        while sum([d for _, d in track_demand.items()]) > 0:

            # set decision values to zero if the corresponding cover type demand is satisfied
            for cover_type, demand in track_demand.items():
                if demand == 0:
                    ind = np.where(np.array(cover_types) == cover_type)[0]
                    dl[:, :, ind] = 0

            # set decision values to zero if the corresponding cell has already changed state
            dl *= xp.expand_dims(xp.logical_not(change), 2)

            # break the loop if all decision values are zero or NaN
            # raise a warning if this happens
            if xp.all(xp.logical_or(dl == 0, xp.isnan(dl))):
                s1 = 'Cover type '
                s2 = ' cells not allocated'
                warnings.warn(
                    """
                    The simulation terminated prematurely because all decision values are zero/NaN.
                    The following cover type demand was not satisfied:
                    {}
                    """.format('\n'.join([
                        s1 + str(c) + ': ' + str(d) + s2 for c, d in track_demand.items() if d > 0
                    ])),
                    RuntimeWarning
                )
                break

            # allocate the cover type to the cell having the highest corresponding decision value
            # if multiple cells share the maximum value, choose one randomly
            maxval = xp.nanmax(dl)
            ind = xp.where(xp.isclose(dl, maxval))

            if len(ind[0]) > 1:
                sample = rng.integers(0, len(ind[0]), dtype=int)
                ind = [i[sample] for i in ind]

            ind = [int(i) for i in ind]
                
            model_space = self.get_model_space()
            model_space._values[ind[0], ind[1]] = label_map[cover_types[ind[2]]]
            self.set_model_space_values(model_space._values)

            # adjust the demand tracker
            track_demand[cover_types[ind[2]]] = track_demand[cover_types[ind[2]]] - 1

            # adjust the change mask
            change[ind[0], ind[1]] = True

            # use the specified update distance and buffer to get more specific slices
            # this will avoid needless computation when updating in the next code block
            # ensure that the slices do not exceed the dimensions of the model space
            if self.update_distance:
                
                min_row = max([0, ind[0] - int(self.update_distance)])
                max_row = min([model_space._values.shape[0], ind[0] + int(self.update_distance)])
                min_col = max([0, ind[1] - int(self.update_distance)])
                max_col = min([model_space._values.shape[1], ind[1] + int(self.update_distance)])

                if self.update_buffer:
                    buffer = self.update_buffer
                    min_row = max([0, min_row - buffer])
                    max_row = min([model_space._values.shape[0], max_row + buffer])
                    min_col = max([0, min_col - buffer])
                    max_col = min([model_space._values.shape[1], max_col + buffer])
                else:
                    buffer = 0

                rows = slice(min_row, max_row)
                columns = slice(min_col, max_col)
            else:
                rows = slice(None)
                columns = slice(None)
                buffer = 0

            # update the function-based layers and decision layers for the next iteration
            self.update_layers(
                rows=rows,
                columns=columns,
                buffer=buffer
                )

            dl_update = self.get_decision_layers(solution,
                rows=rows,
                columns=columns
            )
            dl[rows, columns, :] = xp.asarray(dl_update)

        del dl

    @_doc_function_fields
    def load_layers(self,
        from_cache: Annotated[Optional[bool], Field(
            description='Whether to load layer values from cache.'
        )] = False,
        to_cache: Annotated[Optional[bool], Field(
            description='Whether to store a copy of the loaded values in cache.'
        )] = False
    ) -> None:

        """
        _summary_
        """

        for layer in self.layers:
            if isinstance(layer.source, str) or isinstance(layer.source, Path):
                layer.load(
                    from_cache=from_cache,
                    to_cache=to_cache
                )

    @_doc_function_fields
    def update_layers(self,
        rows: Annotated[Optional[slice], Field(
            description="""Slice object that denotes the rows of the complete layer array that
                are being updated. See documentation Layer.update for more info."""
        )] = slice(None),
        columns: Annotated[Optional[slice], Field(
            description="""Slice object that denotes the columns of the complete layer array that
                are being updated. See documentation Layer.update for more info."""
        )] = slice(None),
        buffer: Annotated[Optional[int], Field(
            ge=0,
            description="""Buffer distance, expressed in cells, over which neighboring cells are
                included when updating layer values. See documentation Layer.update for more
                info."""
        )] = 0,
        from_cache: Annotated[Optional[bool], Field(
            description="""Whether to update values with the values stored in cache."""
        )] = False,
        to_cache: Annotated[Optional[bool], Field(
            description="""Whether to store a copy of the updated values in cache."""
        )] = False
    ) -> None:

        """
        _summary_
        """

        for layer in self.layers:
            if isinstance(layer.source, LayerFunction):
                input_layers = self.get_input_layers(layer.source)
                layer.update(input_layers,
                    rows=rows,
                    columns=columns,
                    buffer=buffer,
                    from_cache=from_cache,
                    to_cache=to_cache
                    )

    def empty_layers(self) -> None:

        """
        _summary_
        """

        for layer in self.layers:
            layer.empty()

    def _check_layer_values_not_empty(self) -> None:

        for layer in self.layers:
            if layer._values is None:
                raise ValueError('All layers must have values to perform this operation')

    @_doc_function_fields
    @validate_all_types
    def get_decision_layers(self,
        solution: Annotated[Iterable[Union[float, int, bool]], Field(...,
            description="""Solution for which the decision layers are calculated. The passed
                solution must be an iterable of numerical values. The length of solution must match
                the total number of coefficients specified in the decision criteria."""
        )],
        rows: Annotated[Optional[slice], Field(
            description="""Slice object that denotes the rows of the complete decision layers arrays
                that are being fetched."""
        )] = slice(None),
        columns: Annotated[Optional[slice], Field(
            description="""Slice object that denotes the columns of the complete decision layer
                arrays that are being fetched."""
        )] = slice(None)
    ) -> np.ndarray:

        """
        _summary_
        """

        self._check_layer_values_not_empty()
        solution = list(solution)
        msg = """The length of solution must match the total number of coefficients specified in the
            decision criteria ({})""".format(self._num_coef)
        assert len(solution) == self._num_coef, msg

        ind = 0
        decision_layers = []

        for deccrit in self.decision_criteria:

            input_layers = self.get_input_layers(deccrit.decision_function)
            inputs = [l[rows, columns] for l in input_layers]
            coefficients = solution[ind:ind + deccrit.decision_function.num_coef]
            dl = deccrit.decision_function._evaluate(inputs, coefficients)
            decision_layers.append(dl)
            ind += deccrit.decision_function.num_coef

        decision_layers = [np.expand_dims(d, 2) for d in decision_layers]
        decision_layers = np.concatenate(decision_layers, axis=2)

        return decision_layers

    @_doc_function_fields
    def evaluate(self) -> Tuple:

        """
        _summary_
        """

        self._check_layer_values_not_empty()

        fitness = []

        for objcrit in self.objective_criteria:
            input_layers = self.get_input_layers(objcrit.objective_function)
            fitness.append(objcrit.objective_function._evaluate(input_layers))

        return tuple(fitness)  # tuple format required for DEAP compatibility

    @_doc_function_fields
    def simulate_evaluate(self,
        solution: Annotated[Iterable[Union[float, int, bool]], Field(
        description="""Solution for which spatial simulation is to be performed. The passed
            solution must be an iterable of numerical values. The length of solution must
            match the total number of coefficients specified in the decision criteria."""
        )]
    ) -> Tuple:

        """
        _summary_
        """

        solution = list(solution)

        msg = """The length of solution must match the total number of coefficients specified in the
            decision criteria"""
        assert len(solution) == self._num_coef, msg

        self.simulate(solution)

        return self.evaluate()

    @_doc_function_fields
    def get_performance_statistics(self,
        objective_criterion: Annotated[str, Field(...,
            description="""Identifier of the objective criterion for which performance statistics
                are returned"""
        )],
        statistic: Annotated[Optional[Literal['avg', 'std', 'min', 'max']], Field(
            description="""Type of performance statistic that are returned"""
        )] = 'avg'
    ) -> List:

        """
        _summary_
        """

        if self._logbook is None:
            msg = 'The optimize method must first be called to get performance statistics'
            raise ValueError(msg)

        if not objective_criterion in [crit.criterion_id for crit in self.objective_criteria]:
            raise ValueError(f'"{objective_criterion}" not found in objective criteria')

        rng = range(self.algorithm.generations)
        stats = []

        for g in rng:
            try:
                stats.append(self._logbook.chapters[objective_criterion][g][statistic])
            except IndexError:
                break

        return stats

    @_doc_function_fields
    def get_solutions(self) -> List[List[float]]:

        """
        _summary_
        """

        if self._population is None:
            raise ValueError('The optimize method must first be called to get solutions')

        return copy.deepcopy(self._population)

    @_doc_function_fields
    def get_best_solution(self,
        objective_criterion: Annotated[str, Field(
            description="""Identifier of the objective criterion for which the best solution is
                returned."""
        )]
    ) -> List[float]:

        """
        _summary_
        """

        if self._population is None:
            raise ValueError('The optimize method must first be called to get the best solution')

        if not objective_criterion in [crit.criterion_id for crit in self.objective_criteria]:
            raise ValueError(f'"{objective_criterion}" not found in objective criteria')

        fitness_ind = [
            crit.criterion_id for crit in self.objective_criteria
        ].index(objective_criterion)
        fitness_values = [solution.fitness.values[fitness_ind] for solution in self._population]

        if self.objective_criteria[fitness_ind].maximize:
            best_ind = np.argmax(fitness_values)
        else:
            best_ind = np.argmin(fitness_values)

        best_solution = copy.deepcopy(self._population[best_ind])

        return best_solution

    @_doc_function_fields
    def get_best_weighted_solution(self,
        objective_criteria: Annotated[Optional[Iterable[str]], Field(
            min_length=1,
            description="""Objective criterion identifiers, for which the solution with the highest
                weighted sum of the corresponding z-scores is to be returned. Z-scores are
                multiplied with the sign of the corresponding criterion (maximize: 1, minimize: -1).
                """
        )] = None,
        weights: Annotated[Optional[Iterable[Union[float, int, bool]]], Field(
            description="""Weights used to calculate the best solution"""
        )] = None
    ) -> List[float]:

        """
        _summary_
        """

        if objective_criteria is None:
            objective_criteria = [objcrit.criterion_id for objcrit in self.objective_criteria]
        else:
            objective_criteria = list(objective_criteria)

        if self._population is None:
            msg = """The optimize method must first be executed to get the best solution"""
            raise ValueError(msg)

        for objective_criterion in objective_criteria:
            if not objective_criterion in [crit.criterion_id for crit in self.objective_criteria]:
                msg = f"""'{objective_criterion}' not found in objective criteria"""
                raise ValueError(msg)

        if weights is not None:
            if len(weights) != len(objective_criteria):
                msg = """The number of weights must match the number of objective criteria
                    specified"""
                raise ValueError(msg)
            weights = np.array(weights)
        else:
            weights = np.ones(len(objective_criteria), dtype=float)
            weights /= weights.sum()

        # gather fitness values
        fitness_values = [list(solution.fitness.values) for solution in self._population]
        fitness_values = np.array(fitness_values)

        # z-score
        fitness_values = (fitness_values - fitness_values.mean(axis=0)) / fitness_values.std(axis=0)

        # account for criterion direction
        direction = np.array([crit.maximize for crit in self.objective_criteria]).astype(bool)
        direction = np.where(direction, 1., -1.)
        fitness_values *= direction

        # weighted sum
        fitness_values *= weights
        fitness_values = fitness_values.sum(axis=1).squeeze()

        best_ind = np.argmax(fitness_values)
        best_weighted_solution = copy.deepcopy(self._population[best_ind])

        return best_weighted_solution        

    @_doc_function_fields
    def to_yaml(self,
        filepath: Annotated[Union[str, Path], Field(
            description="""File path where the YAML-file with the Optimizer attribute values will be
                stored."""
        )]
    ) -> None:

        """
        _summary_
        """

        def serialize(obj: Any) -> Any:

            if obj is None:
                re =  None
            elif isinstance(obj, np.ndarray):
                re = None
            elif hasattr(obj, "model_dump"):
                data_dict = obj.model_dump(exclude_none=True)
                return {
                    k: serialize(v) for k, v in data_dict.items() if not isinstance(v, np.ndarray)
                }
            elif isinstance(obj, dict):
                re = {k: serialize(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
            elif isinstance(obj, (list, tuple, set)):
                re = [serialize(item) for item in obj if not isinstance(item, np.ndarray)]
            elif isinstance(obj, Path):
                re = str(obj)
            elif isinstance(obj, (int, float, bool, str)) or obj is None:
                re = obj
            elif isinstance(obj, rasterio.crs.CRS):
                re = f'EPSG:{obj.to_epsg()}'
            elif isinstance(obj, BaseModel):
                try:
                    data_dict = obj.model_dump(exclude_none=True)
                except AttributeError:  # Pydantic v1 fallback
                    data_dict = obj.dict(exclude_none=True)
                re = {k: serialize(v) for k, v in data_dict.items() if not isinstance(v, np.ndarray)}

                if hasattr(obj, "__private_attributes__"):
                    priv = {k: serialize(getattr(obj, k)) for k in obj.__private_attributes__}
                    re.update(priv)
            elif hasattr(obj, "__dict__"):
                re = {
                    k: serialize(v)
                    for k, v in obj.__dict__.items()
                    if not isinstance(v, np.ndarray)
                }
            else:
                raise TypeError(f"Unsupported type '{type(obj)}' for '{obj}'")

            return re

        data = {
            "yaml_created_on": datetime.now().isoformat(),
            **serialize(self)
        }

        yaml_str = yaml.dump(data, sort_keys=False)

        if filepath:
            with open(filepath, "w", encoding='utf-8') as f:
                f.write(yaml_str)

    @_doc_function_fields
    def to_pickle(self,
        filepath: Annotated[Union[str, Path], Field(
            description="""Location where the pickle file is to be stored"""
        )],
        protocol: Annotated[Optional[int], Field(
            description="""Protocol used to create the pickle file."""
        )] = pickle.HIGHEST_PROTOCOL,
        empty_layers: Annotated[Optional[bool], Field(
            description="""Whether to empty layers prior to saving the Optimizer class instance as a
                pickle file."""
        )] = False
    ) -> None:

        """
        _summary_
        """

        if empty_layers:
            self.empty_layers()

        if isinstance(filepath, str):
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=protocol)
