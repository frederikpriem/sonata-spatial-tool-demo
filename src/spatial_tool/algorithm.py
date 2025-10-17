# spatial_tool/algorithm.py
"""Module containing the Algorithm class definition"""

from typing import Optional, Literal, Annotated
from pydantic import BaseModel, Field

from .support import _doc_class_fields


@_doc_class_fields
class Algorithm(BaseModel):

    """
    _summary_

    """

    num_parent: Annotated[Optional[int], Field(
        gt=0,
        description="""Parent population size. Offspring will always be generated from parent
            population."""
    )] = 10
    num_offspring: Annotated[Optional[int], Field(
        gt=0,
        description="""Offspring population size. Parent + offspring population determines the total
            population size."""
    )] = 20
    mutation_rate: Annotated[Optional[float], Field(
        ge=0,
        le=1,
        description="""The rate at which mutation is applied on coefficients of newly generated
            solutions."""
    )] = 0.5
    mutation_strength: Annotated[Optional[float], Field(
        gt=0,
        description="""The magnitude of mutation. Represents the standard deviation of the normal
            distribution, with mean = 0, from which coefficient mutators are drawn. If mutation
            strength is adaptive, this parameter represents the starting value."""
    )] = 0.25
    learning_rate: Annotated[Optional[float], Field(
        ge=0,
        description="""Learning rate parameter used to control adaptive mutation strength. Higher
            values will make the mutation strength change more in reaction to success rates
            deviating from the target success rate. Setting the learning rate parameter to 0 will
            negate the effect of success rates on mutation strength and keep mutation strength
            static throughout optimization."""
    )] = 0.25
    target_success_rate: Annotated[Optional[float], Field(
        gt=0,
        description="""Target success rate parameter used to control adaptive mutation strength.
            Expresses the expected relative improvement of at least 1 objective criterion. E.g.,
            a value of 0.2 implies an expected relative improvement of 20% for at least 1 criterion
            after each generation. The more the maximum succes rate deviates from the expected
            success rate, the more the mutation strength is scaled. Mutation strength is scaled
            upward if the target is not met, and downward if the target is exceeded."""
    )] = 0.2
    min_mutation_strength: Annotated[Optional[float], Field(
        gt=0,
        description="""Minimal value to enforce on mutation strength."""
    )] = 0.1
    max_mutation_strength: Annotated[Optional[float], Field(
        gt=0,
        description="""Maximum value to enforce on mutation strength."""
    )] = 1.
    crossover_rate: Annotated[Optional[float], Field(
        ge=0,
        le=1,
        description="""The rate at which crossover is applied when generating new solutions."""
    )] = 0.5
    crossover_strength: Annotated[Optional[float], Field(
        ge=0,
        description="""The magnitude of crossover. Represents the factor with which the difference
            of the pair vectors is multiplied during trial vector generation."""
    )] = 0.5
    rank_pressure: Annotated[Optional[float], Field(
        ge=0,
        description="""Rank pressure parameter used to bias the sampling of base and pair vectors,
            for the generation of the trial vector during crossover. Higher values will give more
            weight to solutions having a higher Pareto rank for the sampling of the base vector.
            Using a higher rank pressure will also give more weight to solutions having a lower
            Pareto rank for the sampling of pair vectors. Pareto rank has no effect on sampling if
            the rank pressure parameter is set to 0."""
    )] = 2.
    crowding_pressure: Annotated[Optional[float], Field(
        ge=0,
        description="""Crowding pressure parameter used to bias the sampling of base and pair
            vectors, during the generation of the trial vector (crossover). Higher values will give
            more weight to solutions having a higher crowding distance for the sampling of the base
            vector. Using a higher crowding pressure will also give more weight to solutions having
            a lower crowding distance for the sampling of pair vectors. Crowding distance has no
            effect on sampling if the crowding pressure parameter is set to 0."""
    )] = 1.
    generations: Annotated[Optional[int], Field(
        gt=0,
        description="""Number of generations over which the algorithm will iterate."""
    )] = 20
    coef_ul: Annotated[Optional[float], Field(
        description="""Upper limit enforced on coefficient values."""
    )] = 5.
    coef_ll: Annotated[Optional[float], Field(
        description="""Lower limit enforced on coefficient values."""
    )] = -5.
    sample_scheme: Annotated[Optional[Literal['uniform', 'normal']], Field(
        description="""Sample scheme used to randomly generate the initial population."""
    )] = 'uniform'
    sample_mean: Annotated[Optional[float], Field(
        description="""Mean used for the sampling of the initial population. Only has effect if
            sample_scheme='normal'."""
    )] = 0.
    sample_std: Annotated[Optional[float], Field(
        description="""Standard deviation used for the sampling of the initial population. Only has
            effect if sample_scheme='normal'."""
    )] = 1.
