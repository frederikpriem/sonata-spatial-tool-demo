"""Custom functions used in the given spatial optimization problem"""

from typing import (
    Union,
    Iterable,
    List,
    Annotated
)

import numpy as np
import scipy as sp
import pydantic
from pydantic import Field, ConfigDict

from src.spatial_tool import logistic, combine, standardize


validate_all_types = pydantic.validate_arguments(
    config=ConfigDict(arbitrary_types_allowed=True)
)


# objective functions
@validate_all_types
def agricultural_yield(
    input_layers: Annotated[List[np.ndarray], Field(...,
        description="""List of input layer arrays:
            1. pasture
            2. cropland
            3. RF semi-natural grassland
            4. RF wetland/water surface
            5. sand fraction
            6. soil organic carbon
            7. slope
            8. road access"""
    )]
) -> float:

    # parse the input_layers argument
    past = input_layers[0].astype(bool)
    crop = input_layers[1].astype(bool)
    rf_sng = input_layers[2]
    rf_wsw = input_layers[3]
    sand = input_layers[4]
    soc = input_layers[5]
    slope = input_layers[6]
    ra = input_layers[7]

    # assumed base yields per cover type
    base_ay_past = np.where(past, 40., 0.)
    base_ay_crop = np.where(crop, 100., 0.)

    # assumed mutators on base yield of cropland
    adj_ay_crop = base_ay_crop + 50 * (rf_sng + rf_wsw)
    adj_ay_crop *= np.where(sand < 40, 0.8, 1.)
    adj_ay_crop *= np.where(sand > 60, 0.8, 1.)
    adj_ay_crop *= np.where(soc < 20, 0.8, 1.)
    adj_ay_crop *= np.where(slope > 5, 0.9, 1.)
    adj_ay_crop *= np.where(ra > 3000, 0.95, 1.)

    # assumed mutator on base yield of pasture
    adj_ay_past = base_ay_past + 20 * (rf_sng + rf_wsw)

    # total adjusted yield map
    total_ay = adj_ay_crop + adj_ay_past

    # total yield value
    total_ay = total_ay.sum()

    return total_ay


@validate_all_types
def carbon_sequestration(
    input_layers: Annotated[List[np.ndarray], Field(...,
        description="""List of input layer arrays:
            1. pasture
            2. semi-natural grassland
            3. forest
            4. wetland/surface water
            5. RF pasture
            6. RF semi-natural grassland
            7. RF forest
            8. RF wetland/surface water"""
    )]
) -> float:

    # parse the input_layers argument
    past = input_layers[0].astype(bool)
    sng = input_layers[1].astype(bool)
    fore = input_layers[2].astype(bool)
    wsw = input_layers[3].astype(bool)
    rf_past = input_layers[4]
    rf_sng = input_layers[5]
    rf_fore = input_layers[6]
    rf_wsw = input_layers[7]

    # assumed base carbon sequestration per cover type
    base_cs_past = np.where(past, 10., 0.)
    base_cs_sng = np.where(sng, 35., 0.)
    base_cs_fore = np.where(fore, 50., 0.)
    base_cs_wsw = np.where(wsw, 65., 0.)

    # assumed carbon sequestration mutators
    rf_total = rf_past + rf_sng + rf_fore + rf_wsw
    adj_cs_past = base_cs_past + 5. * rf_total
    adj_cs_sng = base_cs_sng + 15. * rf_total
    adj_cs_fore = base_cs_fore + 25. * rf_total
    adj_cs_wsw = base_cs_wsw + 30. * rf_total

    # total carbon sequestration map
    total_cs = adj_cs_past + adj_cs_sng + adj_cs_fore + adj_cs_wsw

    # total carbon sequestration value
    total_cs = total_cs.sum()

    return total_cs


@validate_all_types
def pollinator_diversity(
    input_layers: Annotated[List[np.ndarray], Field(...,
        description="""List of input layer arrays:
            1. pasture
            2. semi-natural grassland
            3. forest
            4. wetland/surface water
            5. cropland
            6. ecological connectivity insect pollinators"""
    )]
) -> float:

    # parse the input_layers argument
    past = input_layers[0].astype(bool)
    sng = input_layers[1].astype(bool)
    fore = input_layers[2].astype(bool)
    wsw = input_layers[3].astype(bool)
    crop = input_layers[4].astype(bool)
    ec_ip = input_layers[5]

    base_pd_past = np.where(past, 120, 0.)
    adj_pd_past = base_pd_past + 60 * ec_ip

    base_pd_sng = np.where(sng, 200, 0.)
    adj_pd_sng = base_pd_sng + 100 * ec_ip

    base_pd_fore = np.where(fore, 80, 0.)
    adj_pd_fore = base_pd_fore + 40 * ec_ip

    base_pd_wsw = np.where(wsw, 180, 0.)
    adj_pd_wsw = base_pd_wsw + 80 * ec_ip

    base_pd_crop = np.where(crop, 15, 0.)
    adj_pd_crop = base_pd_crop + 15 * ec_ip

    total_pd = adj_pd_past + adj_pd_sng + adj_pd_fore + adj_pd_wsw + adj_pd_crop
    total_pd = total_pd[total_pd > 0]
    mean_pd = total_pd.mean()

    return mean_pd


# decision functions
@validate_all_types
def potential_sng(
    input_layers: Annotated[List[np.ndarray], Field(...,
        description="""List of input layer arrays:
            1. pasture (used as variable and constraint)
            2. cropland (used as variable and constraint)
            3. RF pasture
            4. RF cropland
            5. ecological connectivity insect pollinators
            6. sand fraction
            7. sand fraction squared
            8. soil organic carbon
            9. slope
            10. road access
            11. protected areas (used as constraint)"""
    )],
    coefs: Annotated[Iterable[Union[float, int, bool]], Field(...,
        description="""Coefficients used by the decision function, and to be optimized by the
            evolutionary algorithm. This function requires 11 coefficients in total (10 variables
            and 1 bias)."""
    )]
) -> np.ndarray:

    coefs = list(coefs)

    # get the input layers that are used as constraints
    past = input_layers[0].astype(bool)
    crop = input_layers[1].astype(bool)
    pa = input_layers[10].astype(bool)

    # standardize input layers that are to be weighted (exclude layers that are only constraints)
    input_layers = [standardize([input_layer]) for input_layer in input_layers[:-1]]

    # create a linear combination of the input layers and coefficients
    pot_sng = combine(
        input_layers=input_layers,
        weights=coefs[:-1],
        bias=coefs[-1]
    )

    # perform logistic transformation to get values in [0, 1] range
    pot_sng = logistic([pot_sng])

    # apply constraints
    pot_sng[pa] = 0
    pot_sng[np.logical_not(np.logical_or(past, crop))] = 0

    return pot_sng
