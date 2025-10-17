"""Optimization problem definition"""


# 0.1 import spatial tool classes and functions
from src.spatial_tool import (
    Layer,
    LayerFunction,
    DecisionFunction,
    ObjectiveFunction,
    ObjectiveCriterion,
    DecisionCriterion,
    Algorithm,
    Optimizer,
    indicate,
    power,
    combine,
    convolve
)


# 0.2 import custom functions
from functions import (
    agricultural_yield,
    carbon_sequestration,
    pollinator_diversity,
    potential_sng
)


# 0.3 specify cover types in model space
label_map = {
    'background': 0,
    'pasture': 1,
    'semi-natural grassland': 2,
    'forest': 3,
    'wetland/surface water': 4,
    'cropland': 5
    }


# 0.4 specify layer source files and scale factors
files = {
    'model space': [
        'data/model_space.tif',
        {'model_space': True, 'label_map': label_map}
    ],
    'soil organic carbon': [
        'data/soil_organic_carbon.tif',
        {'scale': 10.}
    ],
    'sand fraction': [
        'data/sand_fraction.tif',
        {'scale': 10.}
    ],
    'slope': [
        'data/slope.tif',
        {}
    ],
    'road access': [
        'data/road_access.tif',
        {}
    ],
    'protected areas': [
        'data/protected_areas.tif',
        {}
    ]
}

# 1. define layers
layers = []


# 1.1. define file-based layers
for layer_id, (source, kwargs) in files.items():

    layers.append(Layer(
        layer_id=layer_id,
        source=source,
        **kwargs
    ))


# 1.2. define function-based layers
# 1.2.1. indicator functions
for label, value in label_map.items():

    layfun = LayerFunction(
        input_layer_ids=['model space'],
        function=indicate,
        kwargs={
            'target': value,
            'predicate': 'eq'
        }
    )
    layer = Layer(
        layer_id=label,
        source=layfun
    )
    layers.append(layer)


# 1.2.2. relative frequencies
for label, _ in label_map.items():

    layfun = LayerFunction(
        input_layer_ids=[label],
        function=convolve,
        kwargs={
            'radius_m': 500,
            'cell_size_m': 100
        }
    )
    layer = Layer(
        layer_id=f'relative frequency {label}',
        source=layfun
    )
    layers.append(layer)


# 1.2.3. ecological connectivity
layfun_hs_ip = LayerFunction(
    input_layer_ids=[
        'pasture',
        'semi-natural grassland',
        'forest',
        'wetland/surface water',
        'cropland'
    ],
    function=combine,
    kwargs={
        'weights': [
            0.5,
            1.0,
            0.3,
            1.0,
            0.1,
            1.0
        ]
    }
)
hs_ip = Layer(
    layer_id='habitat suitability pollinator insects',
    source=layfun_hs_ip
)
layers.append(hs_ip)

layfun_ec_ip = LayerFunction(
    input_layer_ids=['habitat suitability pollinator insects'],
    function=convolve,
    kwargs={
        'radius_m': 500,
        'cell_size_m': 100,
        'decay': 0.002
    }
)
ec_ip = Layer(
    layer_id='ecological connectivity insect pollinators',
    source=layfun_ec_ip
)
layers.append(ec_ip)


# 1.2.4. other
layfun = LayerFunction(
    input_layer_ids=['sand fraction'],
    function=power,
    kwargs={'exponent': 2}
)
layer = Layer(
    layer_id='sand fraction squared',
    source=layfun
)
layers.append(layer)


# 2. define objective criteria
objfun_ay = ObjectiveFunction(
    function=agricultural_yield,
    input_layer_ids=[
        'pasture',
        'cropland',
        'relative frequency semi-natural grassland',
        'relative frequency wetland/surface water',
        'sand fraction',
        'soil organic carbon',
        'slope',
        'road access'
    ]
)
objcrit_ay = ObjectiveCriterion(
    criterion_id='agricultural yield',
    objective_function=objfun_ay,
    maximize=True
)

objfun_cs = ObjectiveFunction(
    function=carbon_sequestration,
    input_layer_ids=[
        'pasture',
        'semi-natural grassland',
        'forest',
        'wetland/surface water',
        'relative frequency pasture',
        'relative frequency semi-natural grassland',
        'relative frequency forest',
        'relative frequency wetland/surface water'
    ]
)
objcrit_cs = ObjectiveCriterion(
    criterion_id='carbon sequestration',
    objective_function=objfun_cs,
    maximize=True
)

objfun_pd = ObjectiveFunction(
    input_layer_ids=[
        'pasture',
        'semi-natural grassland',
        'forest',
        'wetland/surface water',
        'cropland',
        'ecological connectivity insect pollinators'
    ],
    function=pollinator_diversity
)
objcrit_pd = ObjectiveCriterion(
    criterion_id='pollinator diversity',
    objective_function=objfun_pd,
    maximize=True
)

objective_criteria = [
    objcrit_ay,
    objcrit_cs,
    objcrit_pd
]


# 3. define decision criteria
decfun_sng = DecisionFunction(
    input_layer_ids=[
        'pasture',
        'cropland',
        'relative frequency pasture',
        'relative frequency cropland',
        'ecological connectivity insect pollinators',
        'sand fraction',
        'sand fraction squared',
        'soil organic carbon',
        'slope',
        'road access',
        'protected areas'
    ],
    num_coef=11,
    function=potential_sng
)
deccrit_sng = DecisionCriterion(
    criterion_id='potential semi-natural grassland',
    cover_type='semi-natural grassland',
    decision_function=decfun_sng
)

decision_criteria = [
    deccrit_sng
]


# 4. define demand, expressed in cells per cover type
demand = {
    'semi-natural grassland': 100
}


# 5. Specify the algorithm parameters
algorithm = Algorithm(
    num_parent=20,
    num_offspring=20,
    generations=50,
    crossover_rate=.5,
    rank_pressure=0,
    crowding_pressure=0,
    mutation_rate=.5,
    mutation_strength=.2,
    max_mutation_strength=1.,
    min_mutation_strength=0.1,
    target_success_rate=.1,
    learning_rate=.5,
    coef_ll=-5.,
    coef_ul=5.
)
