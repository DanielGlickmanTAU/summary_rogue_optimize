from typing import Dict, List
from itertools import product


def gridsearch(default_params: Dict, params_to_grid_search: Dict[object, List]) -> List[Dict]:
    params_as_dicts = [dict(zip(params_to_grid_search, v)) for v in product(*params_to_grid_search.values())]
    rets = []
    for d in params_as_dicts:
        ret = default_params.copy()
        ret.update(d)
        rets.append(ret)
    return rets
