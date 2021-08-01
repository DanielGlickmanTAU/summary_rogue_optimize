from typing import Dict, List
from itertools import product


def gridsearch(default_params: Dict, params_to_grid_search: Dict[object, List]) -> List[Dict]:
    def prod(params_to_grid_search):
        values = []
        for v in params_to_grid_search.values():
            if isinstance(v, list):
                values.append(v)
            else:
                values.append([v])
        return product(*values)

    params_as_dicts = [dict(zip(params_to_grid_search, v)) for v in prod(params_to_grid_search)]
    rets = []
    for d in params_as_dicts:
        ret = default_params.copy()
        ret.update(d)
        rets.append(ret)
    return rets
