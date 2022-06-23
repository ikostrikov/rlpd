from typing import Any, Dict, Union

import flax
import numpy as np

DataType = Union[np.ndarray, Dict[str, "DataType"]]
PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
