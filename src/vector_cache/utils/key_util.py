from typing import Union, Callable
import uuid


def get_query_index(identifier: Union[str, Callable, None] = None):
    if isinstance(identifier, str):
        vector_id = f"{identifier}_{str(uuid.uuid4())}"
    elif callable(identifier):
        vector_id = identifier()
    else:
        vector_id = str(uuid.uuid4())
    return vector_id
