import importlib
from typing import Any, Type


def get_class(class_name: str) -> Type[Any]:
    """
    Retrieve a class object by its name, supporting both fully qualified
    names (with module path) and simple local class names.

    Args:
        class_name (str): The class name, either a simple name (e.g., "Person")
            or a fully qualified name (e.g., "torch.nn.Embedding").

    Returns:
        Type[Any]: The corresponding class object.

    Raises:
        ValueError: If the class cannot be found in either the specified module
            path or the global scope.
    """
    parts = class_name.split(".")
    for i in range(len(parts), 0, -1):
        module_part = ".".join(parts[:i])
        class_part = ".".join(parts[i:])

        try:
            module = importlib.import_module(module_part)
            cls = getattr(module, class_part)
            if isinstance(cls, type):
                return cls
        except (ImportError, AttributeError):
            continue

    cls = globals().get(class_name)
    if isinstance(cls, type):
        return cls

    raise ValueError(f"Cannot find class: {class_name}")
