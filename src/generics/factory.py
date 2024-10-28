from typing import Dict, List, Type, TypeVar

from loguru import logger

T = TypeVar("T")


class GenericFactory[T]:
    """Factory for creating instances of registered classes."""

    _registry: Dict[str, Type[T]] = {}

    @classmethod
    def register_calculator(cls, name: str, key: Type[T]):
        """
        Register a new class type.

        Args:
            name (str): Name of the calculator.
            key (Type[MetricCalculator]): Class to register.
        """
        cls._registry[name] = key

    @classmethod
    def create(cls, registered_type: str, *args, **kwargs) -> T:
        """
        Create a registered instance.

        Args:
            registered_type (str): Type of registered item to create.
            **kwargs: Additional arguments for the constructor.

        Returns:
            T: Instance of the requested.

        Raises:
            ValueError: If the requested type is unknown.
        """
        _class = cls._registry.get(registered_type)
        if _class is None:
            logger.error(f"Unknown type: {registered_type}")
            raise ValueError(f"Unknown type: {registered_type}")
        return _class(**kwargs)

    @classmethod
    def get_registered_types(cls) -> List[str]:
        """Get list of available registered names.

        Returns:
            List of available registered names.
        """
        return list(cls._registry.keys())

    @classmethod
    def create_all(cls, *args, **kwargs) -> list[T]:
        """
        Create instances of all available registered types.

        Args:
            **kwargs: Additional arguments for the constructors.

        Returns:
            list[MetricCalculator]: List of registered instances.
        """
        return [cls.create(name, *args, **kwargs) for name in cls._registry]
