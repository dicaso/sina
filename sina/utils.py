"""Utility classes or functions
for sina package
"""
# import typing


def logmemory():
    import logging
    import resource
    logging.info(
        'Using max {:.0f}MB'.format(resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024)
    )


class SettingsBase(object):
    """Takes in a list of settings, which will be exposed
    as CLI arguments. Each settings tuple should have the
    following format:
    ('--name', keyword dict for the parser.add_argument function)

    The recommended way to build a SettingsBase object, is to
    inherit from it and define the `setup` method
    (see SettingsBase.setup docstring)

    Args:
      settings (list of tuples | dict with tuple list values)
      groups (bool): if True, settings should be dict with
        lists of the group settings
      parse (bool): if True, already parse arguments
    """

    def __init__(self, settings=[], groups=False, parse=True):
        self.groups = groups
        if settings:
            self._settings = settings
        else:
            self.setup()
        self.make_parser()
        if parse:
            self.parse_args()

    def __getitem__(self, key):
        return self.settings.__getattribute__(key)

    def setup(self):
        """Can be overwritten by inheriting classes.
        Allows defining parameters with type hints.
        Overwritten setup methods need to call `super().setup()`
        at the end.

        Example:
        >>> class Settings(SettingsBase):
        ...     def setup(_, a: int = 5, b: float = .1, c: str = 'a'):
        ...          super().setup()
        ... settings = Settings()
        """
        import inspect
        sig = inspect.signature(self.setup)
        # source = inspect.getsource(self.setup)  # to extract help comments
        # print(sig)
        self._settings = [
            (f'--{p}', {
                'default': sig.parameters[p].default,
                'type': sig.parameters[p].annotation
            }
            )
            for p in sig.parameters
        ]

    def make_parser(self, **kwargs):
        import argparse
        self.parser = argparse.ArgumentParser(**kwargs)
        for grp in self._settings:
            if self.groups:
                parser = self.parser.add_argument_group(grp)
            else:
                parser = self.parser
            for setting in (self._settings[grp] if self.groups else self._settings):
                parser.add_argument(setting[0], **setting[1])
            if not self.groups:
                break  # if no groups need to break

    def parse_args(self):
        self.settings = self.parser.parse_args()
        return self.settings
