# -*- coding: utf-8 -*-
"""sina configuration
"""
from appdirs import AppDirs
appdirs = AppDirs('sina','dicaso')

# Secrets: config for storing user API keys and other sensitive/personal information
from kindi import Secrets
secrets = Secrets(default_section=__package__).instance

