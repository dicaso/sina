# -*- coding: utf-8 -*-
"""sina configuration
"""

# Secrets: config for storing user API keys and other sensitive/personal information
from kindi import Secrets
secrets = Secrets(default_section=__package__).instance
