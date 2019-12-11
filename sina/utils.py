"""Utility classes or functions
for sina package
"""


def logmemory():
    import logging
    import resource
    logging.info(
        'Using max {:.0f}MB'.format(resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024)
    )
