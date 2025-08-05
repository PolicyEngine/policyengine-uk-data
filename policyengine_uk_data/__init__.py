from .datasets import *
from .storage.download_private_prerequisites import (
    download_prerequisites,
    check_prerequisites,
)

# Check prerequisites on import and warn if missing
check_prerequisites()
