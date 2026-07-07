import logging
import sys

from .pipeline import main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
sys.exit(main())
