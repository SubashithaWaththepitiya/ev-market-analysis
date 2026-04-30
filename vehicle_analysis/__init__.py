# vehicle_analysis package
# A modular Python package for EV, Petrol and Diesel vehicle market analysis
# Author: [Student Name]
# Date: April 2026

from . import data_loader
from . import analysis
from . import visualisation
from . import database

__version__ = "1.0.0"
__all__ = ["data_loader", "analysis", "visualisation", "database"]
