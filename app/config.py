import os
import sys
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Config:
    Z_BASE_PATH = os.path.dirname(os.path.dirname(__file__))
    Z_ARTIFACTS_PATH = os.path.join(Z_BASE_PATH, 'app', 'artifacts')
    Z_TIMESTAMP = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    
    os.makedirs(Z_ARTIFACTS_PATH, exist_ok=True)
