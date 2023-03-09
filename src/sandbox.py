import pathlib as pl
import numpy as np
import pandas as pd
import os

d: dict[str, int] = {}
d["majs"] = d.get("majs", 0) + 12
print(d)