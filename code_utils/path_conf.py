from inspect import getsourcefile
from pathlib import Path
from socket import gethostname


loc_data_dir = Path(getsourcefile(lambda: 0)).parent.parent.absolute() / "loc_data"
figure_dir = loc_data_dir.parent / "figures"

assert loc_data_dir.exists()
