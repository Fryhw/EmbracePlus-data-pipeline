from inspect import getsourcefile
from pathlib import Path
from socket import gethostname



_mbrain_root_dir = Path(r"C:/Users\Admin\Documents/GitHub/data-quality-challenges-wearables/loc_data/FoTiPDataset/mBrain21")

    # The path which contains the metadata for the mbrain data
mbrain_metadata_path = _mbrain_root_dir / "metadata"

    # The location in which the processed daily and tz-aware mbrain data is stored
processed_mbrain_path = _mbrain_root_dir / "obelisk_dump"


loc_data_dir = Path(getsourcefile(lambda: 0)).parent.parent.absolute() / "loc_data"
figure_dir = loc_data_dir.parent / "figures"


assert mbrain_metadata_path.exists()
assert processed_mbrain_path.exists()

assert loc_data_dir.exists()


loc_data_dir = Path(getsourcefile(lambda: 0)).parent.parent.absolute() / "loc_data"
figure_dir = loc_data_dir.parent / "figures"

assert loc_data_dir.exists()
