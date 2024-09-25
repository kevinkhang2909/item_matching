from core_pro import Drive
from core_pro.ultilities import make_dir
from pathlib import Path


path = Path.home() / 'Downloads/Data/Item_Matching_Test'
make_dir(path)
drive_id = '1TNgm50QJ3w2-h7cFw5b93cgKAK-zHaY1'
Drive().drive_download(drive_id, str(path))
