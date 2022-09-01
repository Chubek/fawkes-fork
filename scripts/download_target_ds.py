import zipfile

import gdown
from dotenv import dotenv_values

CONFIG = dotenv_values(".env")

id = CONFIG['TARGET_DS_GID']
output = CONFIG['TARGET_OUTPUT']

gdown.download(
    id=id,
    output=output,
)

with zipfile.ZipFile(output) as zf:
    zf.extractall()
