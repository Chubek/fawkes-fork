import zipfile

import gdown

id = "1W_Cm8lS4GPN6TCOqIkPdTzbtH_dFYfJM"
output = "target_data/timg.zip"

gdown.download(
    id=id,
    output=output,
)

with zipfile.ZipFile(output) as zf:
    zf.extractall()
