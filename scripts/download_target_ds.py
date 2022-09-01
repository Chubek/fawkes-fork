import gdown
import zipfile

id = "1W_Cm8lS4GPN6TCOqIkPdTzbtH_dFYfJM"
output = "target_images/timg.zip"

gdown.download(
    id=id, 
    output=output, 
)

with zipfile.ZipFile(output) as zf:
    zf.extractall()