import os
import zipfile
from six.moves.urllib.request import urlretrieve


def download(zip_to_download_name, filename):
  origin = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/"
  origin += zip_to_download_name

  datadir = os.path.join("data")
  if not os.path.exists(datadir):
      os.makedirs(datadir)

  zip_path = os.path.join(datadir, filename)
  zip_path = zip_path + ".zip"

  urlretrieve(origin, zip_path)

  with zipfile.ZipFile(zip_path,"r") as zip_ref:
      zip_ref.extractall(datadir)


download("vangogh2photo.zip", "vangogh")
download("ukiyoe2photo.zip", "ukiyoe")
download("monet2photo.zip", "monet")