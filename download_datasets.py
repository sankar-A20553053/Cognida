import requests, zipfile, io, os

datasets = [
  ("http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip", "data/Market1501"),
  ("http://vision.cs.duke.edu/DukeMTMC/data/mTMCreID.zip",    "data/DukeMTMC-reID")
]

os.makedirs("data", exist_ok=True)

for url, outdir in datasets:
    print(f"Downloading {url} …")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    print(f"Extracting to {outdir} …")
    os.makedirs(outdir, exist_ok=True)
    z.extractall(outdir)

print("All done!")
