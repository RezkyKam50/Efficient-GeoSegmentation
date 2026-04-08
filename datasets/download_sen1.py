import gdown, os, tarfile
 
if not os.path.isfile('sen1floods11_v1.1.tar.gz'):
    gdown.download("https://drive.google.com/uc?id=1lRw3X7oFNq_WyzBO6uyUJijyTuYm23VS", output="./datasets/sen1floods11_v1.1.tar.gz")

tar_path = "./datasets/sen1floods11_v1.1.tar.gz"
extract_dir = "./datasets/"

if os.path.isfile(tar_path):
    print("Extracting archive...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print("Extraction complete.")
else:
    print(f"Archive not found at {tar_path}")