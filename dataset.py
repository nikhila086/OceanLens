import os
from roboflow import Roboflow

def download_roboflow_dataset():
    rf = Roboflow(api_key="c1rtg52qxbZLZviMdfMm")
    dataset = rf.workspace("lumsworkspace").project("river-pollution-master").version(1).download("yolov5")
    print(f" Download complete: {dataset.location}")

if __name__ == "__main__":
    download_roboflow_dataset()





   