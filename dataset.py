import os
from roboflow import Roboflow
from dotenv import load_dotenv

def download_roboflow_dataset():
   
    load_dotenv(dotenv_path="config.env")

    api_key = os.getenv("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=api_key)
    dataset = rf.workspace("lumsworkspace").project("river-pollution-master").version(1).download("yolov5")
    print(f"Download complete: {dataset.location}")

if __name__ == "__main__":
    download_roboflow_dataset()
