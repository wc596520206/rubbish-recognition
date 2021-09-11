import sys
import argparse
import json
import logging
import os
from TrainModel import TrainFace
from InferModel import InferFace

if __name__ == "__main__":
    # Read the input information by the user on the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="config path of model",
                        default=r"E:\3业余资料\10人脸性别分类\code\config\rubbish.json")
    parser.add_argument("--phase", default="train")

    args = parser.parse_args()
    model_file = args.config_file
    with open(model_file, "r", encoding="UTF-8") as fr:
        config = json.load(fr)

    if args.phase == "train":
        config["phase"] = "train"
    if args.phase == "infer":
        config["phase"] = "infer"

    log_path = config["global"]["log_path"]
    task = config["global"]["task"]
    if log_path:
        if not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))
        logger = logging.getLogger(task)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_path, encoding="UTF-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if args.phase == "train":
        config["phase"] = "train"
        TrainFace(config)
    elif args.phase == "infer":
        config["phase"] = "infer"
        InferFace(config)
