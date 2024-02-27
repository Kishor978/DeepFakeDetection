import os
import json
from time import perf_counter
from datetime import datetime
from model.prediction_utils import *

def vids(
    root_dir="sample_prediction_data", dataset=None, num_frames=15, net=None, fp16=False
):
    result = set_result()
    r = 0
    f = 0
    count = 0
    model = load_conswint(net, fp16)

    for filename in os.listdir(root_dir):
        curr_vid = os.path.join(root_dir, filename)

        try:
            if is_video(curr_vid):
                result, accuracy, count, pred = predict(
                    curr_vid,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    "uncategorized",
                    count,
                )
                f, r = (f + 1, r) if "FAKE" == real_or_fake(pred[0]) else (f, r + 1)
                print(
                    f"Prediction: {pred[1]} {real_or_fake(pred[0])} \t\tFake: {f} Real: {r}"
                )
            else:
                print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    return result


def celeb(root_dir="Celeb-DF-v2", dataset=None, num_frames=15, net=None, fp16=False):
    with open(os.path.join("json_file", "celeb_test.json"), "r") as f:
        cfl = json.load(f)
    result = set_result()
    ky = ["Celeb-real", "Celeb-synthesis"]
    count = 0
    accuracy = 0
    model = load_conswint(net, fp16)

    for ck in cfl:
        ck_ = ck.split("/")
        klass = ck_[0]
        filename = ck_[1]
        correct_label = "FAKE" if klass == "Celeb-synthesis" else "REAL"
        vid = os.path.join(root_dir, ck)

        try:
            if is_video(vid):
                result, accuracy, count, _ = predict(
                    vid,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    klass,
                    count,
                    accuracy,
                    correct_label,
                )
            else:
                print(f"Invalid video file: {vid}. Please provide a valid video file.")

        except Exception as e:
            print(f"An error occurred x: {str(e)}")

    return result
