import os
import json
import argparse
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


def predict(
    vid,
    model,
    fp16,
    result,
    num_frames,
    net,
    klass,
    count=0,
    accuracy=-1,
    correct_label="unknown",
    compression=None,
):
    count += 1
    print(f"\n\n{str(count)} Loading... {vid}")

    df = df_face(vid, num_frames, net)  # extract face from the frames
    if fp16:
        df.half()
    y, y_val = (
        pred_video(df, model)
        if len(df) >= 1
        else (torch.tensor(0).item(), torch.tensor(0.5).item())
    )
    result = store_result(
        result, os.path.basename(vid), y, y_val, klass, correct_label, compression
    )

    if accuracy > -1:
        if correct_label == real_or_fake(y):
            accuracy += 1
        print(
            f"\nPrediction: {y_val} {real_or_fake(y)} \t\t {accuracy}/{count} {accuracy/count}"
        )


def gen_parser():
    parser = argparse.ArgumentParser("ConSwinT prediction")
    parser.add_argument("--p", type=str, help="video or image path")
    parser.add_argument(
        "--f", type=int, help="number of frames to process for prediction"
    )
    parser.add_argument(
        "--d", type=str, help="dataset type, dfdc, faceforensics, timit, celeb"
    )
    parser.add_argument("--n", type=str, help="network ed or vae")
    parser.add_argument("--fp16", type=str, help="half precision support")

    args = parser.parse_args()
    path = args.p
    num_frames = args.f if args.f else 15
    dataset = args.d if args.d else "other"
    net = args.n if args.n in ["ed", "vae"] else "genconvit"
    fp16 = True if args.fp16 else False

    return path, dataset, num_frames, net, fp16


def main():
    start_time = perf_counter()
    path, dataset, num_frames, net, fp16 = gen_parser()
    result = (
        globals()[dataset](path, dataset, num_frames, net, fp16)
        if dataset in ["dfdc", "faceforensics", "timit", "celeb"]
        else vids(path, dataset, num_frames, net, fp16)
    )

    curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
    file_path = os.path.join("result", f"prediction_{dataset}_{net}_{curr_time}.json")

    with open(file_path, "w") as f:
        json.dump(result, f)
    end_time = perf_counter()
    print("\n\n--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
    main()
