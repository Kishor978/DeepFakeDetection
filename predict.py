import os
import json
import argparse
from time import perf_counter

from datetime import datetime
from model.prediction_utils import *

def vids(
    root_dir="sample_prediction_data\\", num_frames=10, net="ed", fp16=False
):
    result = set_result()
    r = 0
    f = 0
    count = 0
    model = load_conswint(net, fp16)

    try:
        # # Check if the root directory exists
        # if not os.path.exists(root_dir):
        #     raise FileNotFoundError(f"Directory '{root_dir}' does not exist.")

        # # Check if the root directory is a directory
        # if not os.path.isdir(root_dir):
        #     raise NotADirectoryError(f"'{root_dir}' is not a directory.")

    
        # for filename in os.listdir(root_dir):
        curr_vid = root_dir
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
                    f"\n\nPrediction: {pred[1]*100}% \t\t{real_or_fake(pred[0])} \n\nFake: {f}\t\t Real: {r}"
                )
            else:
                print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
    except Exception as e:
        print(f"Error while processing directory '{root_dir}': {str(e)}")

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
    print(f"\n\n{str(count)} Loading...\t {vid}")

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
            f"\n\nPrediction: {y_val*100}%  {real_or_fake(y)} \t\t {accuracy}/{count} {accuracy/count}"
        )

    return result, accuracy, count, [y, y_val]


# def gen_parser():
#     parser = argparse.ArgumentParser("ConvSwinT prediction")
#     parser.add_argument("--p", type=str, help="video or image path")
#     parser.add_argument(
#         "--f", type=int, help="number of frames to process for prediction"
#     )
#     parser.add_argument("--n", type=str, help="network ed or vae")
#     parser.add_argument("--fp16", type=str, help="half precision support")

#     args = parser.parse_args()
#     path = args.p
#     num_frames = args.f if args.f else 15
#     dataset = "other"
#     net = args.n if args.n in ["ed", "vae"] else "genconvit"
#     fp16 = True if args.fp16 else False

#     return path, dataset, num_frames, net, fp16


# def main():
#     start_time = perf_counter()
#     path, dataset, num_frames, net, fp16 = gen_parser()
#     result =vids(path, dataset, num_frames, net, fp16)


#     curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
#     file_path = os.path.join("result", f"prediction_{dataset}_{net}_{curr_time}.json")

#     with open(file_path, "w") as f:
#         json.dump(result, f)
#     end_time = perf_counter()
    # print("\n\n--- %s seconds ---" % (end_time - start_time))

# def main():
#     path="E:\\Minor_project\\DeepFakeDetection\\sample_prediction_data"
#     start_time = perf_counter()
#     result =vids()


#     curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
#     # file_path = os.path.join("result", f"prediction_{dataset}_{net}_{curr_time}.json")

#     # with open(file_path, "w") as f:
#     #     json.dump(result, f)
#     end_time = perf_counter()
# if __name__ == "__main__":
#     main()
