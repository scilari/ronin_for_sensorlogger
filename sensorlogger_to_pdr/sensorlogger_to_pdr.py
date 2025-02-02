
import pandas as pd
import numpy as np
import os
import glob
import argparse

def process(path, ronin_path, model_path):
    create_ronin_pickle(path)
    run_ronin(path, ronin_path, model_path)
    create_pdr_csv(path)

def create_ronin_pickle(path):
  print(f"Creating RoNIN format pickle from SensorLogger data in {path}" )
  acc = pd.read_csv(path + "/TotalAcceleration.csv")
  gyr = pd.read_csv(path + "/Gyroscope.csv")
  rv = pd.read_csv(path + "/Orientation.csv") 

  df = pd.DataFrame()
  
  # Sample and interpolate data to 200 Hz
  ts = np.arange(acc.seconds_elapsed.iloc[0], acc.seconds_elapsed.iloc[-1], 0.005)

  df["seconds_elapsed"] = ts
  
  df["time"] = ts * 1e9 # to nanoseconds for RoNIN

  df["acce_x"] = np.interp(ts, acc.seconds_elapsed, acc.x)
  df["acce_y"] = np.interp(ts, acc.seconds_elapsed, acc.y)
  df["acce_z"] = np.interp(ts, acc.seconds_elapsed, acc.z)

  df["gyro_x"] = np.interp(ts, gyr.seconds_elapsed, gyr.x)
  df["gyro_y"] = np.interp(ts, gyr.seconds_elapsed, gyr.y)
  df["gyro_z"] = np.interp(ts, gyr.seconds_elapsed, gyr.z)

  df["game_rv_w"] = np.interp(ts, rv.seconds_elapsed, rv.qw)
  df["game_rv_x"] = np.interp(ts, rv.seconds_elapsed, rv.qx)
  df["game_rv_y"] = np.interp(ts, rv.seconds_elapsed, rv.qy)
  df["game_rv_z"] = np.interp(ts, rv.seconds_elapsed, rv.qz)

  filename = path + "/processed/data.pkl"
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  print(f"Saving RoNIN format pickle to {filename}")
  df.to_pickle(filename)

def run_ronin(path, ronin_path, model_path):
    print("Running RoNIN...")
    cmd = f"python {ronin_path}/source/ronin_resnet.py" \
      f" --mode test --dataset ridi --test_path {path}" \
      f" --out_dir {path}/processed --model_path {model_path}"
    os.system(cmd)

def _annotate_pdr(pdr_df, path):
    annotation_path = os.path.join(path, "Annotation.csv")
    if not os.path.exists(annotation_path):
        return
    
    annotations = pd.read_csv(annotation_path)
    print(f"Found {len(annotations)} annotations, adding to PDR CSV")

    pdr_df["annotation"] = ""
    
    for s, text in zip(annotations.seconds_elapsed, annotations.text):
        gt_ix = (pdr_df.seconds_elapsed - s).abs().idxmin()
        pdr_df.loc[gt_ix, "annotation"] = text


def create_pdr_csv(path):
    print("Creating PDR CSV")
    xy_file = glob.glob(path + "/processed/*.npy", recursive=True)[0]
    xy_data = np.load(xy_file)
    x = xy_data[:, 0]
    y = xy_data[:, 1]

    df_ronin = pd.read_pickle(path + "/processed/data.pkl")
    df = pd.DataFrame()
    df["seconds_elapsed"] = df_ronin.seconds_elapsed
    df["t"] = df_ronin.seconds_elapsed # Just for convenience

    df["x"] = x
    df["y"] = y

    annotate_pdr(df, path)

    pdr_path = path + "/processed/pdr.csv"
    print(f"Saving PDR output to {pdr_path}")
    df.to_csv(pdr_path, index=False)

def main():
  parser=argparse.ArgumentParser()

  parser.add_argument("--path", help="Data path for the Sensorlogger data")
  parser.add_argument("--ronin_path", help="RoNIN fork path")
  parser.add_argument("--model_path", help="RoNIN model path")
  
  args=parser.parse_args()

  process(args.path, args.ronin_path, args.model_path)


if __name__ == "__main__":
    main()