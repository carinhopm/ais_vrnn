import pandas as pd
import json
import numpy as np
from scipy import sparse
import torch
import pickle
import datetime
import random


class Preprocess:
    def __init__(self, Config):

        self.max_interval = Config.max_interval
        self.epoch_to_T0 = Config.epoch_to_T0

        self.threshold_trajlen = Config.threshold_trajlen
        self.threshold_dur_min = Config.threshold_dur_min
        self.threshold_dur_max = Config.threshold_dur_max
        self.threshold_moored = Config.threshold_moored

        self.freq = Config.freq

        self.lat_min = Config.lat_min
        self.lat_max = Config.lat_max
        self.long_min = Config.long_min
        self.long_max = Config.long_max

        self.ROI_boundary_long = Config.ROI_boundary_long
        self.ROI_boundary_lat = Config.ROI_boundary_lat

        self.max_knot = Config.max_knot
        self.sog_res = Config.sog_res
        self.sog_columns = Config.sog_columns

        self.cog_res = Config.cog_res
        self.cog_columns = Config.cog_columns

        self.lat_long_res = Config.lat_long_res
        self.lat_columns = Config.lat_columns
        self.long_columns = Config.long_columns

        self.train_cuttime = Config.train_cuttime
        self.val_cuttime = Config.val_cuttime


    def check_moored(self, df):
        perc_still = len(df[df.SOG <= 10]) / len(df)
        moored = perc_still > self.threshold_moored

        return perc_still, moored

    def check_ROI(self, df):
        _latmin, _latmax = min(df.Latitude / 600000), max(df.Latitude / 600000)

        _longmin, _longmax = min(df.Longitude / 600000), max(df.Longitude / 600000)

        not_in_ROI = _latmin < self.lat_min or _latmax > self.lat_max or _longmin < self.long_min or _longmax > self.long_max

        return not_in_ROI

    def resample_interpolate(self, df):

        df["Time"] = pd.to_datetime(df["Time"] + self.epoch_to_T0, unit="s")
        df = df.set_index('Time').resample(str(self.freq) + 'T').mean()
        df = df.interpolate("linear")
        df = df.reset_index(level=0, inplace=False)

        return df

    def split_ROI(self, path):
        df = pd.DataFrame(path, columns=["Time", "Latitude", "Longitude", "SOG", "COG", "Header"])

        out_east = np.array([1 if val / 600000 < self.long_min else 0 for val in df.Longitude])
        long_ROI = np.array([1 if val / 600000 < self.ROI_boundary_long else 0 for val in df.Longitude])
        lat_ROI = np.array([1 if val / 600000 < self.ROI_boundary_lat else 0 for val in df.Latitude])
        out_south = np.array([1 if val / 600000 < self.lat_min else 0 for val in df.Latitude])

        breaks = np.concatenate((np.where(long_ROI[:-1] != long_ROI[1:])[0] + 1,
                                 np.where(lat_ROI[:-1] != lat_ROI[1:])[0] + 1,
                                 np.where(out_east[:-1] != out_east[1:])[0] + 1,
                                 np.where(out_south[:-1] != out_south[1:0])[0] + 1), axis=0)

        subpaths = np.array_split(df, np.sort(breaks))

        return subpaths

    def create_bins(self, feature_array, resolution, scaling_factor, upperbound=None, lowerbound=None, kind = None):
        if upperbound:
            upperbound = upperbound * scaling_factor
            feature_array = [upperbound if coord > upperbound else coord for coord in feature_array]
        if lowerbound:
            lowerbound = lowerbound * scaling_factor
            feature_array = [lowerbound if coord < lowerbound else coord for coord in feature_array]

        if kind == "coordinate":
            feature_array = [np.round(np.round(coord / (resolution * scaling_factor)) * resolution,2) for coord in feature_array]
            return feature_array

        feature_array = [np.round(coord / (resolution * scaling_factor)) * resolution for coord in feature_array]

        return feature_array

    def one_hot_and_fill(self, feature_array, cols):
        dummies = pd.get_dummies(feature_array)
        dummies = dummies.reindex(columns=cols).fillna(0)

        return dummies

    def four_hot_encode(self, path, ROI):
        sog = self.create_bins(path["SOG"], self.sog_res, scaling_factor=10, upperbound=self.max_knot)
        sog = self.one_hot_and_fill(sog, self.sog_columns)

        cog = self.create_bins(path["COG"], self.cog_res, scaling_factor=10)
        cog = self.one_hot_and_fill(cog, self.cog_columns)

        lat = self.create_bins(path["Latitude"], self.lat_long_res, scaling_factor=600000, upperbound=self.lat_max,
                          lowerbound=self.lat_min, kind="coordinate")
        lat = self.one_hot_and_fill(lat, cols = pd.Float64Index(np.round(self.lat_columns[ROI], 2)))

        long = self.create_bins(path["Longitude"], self.lat_long_res, scaling_factor=600000, upperbound=self.long_max,
                           lowerbound=self.long_min, kind="coordinate")
        long = self.one_hot_and_fill(long, cols = pd.Float64Index(np.round(self.long_columns[ROI], 2)))

        data = sparse.csr_matrix(np.concatenate((lat, long, sog, cog), axis=1))

        return data

    def prepare_output(self, js, subsubpath, ROI):
        js1 = js.copy()
        df = subsubpath.copy()
        df["Time"] = df["Time"].astype(str)
        js1["path"] = df.to_dict("list")
        js1["FourHot"] = self.four_hot_encode(js1["path"], ROI)

        return js1

    def split_and_collect_trajectories(self, files, month, category):  #Takes as input a list of files each with shiptype "cat" in month "month"
        print(f"Processing files from month: {month}")
        print(f"===> of type: {category}")
        data_train_bh = []
        data_val_bh = []
        data_test_bh = []

        data_train_sk = []
        data_val_sk = []
        data_test_sk = []

        data_train_blt = []
        data_val_blt = []
        data_test_blt = []

        disc_short = 0
        disc_ROI = 0
        disc_moored = 0
        disc_dur_min = 0


        for i, file in enumerate(files): #For each file
            if i % 100 == 0:
                print(f"Preparing file {i}/{len(files)}")
            with open(file) as json_file:
                js = json.load(json_file)[0]

            res = [js["path"][i + 1][0] - js["path"][i][0] for i in range(len(js["path"]) - 1)] #Make a list of time differences between 2 consecutive updates

            break_points = [ids + 1 for ids, diff in enumerate(res) if diff > self.max_interval] #Find updates to split up based on time difference (Prepare Step 2.4)

            paths = np.split(js["path"], break_points) #Do step 2.4

            for path in paths: #For each track

                # Dismissing path if less than **threshold** messages       Step 2.5
                if len(path) < self.threshold_trajlen:
                    disc_short += 1
                    continue

                subpaths = self.split_ROI(path)    #Filter updates outside ROI. Part of Step 2.2

                for subpath in subpaths:

                    if any(subpath.Longitude < self.long_min * 600000):    #Does the longitude filtering of updates outside ROI. Part of Step 2.2
                        disc_ROI += 1
                        continue

                    if any(subpath.Longitude > self.long_max * 600000):    #Does the longitude filtering of updates outside ROI. Part of Step 2.2
                        disc_ROI += 1
                        continue

                    if any(subpath.Latitude < self.lat_min * 600000):      #Does the latitude filtering of updates outside ROI. Part of Step 2.2
                        disc_ROI += 1
                        continue

                    if len(subpath) < self.threshold_trajlen:       # Do Step 2.5 again since filtering might have made some track too short
                        disc_short += 1
                        continue

                    # Resampling and interpolating
                    df = self.resample_interpolate(subpath)          # Step 2.6

                    # Check and split path if longer than **threshold**
                    idx = (np.arange(0, len(df) * self.freq * 60, self.threshold_dur_max) / (self.freq * 60)).astype(int)[1:]    # Step 2.7

                    subsubpaths = np.split(df, idx)

                    for subsubpath in subsubpaths:                      # For each segment

                        # dismissing path if shorter than **threshold** duration
                        if (subsubpath["Time"].iloc[-1] - subsubpath["Time"].iloc[0]).total_seconds() < self.threshold_dur_min:   #Again check step 2.5
                            disc_dur_min += 1
                            continue

                        perc_still, moored = self.check_moored(subsubpath)                  #Step 2.3

                        if moored:
                            disc_moored += 1
                            continue

                        # Splitting based on region                 #Marie made her code to work some 3 different ROIs and does train/test split based on time. So she saves the final track to the "correct" dictionary
                        if all(subsubpath["Longitude"] > self.ROI_boundary_long * 600000):
                            js1 = self.prepare_output(js, subsubpath, ROI="bh")
                            if all(subsubpath["Time"] < self.train_cuttime):
                                data_train_bh.append(js1)
                            elif all(subsubpath["Time"] < self.val_cuttime):
                                data_val_bh.append(js1)
                            else:
                                data_test_bh.append(js1)
                        elif all(subsubpath["Latitude"] > self.ROI_boundary_lat * 600000):
                            js1 = self.prepare_output(js, subsubpath, ROI="sk")
                            if all(subsubpath["Time"] < self.train_cuttime):
                                data_train_sk.append(js1)
                            elif all(subsubpath["Time"] < self.val_cuttime):
                                data_val_sk.append(js1)
                            else:
                                data_test_sk.append(js1)
                        else:
                            js1 = self.prepare_output(js, subsubpath, ROI="blt")
                            if all(subsubpath["Time"] < self.train_cuttime):
                                data_train_blt.append(js1)
                            elif all(subsubpath["Time"] < self.val_cuttime):
                                data_val_blt.append(js1)
                            else:
                                data_test_blt.append(js1)

        stats = {"disc_short": disc_short,
                 "disc_ROI": disc_ROI,
                 "disc_moored": disc_moored,
                 "disc_dur_min": disc_dur_min,
                 "train_no_bh": len(data_train_bh),
                 "val_no_bh": len(data_val_bh),
                 "test_no_bh": len(data_test_bh),
                 "train_no_sk": len(data_train_sk),
                 "val_no_sk": len(data_val_sk),
                 "test_no_sk": len(data_test_sk),
                 "train_no_blt": len(data_train_blt),
                 "val_no_blt": len(data_val_blt),
                 "test_no_blt": len(data_test_blt)
                 }

        train = {"bh": data_train_bh,
                 "sk": data_train_sk,
                 "blt": data_train_blt}

        val = {"bh": data_val_bh,
                "sk": data_val_sk,
                "blt": data_val_blt}

        test = {"bh": data_test_bh,
                "sk": data_test_sk,
                "blt": data_test_blt}

        return train, val, test, stats
