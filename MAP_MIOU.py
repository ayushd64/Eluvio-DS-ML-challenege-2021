#!/usr/bin/env python
# coding: utf-8

# <h2>Importing Libraries</h2>

# In[220]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, f1_score, roc_curve
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn import svm
import matplotlib.pyplot as plt
import pickle
import torch
import os
import sys
import glob
import json


# <h2>Data Reading and Model Training</h2>

# In[278]:


# read and train the model
probabilities = dict()
def read_files():
    file_names = glob.glob('tt*')
    for i in file_names:
        print('-->For '+str(i))
        x_train = pd.read_csv(i+'/x_train.csv', index_col=0)
        y_train = pd.read_csv(i+'/y_train.csv', index_col=0)
        x_test = pd.read_csv(i+'/x_test.csv', index_col=0)
        y_test = pd.read_csv(i+'/y_test.csv', index_col=0)
        clf = RandomForestClassifier(random_state=42)
        clf = clf.fit(x_train, y_train.values.ravel())
        prob = clf.predict_proba(x_test)
#         y_score = clf.decision_function(x_test)
        probabilities[i] = prob


# In[279]:


# here the list of files that are read by the read_files()
read_files()


# In[280]:


# Fetching ground-truth values for each .pkl_files
test_ground_truth = dict()
def read_y_test():
    file_names = glob.glob('tt*')
    for i in file_names:
        y_test = pd.read_csv(i+'/y_test.csv', index_col=0).values.ravel()
        test_ground_truth[i] = y_test
read_y_test()
test_ground_truth


# <h2>Calculating Mean Average Precision</h2>

# In[447]:


ap = dict()
gt = dict()
pt = dict()
for i in test_ground_truth.keys():
    ap[i] = average_precision_score(test_ground_truth[i], probabilities[i][:,1:].ravel(), pos_label=0)
    gt[i] = test_ground_truth[i]
    pt[i] = probabilities[i][:,1:].ravel()
mAP = sum(ap.values()) / len(ap)
print('Mean Average Precision: ', mAP)


# <h2>Function to calculate MIOU</h2>

# In[450]:


def calc_miou(gt_dict, pr_dict, shot_to_end_frame_dict, threshold=0.5):
    """Maximum IoU (Miou) for scene segmentation.
    Miou measures how well the predicted scenes and ground-truth scenes overlap. The descriptions can be found in
    https://arxiv.org/pdf/1510.08893.pdf. Note the length of intersection or union is measured by the number of frames.
    Args:
        gt_dict: Scene transition ground-truths.
        pr_dict: Scene transition predictions.
        shot_to_end_frame_dict: End frame index for each shot.
        threshold: A threshold to filter the predictions.
    Returns:
        Mean MIoU, and a dict of MIoU for each movie.
    """
    def iou(x, y):
        s0, e0 = x
        s1, e1 = y
        smin, smax = (s0, s1) if s1 > s0 else (s1, s0)
        emin, emax = (e0, e1) if e1 > e0 else (e1, e0)
        return (emin - smax + 1) / (emax - smin + 1)

    def scene_frame_ranges(scene_transitions, shot_to_end_frame):
        end_shots = np.where(scene_transitions)[0]
        scenes = np.zeros((len(end_shots) + 1, 2), dtype=end_shots.dtype)
        scenes[:-1, 1] = shot_to_end_frame[end_shots]
        scenes[-1, 1] = shot_to_end_frame[len(scene_transitions)]
        scenes[1:, 0] = scenes[:-1, 1] + 1
        return scenes

    def miou(gt_array, pr_array, shot_to_end_frame):
        gt_scenes = scene_frame_ranges(gt_array, shot_to_end_frame)
        pr_scenes = scene_frame_ranges(pr_array >= threshold, shot_to_end_frame)
        assert gt_scenes[-1, -1] == pr_scenes[-1, -1]

        m = gt_scenes.shape[0]
        n = pr_scenes.shape[0]

        # IoU for (gt_scene, pr_scene) pairs
        iou_table = np.zeros((m, n))

        j = 0
        for i in range(m):
            # j start prior to i end
            while pr_scenes[j, 0] <= gt_scenes[i, 1]:
                iou_table[i, j] = iou(gt_scenes[i], pr_scenes[j])
                if j < n - 1:
                    j += 1
                else:
                    break
            # j end prior to (i + 1) start
            if pr_scenes[j, 1] < gt_scenes[i, 1] + 1:
                break
            # j start later than (i + 1) start
            if pr_scenes[j, 0] > gt_scenes[i, 1] + 1:
                j -= 1
        assert np.isnan(iou_table).sum() == 0
        assert iou_table.min() >= 0

        # Miou
        return (iou_table.max(axis=0).mean() + iou_table.max(axis=1).mean()) / 2

    assert gt_dict.keys() == pr_dict.keys()

    miou_dict = dict()

    for imdb_id in gt_dict.keys():
        miou_dict[imdb_id] = miou(gt_dict[imdb_id], pr_dict[imdb_id], shot_to_end_frame_dict[imdb_id])
    mean_miou = sum(miou_dict.values()) / len(miou_dict)

    return mean_miou, miou_dict


# <h2>Calculating mean MIOU score</h2>

# In[451]:
hola = read_sef()
hola

scores = dict()
scores["Miou"], _ = calc_miou(gt, pt, hola)
print("Mean MIOU: ",scores["Miou"])


# <h2>Reading 'Shot-End Frame' for each .pkl file</h2>

# In[347]:


# Function to read 'Shot-End Frame' for each .pkl files'
def read_sef():
    sed = dict()
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    for file in read_files():
        with open(file, 'rb') as f:
            pickle_model = pickle.load(f)
        imdb_ids = pickle_model.get('imdb_id')
        shot_end_frame = torch.as_tensor(pickle_model.get('shot_end_frame'))
        sed[imdb_ids] = shot_end_frame
    return sed


# In[348]:


# Function to read .pkl files
def read_files():
    file_names = glob.glob('data/tt*.pkl')
    return file_names


# In[350]:





# In[ ]:




