import datetime

import face_model
import argparse
import cv2
import sys
import os
import numpy as np
from skimage.transform import resize


def show_images(person_name, image_dir):
    image_nams = os.listdir(os.path.join(image_dir, person_name))
    image_path1 = os.path.join(data_dir, person, image_nams[0])
    image_path2 = os.path.join(data_dir, person, image_nams[-1])
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    aligned_image1 = model.get_input(image1)
    aligned_image2 = model.get_input(image2)
    aligned_image1 = np.transpose(aligned_image1, (1, 2, 0))
    aligned_image1 = cv2.cvtColor(aligned_image1, cv2.COLOR_RGB2BGR)
    aligned_image2 = np.transpose(aligned_image2, (1, 2, 0))
    aligned_image2 = cv2.cvtColor(aligned_image2, cv2.COLOR_RGB2BGR)
    numpy_horizontal_concat = np.concatenate((image1, aligned_image1, image2, aligned_image2), axis=1)
    cv2.imshow(person_name + " aligned", numpy_horizontal_concat)

    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()


def show_wrong_images(wrong_names, image_dir):
    if len(wrong_names) == 0:
        return
    image_to_show = None
    numpy_horizontal_concat = None
    for person_name in wrong_names:
        image_nams = os.listdir(os.path.join(image_dir, person_name))
        image_path1 = os.path.join(data_dir, person_name, image_nams[0])
        image_path2 = os.path.join(data_dir, person_name, image_nams[-1])
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)
        aligned_image1 = model.get_input(image1)
        aligned_image2 = model.get_input(image2)
        aligned_image1 = np.transpose(aligned_image1, (1, 2, 0))
        aligned_image1 = cv2.cvtColor(aligned_image1, cv2.COLOR_RGB2BGR)
        aligned_image2 = np.transpose(aligned_image2, (1, 2, 0))
        aligned_image2 = cv2.cvtColor(aligned_image2, cv2.COLOR_RGB2BGR)
        numpy_horizontal_concat = np.concatenate((image1, image2, aligned_image1, aligned_image2), axis=1)
        if image_to_show is None:
            image_to_show = numpy_horizontal_concat
        else:
            image_to_show = np.concatenate((image_to_show, numpy_horizontal_concat), axis=0)
    saved_image_name = str(num_classes_to_skip) + "_" + str(num_classes) + "_insightface.jpg"
    save_dir = os.path.join("C:\\Users\\Paul\\insightface\\datasets\\test_results", saved_image_name)
    cv2.imwrite(save_dir, image_to_show)

    time_now = datetime.datetime.now()
    diff = time_now - time0
    print('running time', diff.total_seconds(), "seconds")

    print("correct: " + str(correct))
    print("wrong: " + str(wrong))
    print("{:.2%} correct".format(correct / len(person_list)))
    print(wrong_list)

    #cv2.imshow("wrong", image_to_show)
    #k = cv2.waitKey(0)
    #if k == 27:  # wait for ESC key to exit
    #    cv2.destroyAllWindows()


time0 = datetime.datetime.now()

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

args.det = 0
args.model = "C:\\Users\\Paul\\insightface\\models\\model-r50-am-lfw\\,0"
num_classes = 40
num_classes_to_skip = 0
model = face_model.FaceModel(args)

correct = 0
wrong = 0
data_dir = "C:\\Users\\Paul\\LFW-RESIZE\\lfw-112"
person_list = os.listdir(data_dir)[num_classes_to_skip:num_classes_to_skip + num_classes]
done = 0
feature_list = []
for person in person_list:
    print(person)
    image_names = os.listdir(os.path.join(data_dir, person))
    image_path = os.path.join(data_dir, person, image_names[0])
    image = cv2.imread(image_path)
    image = model.get_input(image)
    features = model.get_feature(image)
    feature_list.append(features)
    done += 1
    print("{:.2%}".format(done / len(person_list)))
print("done with features for reference images")
done = 0
wrong_list = []
for i, person in enumerate(person_list):
    image_names = os.listdir(os.path.join(data_dir, person))
    image_path = os.path.join(data_dir, person, image_names[-1])
    image = cv2.imread(image_path)
    image = model.get_input(image)
    features = model.get_feature(image)
    best_sim = 0
    index_of_best_sim = None
    for j, comparison_features in enumerate(feature_list):
        #dist = np.sum(np.square(comparison_features - features))
        sim = np.dot(comparison_features, features.T)
        if sim > best_sim:
            index_of_best_sim = j
            best_sim = sim
    print(best_sim)
    if index_of_best_sim == i:
        correct += 1
    else:
        wrong += 1
        wrong_list.append(person)
        #show_images(person, data_dir)
    done += 1
    print("{:.2%}".format(done / len(person_list)))
show_wrong_images(wrong_list, data_dir)


