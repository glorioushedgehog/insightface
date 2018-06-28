import argparse
import sys

import cv2

from deploy import face_model


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='path to the image')
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    return parser.parse_args(argv)


def main(args):
    args.det = 0
    args.model = "C:\\Users\\Paul\\insightface\\models\\model-r50-am-lfw\\,0"
    model = face_model.FaceModel(args)
    img = cv2.imread(args.image)
    img = cv2.resize(img, (112, 112))
    img = model.get_input(img)
    f1 = model.get_feature(img)
    print(f1)
    #dist = np.sum(np.square(f1 - f2))
    #print(dist)
    #sim = np.dot(f1, f2.T)
    #print(sim)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))