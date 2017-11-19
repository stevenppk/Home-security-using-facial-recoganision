#!/usr/bin/env python2
# modified by steven thomas

# Email notification
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEBase import MIMEBase
from email import encoders

import argparse
import cv2
import os
import pickle
import sys

import time
start = time.time()

from operator import itemgetter
import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def getRep(imgPath, multiple=True):
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        #raise Exception("Unable to load image: {}".format(imgPath))
	os.system("./imgdetection.sh")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)
    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
    if len(bbs) == 0 or (not multiple and bb1 is None):
        #raise Exception("Unable to find a face: {}".format(imgPath))
	os.system("./imgdetection.sh")# since no face is detected the process is

    reps = []
    for bb in bbs:
        alignedFace = align.align(
            args.imgDim,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            #raise Exception("Unable to align image: {}".format(imgPath))
	    os.system("./imgdetection.sh")

        # passing the aligned faces to the openface network
        rep = net.forward(alignedFace)
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps


def train(args):
    print("Loading embeddings.")
    fname = "{}/labels.csv".format(args.workDir)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    fname = "{}/reps.csv".format(args.workDir)
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print nClasses
    print("Training for {} classes.".format(nClasses))

    # Radial Basis Function kernel,works better with C = 1 and gamma = 2
    clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)

    clf.fit(embeddings, labelsNum)

    fName = "{}/classifier.pkl".format(args.workDir)
    print("Saving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)


def infer(args, multiple=True):
    with open(args.classifierModel, 'rb') as f:
        if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
        else:
                (le, clf) = pickle.load(f, encoding='latin1')

    for img in args.imgs:
        print("\n=== {} ===".format(img))
        reps = getRep(img, multiple)
        if len(reps) > 1:
            print("List of faces in image from left to right")
        for r in reps:
            rep = r[1].reshape(1, -1)
            bbx = r[0]
            predictions = clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = le.inverse_transform(maxI)
            confidence = predictions[maxI]
	    if confidence > 0.70000000000:
	            if multiple:
	                print("The person in the image : {} with {:.2f} confidence.".format(person.decode('utf-8'), bbx , confidence))
	            else:
                	print("The person in the image : {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
	    else:
		    print ("The person in the image is not recognized \n")
    sent_mail(args,person,confidence)

def sent_mail(args,person,confidence):

	img = args.imgs
	localtime = time.asctime( time.localtime(time.time()) )
	fromaddr = "123ppksteven@gmail.com"
	toaddr = "steventhomaspuli@gmail.com"

	msg = MIMEMultipart()

	msg['From'] = fromaddr
	msg['To'] = toaddr
	msg['Subject'] = "Someone is at your door"
	
	if confidence > 0.50000000000:
		body = ("{} is at your door now :: {} :: ".format(person.decode('utf-8'),localtime))
	else:
		body = ("Unknown person is at your door")

	msg.attach(MIMEText(body, 'plain'))

	filename = "person.jpeg"
	attachment = open(img[0], "rb")

	part = MIMEBase('application', 'octet-stream')
	part.set_payload((attachment).read())
	encoders.encode_base64(part)
	part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

	msg.attach(part)

	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.starttls()
	server.login(fromaddr, "123ppksteven")
	text = msg.as_string()
	server.sendmail(fromaddr, toaddr, text)
	server.quit()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dlibFacePredictor',type=str ,help="Path to dlib's face predictor.")
    parser.add_argument('--networkModel',type=str,help="Path to Torch network model.")
    parser.add_argument('--imgDim', type=int , help="Default image dimension.", default=96)
    parser.add_argument('--cuda', action='store_true')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    trainParser = subparsers.add_parser('train',help="Train a new classifier.")
    trainParser.add_argument('workDir',type=str,help="The input work directory containing 'reps.csv' and 'labels.csv'")

    inferParser = subparsers.add_parser('infer', help='Predict who an image contains from a trained classifier.')
    inferParser.add_argument('classifierModel',type=str,help='The Python pickle representing the classifier.')
    inferParser.add_argument('imgs', type=str, nargs='+',help="Input image.")
    inferParser.add_argument('--multi', help="Infer multiple faces in image",action="store_true")

    args = parser.parse_args()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,cuda=args.cuda)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        infer(args, args.multi)
