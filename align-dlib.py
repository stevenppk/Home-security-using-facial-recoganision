#!/usr/bin/env python3


import argparse
import cv2
import numpy as np
import os
import random
import shutil

import openface
import openface.helper
from openface.data import iterImgs

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def write(vals, fName):
    if os.path.isfile(fName):
        print("{} exists. Backing up.".format(fName))
        os.rename(fName, "{}.bak".format(fName))
    with open(fName, 'w') as f:
        for p in vals:
            f.write(",".join(str(x) for x in p))
            f.write("\n")

def alignMain(args):
    openface.helper.mkdirP(args.outputDir)

    imgs = list(iterImgs(args.inputDir))

    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)

    landmarkMap = {
        'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE
	}
    if args.landmarks not in landmarkMap:
        raise Exception("Landmarks unrecognized: {}".format(args.landmarks))

    landmarkIndices = landmarkMap[args.landmarks]

    align = openface.AlignDlib(args.dlibFacePredictor)

    for imgObject in imgs:
        print("=== {} ===".format(imgObject.path))
        outDir = os.path.join(args.outputDir, imgObject.cls)
        openface.helper.mkdirP(outDir)
        outputPrefix = os.path.join(outDir, imgObject.name)
        imgName = outputPrefix + ".png"

        if os.path.isfile(imgName):
            if args.verbose:
                print("  + Already found, skipping.")
        else:
            rgb = imgObject.getRGB()
            if rgb is None:
                if args.verbose:
                    print("  + Unable to load.")
                outRgb = None
            else:
                outRgb = align.align(args.size, rgb,
                                     landmarkIndices=landmarkIndices)
                if outRgb is None and args.verbose:
                    print("  + Unable to align.")

           
            if outRgb is not None:
                if args.verbose:
                    print("  + Writing aligned file to disk.")
                outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(imgName, outBgr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('inputDir', type=str, help="Input image directory.")
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.")

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    alignmentParser = subparsers.add_parser('align', help='Align a directory of images.')
    alignmentParser.add_argument('landmarks', type=str, help='The landmarks to align to.',default = "outerEyesAndNose")
    alignmentParser.add_argument('outputDir', type=str, help="Output directory of aligned images.")
    alignmentParser.add_argument('--size', type=int, help="Default image size.",default=96)
    alignmentParser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    
    alignMain(args)
