
rm -r ~/Documents/openface-master/aligned-images
mkdir ~/Documents/openface-master/aligned-images

cd ~/Documents/openface-master
./align-dlib.py ./training-images/ --dlibFacePredictor shape_predictor_68_face_landmarks.dat align outerEyesAndNose ./aligned-images/ --size 96
./batch-represent/main.lua -outDir ./generated-embeddings/ -data ./aligned-images/  -model nn4.small2.v1.t7
./classifier.py --dlibFacePredictor shape_predictor_68_face_landmarks.dat --networkModel nn4.small2.v1.t7 train  ./generated-embeddings/
