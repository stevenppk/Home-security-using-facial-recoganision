
counter="person"
fswebcam -d /dev/video0 -r 960x720 --jpeg 85 -F 20 $counter.jpg

cd ~/Documents/openface-master

./classifier.py --dlibFacePredictor shape_predictor_68_face_landmarks.dat --networkModel nn4.small2.v1.t7 infer ./generated-embeddings/classifier.pkl person.jpg
