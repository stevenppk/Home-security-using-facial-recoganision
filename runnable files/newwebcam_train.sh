echo "Enter the name of the person"
read name
cd ~/Documents/openface-master/training-images
mkdir $name
cd $name
counter=1
while [ $counter -le 15 ]
	do
	fswebcam -d /dev/video1 -r 640x480 --jpeg 85 -F 20 $counter.jpg
	((counter++))
	done






