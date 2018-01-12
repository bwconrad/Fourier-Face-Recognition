# Fourier Face Recognition
Face recognition implemented in MatLab using Fourier space algorithms as described in "Face Recognition in Fourier Space" by Hagen Spies and Ian Ricketts. 
</br>
</br>
Original Paper: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.1339&rank=1&q=Face%20Recognition%20in%20Fourier%20Space&osm=&ossid=
</br>
</br>


## How to Run
Uncomment the desired functions in main() to run the program.

1. Run on the entire database and calculate accuracy:
```
correct = faceRecognitionTesting();
disp(['Face Recognition Accuracy: ', num2str(correct),'/400'])
```

2. Show example matches:
```
faceRecognitionShow()
```

3. Run on the entire database with rotated training images:
```
correct = faceRecognitionRotationTesting(angle);
disp(['Face Recognition Accuracy: ', num2str(correct),'/400'])
```
Change the argument in faceRecognitionRotationTesting(angle) to modify the angle the images are rotated. 

4. Show example matches with rotated images:
```
faceRecognitionRotationShow(angle)
```

## Results
Testing was done on the AT&T face database and achieved an accuracy of 97.75% (391/400).

