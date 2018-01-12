function [] = main()
    
    % Accuracy on the entire database
    correct = faceRecognitionTesting();
    disp(['Face Recognition Accuracy: ', num2str(correct),'/400'])
    
    % Demo showing 3 example matches
    %faceRecognitionShow()
    
    % Accuracy with rotated test images
    %correct = faceRecognitionRotationTesting(5);
    %disp(['Face Recognition Accuracy: ', num2str(correct),'/400'])
    
    % Demo showing 3 example rotated matches
    %faceRecognitionRotationShow(180)
    
end 

function correct = faceRecognitionTesting()
% FACERECOGNITIONTESTING - test the accuracy on 400 test images and return
% the number of correct matches

    testIndex = randi([1 10],1,40); % Get random indexes for 1 test image of each person
    
    correct = 0;
    % Train and test on 40 images 10 times -> 400 tests
    for i = 1:10
       testIndex = mod(testIndex, 10)+1; % Shift the indexes of test imgs
       [trainingSet, testingSet] = training(testIndex); % Train on new training/testing split
       
       % Test on 40 testing imgs
       for j = 1:40
           [val, index] = faceRecognition(testingSet(j,:), j, trainingSet);
           correct = correct + val;
       end
    end
end

function [] = faceRecognitionShow()
% FACERECOGNITIONSHOW - demo display matching images results

    testIndex = [4,9,7,2,6,5,8,10,1,3,8,2,8,10,2,4,7,5,5,5,9,10,5,4,7,1,9,10,10,5,6,10,3,7,8,8,5,10,9,10];
    correct = 0;
    [trainingSet, testingSet] = training(testIndex); % Train on training/testing split
    
    
    % Test and display 3 testing imgs
    [val, index] = faceRecognition(testingSet(1,:), 1, trainingSet);
    test1 = double(imread(sprintf('ATT_Face_Database\\s%d\\%d.pgm', 1, 4)))/255;
    test1 = padarray(test1, [8 18], 'replicate', 'both'); % Pad the image to 128x128
    person = ceil(index/9);
    person_img = mod(index+person,10);
    match1 = double(imread(sprintf('ATT_Face_Database\\s%d\\%d.pgm', person, person_img)))/255;
    match1 = padarray(match1, [8 18], 'replicate', 'both'); % Pad the image to 128x128
    figure(); imshowpair(test1, match1, 'montage'); title('Test - Left | Match - Right')
    
    [val, index] = faceRecognition(testingSet(4,:), 4, trainingSet);
    test4 = double(imread(sprintf('ATT_Face_Database\\s%d\\%d.pgm', 4, 2)))/255;
    test4 = padarray(test4, [8 18], 'replicate', 'both'); % Pad the image to 128x128
    person = ceil(index/9);
    person_img = mod(index+person,10);
    match4 = double(imread(sprintf('ATT_Face_Database\\s%d\\%d.pgm', person, person_img)))/255;
    match4 = padarray(match4, [8 18], 'replicate', 'both'); % Pad the image to 128x128
    figure(); imshowpair(test4, match4, 'montage'); title('Test - Left | Match - Right')
    
    [val, index] = faceRecognition(testingSet(9,:), 9, trainingSet);
    test9 = double(imread(sprintf('ATT_Face_Database\\s%d\\%d.pgm', 9, 1)))/255;
    test9 = padarray(test9, [8 18], 'replicate', 'both'); % Pad the image to 128x128
    person = ceil(index/9);
    person_img = mod(index+person,10);
    match9 = double(imread(sprintf('ATT_Face_Database\\s%d\\%d.pgm', person, person_img)))/255;
    match9 = padarray(match9, [8 18], 'replicate', 'both'); % Pad the image to 128x128
    figure(); imshowpair(test9, match9, 'montage'); title('Test - Left | Match - Right')
 
end


function correct = faceRecognitionRotationTesting(angle)
% FACERECOGNITIONROTATIONTESTING - test the accuracy on 400 test images
% that have been rotated counterclockwise by angle and return the number of
% correct matches
    % angle - counterclockwise angle the test images are rotated

    testIndex = randi([1 10],1,40); % Get random indexes for 1 test image of each person
    
    correct = 0;
    % Train and test on 40 images 10 times -> 400 tests
    for i = 1:10
       testIndex = mod(testIndex, 10)+1; % Shift the indexes of test imgs
       [trainingSet, testingSet] = trainingRotation(testIndex, angle); % Train on new training/testing split
       
       % Test on 40 testing imgs
       for j = 1:40
           [val, index] = faceRecognition(testingSet(j,:), j, trainingSet);
           correct = correct + val;
       end
    end
end

function [] = faceRecognitionRotationShow(angle)
% FACERECOGNITIONSHOW - demo display matching rotated images results

    testIndex = [4,9,7,2,6,5,8,10,1,3,8,2,8,10,2,4,7,5,5,5,9,10,5,4,7,1,9,10,10,5,6,10,3,7,8,8,5,10,9,10];
    correct = 0;
    [trainingSet, testingSet] = trainingRotation(testIndex, angle); % Train on training/testing split
    
    
    % Test and display 3 testing imgs
    [val, index] = faceRecognition(testingSet(1,:), 1, trainingSet);
    test1 = double(imread(sprintf('ATT_Face_Database\\s%d\\%d.pgm', 1, 4)))/255;
    test1 = padarray(test1, [8 18], 'replicate', 'both'); % Pad the image to 128x128
    test1 = imrotate(test1,angle);  
    c = floor(size(test1,1)/2); % Center point 
    test1 = imcrop(test1, [(c-64) (c-64) 128 128]); % Crop the center 128x128 section 
    test1 = imresize(test1, [128 128]); % Fix rounding
    person = ceil(index/9);
    person_img = mod(index+person,10);
    match1 = double(imread(sprintf('ATT_Face_Database\\s%d\\%d.pgm', person, person_img)))/255;
    match1 = padarray(match1, [8 18], 'replicate', 'both'); % Pad the image to 128x128
    figure(); imshowpair(test1, match1, 'montage'); title('Test - Left | Match - Right')
    
    [val, index] = faceRecognition(testingSet(4,:), 4, trainingSet);
    test4 = double(imread(sprintf('ATT_Face_Database\\s%d\\%d.pgm', 4, 2)))/255;
    test4 = padarray(test4, [8 18], 'replicate', 'both'); % Pad the image to 128x128
    test4 = imrotate(test4,angle);  
    c = floor(size(test4,1)/2); % Center point 
    test4 = imcrop(test4, [(c-64) (c-64) 128 128]); % Crop the center 128x128 section 
    test4 = imresize(test4, [128 128]); % Fix rounding
    person = ceil(index/9);
    person_img = mod(index+person,10);
    match4 = double(imread(sprintf('ATT_Face_Database\\s%d\\%d.pgm', person, person_img)))/255;
    match4 = padarray(match4, [8 18], 'replicate', 'both'); % Pad the image to 128x128
    figure(); imshowpair(test4, match4, 'montage'); title('Test - Left | Match - Right')
    
    [val, index] = faceRecognition(testingSet(9,:), 9, trainingSet);
    test9 = double(imread(sprintf('ATT_Face_Database\\s%d\\%d.pgm', 9, 1)))/255;
    test9 = padarray(test9, [8 18], 'replicate', 'both'); % Pad the image to 128x128
    test9 = imrotate(test9,angle);  
    c = floor(size(test9,1)/2); % Center point 
    test9 = imcrop(test9, [(c-64) (c-64) 128 128]); % Crop the center 128x128 section 
    test9 = imresize(test9, [128 128]); % Fix rounding
    person = ceil(index/9);
    person_img = mod(index+person,10);
    match9 = double(imread(sprintf('ATT_Face_Database\\s%d\\%d.pgm', person, person_img)))/255;
    match9 = padarray(match9, [8 18], 'replicate', 'both'); % Pad the image to 128x128
    figure(); imshowpair(test9, match9, 'montage'); title('Test - Left | Match - Right')
 
end


function [trainingSet, testingSet] = training(testIndex)
% TRAINING - computes the variance frequency feature vectors for the test
% and training imgs
    % testIndex - 1x40 array of 1:10 indexes denoting the test image index
    % for all 40 people
    
    allImgs = zeros(360, 128, 128);
    i=0;
    
    % Get imgs and convert to fft 
    for d = 1:40
        for f = 1:10
            if(testIndex(d) == f)
                continue
            end
            i = i+1;
            img = double(imread(sprintf('ATT_Face_Database\\s%d\\%d.pgm', d, f)))/255; % Get the image 
            img = padarray(img, [8 18], 'replicate', 'both'); % Pad the image to 128x128
            fftImg = real(fftshift(fft2(img))); % Get the fourier real component
            %fftImg = abs(fftshift(fft2(img))); % Get the fourier magnitude
            allImgs(i, :, :) = fftImg; % Add the img to the database matrix 
        end
    end
    
    % Calculate variance
    variances = zeros(128,128);
    
    for r = 1:128
       for c = 1:128
           variances(r,c) = var(allImgs(:,r,c)); % Get variance from the (r,c) cell in every img's fft
       end
    end
    
    [sortVarVals, sortVarIndexes] = sort(variances(:), 'descend'); % Sort variances
    maxVarIndexes = sortVarIndexes(1:30); % Get indexes of 30 largest variances
    [rVarIndexes, cVarIndexes] = ind2sub(size(variances), maxVarIndexes); % Convert indexes to (r,c)
    
    % Create feature vectors for the training
    trainingSet = zeros(360, 30);
    for i=1:360
       for f = 1:30
          trainingSet(i,f) = allImgs(i, rVarIndexes(f), cVarIndexes(f)); %Get the 30 freqs from the fft and add to vector
       end
    end
    
    % Create feature vectors for the training
    testingSet = zeros(40,30);
    for d=1:40
        img = double(imread(sprintf('ATT_Face_Database\\s%d\\%d.pgm', d, testIndex(d))))/255; % Get the image 
        img = padarray(img, [8 18], 'replicate', 'both'); % Pad the image to 128x128
        fftImg = real(fftshift(fft2(img))); % Get the fourier real component
        %fftImg = abs(fftshift(fft2(img))); % Get the fourier magnitude
        for f = 1:30
          testingSet(d,f) = fftImg(rVarIndexes(f), cVarIndexes(f)); %Get the 30 freqs from the fft and add to vector
       end
    end
    
end

function [trainingSet, testingSet] = trainingRotation(testIndex, angle)
% TRAININGROTATION - computes the variance frequency feature vectors for
% the training and rotated by angle test images
    % testIndex - 1x40 array of 1:10 indexes denoting the test image index
    % for all 40 people
    % angle - counterclockwise angle the test images are rotated 
    
    allImgs = zeros(360, 128, 128);
    i=0;
    
    % Get imgs and convert to fft 
    for d = 1:40
        for f = 1:10
            if(testIndex(d) == f)
                continue
            end
            i = i+1;
            img = double(imread(sprintf('ATT_Face_Database\\s%d\\%d.pgm', d, f)))/255; % Get the image 
            img = padarray(img, [8 18], 'replicate', 'both'); % Pad the image to 128x128
            fftImg = real(fftshift(fft2(img))); % Get the fourier magnitude
            allImgs(i, :, :) = fftImg; % Add the img to the database matrix 
        end
    end
    
    % Calculate variance
    variances = zeros(128,128);
    
    for r = 1:128
       for c = 1:128
           variances(r,c) = var(allImgs(:,r,c)); % Get variance from the (r,c) cell in every img's fft
       end
    end
    
    [sortVarVals, sortVarIndexes] = sort(variances(:), 'descend'); % Sort variances
    maxVarIndexes = sortVarIndexes(1:30); % Get indexes of 30 largest variances
    [rVarIndexes, cVarIndexes] = ind2sub(size(variances), maxVarIndexes); % Convert indexes to (r,c)
    
    % Create feature vectors for the training
    trainingSet = zeros(360, 30);
    for i=1:360
       for f = 1:30
          trainingSet(i,f) = allImgs(i, rVarIndexes(f), cVarIndexes(f)); %Get the 30 freqs from the fft and add to vector
       end
    end
    
    % Create feature vectors for the training
    testingSet = zeros(40,30);
    for d=1:40
        img = double(imread(sprintf('ATT_Face_Database\\s%d\\%d.pgm', d, testIndex(d))))/255; % Get the image 
        img = padarray(img, [8 18], 'replicate', 'both'); % Pad the image to 128x128
        % Rotate by angle
        img = imrotate(img,angle);  
        c = floor(size(img,1)/2); % Center point 
        img = imcrop(img, [(c-64) (c-64) 128 128]); % Crop the center 128x128 section 
        img = imresize(img, [128 128]); % Fix rounding
        
        fftImg = real(fftshift(fft2(img))); % Get the fourier magnitude
        for f = 1:30
          testingSet(d,f) = fftImg(rVarIndexes(f), cVarIndexes(f)); %Get the 30 freqs from the fft and add to vector
       end
    end
end


function [found, minIndex] = faceRecognition(img, correct_face, trainingSet)
% FACERECOGNITION - find the best matching face from the training for the
% test image and return 1 if the prediciton was correct or 0 if not.
    % IMG - 1x30 feature vector of the test image
    % correct_face - label of the test image's correct output
    % trainingSet - 360x30 matrix of the training image's feature vectors
    
    distance = zeros(1,360);
    
    % Get the euclidean distance between the test img and all training imgs
    for i = 1:360
       distance(1,i) = sqrt(sum((trainingSet(i,:) - img) .^ 2)); 
    end
    
    % Find the closest match from the training imgs
    [minDist, minIndex] = min(distance(:)); 
    face = ceil(minIndex/9); 
    
    if(face == correct_face)
       found = 1; 
    else
       found = 0;
    end

end
