%% Final assignment:Multiclassification

% we use the CIFAR-10 data set
% The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
% with 6000 imagesper class. There are 50000 training images and 10000 test 
% images.

%[ 0     1     2     3     4     5     6     7     8     9]
%[ airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck] 


%% Step 0 : Initialise constants and parameters
tic;

inputSize = 32*32*3 ; % size of input vector(CIFAR-10 images are 32x32 with RGB)
numClasses = 10; % num of classes(CIFAR-10 images fall into 10 classes)

lambda = 1e-4; % weight decay parameter

%% Step 1 : Load data

load('data_batch_1to4_double.mat');
%            data: [3072x40000 double]
%          labels: [40000x1 double]
%     batch_label: 'training batch 1 to 4'

% % visualize the data
% figure('name','Raw images');
% randsel = randi(size(data,2),200,1); % A random selection of samples for visualization
% display_network(data(:,randsel));

inputData = data;
labels(labels==0) = 10; % Remap 0 to 10

% Randomly initialise theta
theta = 0.005 * randn(numClasses * inputSize, 1);

pause;
%% Step 2 : Implement softmaxCost
%
%  Implement softmaxCost in softmaxCost.m. 

[cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, inputData, labels);


%% STEP 3: Gradient checking
%
%  As with any learning algorithm, you should always check that your
%  gradients are correct before learning the parameters.
% 

%% STEP 4: Learning parameters
%
%  Once you have verified that your gradients are correct, 
%  you can start training your softmax regression code using softmaxTrain
%  (which uses minFunc).

options.maxIter = 200;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            inputData, labels, options);
                          
% Although we only use 100 iterations here to train a classifier for the 
% CIFAR-10 data set, in practice, training for more iterations is usually
% beneficial.

%% STEP 5: Testing
%
%  You should now test your model against the test images.
%  To do this, you will first need to write softmaxPredict
%  (in softmaxPredict.m), which should return predictions
%  given a softmax model and the input data.

load('test_batch_doubles.mat');
%            data: [10000x3072 double]
%          labels: [10000x1 double]
%     batch_label: 'testing batch 1 of 1'

inputData = data;
labels(labels==0) = 10; % Remap 0 to 10

[pred] = softmaxPredict(softmaxModel, inputData);

acc = mean(labels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);


toc;
