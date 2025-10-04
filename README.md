### Current plan: 
#### 1. use OpenCV to: 
####    preprocess images
####    apply color space conversion to grayscale
####    Apply gaussian blur to remove imperfections 
####    Apply thresholding to create a high contrast image where microplastics appear bright white and the background appears dark black 

#### 2. use Pytorch to:
####    teach the model what a microplastic is 

#### 3. use YOLO to: 
####    serve as a blueprint to train our Pytorch model as to what exactly a microplastic is 

### Why cant we just apply countours using OpenCV and call it a day? 
####    If two microplastics are close to each other or touching, countours would assume these two as one microplastic leading to inaccurate model 

### Links: 
#### Research: https://docs.google.com/document/d/1IidHKuvcOZBTAYJ0285bdjD_cWB6SS_9NLJEa-lFVb8/edit?usp=sharing
