# Image Colorization
## Fall 2021
## COMS 4995 - Deep Learning for Computer Vision

### Team Members
1. Aishwarya Sarangu (als2389)
2. Vaibhav Goyal (vg2498)


#### 1. Presentation Video Link
Check [this link](https://youtu.be/JeA7tZ-yUAM)

#### 2. Project Structure

1. Final_project.py - Driver file to pre-process, Train, and Test the model
2. Preprocess.py - File to preprocess images
3. Train.py - File to Train and test the model

#### 3. Preprocessing
In order to only pre-process input images, we run:

```
python Final_project.py --Task "Preprocess" \
--train "<path to raw data folder>" --preprocessed_train \
"<location where you want to save preprocessed images>"
```

#### 4. Training
In order to Train the model, we run: (Note: images must be pre-processed before training):

```
python Final_project.py --Task "Train" \
--preprocessed_train "<location of pre-processed images>" \
--epochs "1" --loss_function "<can be mse or 2DCrossEntropy>" \
--batch_size "32" --model_path "<path to save model>"
```

#### 5. Testing
In order to just test model using images, run:

```
python Final_project.py --Task "Test" \
--model_path "<path to saved model>" \
--test_image_path "<path to test image>" \
--test_image_path_predicted "<path to save  final predicted image>"
```