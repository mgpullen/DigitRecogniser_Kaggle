# DigitRecogniser_Kaggle
My Convolutional Neural Network submission for the Digit Recogniser Kaggle competition.

Remember to download the files from the Kaggle website and put them in the '/data/' sub-directory before running the code.
https://www.kaggle.com/c/digit-recognizer

This code should give you ~99.4% on the test data, which at the time of posting has me at position 97 (top 8%).

Note that I have recently become aware that people have been using the rest of the MNIST data to train their models (thus achieving 100%). My view is that this is cheating so I have ONLY used data available from the Kaggle website. Presumably my ranking would increase if these submissions were excluded.

In this version I have effectively doubled the data size by adding noise to the training data. The improvements were only marginal so feel free to remove the relevant lines of code to make the training faster.

The training time for me was about 2.5 hours on the following machine:
MacBook Pro (13-inch, Mid 2012)
Processor 2.5 GHz Intel Core i5
Memory 4 GB 1600 MHz DDR3
OS X 10.10.5

Many improvements could be made such as image augmentation (translations and rotations) in a package such as EBImage, however, my system is not particularly well suited to training such large data sets. Therefore I will have to make do with the current result.

Mick
2016/12/05
