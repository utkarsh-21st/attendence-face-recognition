# attendence-face-recognition
This is my attempt to build an attendence system based on face-recognition.

To build a Face-Recognition system, we, ofcourse, are going to employ techiniques based on Convolution Neural Network.
Now one way is to first gather some images of all the identities and train a classification-algorithm over those images and there it is, a Face-Recognition algorithm, ready to recognize whether a new image is among those identities or is some stranger. This algorithm, if trained well, can definitely provide a robust performance, however, this method is impractical due to: It must be retrained whenever a person leaves or joins the institution, The data has to have atleast 5-10 images of each identity

The other approach is whats called "one-shot-learning", which is what we are going to do here. Unlike the previous approach, this algorithm once trained is all there is, no need to retrain in case of nay change in identities. Moreover, it only needs one or a few images of each identity. Given an image, it calculates a low dimensional feature-vector for that image called embeddings. And the model is designed such that the Euclidean distance between two embeddings quantifies how similar two images are (whether they belong to the same identities).

A convinient interface is also provided to manage the system (some snapshots):





