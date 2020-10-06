# attendence-face-recognition-one-shot-learning
This is my attempt to build an attendence system based on face-recognition.

To build a Face-Recognition system, we, ofcourse, are going to employ techiniques based on Convolution Neural Network.

One way is to first gather some images of all the identities and train a classification-algorithm over those images and there it is, a Face-Recognition algorithm, ready to recognize whether a new image is among those identities or is some stranger. This algorithm, if trained well, can definitely provide a robust performance, however, this method is impractical due to: It must be retrained whenever a person leaves or joins the institution, The data has to have atleast 5-10 images of each identity

The other approach is whats called "one-shot-learning", which is what has been used here. Unlike the previous approach, this algorithm once trained is all there is, no need to retrain in case of nay change in identities. Moreover, it only needs one or a few images of each identity. Given an image, it calculates, using siamese network, a low dimensional feature-vector for that image called embedding. And the model is designed such that the Euclidean distance between two embeddings quantifies how similar two images are (whether they belong to the same identities).

A convinient interface is also provided to manage the system (some snapshots):
![interface](https://github.com/utkarsh-21st/attendence-face-recognition/blob/master/sample%20images/sample1.png "interface")![interace](https://github.com/utkarsh-21st/attendence-face-recognition/blob/master/sample%20images/sample3.png "interace")![interface](https://github.com/utkarsh-21st/attendence-face-recognition/blob/master/sample%20images/sample4.png "interface")


How is attendence taken?
- A database of all identities can be created by using the interface.
- A face-detection algorithm attempts to find a face in an image
- Detected-face is then carried out to the face-recognition model which calculates an embedding for that face.
- Eucledian distance is calculated between this embedding against all stored embeddings in database.
- The minmum of all distances is chosen if it is also less than a certain threshold.

**Face Detection** algorithm used here is MTCNN. 
**Face Recognition** algorithm used here is a  pre-trained model, keras version of [OpenFace](https://github.com/cmusatyalab/openface "OpenFace").

### References
- [MTCNN](https://pypi.org/project/mtcnn/ "MTCNN")
- Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf "FaceNet: A Unified Embedding for Face Recognition and Clustering")
- Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf "DeepFace: Closing the gap to human-level performance in face verification")
- The pretrained model used is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
