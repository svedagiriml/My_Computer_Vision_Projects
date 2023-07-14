## Signature Fraud Detection Microservice
   The application we are building is **Signature Fraud Detection Microservice** . This service can be plugged in as a component of a larger **cheque processing application** in Banks. This service is not only useful in Banks but in any office where client's signature needs to be verified, for example in Government offices , and Courts.
### The Solution
![image](https://user-images.githubusercontent.com/85877266/191704010-93965294-50d8-4101-8b22-320312c33fad.png)
The solution consists of a User Interface which accepts the Customer ID, and signature to be verified (uploaded signature). These information are passed to a Flask application . The flask application searches for reference signature from database, and passes the uploaded signature and the reference signature as inputs to the Deep learning model. The deep learning model responds with the final prediction as to whether the signature uploaded is Genuine or Forged which is passed back to the User Interface.
   We have used the open **CEDAR signature dataset** to train a Deep learning model to detect signature fraud. The deep learning model has **Siamese Convolutional Neural Networks** whose output is concatenated and fed to a fully connected layer to make the final prediction as shown in the figure below.
### The Deep Learning Model
![deep learning](https://user-images.githubusercontent.com/85877266/191706026-995053d3-5487-4138-ac3d-c9c1e3631ad4.png)
The Deep Learning model consisted of two SIamese Convolutional Neural Networks. The reference signature, and uploaded signature were fed as input to the Siamese CNN network. The output features of the Siamese CNN were concatenated and fed as input to a fully connected layer followed by Softmax output layer.
### The CEDAR Dataset
CEDAR Signature dataset is a database of off-line signatures for signature verification. Each of 55 individuals contributed 24 signatures thereby creating 1,320 genuine signatures. Some were asked to forge three other writers’ signatures, eight times per subject, thus creating 1,320 forgeries. Each signature was scanned at 300 dpi gray-scale and binarized using a gray-scale histogram. Salt pepper noise removal and slant normalization were two steps involved in image preprocessing. The database has 24 genuines and 24 forgeries available for each writer.

### Training the Deep Learning Model
During training the Deep Learning Model, Binary Cross Entropy was used as Loss function, and RMSProp optimizer was used. The validation loss was decreasing after every epoch of training, however after 8th epoch the validation loss increased, hence stopped training the model further and used 8th epoch model as final model.
The Dataset was split into Training set, Validation set, and Test set in the ratio 42:7:6. While training the Deep Learning Model, Binary Cross Entropy was used as Loss function, and RMSProp optimizer was used. The validation loss was decreasing after every epoch of training, however after 8th epoch the validation loss increased, hence stopped training the model further and used 8th epoch model as final model. The **accuracy of the final model was 80.1%**

#### Challenges in Training the Deep Learning Model
Nan values occured for loss from 1st epoch during training…..reduced learning rate to 1e-04…it worked….but Nan values for loss started to occur from 5th epoch during training…..first checked if input contains Nan….which was not the case….next….changed contrastive loss function implementation….this worked….but still the model validation accuracy was very poor….did data pre-processing to invert the images and then train…..both training accuracy and validation accuracy where both still poor …changed the model by removing contrastive loss, and instead using a small cnn to train the output features of the Siamese cnn….used binary cross entropy as loss function….this increased accuracy a lot…hence went for this implementation….
### Scope for improvement
There is scope for improvement in increasing the accuracy of the model further. We can try Transfer learning models like VGG-19 etc for training . However this was not tried due to time constraints. 
### Applications of the Microservice
 This microservice can be plugged in as a component of a larger **cheque processing application** in Banks as shown inthe figure below.
![cheque processing](https://user-images.githubusercontent.com/85877266/191708852-5c50bde0-fa1c-4415-80aa-2fdc2d067e6a.png)
We can use image processing techniques to extract the signature, amount, date, and customer ID from the cheque leaf after scanning. These extracted features can be processed seperately and verified. The signature verification can be done using the model developed here, and based on the outcome either the cheque could be approved or sent for further review.

