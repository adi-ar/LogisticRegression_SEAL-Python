## Logistic Regression with SEAL-Python

This is a fork of the SEAL-Python repository, python binding of MS SEAL library.

Logistic Regression is implemented using the CKKS Encryption scheme implemented in SEAL.

Credit Card Fraud Detection data is used for building the logistic regression model. Please download the data from  and place in working directory.
Output logs and graphs are generated in out/

The model is build with 4 features. (adding or removing features to encrypted model will require significant change in code)

The Encrypted model can be trained by running Enc_train_and_eval.
Plaintext_train_and_eval is the plaintext version of the encrypred model training algorithm.
