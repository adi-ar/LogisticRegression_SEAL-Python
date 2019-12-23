#!/usr/bin/env python
# coding: utf-8

# ## Logistic Regression in Ciphertext : train and evaluation
# 
# The file Enc_train_and_eval contains the code for running Logistic Regression using NAG for Ciphertexts, and evaluation of the model.
# Since this section of the code is run on Openstack VM, the output can not be shared in an interactive notebook format.
# However, the results are shared as images below the code.

# ##NOTE: 
# During our testing, creating keys with poly_modulus_degree=32768 on a 8 gb ram fails and freezes the system. Please change poly_modulus_degree argument to 16384 if key kreation fails

# In[1]:


import math
from seal import *
from seal_helper import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime


### Setup encryption scheme

parms = EncryptionParameters(scheme_type.CKKS)



# poly_modulus_degree = 16384
# parms.set_poly_modulus_degree(poly_modulus_degree)
# parms.set_coeff_modulus(CoeffModulus.Create(
#     poly_modulus_degree, [40,40,40,40,40,40,40,40,40,40]))

#poly_modulus_degree = 16384
#parms.set_poly_modulus_degree(poly_modulus_degree)
#parms.set_coeff_modulus(CoeffModulus.Create(
#    poly_modulus_degree, [30,30,30,30,30,30,30,30,30,30,30,30,30,30]))

poly_modulus_degree = 32768
parms.set_poly_modulus_degree(poly_modulus_degree)
parms.set_coeff_modulus(CoeffModulus.Create(
    poly_modulus_degree, [30,30,30,30,30,30,30,30,30,30,30,30,30,30,
                         30,30,30,30,30,30,30,30,30,30,30,30,30,30]))

#poly_modulus_degree = 32768
#parms.set_poly_modulus_degree(poly_modulus_degree)
#parms.set_coeff_modulus(CoeffModulus.Create(
#    poly_modulus_degree, [60,40,40,40,40,40,40,40,40,40,
#                         40,40,40,40,40,40,40,40,40,60]))

scale = pow(2.0, 30)
context = SEALContext.Create(parms)
print_parameters(context)

keygen = KeyGenerator(context)
public_key = keygen.public_key()
secret_key = keygen.secret_key()
relin_keys = keygen.relin_keys()
gal_keys = keygen.galois_keys()

encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)
decryptor = Decryptor(context, secret_key)

encoder = CKKSEncoder(context)
slot_count = encoder.slot_count()
print("Number of slots: " + str(slot_count))


# In[2]:


import sys
print(sys.getsizeof(public_key),
sys.getsizeof(secret_key),
sys.getsizeof(relin_keys),
sys.getsizeof(gal_keys))


# In[3]:


## Read data
inputData = pd.read_csv("./creditcard.csv")
data = inputData.iloc[:,[4,10,14,16]]
y = inputData.iloc[:,30] 

data = (data - data.mean())/data.std()
data.iloc[data>2.5] = 2.5

np.random.seed(2)
###################### train-test split #################
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=2)


nm = NearMiss()
x_res, y_res = nm.fit_resample(X_train, y_train.ravel())

############################################################################

x_res = pd.DataFrame(x_res)
y_res = pd.Series(y_res)


# In[4]:


###### all computation for client server / data owner IN THIS BLOCK ONLY

### read data and encrypt column wise


x1 = DoubleVector()
x2 = DoubleVector()
x3 = DoubleVector()
x4 = DoubleVector()
y = DoubleVector()

for i in range(len(x_res)):
    x1.push_back(x_res.iloc[i,0])
    x2.push_back(x_res.iloc[i,1])
    x3.push_back(x_res.iloc[i,2])
    x4.push_back(x_res.iloc[i,3])
    y.push_back(y_res.iloc[i])

x1_plain = Plaintext()
x2_plain = Plaintext()
x3_plain = Plaintext()
x4_plain = Plaintext()
y_plain = Plaintext()

encoder.encode(x1, scale, x1_plain)
encoder.encode(x2, scale, x2_plain)
encoder.encode(x3, scale, x3_plain)
encoder.encode(x4, scale, x4_plain)
encoder.encode(y, scale, y_plain)

x1_enc = Ciphertext()
x2_enc = Ciphertext()
x3_enc = Ciphertext()
x4_enc = Ciphertext()
y_enc = Ciphertext()

encryptor.encrypt(x1_plain, x1_enc)
encryptor.encrypt(x2_plain, x2_enc)
encryptor.encrypt(x3_plain, x3_enc)
encryptor.encrypt(x4_plain, x4_enc)
encryptor.encrypt(y_plain, y_enc)



### Calculate gradient Multiplication Factor m = (alpha/n) * X and encrypt

## m1:m4 can be in same dataframe and subsetted in push_front

alpha = 0.0001

m1 = data.iloc[:,0] *(alpha/(len(x_res)))
m2 = data.iloc[:,1] *(alpha/(len(x_res)))
m3 = data.iloc[:,2] *(alpha/(len(x_res)))
m4 = data.iloc[:,3] *(alpha/(len(x_res)))
m0 = pd.Series(np.ones([len(x_res)])) * (alpha/(len(x_res)))

m0_vec = DoubleVector()
m1_vec = DoubleVector()
m2_vec = DoubleVector()
m3_vec = DoubleVector()
m4_vec = DoubleVector()

for i in range(len(x_res)):
    m0_vec.push_back(m0.iloc[i])
    m1_vec.push_back(m1.iloc[i])
    m2_vec.push_back(m2.iloc[i])
    m3_vec.push_back(m3.iloc[i])
    m4_vec.push_back(m4.iloc[i])


m1_plain = Plaintext()
m2_plain = Plaintext()
m3_plain = Plaintext()
m4_plain = Plaintext()
m0_plain = Plaintext()

m0_enc = Ciphertext()
m1_enc = Ciphertext()
m2_enc = Ciphertext()
m3_enc = Ciphertext()
m4_enc = Ciphertext()

encoder.encode(m0_vec, scale, m0_plain)
encoder.encode(m1_vec, scale, m1_plain)
encoder.encode(m2_vec, scale, m2_plain)
encoder.encode(m3_vec, scale, m3_plain)
encoder.encode(m4_vec, scale, m4_plain)

encryptor.encrypt(m0_plain, m0_enc)
encryptor.encrypt(m1_plain, m1_enc)
encryptor.encrypt(m2_plain, m2_enc)
encryptor.encrypt(m3_plain, m3_enc)
encryptor.encrypt(m4_plain, m4_enc)


################### weights precomputation #####################

x_res = pd.DataFrame(x_res)
avOfMeans = (x_res.loc[y_res==0,:].mean() + x_res.loc[y_res==1,:].mean() )/2

weights_precomp = [-1*np.log(0.66)/x for x in avOfMeans]

bias = [0.6]

beta = pd.Series(bias + weights_precomp)
beta=pd.to_numeric(beta)

# encrypt weights to separate ciphertexts

beta0_plain = Plaintext()
beta1_plain = Plaintext()
beta2_plain = Plaintext()
beta3_plain = Plaintext()
beta4_plain = Plaintext()

beta0_enc = Ciphertext()
beta1_enc = Ciphertext()
beta2_enc = Ciphertext()
beta3_enc = Ciphertext()
beta4_enc = Ciphertext()

encoder.encode(beta.iloc[0], scale, beta0_plain)
encoder.encode(beta.iloc[1], scale, beta1_plain)
encoder.encode(beta.iloc[2], scale, beta2_plain)
encoder.encode(beta.iloc[3], scale, beta3_plain)
encoder.encode(beta.iloc[4], scale, beta4_plain)

encryptor.encrypt(beta0_plain, beta0_enc)
encryptor.encrypt(beta1_plain, beta1_enc)
encryptor.encrypt(beta2_plain, beta2_enc)
encryptor.encrypt(beta3_plain, beta3_enc)
encryptor.encrypt(beta4_plain, beta4_enc)


# In[5]:


### algo to return sum of elements of ciphertext, in all elements of resulting ciphertext

def sumVector(cipher, size):
    
    
    ### allSum algorithm as described by Kim et al, discussed in Implementation section of the report.
    
    for i in range(0, int(math.log(size,2))):
        x1_temp = Ciphertext()
        evaluator.rotate_vector(cipher, int(math.pow(2,i)), gal_keys, x1_temp)
        evaluator.add_inplace(cipher, x1_temp)
    
    return cipher


# In[6]:


### get linear response
## Depth = 1

def getLinResponse_enc(beta0_enc, beta1_enc, beta2_enc, beta3_enc, beta4_enc, x1_enc, x2_enc, x3_enc, x4_enc):
### get inner product b0 + b1x1 + ....
## input encrypted weight vectors and features, returns encrypted linear response

    x1b1 = Ciphertext()
    x2b2 = Ciphertext()
    x3b3 = Ciphertext()
    x4b4 = Ciphertext()
    
    if(context.get_context_data(x1_enc.parms_id()).chain_index() != context.get_context_data(beta1_enc.parms_id()).chain_index()):
        evaluator.mod_switch_to_inplace(x1_enc, beta1_enc.parms_id())
        evaluator.mod_switch_to_inplace(x2_enc, beta1_enc.parms_id())
        evaluator.mod_switch_to_inplace(x3_enc, beta1_enc.parms_id())
        evaluator.mod_switch_to_inplace(x4_enc, beta1_enc.parms_id())
        
        x1_enc.set_scale(beta1_enc.scale())
        x2_enc.set_scale(beta1_enc.scale())
        x3_enc.set_scale(beta1_enc.scale())
        x4_enc.set_scale(beta1_enc.scale())

    evaluator.multiply(x1_enc, beta1_enc, x1b1)
    evaluator.multiply(x2_enc, beta2_enc, x2b2)
    evaluator.multiply(x3_enc, beta3_enc, x3b3)
    evaluator.multiply(x4_enc, beta4_enc, x4b4)

    # relinearize and rescale
    evaluator.relinearize_inplace(x1b1, relin_keys)
    evaluator.relinearize_inplace(x2b2, relin_keys)
    evaluator.relinearize_inplace(x3b3, relin_keys)
    evaluator.relinearize_inplace(x4b4, relin_keys)
    evaluator.rescale_to_next_inplace(x1b1)
    evaluator.rescale_to_next_inplace(x2b2)
    evaluator.rescale_to_next_inplace(x3b3)
    evaluator.rescale_to_next_inplace(x4b4)

    evaluator.add_inplace(x3b3, x4b4)
    evaluator.add_inplace(x2b2, x3b3)
    evaluator.add_inplace(x1b1, x2b2)



    # switch down beta0 and add
    evaluator.mod_switch_to_inplace(beta0_enc, x1b1.parms_id())
    #beta0_enc.set_scale(x1b1.scale())
    x1b1.set_scale(beta0_enc.scale())

    lr_enc = Ciphertext()
    evaluator.add(x1b1, beta0_enc, lr_enc)
    
    return lr_enc


# In[7]:


######## get LSA_Sigmoid approximation of lr_enc

## sigmoid = 0.5 + ((1.20096/8) * lin_response) + ((-0.81562/pow(8,3)) * lin_response**3) 

# (x * x) * (-0.00159 * x) = cube term - depth 3
# x * 0.15012 = x term - depth 2
# 0.5 - depth 1
## Total depth = 3

def getLsaSigmoid(lr_enc):
    sigApprox = Ciphertext()
    lr_sq = Ciphertext()
    lr_cube = Ciphertext()
    lr_term = Ciphertext()
    
    # lr_sq = lr * lr
    evaluator.square(lr_enc, lr_sq) 
    evaluator.relinearize_inplace(lr_sq, relin_keys)
    evaluator.rescale_to_next_inplace(lr_sq)
    
    ## temp term lr * (-0.00159)
    temp1 = Plaintext()
    temp1_enc = Ciphertext()
    
    encoder.encode(-0.00159, scale, temp1)
    encryptor.encrypt(temp1, temp1_enc)
    
    # shift temp to mod(lr)
    evaluator.mod_switch_to_inplace(temp1_enc, lr_enc.parms_id())
    #temp1_enc.set_scale(lr_enc.scale())
    lr_enc.set_scale(temp1_enc.scale())
    
    # temp * lr_enc
    evaluator.multiply_inplace(temp1_enc, lr_enc)
    evaluator.relinearize_inplace(temp1_enc, relin_keys)
    evaluator.rescale_to_next_inplace(temp1_enc)
    
    # temp1 * lr_sq = lr_cube
    # scale and mod of both should be same here
    evaluator.multiply(lr_sq, temp1_enc, lr_cube)
    evaluator.relinearize_inplace(lr_cube, relin_keys)
    evaluator.rescale_to_next_inplace(lr_cube)
    
    
    ## lr * 0.15012
    temp2 = Plaintext()
    temp2_enc = Ciphertext()
    
    encoder.encode(0.15012, scale, temp2)
    encryptor.encrypt(temp2, temp2_enc)
    

    # shift temp2_enc to lr
    evaluator.mod_switch_to_inplace(temp2_enc, lr_enc.parms_id())
    #temp2_enc.set_scale(lr_enc.scale())
    lr_enc.set_scale(temp2_enc.scale())
    # lr * temp2
    evaluator.multiply(lr_enc, temp2_enc, lr_term)
    evaluator.relinearize_inplace(lr_term, relin_keys)
    evaluator.rescale_to_next_inplace(lr_term)
    
    ## encode const term 0.5
    const = Plaintext()
    const_enc = Ciphertext()
    encoder.encode(0.5, scale, const)
    encryptor.encrypt(const, const_enc)
    
    ## shift const and lr_term to lr_cube
    evaluator.mod_switch_to_inplace(const_enc, lr_cube.parms_id())
    #const_enc.set_scale(lr_cube.scale())
    lr_cube.set_scale(const_enc.scale())

    evaluator.mod_switch_to_inplace(lr_term, lr_cube.parms_id())
    #lr_term.set_scale(lr_cube.scale())
    lr_term.set_scale(const_enc.scale())
    ## add all terms
    evaluator.add_inplace(lr_cube, lr_term)
    evaluator.add_inplace(lr_cube, const_enc)
    
    return lr_cube
    


# In[8]:


### calculate error as (sigmoid - y) ###
# 0 multiplications
def getError(sig_approx, y_enc_negative):
    
    if(context.get_context_data(y_enc.parms_id()).chain_index() != context.get_context_data(sig_approx.parms_id()).chain_index()):
        evaluator.mod_switch_to_inplace(y_enc_negative, sig_approx.parms_id())
        y_enc_negative.set_scale(sig_approx.scale())

    error = Ciphertext()
    evaluator.add(sig_approx, y_enc_negative, error)
    
    return error


# In[9]:


### get gradient vectors 1 - d
## depth = 1
## Total multiplications / iter = 4

def getGradient(error_enc, m_enc):
    ## m = X * (2/n) as defined above; gradient(d) = error * m(d)
    gradient = Ciphertext()

    
    evaluator.mod_switch_to_inplace(m_enc, error_enc.parms_id())

    m_enc.set_scale(error_enc.scale())

    evaluator.multiply(m_enc, error_enc, gradient)

    
    evaluator.relinearize_inplace(gradient, relin_keys)

    
    evaluator.rescale_to_next_inplace(gradient)

    
    return gradient


# In[10]:


def updateWeight(weight, sumGradient):
    evaluator.mod_switch_to_inplace(weight)
    weight.set_scale(sumGradient.scale())
    
    evaluator.add_inplace(weight, sumGradient)
    
    return weight


# In[11]:


# calculate -1 *y_enc and rescale; scale and mod then match lr_enc
temp = Plaintext()
encoder.encode(-1.0, scale, temp)
y_enc_neg = Ciphertext()
evaluator.multiply_plain(y_enc, temp, y_enc_neg)
evaluator.relinearize_inplace(y_enc_neg, relin_keys)
evaluator.rescale_to_next_inplace(y_enc_neg)

print("starting scale of x: ", context.get_context_data(x1_enc.parms_id()).chain_index())

i = 0


f = open("logfile.txt", "w+")

while(context.get_context_data(beta1_enc.parms_id()).chain_index() > 4): ## replace with calculated final depth per iteration
    
    print("starting iter: ", i)
    print(datetime.datetime.now())
    f.write(str(datetime.datetime.now()))
    
    # with beta_temp calculate Gradient and updated mGrad:
    

    lr_enc = getLinResponse_enc(beta0_enc, beta1_enc, beta2_enc, beta3_enc, beta4_enc, x1_enc, x2_enc, x3_enc, x4_enc)
    
    ##########################################################################################
    print("scale of lr: ", context.get_context_data(lr_enc.parms_id()).chain_index())
    print("scale of lr:   ", lr_enc.scale(), "   ", math.log(lr_enc.scale(), 2), " bits\n")
    f.write("scale of lr: ")
    f.write(str(context.get_context_data(lr_enc.parms_id()).chain_index()))
    f.write("\n")
    f.write("scale of lr:   ")
    f.write(str(math.log(lr_enc.scale(), 2)))
    f.write("\n")
    ############################################################################################
    
    
    sig_approx = getLsaSigmoid(lr_enc)
    ############################################################################################
    print("scale of sig: ", context.get_context_data(sig_approx.parms_id()).chain_index())
    print("scale of sig:   ", sig_approx.scale(), "   ", math.log(sig_approx.scale(), 2), " bits\n")
    f.write("scale of sig: ") 
    f.write(str(context.get_context_data(sig_approx.parms_id()).chain_index()))
    f.write("\n")
    f.write("scale of sig:   ")
    f.write(str(math.log(sig_approx.scale(), 2)))
    f.write("\n")
    ##########################################################################################
    
    
    error = getError(sig_approx, y_enc_neg)
    ##########################################################################################
    print("scale of error: ", context.get_context_data(error.parms_id()).chain_index())
    print("scale of error:   ", error.scale(), "   ", math.log(error.scale(), 2), " bits\n")
    f.write("scale of error: ")
    f.write(str(context.get_context_data(error.parms_id()).chain_index()))
    f.write("\n")
    f.write("scale of error:   ")
    f.write(str(math.log(error.scale(), 2)))
    f.write("\n")
    ##########################################################################################
    
    gradient0 = getGradient(error, m0_enc) ## getGradient: calculates (sig-y)*x_factor, x_factor = alpha * x/n
    gradient1 = getGradient(error, m1_enc)
    gradient2 = getGradient(error, m2_enc)
    gradient3 = getGradient(error, m3_enc)
    gradient4 = getGradient(error, m4_enc)
    
    print("calculating sum of gradient vectors")

    sumGradient0 = sumVector(gradient0, len(x_res))
    sumGradient1 = sumVector(gradient1, len(x_res))
    sumGradient2 = sumVector(gradient2, len(x_res))
    sumGradient3 = sumVector(gradient3, len(x_res))
    sumGradient4 = sumVector(gradient4, len(x_res))
    
    #################### Update beta = beta + sumGradient ##################################

    evaluator.mod_switch_to_inplace(beta0_enc, sumGradient0.parms_id())
    evaluator.mod_switch_to_inplace(beta1_enc, sumGradient0.parms_id())
    evaluator.mod_switch_to_inplace(beta2_enc, sumGradient0.parms_id())
    evaluator.mod_switch_to_inplace(beta3_enc, sumGradient0.parms_id())
    evaluator.mod_switch_to_inplace(beta4_enc, sumGradient0.parms_id())

    beta0_enc.set_scale(sumGradient0.scale())
    beta1_enc.set_scale(sumGradient0.scale())
    beta2_enc.set_scale(sumGradient0.scale())
    beta3_enc.set_scale(sumGradient0.scale())
    beta4_enc.set_scale(sumGradient0.scale())

    evaluator.add_inplace(beta0_enc, sumGradient0)
    evaluator.add_inplace(beta1_enc, sumGradient1)
    evaluator.add_inplace(beta2_enc, sumGradient2)
    evaluator.add_inplace(beta3_enc, sumGradient3)
    evaluator.add_inplace(beta4_enc, sumGradient4)
    ##########################################################################################

    #########################################################################################
    # 21Dec: Add additional set scale to original scale at end to stabilize slowly growing scale
    #beta0_enc.set_scale(scale)
    #beta1_enc.set_scale(scale)
    #beta2_enc.set_scale(scale)
    #beta3_enc.set_scale(scale)
    #beta4_enc.set_scale(scale)
    ###########################################################################################

    print("End of iter: ", i, ", index reached: ", context.get_context_data(beta1_enc.parms_id()).chain_index(), "\n")
    f.write("End of iter \n")
    i += 1


print("reached max depth in ", i, " iterations; gradient descent complete")
f.write("reached max depth in iter: ")
f.write(str(i))
f.write("\n")


# ### Performance on training data

# In[12]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix


sigmoid_out = Plaintext()
sigmoid_out_vec = DoubleVector()

decryptor.decrypt(sig_approx, sigmoid_out)
encoder.decode(sigmoid_out, sigmoid_out_vec)

pred_prob = []
for i in range(len(x_res)):
    pred_prob.append(sigmoid_out_vec[i])
    
sigmoid_l = np.array(pred_prob)

y_pred = sigmoid_l>0.6

nag_train_accuracy = sum(y_pred == y_res)/len(y_res)
nag_train_auc = roc_auc_score(y_res, y_pred)
nag_train_fpr, nag_train_tpr, _ = roc_curve(y_res, sigmoid_l)
nag_train_precision, nag_train_recall, _ = precision_recall_curve(y_res, sigmoid_l)
nag_train_mcc = matthews_corrcoef(y_res, y_pred)

tn, fp, fn, tp = confusion_matrix(y_res, y_pred).ravel()
print("tn, fp, fn, tp: ",tn, fp, fn, tp)
print("Train AUC: ",nag_train_auc,"  Train MCC", nag_train_mcc)

fig = plt.figure(figsize = (10, 5))

plt.subplot(1, 2, 1)
plt.plot(nag_train_fpr, nag_train_tpr)
plt.title("ROC-AUC curve")

plt.subplot(1, 2, 2)
plt.plot(nag_train_recall, nag_train_precision)
plt.title("Precision-Recall curve ")

fig.suptitle("Training data ROC and Precision Recall Curve with Encrypted data")

plt.show()

plt.savefig('./out/Enc_train_evaluation.png')


# ### Evaluation on traning data
# 
# We will decrypt the weights learnt from encrypted algorithm and evaluate performance against test data

# In[14]:


## Decrypt weights

beta0_out = Plaintext()
beta1_out = Plaintext()
beta2_out = Plaintext()
beta3_out = Plaintext()
beta4_out = Plaintext()

decryptor.decrypt(beta0_enc, beta0_out)
decryptor.decrypt(beta1_enc, beta1_out)
decryptor.decrypt(beta2_enc, beta2_out)
decryptor.decrypt(beta3_enc, beta3_out)
decryptor.decrypt(beta4_enc, beta4_out)

beta0_out_vec = DoubleVector()
beta1_out_vec = DoubleVector()
beta2_out_vec = DoubleVector()
beta3_out_vec = DoubleVector()
beta4_out_vec = DoubleVector()

encoder.decode(beta0_out, beta0_out_vec)
encoder.decode(beta1_out, beta1_out_vec)
encoder.decode(beta2_out, beta2_out_vec)
encoder.decode(beta3_out, beta3_out_vec)
encoder.decode(beta4_out, beta4_out_vec)

print_vector(beta0_out_vec, 4,4)
print_vector(beta1_out_vec, 4,4)
print_vector(beta2_out_vec, 4,4)
print_vector(beta3_out_vec, 4,4)
print_vector(beta4_out_vec, 4,4)

f.write("{} {} {} {} {}".format(beta0_out_vec[0],beta1_out_vec[0],beta2_out_vec[0],beta3_out_vec[0],beta4_out_vec[0]))
f.write("\n")
f.write("{} {} {} {} {}".format(beta0_out_vec[5],beta1_out_vec[5],beta2_out_vec[5],beta3_out_vec[5],beta4_out_vec[0]))
f.write("\n")
f.write("{} {} {} {} {}".format(beta0_out_vec[15],beta1_out_vec[15],beta2_out_vec[15],beta3_out_vec[15],beta4_out_vec[0]))
f.write("\n")
f.write("{} {} {} {} {}".format(beta0_out_vec[105],beta1_out_vec[105],beta2_out_vec[105],beta3_out_vec[105],beta4_out_vec[0]))


# In[19]:


## as the whole beta_enc vectors have the same values in all slots, we can take the 0th value as the weight

beta_out = [beta0_out_vec[0], beta1_out_vec[0], beta2_out_vec[0], beta3_out_vec[0], beta4_out_vec[0]]


## Get LSA sigmoid values with these weights


def getLinResponse(train_data, beta):
    
    ## replace with pd dot function
    lin_response = pd.Series(np.dot(train_data, beta))
    return lin_response

def lsa3Sigmoid(lin_response):
    sigmoid = 0.5 + ((1.20096/8) * lin_response) + ((-0.81562/pow(8,3)) * lin_response**3)    
    return sigmoid

X_test = np.array(X_test)

const1 = np.ones(len(X_test))
const1 = const1.reshape(len(X_test), 1)
test_data = np.concatenate((const1, X_test), axis = 1)


# In[21]:


test_resp = getLinResponse(test_data, beta_out)
test_sig = lsa3Sigmoid(test_resp)
test_out = test_sig > 0.6

test_out = np.array(test_out)
y_test = np.array(y_test)


tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, test_out).ravel()
nag_test_accuracy = sum(test_out == y_test)/len(y_test)
nag_test_auc = roc_auc_score(y_test, test_out)
nag_test_mcc = matthews_corrcoef(y_test, test_out)
nag_test_fpr, nag_test_tpr, _ = roc_curve(y_test, test_sig)
nag_test_precision, nag_test_recall, _ = precision_recall_curve(y_test, test_sig)
print(tn_test, fp_test, fn_test, tp_test)


# In[23]:


# Print plot and scores

fig = plt.figure(figsize = (15,5))

plt.subplot(1, 2, 1)
plt.plot(nag_test_fpr, nag_test_tpr)
plt.title("AUC curve")

plt.subplot(1, 2, 2)
plt.plot(nag_test_recall, nag_test_precision)
plt.title("Precision-Recall curve")

plt.savefig('./out/Enc_test_evaluation.png', dpi = 100)
plt.show()

print("Encrypted model AUC = ", nag_test_auc,"\n")
print("Encrypted model MCC = ", nag_test_mcc)

