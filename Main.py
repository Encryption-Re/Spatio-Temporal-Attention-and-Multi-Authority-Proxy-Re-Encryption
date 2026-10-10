import random
import string
import hashlib
import json
import pandas as pd
from numpy import matlib
from sklearn.utils import shuffle
import random as rn
from AES import Model_AES
from Attributes import Create_Attributes
from CWO import CWO
from DES import Model_DES
from ECC import Model_ECC
from Glob_Vars import Glob_Vars
from Model_DNN import Model_DNN
from Model_KNN import Model_KNN
from Model_Proposed import Model_PROPOSED
from Model_RNN import Model_STA_ABi_RNN
from Model_SVM import Model_SVM
from Objective_Function import Objfun_Cls
from PROPOSED import PROPOSED
from Plot_Crypt import *
from Plot_Results import *
from RSA import Model_RSA
from SOA import SOA
from TOT import TOT
from WOA import WOA


def Data_Change(Data):
    Data = pd.DataFrame(Data)
    original_data = np.asarray(Data)
    for i in range(original_data.shape[1]):
        if isinstance(original_data[0, i], str):
            print(i, original_data.shape[1])
            data = Data.iloc[:, i]
            original_data[:, i] = string_Change(data)
    return original_data


def string_Change(data):
    numeric_values = []
    for k in range(len(data)):
        string_to_number = int(''.join(str(ord(char)) for char in data[k]))
        numeric_values.append(string_to_number)
    return numeric_values


# Function to generate a unique random string
def random_string(length=16, existing_set=None):
    while True:
        result = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        if existing_set is None or result not in existing_set:
            if existing_set is not None:
                existing_set.add(result)
            return result


# Function to generate a unique hash
def generate_hash(data, existing_hashes):
    while True:
        hash_value = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        if hash_value not in existing_hashes:
            existing_hashes.add(hash_value)
            return hash_value


# Sets to track existing values
existing_private_keys = set()
existing_signatures = set()
existing_hashes = set()
existing_public_keys = set()
existing_usernames = set()
existing_ids = set()


# Function to generate a unique username
def generate_username():
    while True:
        username = f"user{random.randint(1000, 9999)}"
        if username not in existing_usernames:
            existing_usernames.add(username)
            return username


# Function to generate a unique transaction
def generate_transaction():
    return [
        random_string(32, existing_private_keys),
        random_string(64, existing_signatures),
        generate_hash(random_string(64), existing_hashes),
        random_string(128),
        random_string(32, existing_public_keys)
    ]


# Function to generate a block
def generate_block(previous_hash="0" * 64):
    transactions = [generate_transaction() for _ in range(random.randint(1, 5))]
    merkle_root = generate_hash(transactions, existing_hashes)

    block = [
        random_string(16, existing_ids),  # Unique ID
        generate_username(),  # Unique Username
        random_string(12),  # Unique Password
        previous_hash,
        merkle_root,
        transactions,
        random.randint(1609459200, 1700000000)  # Timestamp
    ]
    return block


# Attributes
an = 0
if an == 1:
    No_of_Data_Count = 1000
    blockchain = []
    previous_hash = "0" * 64  # Genesis block previous hash
    for _ in range(No_of_Data_Count):
        block = generate_block(previous_hash)
        previous_hash = generate_hash(block, existing_hashes)  # Hash current block to link
        blockchain.append(block)
    # Save blockchain data
    np.save('Blockchain_Attributes.npy', np.asarray(blockchain, dtype=object))

# Datas
an = 0
if an == 1:
    path = './Dataset/diabetes.csv'
    data = pd.read_csv(path)
    Tar = data.iloc[:, -1]
    Data = data.iloc[:, :-1]
    np.save('Data.npy', Data)
    np.save('Target.npy', Tar)

# Data Creation
an = 0
if an == 1:
    Attributes = np.load('Blockchain_Attributes.npy', allow_pickle=True)
    New_attributes = Create_Attributes(len(Attributes))
    Data = np.vstack((Attributes[:, [0, 2]], np.asarray(New_attributes, dtype=object)))  # User id and user Password
    Target = np.zeros((Data.shape[0], 1)).astype('int32')
    Target[:Attributes.shape[0]] = 1
    Data, Target = shuffle(Data, Target)
    np.save('Authendication_Attributes.npy', Data)
    np.save('Authendication_Target.npy', Target)

# Optimization for Access Control
an = 0
if an == 1:
    Data = np.load('Authendication_Attributes.npy', allow_pickle=True)
    Target = np.load('Authendication_Target.npy', allow_pickle=True)
    Glob_Vars.Data = Data
    Glob_Vars.Target = Target
    fname = Objfun_Cls
    Npop = 10
    Chlen = 3
    max_iter = 50
    xmin = matlib.repmat(([5, 0.01, 1]), Npop, 1)
    xmax = matlib.repmat(([255, 0.99, 5]), Npop, 1)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])

    print('CWO....')
    [bestfit1, fitness1, bestsol1, Time1] = CWO(initsol, fname, xmin, xmax, max_iter)

    print('TOT....')
    [bestfit2, fitness2, bestsol2, Time2] = TOT(initsol, fname, xmin, xmax, max_iter)

    print('WOA....')
    [bestfit3, fitness3, bestsol3, Time3] = WOA(initsol, fname, xmin, xmax, max_iter)

    print('SOA....')
    [bestfit4, fitness4, bestsol4, Time4] = SOA(initsol, fname, xmin, xmax, max_iter)

    print('PROPOSED....')
    [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

    sol = ([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
    fitness = ([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])

    np.save('Bestsol.npy', sol)
    np.save('Fitness.npy', fitness)

# Authentication Verification
an = 0
if an == 1:
    Data = np.load('Authendication_Attributes.npy', allow_pickle=True)
    Target = np.load('Authendication_Target.npy', allow_pickle=True)
    Data = Data_Change(Data)
    sol = np.load('Bestsol.npy', allow_pickle=True)
    k_fold = 5
    Eval = []
    for m in range(k_fold):
        Total_Index = np.arange(Data.shape[0])
        Test_index = np.arange(((m - 1) * (Data.shape[0] / k_fold)) + 1, m * (Data.shape[0] / k_fold))
        Train_Index = np.setdiff1d(Total_Index, Test_index)
        train_data = Data[Train_Index, :]
        train_target = Target[Train_Index, :]
        test_data = Data[Test_index.astype('int'), :]
        test_target = Target[Test_index.astype('int'), :]
        EVAL = np.zeros((10, 25))
        for i in range(5):  # for all algorithms
            EVAL[i, :], pred4 = Model_STA_ABi_RNN(train_data, train_target, test_data, test_target, m,
                                                  sol[i].astype('int'))
        EVAL[5, :] = Model_SVM(train_data, train_target, test_data, test_target, m)
        EVAL[6, :] = Model_KNN(train_data, train_target, test_data, test_target, m)
        EVAL[7, :] = Model_DNN(train_data, train_target, test_data, test_target, m)
        EVAL[8, :] = Model_STA_ABi_RNN(train_data, train_target, test_data, test_target, m)
        EVAL[9, :] = EVAL[4, :]
        Eval.append(EVAL)
    np.save('Eval_All_Kfold.npy', Eval)
    np.save('Predict.npy', pred4)

# Cryptography
an = 0
if an == 1:
    Eval_all = []
    Data = np.load('Data.npy', allow_pickle=True)
    Pred = np.load('Predict.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Blocksize = [1, 2, 3, 4, 5]
    for m in range(len(Blocksize)):
        per = round(Data.shape[0] * (Blocksize[m] / 100))
        EVAL = np.zeros((5, 4))
        data = Data[:per, :]
        pred_data = Pred[:per, :]
        EVAL[0, :], Encrypt_Data0, Decrypt_Data0 = Model_DES(data, pred_data)
        EVAL[1, :], Encrypt_Data1, Decrypt_Data1 = Model_AES(data, pred_data)
        EVAL[2, :], Encrypt_Data2, Decrypt_Data2 = Model_ECC(data, pred_data)
        EVAL[3, :], Encrypt_Data3, Decrypt_Data3 = Model_RSA(data, pred_data)
        EVAL[4, :], Encrypt_Data4, Decrypt_Data4 = Model_PROPOSED(data, pred_data)
        Eval_all.append(EVAL)
    np.save('Eval_time.npy', Eval_all)
    np.save('Decrypted_Data.npy', Decrypt_Data4)
    np.save('Encrypeted_Data.npy', Encrypt_Data4)

plot_ENC_DEC_CONS_MEM_1()
plot_CPA_KPA()
ROC_curve()
plot_results_conv()
Plot_table()
Plot_Confusion()
Plot_Results()
