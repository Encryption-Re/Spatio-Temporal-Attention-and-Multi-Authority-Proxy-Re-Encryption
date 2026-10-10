import numpy as np
from Crypto.Cipher import DES as des
from Crypto.Util.Padding import pad, unpad
import time
import tracemalloc
import sys


def encrypt_des(plaintext, key):
    cipher = des.new(key, des.MODE_ECB)
    padded_plaintext = pad(plaintext.encode(), des.block_size)
    ciphertext = cipher.encrypt(padded_plaintext)
    return ciphertext


def decrypt_des(ciphertext, key):
    cipher = des.new(key, des.MODE_ECB)
    decrypted_plaintext = cipher.decrypt(ciphertext)
    unpadded_plaintext = unpad(decrypted_plaintext, des.block_size)
    return unpadded_plaintext.decode()


def get_memory_size(obj):
    return sys.getsizeof(obj)


def Model_DES(Data, pred):
    index = pred.index(1)
    data = Data[index]
    strData = ''
    for i in range(len(data)):
        strData = strData + '[' + ','.join([str(j) for j in data[i, :]]) + '],'

    key = b"01234567"  # 8-byte key
    tracemalloc.start()
    ct = time.time()
    ciphertext = encrypt_des(strData, key)

    ENC_time = time.time() - ct
    decrypted_plaintext = decrypt_des(ciphertext, key)

    splitDe = decrypted_plaintext.split('],[')
    splitDe[0] = splitDe[0].split('[')[-1]
    splitDe[-1] = splitDe[-1].split(']')[0]
    List = []
    for i in range(len(splitDe)):
        List.append([float(j) for j in splitDe[i].split(',')])
    Decrypt_Data = np.asarray(List)
    mem_size = get_memory_size(strData)
    DEC_Time = time.time() - ENC_time
    Compt_Time = ENC_time + DEC_Time
    return [ENC_time, DEC_Time, mem_size, Compt_Time],ciphertext, Decrypt_Data

