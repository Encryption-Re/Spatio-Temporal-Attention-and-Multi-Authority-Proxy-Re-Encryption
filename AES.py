import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import time
import tracemalloc
import sys


def encrypt_aes(plaintext, key):
    cipher = AES.new(key, AES.MODE_ECB)
    padded_plaintext = pad(plaintext.encode(), AES.block_size)
    ciphertext = cipher.encrypt(padded_plaintext)
    return ciphertext


def decrypt_aes(ciphertext, key):
    cipher = AES.new(key, AES.MODE_ECB)
    decrypted_plaintext = cipher.decrypt(ciphertext)
    unpadded_plaintext = unpad(decrypted_plaintext, AES.block_size)
    return unpadded_plaintext.decode()


def get_memory_size(obj):
    return sys.getsizeof(obj)


def Model_AES(Data, pred):
    index = pred.index(1)
    data = Data[index]
    strData = ''
    for i in range(len(data)):
        strData = strData + '[' + ','.join([str(j) for j in data[i, :]]) + '],'

    key = b"0123456789abcdef"  # 16-byte key (128-bit key for AES)
    tracemalloc.start()
    ct = time.time()
    ciphertext = encrypt_aes(strData, key)

    ENC_time = time.time() - ct
    decrypted_plaintext = decrypt_aes(ciphertext, key)

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
    return [ENC_time, DEC_Time, mem_size, Compt_Time], ciphertext, Decrypt_Data
