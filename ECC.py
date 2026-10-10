import numpy as np
import time
import sys
import tracemalloc
from tinyec import registry
import secrets


def get_memory_size(obj):
    return sys.getsizeof(obj)


def encrypt_ecc(plaintext, pubKey, curve):
    """Encrypts a message using ECC with ElGamal encryption"""
    msg_bytes = plaintext.encode()
    k = secrets.randbelow(curve.field.n)
    R = k * curve.g  # Random point on the curve
    S = k * pubKey  # Shared secret
    ciphertext = bytes([msg_bytes[i] ^ S.x.to_bytes((S.x.bit_length() + 7) // 8, 'big')[
        i % len(S.x.to_bytes((S.x.bit_length() + 7) // 8, 'big'))] for i in range(len(msg_bytes))])
    return (R, ciphertext)


def decrypt_ecc(encrypted_data, privKey):
    """Decrypts ECC-encrypted message"""
    R, ciphertext = encrypted_data
    S = privKey * R  # Compute shared secret
    msg_bytes = bytes([ciphertext[i] ^ S.x.to_bytes((S.x.bit_length() + 7) // 8, 'big')[
        i % len(S.x.to_bytes((S.x.bit_length() + 7) // 8, 'big'))] for i in range(len(ciphertext))])
    return msg_bytes.decode()


def Model_ECC(Data, pred):
    index = pred.index(1)
    data = Data[index]
    strData = ''
    for i in range(len(data)):
        strData += '[' + ','.join(map(str, data[i, :])) + '],'

    curve = registry.get_curve('secp256r1')  # Choose ECC curve
    privKey = secrets.randbelow(curve.field.n)
    pubKey = privKey * curve.g  # Public key

    tracemalloc.start()
    ct = time.time()
    encrypted_data = encrypt_ecc(strData, pubKey, curve)
    ENC_time = time.time() - ct

    decrypted_plaintext = decrypt_ecc(encrypted_data, privKey)

    splitDe = decrypted_plaintext.split('],[')
    splitDe[0] = splitDe[0].split('[')[-1]
    splitDe[-1] = splitDe[-1].split(']')[0]
    List = [[float(j) for j in splitDe[i].split(',')] for i in range(len(splitDe))]

    Decrypt_Data = np.asarray(List)
    mem_size = get_memory_size(strData)
    DEC_Time = time.time() - ENC_time
    Compt_Time = ENC_time + DEC_Time

    return [ENC_time, DEC_Time, mem_size, Compt_Time], encrypted_data, Decrypt_Data


