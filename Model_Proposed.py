from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import time
import sys


def setup():
    """ Generate a random AES key for encryption. """
    key = get_random_bytes(32)  # 256-bit AES key
    return key


def encrypt(key, message):
    """ Encrypt message using AES-GCM. """
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(message.encode())
    return cipher.nonce, ciphertext, tag


def decrypt(key, nonce, ciphertext, tag):
    """ Decrypt AES-GCM ciphertext. """
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag).decode()


# Function to get the memory size of an object
def get_memory_size(obj):
    return sys.getsizeof(obj)


# Example Usage
def Model_PROPOSED(Data, pred):
    index = pred.index(1)
    data = Data[index]
    strData = ''
    for i in range(len(data)):
        strData = strData + '[' + ','.join([str(j) for j in data[i, :]]) + '],'

    # Setup AES system
    aes_key = setup()

    # Encrypt a message
    start_time = time.time()
    nonce, ciphertext, tag = encrypt(aes_key, strData)
    ENC_time = time.time() - start_time

    # Simulated Re-encryption (in AES, we just share the key)
    re_enc_key = aes_key  # In a full system, key sharing would use a secure method.

    # Decryption
    start_time = time.time()
    decrypted_message = decrypt(re_enc_key, nonce, ciphertext, tag)
    DEC_Time = time.time() - start_time

    print("Decrypted Message:", decrypted_message)

    # Get memory size and total computation time
    mem_size = get_memory_size(strData)
    Compt_Time = ENC_time + DEC_Time  # Total computation time

    return [ENC_time, DEC_Time, mem_size, Compt_Time], ciphertext, decrypted_message
