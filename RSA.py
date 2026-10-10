import numpy as np
import time
import tracemalloc
import sys
from Crypto.PublicKey import RSA as CryptoRSA
from Crypto.Cipher import PKCS1_OAEP


# Function to generate RSA keys
def generate_rsa_keys(bitsize=2048):
    key = CryptoRSA.generate(bitsize)  # Ensure RSA is being used correctly from Crypto.PublicKey
    private_key = key
    public_key = key.publickey()
    return public_key, private_key


# Function to encrypt with RSA
def encrypt_rsa(plaintext, public_key):
    cipher = PKCS1_OAEP.new(public_key)
    ciphertext = cipher.encrypt(plaintext.encode())
    return ciphertext


# Function to decrypt with RSA
def decrypt_rsa(ciphertext, private_key):
    cipher = PKCS1_OAEP.new(private_key)
    decrypted_plaintext = cipher.decrypt(ciphertext)
    return decrypted_plaintext.decode()


# Function to get the memory size of an object
def get_memory_size(obj):
    return sys.getsizeof(obj)


# Main function to handle RSA encryption and decryption, similar to your DES function
def Model_RSA(Data, pred):
    index = pred.index(1)
    data = Data[index]
    strData = ''
    for i in range(len(data)):
        strData = strData + '[' + ','.join([str(j) for j in data[i, :]]) + '],'

    # RSA Key Generation
    public_key, private_key = generate_rsa_keys(bitsize=1024)  # Generate keys (1024-bit size)

    # Start memory tracking and time measurement
    tracemalloc.start()
    start_time = time.time()

    # Encrypt the data using the public key
    ciphertext = encrypt_rsa(strData[0], public_key)

    ENC_time = time.time() - start_time  # Encryption time

    # Decrypt the ciphertext using the private key
    decrypted_plaintext = decrypt_rsa(ciphertext, private_key)

    splitDe = decrypted_plaintext.split('],[')
    splitDe[0] = splitDe[0].split('[')[-1]
    splitDe[-1] = splitDe[-1].split(']')[0]
    List = []
    for i in range(len(splitDe)):
        List.append([float(j) for j in splitDe[i].split(',')])
    Decrypt_Data = np.asarray(List)

    # Get memory size and total computation time
    mem_size = get_memory_size(strData)
    DEC_Time = time.time() - ENC_time  # Decryption time
    Compt_Time = ENC_time + DEC_Time  # Total computation time

    return [ENC_time, DEC_Time, mem_size, Compt_Time], ciphertext, Decrypt_Data
