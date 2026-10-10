import random
import string


def generate_unique_id(existing_ids):
    while True:
        unique_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        if unique_id not in existing_ids:
            existing_ids.add(unique_id)
            return unique_id


def Create_Attributes(No_of_Data_Count):
    User = []
    existing_ids = set()
    for _ in range(No_of_Data_Count):
        unique_id = generate_unique_id(existing_ids)
        user = [
             unique_id,
             ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        ]
        User.append(user)
    return User
