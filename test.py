from logger_config import setup_logger
import random
import os
import json

FILE_NAME = "test.json"

logger = setup_logger()

logger.info("Started script")
random_num = random.randint(1, 1000)
logger.info(f"Picked random number: {random_num}")

if os.path.exists(FILE_NAME):
    with open(FILE_NAME, "r") as f:
        nums = json.load(f)
    logger.info(f"Loaded {FILE_NAME}. There are {len(nums)} numbers in the file.")
    logger.info(f"Current list: {nums}")
else:
    nums = []
    logger.info(f"{FILE_NAME} not found. Initializing an empty list.")

nums.append(random_num)
logger.info(f"Random number {random_num} appended to list")
logger.info(f"Current list: {nums}")

with open(FILE_NAME, "w") as f:
    json.dump(nums, f)
logger.info(f"List written to file: {FILE_NAME}")

logger.info("Re-opening file")
with open(FILE_NAME, "r") as f:
    nums = json.load(f)

logger.info(f"List: {nums}\nThere are {len(nums)} number in the file.")
logger.info("Finished script")
