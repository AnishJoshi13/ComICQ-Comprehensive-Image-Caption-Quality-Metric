import re

def getSubject(input_string):
    subject_pattern = r"Subject: (.*?)Object:"
    subject_match = re.search(subject_pattern, input_string)
    if subject_match:
        return subject_match.group(1)
    else:
        return None


def getObject(input_string):
    object_pattern = r"Object: (.*?)$"
    object_match = re.search(object_pattern, input_string)
    if object_match:
        return object_match.group(1)
    else:
        return None

# Input string
input_string = "Subject: Men Object: Knives"

# # Extract subject and object
# subject = getSubject(input_string)
# object = getObject(input_string)
#
# if subject is not None:
#     print("Subject:", subject)
# else:
#     print("Subject not found")
#
# if object is not None:
#     print("Object:", object)
# else:
#     print("Object not found")
