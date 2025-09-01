import pandas as pd

# Layer type file path
layer_type = pd.read_csv(r"layer_type.csv")

# SPD or CSV file for via information

input_path = r"via_throughhole.spd"


# stackup file path

stack_up_csv_path = r"StackUp_throughhole.csv"


# touchstone file path

touchstone_path = r"via_throughhole_052725_160313_21096.S3P"


#Check whether the input path is a '.spd' or '.csv' file.
if input_path.lower().endswith(".spd"):
    input_type = "spd"
elif input_path.lower().endswith(".csv"):
    input_type = "csv"
else:
    raise ValueError("The file format is not supported. Please upload a '.spd' or '.csv' file.")
