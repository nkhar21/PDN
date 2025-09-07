import pandas as pd

# ## Original
# # Layer type file path
# layer_type = pd.read_csv(r"layer_type.csv")
# # SPD or CSV file for via information
# input_path = r"via_throughhole.spd"
# # stackup file path
# stack_up_csv_path = r"StackUp_throughhole.csv"
# # touchstone file path
# touchstone_path = r"via_throughhole_052725_160313_21096.S3P"


# # Board 1
# # Layer type file path
# layer_type = pd.read_csv(r"b1_layer_type.csv")
# # SPD or CSV file for via information
# input_path = r"b1.spd"
# # stackup file path
# stack_up_csv_path = r"b1_stackup.csv"
# # touchstone file path
# touchstone_path = r"b1.S3P"

# # Board 2
# # Layer type file path
# layer_type = pd.read_csv(r"b2_layer_type.csv")
# # SPD or CSV file for via information
# input_path = r"b2.spd"
# # stackup file path
# stack_up_csv_path = r"b2_stackup.csv"
# # touchstone file path
# touchstone_path = r"b2.S5P"

# # Board 3
# # Layer type file path
# layer_type = pd.read_csv(r"b3_layer_type.csv")
# # SPD or CSV file for via information
# input_path = r"b3.spd"
# # stackup file path
# stack_up_csv_path = r"b3_stackup.csv"
# # touchstone file path
# touchstone_path = r"b3.S3P"

# Board 4.1
# Layer type file path
layer_type = pd.read_csv(r"b4_1_layer_type.csv")
# SPD or CSV file for via information
input_path = r"b4_1.spd"
# stackup file path
stack_up_csv_path = r"b4_1_stackup.csv"
# touchstone file path
touchstone_path = r"b4_1.S3P"




#Check whether the input path is a '.spd' or '.csv' file.
if input_path.lower().endswith(".spd"):
    input_type = "spd"
elif input_path.lower().endswith(".csv"):
    input_type = "csv"
else:
    raise ValueError("The file format is not supported. Please upload a '.spd' or '.csv' file.")
