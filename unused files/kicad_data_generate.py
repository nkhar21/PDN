import pandas as pd
import re

# VIA INFORMATION CSV FILE
# via_path = r"C:\Users\al7k7\Desktop\Is tool\final_comlication.csv"
# via_path = r"C:\Users\al7k7\Desktop\Is tool\via_throughhole_sample.csv"
# via_path = r"C:\Users\al7k7\Desktop\Is tool\via_stack_final.csv"
# via_path = r"C:\Users\al7k7\Desktop\Is tool\via_buried_final.csv"
via_path = r"C:\Users\al7k7\Desktop\final_complicated\final_comlicate1.csv"

# STACKUP FILE
# stackup_path = r"C:\Users\al7k7\Desktop\Is tool\StackUp_final_complicate.csv"
# stackup_path = r"C:\Users\al7k7\Desktop\Is tool\StackUp_throughhole.csv"
# stackup_path = r"C:\Users\al7k7\Desktop\Is tool\StackUp_stackvia_final.csv"
stackup_path = r"C:\Users\al7k7\Desktop\Is tool\StackUp_final_complicate.csv"

# LAYER TYPE
layer_type_path = r"C:\Users\al7k7\Desktop\Is tool\layer_type.csv"

#OUTPUT FILE
output_path = r"C:\Users\al7k7\Desktop\Is tool\generated.kicad_pcb"

# Read data from CSV file
via_df = pd.read_csv(via_path).dropna(subset=['via_id'])
stackup_df = pd.read_csv(stackup_path)
layer_type_df = pd.read_csv(layer_type_path)

# Function to rename signal layer names
def normalize_signal_name(name):
    match = re.match(r'(Signal)(\d+)', str(name))
    if match:
        base, num = match.groups()
        return f"{base}{int(num):03d}"
    return name

# Mapping of signal layer names to new identifiers
signal_name_map = {}
normalized_layer_names = []

layers = []
index = 0
for name in stackup_df["Layer Name"].dropna():
    norm_name = normalize_signal_name(name)
    signal_name_map[name] = norm_name
    if "Signal" in name or ".Cu" in name:
        normalized_layer_names.append(norm_name)
        layers.append(f'    ({index} "{norm_name}" signal)')
        index += 1

# total number of layers
max_layer_number = len(normalized_layer_names)

# layer type mapping
layer_type_df['layer'] = layer_type_df['layer'].apply(normalize_signal_name)
layer_types = {row['layer']: row['type'] for _, row in layer_type_df.iterrows()}

# Maximum Y value including zones and vias
zone_y_cols = [col for col in via_df.columns if col.startswith('b_y')]
y_max_zone = via_df[zone_y_cols].max().max()
y_max_via = via_df['via_y_(mm)'].max()
Y_MAX = max(y_max_zone, y_max_via)

# In KiCad, X increases to the left, and Y increases as you go downward.
# Center shift calculation
num_zones = len([col for col in via_df.columns if col.startswith('b_x')])
# Collect all coordinates used in zone polygons
all_coords = []
for i in range(1, num_zones + 1):
    x_col = f'b_x{i}(mm)'
    y_col = f'b_y{i}(mm)'
    if x_col in via_df.columns and y_col in via_df.columns:
        shape_df = via_df[[x_col, y_col]].dropna()
        all_coords += shape_df.values.tolist()
# Calculate the geometric center (centroid) of all zone shapes
if all_coords:
    x_vals, y_vals = zip(*all_coords)
    x_center = (max(x_vals) + min(x_vals)) / 2
    y_center = (max(y_vals) + min(y_vals)) / 2
else:
    x_center = y_center = 0 # Default to origin if no shapes are present

# A4 paper center coordinates in millimeters (used for aligning layout)
a4_x_center = 105.0
a4_y_center = 148.5

# Calculate how much to shift all coordinates to center the layout on the A4 page
shift_x = a4_x_center - x_center
shift_y = a4_y_center - y_center - 20  

# stackup generation
stackup = '    (stackup\n'
for i, row in stackup_df.iterrows():
    lname = row["Layer Name"]
    norm_name = signal_name_map.get(lname, lname)
    thick = float(row["Thickness(mm)"])
    er = row.get("Er", 4.5)
    loss = row.get("Loss Tangent", 0.02)

    if "Signal" in lname:
        stackup += f'      (layer "{norm_name}" (type "copper") (thickness {thick:.5f}))\n'
    else:
        stackup += f'      (layer "dielectric {i}" (type "core") (thickness {thick:.5f}) (material "FR4") (epsilon_r {er}) (loss_tangent {loss}))\n'
stackup += '    )\n'

# via list
vias = ''
# Group vias by their X/Y coordinates and type (e.g., GND or PWR)
grouped = via_df.dropna(subset=['via_x_(mm)', 'via_y_(mm)', 'via_type']).groupby(
    ['via_x_(mm)', 'via_y_(mm)', 'via_type']
)
# Iterate through each group of vias at the same location with the same type
for (x, y, net), group in grouped:
    layers_connected = set()
    # Collect all layers connected by vias at this location
    for _, row in group.iterrows():
        start = int(row['start_layer'])
        stop = int(row['stop_layer'])
        layers_connected.update(range(start, stop + 1))

    # Check if this is a through-hole via (connects from top to bottom layer)
    is_through_hole = layers_connected == set(range(1, max_layer_number + 1))
    blind = "" if is_through_hole else "blind"

    # Determine which two layers the via connects
    start_layer = min(layers_connected)
    stop_layer = max(layers_connected)
    layer1 = normalized_layer_names[start_layer - 1]
    layer2 = normalized_layer_names[stop_layer - 1]

    # Assign net number: 1 for GND, 2 for PWR
    net_num = 1 if net == "GND" else 2

    # Convert coordinates (Y-flip and origin shift for KiCad coordinate system)
    flipped_y = Y_MAX - y
    shifted_x = x + shift_x
    shifted_y = flipped_y + shift_y

    # Append the via definition to the vias string in KiCad format
    vias += f'''  (via {blind}
    (at {shifted_x:.4f} {shifted_y:.4f})
    (size 0.8)
    (drill 0.4)
    (layers "{layer1}" "{layer2}")
    (net {net_num})
  )\n\n'''

# zone generation
# Initialize the zones string to accumulate zone definitions
zones = ''

# Iterate through each potential zone index
for i in range(1, num_zones + 1):
    # Define the column names for the X and Y coordinates of the zone polygon
    x_col = f'b_x{i}(mm)'
    y_col = f'b_y{i}(mm)'

    # Skip this zone if the required coordinate columns are missing
    if x_col not in via_df.columns or y_col not in via_df.columns:
        continue

    # Extract the shape's X and Y coordinates and remove any rows with NaN values
    shape_df = via_df[[x_col, y_col]].dropna()
    shape_coords = shape_df.values.tolist()

    # Skip if the polygon has fewer than 3 points (not a valid shape)
    if len(shape_coords) < 3:
        continue
    # Try to get the corresponding layer information from the layer_type_df
    try:
        layer_row = layer_type_df.iloc[i - 1]
    except IndexError:
        continue

    layer_name = layer_row['layer']
    layer_type = layer_row['type']

    # Only include zones that are on signal layers
    if not str(layer_name).startswith("Signal"):
        continue

    # Assign net and net number based on layer type (0 = GND, 1 = PWR)
    if layer_type == 0:
        net_type = 'GND'
        zone_net = 1
    elif layer_type == 1:
        net_type = 'PWR'
        zone_net = 2
    else:
        continue

 # Append zone definition here (not shown in your code snippet, assumed below)
    zones += f'''
  (zone
    (net {zone_net})
    (net_name "{net_type}")
    (layer "{layer_name}")
    (hatch edge 0.508)
    (connect_pads (clearance 1.0))
    (min_thickness 0.2)
    (fill yes)
    (polygon
      (pts
'''
    for bx, by in shape_coords:
        flipped_y = Y_MAX - by
        shifted_x = bx + shift_x
        shifted_y = flipped_y + shift_y
        zones += f'        (xy {shifted_x:.4f} {shifted_y:.4f})\n'

    zones += '''      )
    )
  )
'''

# This section defines the nets
kicad_pcb = f'''(kicad_pcb (version 20211014) (generator py)

  (general
    (thickness 6.46)
  )

  (paper "A4")
  (layers
{chr(10).join(layers)}
  )

  (setup
{stackup.strip()}
    (pad_to_mask_clearance 0)
    (grid_origin 0 0)
  )

  (net 0 "")
  (net 1 "GND")
  (net 2 "PWR")

{vias.strip()}

{zones.strip()}


)
'''

# save
with open(output_path, "w", encoding="utf-8") as f:
    f.write(kicad_pcb.strip())

print("KiCad PCB file has been successfully generated:", output_path)
