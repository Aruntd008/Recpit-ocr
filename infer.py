import subprocess
import pandas as pd
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json

# Paths to scripts and environments
script1_path = "script1.py"  # Path to the first script
script2_path = "script2.py"  # Path to the second script
output_json = "output.json"  # Output file to store JSON data from the first script
image_file = r"C:\Users\Arun\pytorch\datasets\bills\sample\OCR Receipts\img\18.jpg"  # Path to the input image

# Function to convert bounding box to polygon
def convert_to_polygon(coords):
    """
    Convert bounding box coordinates in [xmin, ymin, xmax, ymax] format
    to a polygon format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    """
    xmin, ymin, xmax, ymax = coords
    return [
        [xmin, ymin],  # Top-left
        [xmax, ymin],  # Top-right
        [xmax, ymax],  # Bottom-right
        [xmin, ymax]   # Bottom-left
    ]

# Step 1: Run the first script in the current environment
print("Running first script...")

# Load image
img = DocumentFile.from_images([image_file])

# Load OCR model
model = ocr_predictor(
    det_arch='db_resnet50', 
    reco_arch='crnn_vgg16_bn', 
    pretrained=True, 
    straighten_pages=True
)

# Analyze the image
result = model(img)
result_export = result.export()

# Process OCR results
data = []
for i, page in enumerate(result_export['pages']):
    page_dims = page['dimensions']  # Dimensions of the page
    for block in page['blocks']:
        for line in block['lines']:
            for word in line['words']:
                # Calculate absolute coordinates for the word
                abs_coords = [
                    int(round(word['geometry'][0][0] * page_dims[1])),
                    int(round(word['geometry'][0][1] * page_dims[0])),
                    int(round(word['geometry'][1][0] * page_dims[1])),
                    int(round(word['geometry'][1][1] * page_dims[0])),
                ]

                # Convert to polygon format
                polygon = convert_to_polygon(abs_coords)

                # Append transcription and coordinates to the data list
                data.append({
                    "transcription": word['value'],
                    "points": polygon,
                    "label": f"Tax_key"
                })

# Print or process the data
for item in data:
    print(item)

# Convert the list to a DataFrame
df = pd.DataFrame(data)

# Save the data as JSON
json_data = df.to_json(orient="records", lines=False)

# Save OCR results as JSON
json_label_file = r"C:\Users\Arun\dl_project\ocr_results.json"  # Path for the OCR result file
with open(json_label_file, "w") as f:
    json.dump(data, f)

# Add the JSON label file path to the subprocess command
command = (
    f"conda run -n paddle_env3 python C:/Users/Arun/PaddleOCR/tools/infer_one.py "
    f"-c C:/Users/Arun/PaddleOCR/configs/kie/kie_unet_sdmgr.yml "
    f"-o Global.checkpoints=C:/Users/Arun/PaddleOCR/kie_vgg16/best_accuracy "
    f"Global.infer_img=\"{image_file}\" "
    f"Global.label_file=\"{json_label_file}\" "
    f"Global.save_res_path=C:/Users/Arun/PaddleOCR/output/results.txt"
)

# Run the command
subprocess.run(command, shell=True, check=True)

print("Both processes completed successfully.")
