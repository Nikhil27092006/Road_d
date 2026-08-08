import os
import sys
from dotenv import load_dotenv

load_dotenv()
os.environ["MODEL_VALIDATION_DISABLED"] = "True"

import cv2
import supervision as sv
from inference import get_model

API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Connect to pretrained model (downloads weights on first run, caches locally)
MODEL_ID = "road-damage-detection-i40w8/3"   # replace with your chosen model
model = get_model(model_id=MODEL_ID, api_key=API_KEY)

# Load image from command line argument or default
image_path = sys.argv[1] if len(sys.argv) > 1 else "test_road.jpg"

if not os.path.exists(image_path):
    print(f"Error: Image '{image_path}' not found. Usage: python detect.py <path_to_image>")
    sys.exit(1)

image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not decode image file '{image_path}'")
    sys.exit(1)


# Run inference
results = model.infer(image, confidence=0.4)[0]

# Convert to supervision Detections for easy drawing
detections = sv.Detections.from_inference(results)

# Annotate image with boxes + labels
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(detections.data["class_name"], detections.confidence)
]

annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# Save result
cv2.imwrite("output_annotated.jpg", annotated_image)
print("Done. Saved as output_annotated.jpg")
print(results)