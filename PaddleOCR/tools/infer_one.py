import os
import sys
import time
import json
import numpy as np
import paddle
import paddle.nn.functional as F
import cv2

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.utils.save_load import load_model
import tools.program as program

def read_class_list(filepath):
    """Reads class labels from a file."""
    classes = {}
    with open(filepath, "r") as f:
        for idx, line in enumerate(f.readlines()):
            classes[idx] = line.strip("\n")
    return classes

def draw_kie_result(batch, node, idx_to_cls, save_path):
    """
    Draws KIE results on the image and saves the visualized image.

    Args:
        batch: The batch of data.
        node: Node predictions.
        idx_to_cls: Mapping of indices to class labels.
        save_path: Path to save the visualized image.
    """
    img = batch[6].copy()
    boxes = batch[7]
    h, w = img.shape[:2]
    pred_img = np.ones((h, w * 2, 3), dtype=np.uint8) * 255
    max_value, max_idx = paddle.max(node, -1), paddle.argmax(node, -1)
    node_pred_label = max_idx.numpy().tolist()
    node_pred_score = max_value.numpy().tolist()

    for i, box in enumerate(boxes):
        if i >= len(node_pred_label):
            break

        new_box = [
            [box[0], box[1]],
            [box[2], box[1]],
            [box[2], box[3]],
            [box[0], box[3]],
        ]
        Pts = np.array([new_box], np.int32)
        cv2.polylines(img, [Pts.reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)

        x_min = int(min([point[0] for point in new_box]))
        y_min = int(min([point[1] for point in new_box]))

        pred_label = node_pred_label[i]
        pred_label = idx_to_cls.get(pred_label, pred_label)
        pred_score = f"{node_pred_score[i]:.2f}"
        text = f"{pred_label}({pred_score})"

        cv2.putText(pred_img, text, (x_min * 2, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    vis_img = np.ones((h, w * 3, 3), dtype=np.uint8) * 255
    vis_img[:, :w] = img
    vis_img[:, w:] = pred_img
    cv2.imwrite(save_path, vis_img)
    print(f"The KIE Image is saved at {save_path}")

def prepare_data(image_path, label_path):
    """
    Prepare data for KIE processing.

    Args:
        image_path (str): Path to the input image.
        label_path (str): Path to the label JSON file.

    Returns:
        dict: Prepared data for processing
    """
    # Read image
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Read label
    with open(label_path, "r") as f:
        label_data = json.load(f)

    # Prepare data dictionary
    data = {
        "img_path": image_path, 
        "label": json.dumps(label_data), 
        "image": image_data
    }
    
    return data

def main():
    """
    Main function to process KIE inference.
    Uses configuration from command-line arguments.
    """
    # Load configuration and preprocess
    config, device, logger, vdl_writer = program.preprocess()
    global_config = config["Global"]

    # Build model
    model = build_model(config["Architecture"])
    load_model(config, model)

    # Create data operators
    transforms = create_operators(config["Eval"]["dataset"]["transforms"], global_config)

    # Read class labels
    class_path = global_config.get("class_path", "")
    idx_to_cls = read_class_list(class_path) if class_path and os.path.exists(class_path) else {}

    # Prepare data
    image_path = global_config.get("infer_img", "")
    label_path = global_config.get("label_file", "")
    
    if not image_path or not label_path:
        raise ValueError("Both image path and label path must be provided")

    # Prepare data for processing
    data = prepare_data(image_path, label_path)
    
    batch = transform(data, transforms)
    batch_pred = [paddle.to_tensor(np.expand_dims(batch[i], axis=0)) for i in range(len(batch))]

    # Perform inference
    model.eval()
    start_time = time.time()
    node, edge = model(batch_pred)
    node = F.softmax(node, -1)
    print(f"Inference time: {time.time() - start_time:.4f} seconds")

    # Save results
    save_res_path = global_config.get("save_res_path", "./output/results.txt")
    save_kie_path = os.path.join(os.path.dirname(save_res_path), "kie_results")
    os.makedirs(save_kie_path, exist_ok=True)
    save_path = os.path.join(save_kie_path, "result.png")
    draw_kie_result(batch, node, idx_to_cls, save_path)

    # Save text results
    with open(save_res_path, "w") as f:
        max_value, max_idx = paddle.max(node, -1), paddle.argmax(node, -1)
        node_pred_label = max_idx.numpy().tolist()
        node_pred_score = max_value.numpy().tolist()
        
        for i, (label, score) in enumerate(zip(node_pred_label, node_pred_score)):
            pred_label = idx_to_cls.get(label, label)
            f.write(f"Box {i}: Label = {pred_label}, Score = {score:.2f}\n")

if __name__ == "__main__":
    main()