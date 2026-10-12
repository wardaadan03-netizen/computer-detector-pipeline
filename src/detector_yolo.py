"""YOLOv8 wrapper for object detection."""
from ultralytics import YOLO 
import numpy as np 

class YOLODetector:
    def __init__(self,cfg:dict):
        self.model=YOLO(cfg["model_path"])
        self.conf=cfg["conf_threshold"]
        self.iou=cfg["iou_threshold"]
        self.device=cfg["device"]
        self.classes=cfg["classes"]
    
    def infer(self,frame:np.ndarray):
        """Return list of dicts:{bbox,conf,cls_id,cls_name}."""
        results=self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )[0]

        detections=[]
        names=results.names
        for box in results.boxes:
            xyxy=box.xyxy[0].cpu().numpy().astype(int)
            conf=float(box.conf[0].cpu().numpy())
            cls_id=int(box.cls[0].cpu().numpy())
            detections.append({
                "bbox":xyxy,
                "conf":conf,
                "cls_id":cls_id,
                "cls_name":names[cls_id],
            })

        return detections