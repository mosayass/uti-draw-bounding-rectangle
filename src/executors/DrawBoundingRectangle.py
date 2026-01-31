
import os
import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.DrawBoundingRectangle.src.utils.response import build_response
from components.DrawBoundingRectangle.src.models.PackageModel import PackageModel


class DrawBoundingRectangle(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.detections = self.request.get_param("inputDetections")
        self.color_axis = self.request.get_param("ConfigColorAxis")
        self.load_parameters()
        self.load_colors()

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {"colors": [], "palette_name": [], "palette": [], "color_map": {}}

    def load_parameters(self) -> None:
        self.colors = self.bootstrap["colors"]
        self.palette_name = self.request.get_param("ConfigColorPalette")
        if self.palette_name == "custom":
            self.palette = self.request.get_param("configCustomColors")
        else:
            self.palette = self.request.get_param("ConfigPaletteSize")
        self.config_thickness = self.request.get_param("ConfigThickness")
        self.radius = self.request.get_param("ConfigRadius")

    def load_colors(self) -> None:
        if self.colors is None or self.palette_name != self.bootstrap["palette_name"] or self.palette != self.bootstrap["palette"]:
            self.colors = []
            self.bootstrap["color_map"] = {}
            if self.palette_name == "custom":
                for color_str in self.palette.split(','):
                    clean_hex = color_str.strip().lstrip('#')

                    if len(clean_hex) == 6:
                        bgr = (
                            int(clean_hex[4:6], 16),
                            int(clean_hex[2:4], 16),
                            int(clean_hex[0:2], 16)
                        )
                        self.colors.append(bgr)
            else:
                cmap = plt.get_cmap(self.palette_name).colors
                self.colors = [
                    tuple(int(255 * r) for r in cmap[i % len(cmap)])
                    for i in range(self.palette)
                ]
        self.bootstrap["colors"] = self.colors
        self.bootstrap["palette_name"] = self.palette_name
        self.bootstrap["palette"] = self.palette

    def select_color(self):
        num_items = len(self.detections)
        color_map = self.bootstrap["color_map"]
        if self.color_axis == "Class":
            class_ids = [det["classId"] for det in self.detections]
            color_dict = {}
            for class_id in class_ids:
                if class_id not in color_map:
                    next_index = len(color_map) % len(self.colors)
                    color_map[class_id] = self.colors[next_index]
                color_dict[class_id] = color_map[class_id]
            self.bootstrap["color_map"] = color_map

        elif self.color_axis == "Index":
            color_dict = {idx: self.colors[idx % len(self.colors)] for idx in range(num_items)}

        else:
            track_ids = [det["trackerID"] for det in self.detections]
            color_dict = {}
            for track_id in track_ids:
                if track_id not in color_map:
                    next_index = len(color_map) % len(self.colors)
                    color_map[track_id] = self.colors[next_index]
                color_dict[track_id] = color_map[track_id]

            self.bootstrap["color_map"] = color_map

        return color_dict

    def draw_bounding_Rectangle(self, image, color_dict):
        for idx, detection in enumerate(self.detections):
            if not detection["boundingRectangle"]:
                continue

            left = detection["boundingRectangle"]["left"]
            top = detection["boundingRectangle"]["top"]
            width = detection["boundingRectangle"]["width"]
            height = detection["boundingRectangle"]["height"]

            x1 = int(left)
            y1 = int(top)
            w_int = int(width)
            h_int = int(height)

            if self.color_axis == "Class":
                color = color_dict[detection["classId"]]
            elif self.color_axis == "Index":
                color = color_dict[idx]
            else:
                color = color_dict[detection.get("trackerID", 0)]

            angle = detection.get("angle")

            if angle is not None:  #  OBB
                angle_deg = math.degrees(angle)
                center_x = left + width / 2.0
                center_y = top + height / 2.0
                rect = ((center_x, center_y), (width, height), angle_deg)
                Rectangle = cv2.RectanglePoints(rect)
                Rectangle = np.int0(Rectangle)
                cv2.drawContours(image, [Rectangle], 0, color, self.config_thickness)
            else:
                if self.radius > 0:
                    cv2.ellipse(image, (x1 + self.radius, y1 + self.radius), (self.radius, self.radius), 180, 0, 90, color, self.config_thickness)
                    cv2.ellipse(image, (x1 + w_int - self.radius, y1 + self.radius), (self.radius, self.radius), 270, 0, 90, color, self.config_thickness)
                    cv2.ellipse(image, (x1 + self.radius, y1 + h_int - self.radius), (self.radius, self.radius), 90, 0, 90, color, self.config_thickness)
                    cv2.ellipse(image, (x1 + w_int - self.radius, y1 + h_int - self.radius), (self.radius, self.radius), 0, 0, 90, color, self.config_thickness)
                    cv2.line(image, (x1 + self.radius, y1), (x1 + w_int - self.radius, y1), color, self.config_thickness)
                    cv2.line(image, (x1 + self.radius, y1 + h_int), (x1 + w_int - self.radius, y1 + h_int), color, self.config_thickness)
                    cv2.line(image, (x1, y1 + self.radius), (x1, y1 + h_int - self.radius), color, self.config_thickness)
                    cv2.line(image, (x1 + w_int, y1 + self.radius), (x1 + w_int, y1 + h_int - self.radius), color, self.config_thickness)
                else:
                    x2 = int(x1 + w_int)
                    y2 = int(y1 + h_int)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, self.config_thickness)

        return image

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img.value = self.draw_bounding_Rectangle(img.value, self.select_color())
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        packageModel = build_response(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()