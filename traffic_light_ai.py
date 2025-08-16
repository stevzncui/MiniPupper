#!/usr/bin/env python3
# traffic_light_ai.py
# Detects a traffic light with TFLite SSD; classifies RED/YELLOW/GREEN in ROI (HSV).
# Publishes ROS 2 Twist to /cmd_vel. Prints "GREEN" periodically while moving (headless-friendly).

import argparse, time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from tflite_runtime.interpreter import Interpreter


# ----------------- helpers -----------------
def load_labels(path):
    with open(path, 'r') as f:
        return [l.strip() for l in f.readlines()]

def preprocess(frame, input_size):
    ih, iw = input_size
    img = cv2.resize(frame, (iw, ih))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)[None, ...]  # [1,H,W,3]

def classify_light_color(bgr_roi):
    """Return 'RED'|'YELLOW'|'GREEN'|'UNKNOWN' via HSV with vertical-band emphasis."""
    if bgr_roi is None or bgr_roi.size == 0:
        return 'UNKNOWN'
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    H, W = hsv.shape[:2]
    band_h = max(4, H // 3)

    # HSV masks
    red1   = cv2.inRange(hsv,  (0,   100, 70),  (10,  255, 255))
    red2   = cv2.inRange(hsv,  (170, 100, 70),  (180, 255, 255))
    yellow = cv2.inRange(hsv,  (20,  100, 100), (35,  255, 255))
    green  = cv2.inRange(hsv,  (35,  80,  60),  (85,  255, 255))

    def frac(mask):  # fraction of mask pixels
        return float(np.count_nonzero(mask)) / max(1, mask.size)

    score_red    = 1.1*frac(red1[0:band_h, :]) + 1.1*frac(red2[0:band_h, :]) + 0.4*frac(red1) + 0.4*frac(red2)
    score_yellow = 1.2*frac(yellow[H//3:2*H//3, :]) + 0.4*frac(yellow)
    score_green  = 1.2*frac(green[-band_h:, :]) + 0.4*frac(green)

    scores = {'RED': score_red, 'YELLOW': score_yellow, 'GREEN': score_green}
    label = max(scores, key=scores.get)
    return label if scores[label] >= 0.02 else 'UNKNOWN'


# ----------------- node -----------------
class TrafficLightAIBot(Node):
    def __init__(self, cam_index, model_path, labels_path, conf_thresh,
                 go_speed, slow_speed, min_box, topic, show,
                 log_interval, detect_every, width, height, best_effort):
        super().__init__('traffic_light_ai_bot')

        # QoS (some bases expect BEST_EFFORT)
        qos = QoSProfile(depth=10)
        if best_effort:
            qos.reliability = QoSReliabilityPolicy.BEST_EFFORT

        # Publisher
        self.pub = self.create_publisher(Twist, topic, qos)

        # Params
        self.go_speed = float(go_speed)
        self.slow_speed = float(slow_speed)
        self.min_box = float(min_box)
        self.conf_thresh = float(conf_thresh)
        self.show = bool(show)
        self.log_interval = float(log_interval)
        self.detect_every = max(1, int(detect_every))
        self.state = 'STOP'
        self.last_publish = 0.0
        self.last_log = 0.0
        self.frame_id = 0
        self.last_bbox = None  # (x1,y1,x2,y2)

        # Camera (prefer V4L2 on headless)
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open camera index {cam_index}')
        if width > 0:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        if height > 0: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # TFLite
        self.labels = load_labels(labels_path)
        self.interpreter = Interpreter(model_path=model_path, num_threads=2)
        self.interpreter.allocate_tensors()
        self.inp_index = self.interpreter.get_input_details()[0]['index']
        outs = self.interpreter.get_output_details()
        # Typical SSD Mobilenet v1 output order:
        self.out_boxes   = outs[0]['index']  # [1,N,4]
        self.out_classes = outs[1]['index']  # [1,N]
        self.out_scores  = outs[2]['index']  # [1,N]
        self.out_count   = outs[3]['index']  # [1]

        _, self.inp_h, self.inp_w, _ = self.interpreter.get_input_details()[0]['shape']

        # Loop timer
        self.timer = self.create_timer(0.05, self.loop_once)  # ~20 Hz

    def loop_once(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn('No camera frame; stopping.')
            self.stop_robot()
            return
        h, w = frame.shape[:2]

        # Detect every N frames for speed
        run_detect = (self.frame_id % self.detect_every == 0) or (self.last_bbox is None)
        if run_detect:
            img = preprocess(frame, (self.inp_h, self.inp_w))
            self.interpreter.set_tensor(self.inp_index, img)
            self.interpreter.invoke()
            boxes   = self.interpreter.get_tensor(self.out_boxes)[0]
            classes = self.interpreter.get_tensor(self.out_classes)[0].astype(int)
            scores  = self.interpreter.get_tensor(self.out_scores)[0]
            count   = int(self.interpreter.get_tensor(self.out_count)[0])

            self.last_bbox = None
            for i in range(count):
                sc = float(scores[i])
                if sc < self.conf_thresh:
                    continue
                cls_id = classes[i]
                name = self.labels[cls_id] if 0 <= cls_id < len(self.labels) else ''
                if 'traffic light' not in name.lower():
                    continue
                ymin, xmin, ymax, xmax = boxes[i]
                x1, y1 = int(xmin * w), int(ymin * h)
                x2, y2 = int(xmax * w), int(ymax * h)
                bw, bh = max(0, x2 - x1), max(0, y2 - y1)
                if bw * bh < self.min_box * (w * h):
                    continue
                self.last_bbox = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
                break  # first good detection

        # Classify color inside the box
        color = 'UNKNOWN'
        if self.last_bbox is not None:
            x1, y1, x2, y2 = self.last_bbox
            roi = frame[y1:y2, x1:x2]
            color = classify_light_color(roi)

        # State machine
        if color == 'GREEN':
            self.set_state('GO', 'green detected')
        elif color == 'YELLOW':
            self.set_state('SLOW', 'yellow detected')
        elif color == 'RED':
            self.set_state('STOP', 'red detected')
        else:
            self.set_state('STOP', 'no/unknown light')

        # Actuate + optional preview/logs
        self.actuate()

        if self.show:
            if self.last_bbox is not None:
                x1, y1, x2, y2 = self.last_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"State={self.state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.imshow('traffic_light_ai', frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                rclpy.shutdown()

        self.frame_id += 1

    def set_state(self, new_state, reason=''):
        if new_state != self.state:
            self.state = new_state
            self.get_logger().info(f"STATE -> {self.state} ({reason})")

    def actuate(self):
        now = time.time()
        if now - self.last_publish < 0.03:
            return
        msg = Twist()
        if self.state == 'GO':
            msg.linear.x = self.go_speed
        elif self.state == 'SLOW':
            msg.linear.x = self.slow_speed
        else:
            msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.pub.publish(msg)
        self.last_publish = now
        # print "GREEN" periodically while moving
        if self.state == 'GO' and (now - self.last_log) >= self.log_interval:
            print("GREEN", flush=True)
            self.last_log = now

    def stop_robot(self):
        self.pub.publish(Twist())

    def destroy_node(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        super().destroy_node()


# ----------------- main -----------------
def main():
    parser = argparse.ArgumentParser(description="AI traffic light controller (TFLite SSD + HSV).")
    parser.add_argument('--camera', type=int, default=0, help='Webcam index')
    parser.add_argument('--model', type=str, default='detect.tflite', help='TFLite model path')
    parser.add_argument('--labels', type=str, default='labelmap.txt', help='Label map path')
    parser.add_argument('--conf', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--min-box', type=float, default=0.002, help='Min bbox area fraction of frame')
    parser.add_argument('--topic', type=str, default='/cmd_vel', help='Twist topic')
    parser.add_argument('--go-speed', type=float, default=0.15, help='Speed when GREEN (m/s)')
    parser.add_argument('--slow-speed', type=float, default=0.05, help='Speed when YELLOW (m/s)')
    parser.add_argument('--show', type=int, default=0, help='Try to open a window (0 for SSH)')
    parser.add_argument('--log-interval', type=float, default=0.5, help='Seconds between printing GREEN')
    parser.add_argument('--detect-every', type=int, default=2, help='Run detector every N frames')
    parser.add_argument('--width', type=int, default=640, help='Camera width (0=leave default)')
    parser.add_argument('--height', type=int, default=480, help='Camera height (0=leave default)')
    parser.add_argument('--best-effort', action='store_true', help='Use BEST_EFFORT QoS')
    args = parser.parse_args()

    rclpy.init()
    node = None
    try:
        node = TrafficLightAIBot(
            cam_index=args.camera,
            model_path=args.model,
            labels_path=args.labels,
            conf_thresh=args.conf,
            go_speed=args.go_speed,
            slow_speed=args.slow_speed,
            min_box=args.min_box,
            topic=args.topic,
            show=bool(args.show),
            log_interval=args.log_interval,
            detect_every=args.detect_every,
            width=args.width,
            height=args.height,
            best_effort=args.best_effort,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
