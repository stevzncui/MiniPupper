#!/usr/bin/env python3
# GREEN = go, YELLOW = slow, RED = stop 
# Publishes ROS 2 Twist to /cmd_vel and logs while moving

import argparse, time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class ColorGo(Node):
    def __init__(self, cam_index=0, topic='/cmd_vel',
                 go_speed=0.15, slow_speed=0.05,
                 show=False, min_area=0.003,        
                 consecutive=5, width=640, height=480,
                 log_interval=0.5):
        super().__init__('traffic_color_controller')

        self.pub = self.create_publisher(Twist, topic, 10)
        self.go_speed = float(go_speed)
        self.slow_speed = float(slow_speed)
        self.show = bool(show)
        self.min_area = float(min_area)
        self.consecutive = int(consecutive)
        self.state = 'STOP'  
        self.last_state = self.state
        self.counts = {'GREEN':0, 'YELLOW':0, 'RED':0, 'NONE':0}
        self.last_publish = 0.0
        self.log_interval = float(log_interval)
        self.last_log = 0.0

        #cam
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open camera index {cam_index}')
        if width > 0:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        if height > 0: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        #kernal morph
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

        # HSV ranges (tweak if your lighting differs)
        self.green_lower  = np.array([35,  80,  60]); self.green_upper  = np.array([85, 255, 255])
        self.yellow_lower = np.array([20, 100, 100]); self.yellow_upper = np.array([35, 255, 255])
        self.red1_lower   = np.array([0,  100,  70]); self.red1_upper   = np.array([10, 255, 255])
        self.red2_lower   = np.array([170,100,  70]); self.red2_upper   = np.array([180,255,255])

        #loop 
        self.timer = self.create_timer(0.05, self.loop_once)

    def loop_once(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().warn('No camera frame; stopping.')
            self.stop_robot()
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_g = cv2.inRange(hsv, self.green_lower,  self.green_upper)
        mask_y = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        mask_r = cv2.inRange(hsv, self.red1_lower,   self.red1_upper) \
               | cv2.inRange(hsv, self.red2_lower,   self.red2_upper)

        # clean the masks
        mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN,  self.kernel, iterations=1)
        mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN,  self.kernel, iterations=1)
        mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN,  self.kernel, iterations=1)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        total_px   = frame.shape[0] * frame.shape[1]
        frac_green = float(np.count_nonzero(mask_g)) / max(1, total_px)
        frac_yel   = float(np.count_nonzero(mask_y)) / max(1, total_px)
        frac_red   = float(np.count_nonzero(mask_r)) / max(1, total_px)

        # observation choice
        if frac_red >= self.min_area:
            obs = 'RED'
        elif frac_yel >= self.min_area:
            obs = 'YELLOW'
        elif frac_green >= self.min_area:
            obs = 'GREEN'
        else:
            obs = 'NONE'

        # counting the frames in the observation
        for k in self.counts: self.counts[k] = self.counts[k] + 1 if k == obs else 0

        # the transition states red-> yellow-> green
        new_state = self.state
        if self.counts['RED'] >= self.consecutive:
            new_state = 'STOP'
        elif self.counts['YELLOW'] >= self.consecutive:
            new_state = 'SLOW'
        elif self.counts['GREEN'] >= self.consecutive:
            new_state = 'GO'
        elif self.counts['NONE'] >= self.consecutive:
            new_state = 'STOP'

        if new_state != self.state:
            self.state = new_state
            self.get_logger().info(f"STATE -> {self.state} (g={frac_green:.3f} y={frac_yel:.3f} r={frac_red:.3f})")


        if self.state == 'GO':
            self.move_robot(self.go_speed, say='GREEN')
        elif self.state == 'SLOW':
            self.move_robot(self.slow_speed, say='YELLOW')
        else:
            self.stop_robot()

        if self.show:
            overlay = frame.copy()
            cv2.putText(overlay, f"State: {self.state}  g={frac_green:.3f} y={frac_yel:.3f} r={frac_red:.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
            small_g = cv2.resize(mask_g, (0,0), fx=0.25, fy=0.25)
            small_y = cv2.resize(mask_y, (0,0), fx=0.25, fy=
