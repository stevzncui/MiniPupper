#!/usr/bin/env python3
# line_follower.py
# Follow a BLACK line using a USB webcam (Raspberry Pi) and publish ROS 2 /cmd_vel.
# Headless-friendly (no GUI). Uses grayscale + adaptive/otsu thresholding.
#
# Default: moves forward at --speed, steers with PID on line centroid error.
# If the line is lost, it stops and slowly rotates to re-acquire.

import argparse, time, math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class PID:
    def __init__(self, kp=0.9, ki=0.0, kd=0.05, out_min=-1.0, out_max=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.prev_err = 0.0
        self.i_term = 0.0
        self.prev_t = None

    def reset(self):
        self.prev_err = 0.0
        self.i_term = 0.0
        self.prev_t = None

    def step(self, err):
        t = time.time()
        if self.prev_t is None:
            dt = 0.05
        else:
            dt = max(1e-3, t - self.prev_t)
        self.prev_t = t

        self.i_term += self.ki * err * dt
        self.i_term = clamp(self.i_term, self.out_min, self.out_max)

        d = (err - self.prev_err) / dt
        self.prev_err = err

        out = self.kp * err + self.i_term + self.kd * d
        return clamp(out, self.out_min, self.out_max)

class BlackLineFollower(Node):
    def __init__(self, cam_index, topic, speed, max_ang, kp, ki, kd,
                 roi_height_frac, min_area_frac, thresh, show, width, height,
                 best_effort, search_ang, lost_frames_limit):
        super().__init__('black_line_follower')

        # QoS
        qos = QoSProfile(depth=10)
        if best_effort:
            qos.reliability = QoSReliabilityPolicy.BEST_EFFORT

        self.pub = self.create_publisher(Twist, topic, qos)

        # Params
        self.speed = float(speed)
        self.max_ang = float(max_ang)
        self.roi_hf = float(roi_height_frac)
        self.min_area_frac = float(min_area_frac)
        self.thresh_mode = thresh  # 'auto' | 'otsu' | int(0..255)
        self.show = bool(show)
        self.search_ang = float(search_ang)
        self.lost_frames_limit = int(lost_frames_limit)

        # PID on normalized horizontal error (left negative, right positive)
        self.pid = PID(kp=kp, ki=ki, kd=kd, out_min=-1.0, out_max=1.0)

        self.lost_count = 0
        self.last_publish = 0.0

        # Camera (prefer V4L2 on headless)
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open camera index {cam_index}')
        if width > 0:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        if height > 0: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Loop ~20 Hz
        self.timer = self.create_timer(0.05, self.loop_once)

        # Morph kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

        self.get_logger().info('Black line follower started.')

    def loop_once(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().warn('No camera frame; stopping.')
            self.stop_robot()
            return

        H, W = frame.shape[:2]
        roi_h = max(8, int(self.roi_hf * H))
        roi = frame[H - roi_h : H, :]

        # --- Preprocess ---
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        # Threshold to get BLACK line as WHITE blob (invert)
        if self.thresh_mode == 'auto':
            # Adaptive threshold works well under uneven lighting
            bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 10)
        elif self.thresh_mode == 'otsu':
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # fixed integer threshold
            t = int(self.thresh_mode)
            _, bw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)

        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, self.kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        # --- Find largest contour (assume it's the line) ---
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if area > best_area:
                best_area = area
                best = c

        found = False
        cx_norm = 0.0
        if best is not None and best_area >= self.min_area_frac * (bw.shape[0]*bw.shape[1]):
            M = cv2.moments(best)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                # Normalize error: center_x in [-1,1]
                cx_norm = (cx - (W/2)) / (W/2)
                found = True

        if found:
            self.lost_count = 0
            # PID on error (positive means line is to the right -> turn right (positive ang))
            u = self.pid.step(cx_norm)
            ang = clamp(u * self.max_ang, -self.max_ang, self.max_ang)
            self.move(self.speed, ang)
            # Minimal console breadcrumbs
            # print(f"FOLLOW: err={cx_norm:+.3f} ang={ang:+.3f}", flush=True)
        else:
            self.lost_count += 1
            self.pid.reset()
            if self.lost_count > self.lost_frames_limit:
                # Search: rotate slowly to reacquire
                self.move(0.0, self.search_ang)
                # print("SEARCHING...", flush=True)
            else:
                # Brief stop while still unsure
                self.stop_robot()

        if self.show:
            vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
            if found:
                cv2.line(vis, (int(W/2), 0), (int(W/2), bw.shape[0]-1), (0,255,255), 2)
                cx_pix = int((cx_norm*(W/2)) + (W/2))
                cv2.line(vis, (cx_pix, 0), (cx_pix, bw.shape[0]-1), (0,255,0), 2)
            cv2.imshow('roi_bw', vis)
            if (cv2.waitKey(1) & 0xFF) == 27:
                rclpy.shutdown()

    def move(self, vx, wz):
        now = time.time()
        if now - self.last_publish < 0.03:
            return
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.pub.publish(msg)
        self.last_publish = now

    def stop_robot(self):
        self.pub.publish(Twist())

    def destroy_node(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            if self.show:
                cv2.destroyAllWindows()
        except:
            pass
        super().destroy_node()

def main():
    ap = argparse.ArgumentParser(description='Black line follower (webcam -> ROS 2 Twist)')
    ap.add_argument('--camera', type=int, default=0, help='Webcam index')
    ap.add_argument('--topic', type=str, default='/cmd_vel', help='Velocity topic')
    ap.add_argument('--speed', type=float, default=0.12, help='Forward speed (m/s)')
    ap.add_argument('--max-ang', type=float, default=1.2, help='Max angular speed (rad/s)')
    ap.add_argument('--kp', type=float, default=0.9, help='PID Kp')
    ap.add_argument('--ki', type=float, default=0.0, help='PID Ki')
    ap.add_argument('--kd', type=float, default=0.05, help='PID Kd')
    ap.add_argument('--roi-height-frac', type=float, default=0.45, help='Bottom ROI height fraction (0..1)')
    ap.add_argument('--min-area-frac', type=float, default=0.003, help='Min contour area fraction in ROI')
    ap.add_argument('--thresh', default='auto', help='auto | otsu | <0..255 int> for fixed threshold')
    ap.add_argument('--show', type=int, default=0, help='Show debug windows (0 for SSH)')
    ap.add_argument('--width', type=int, default=640, help='Camera width (0=leave)')
    ap.add_argument('--height', type=int, default=480, help='Camera height (0=leave)')
    ap.add_argument('--best-effort', action='store_true', help='Use BEST_EFFORT QoS for publisher')
    ap.add_argument('--search-ang', type=float, default=0.4, help='Search angular speed when line lost (rad/s)')
    ap.add_argument('--lost-frames-limit', type=int, default=6, help='Frames before search starts')
    args = ap.parse_args()

    rclpy.init()
    node = None
    try:
        node = BlackLineFollower(
            cam_index=args.camera, topic=args.topic, speed=args.speed, max_ang=args.max_ang,
            kp=args.kp, ki=args.ki, kd=args.kd, roi_height_frac=args.roi_height_frac,
            min_area_frac=args.min_area_frac, thresh=args.thresh, show=bool(args.show),
            width=args.width, height=args.height, best_effort=args.best_effort,
            search_ang=args.search_ang, lost_frames_limit=args.lost_frames_limit
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
