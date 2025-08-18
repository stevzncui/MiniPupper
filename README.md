1. Traffic Light Color Controller (traffic_color_go.py)

This script uses a USB webcam to detect traffic light colors (green, yellow, red) and control the robot:

Green → the robot moves forward

Yellow → the robot slows down

Red or no light → the robot stops

It publishes velocity commands on the ROS 2 topic /cmd_vel.
You can run it headless over SSH, or enable a preview window to see the camera feed.

2. Black Line Follower (black_line_follower.py)

This script is intended to make the robot follow a black line on the ground using a webcam and a PID controller.

It looks at the bottom region of the camera image.

If it finds the line, it steers the robot to stay centered.

If it loses the line, it should stop and slowly rotate to search for it again.

Note: At the moment, this script is not working correctly. It still needs fixes before it can reliably follow a line.

3. Gemini Text-to-Speech Loop (gemini_tts.py)

This script connects to the Google Gemini API and uses gTTS (Google Text-to-Speech) to make the robot speak.

It randomly picks prompts (like “tell me a joke” or “give me a fun fact”).

Gemini generates a response.

The response is converted to speech and played out loud.

Every 20 seconds, it repeats with a new random prompt.
