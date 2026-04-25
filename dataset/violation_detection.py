import cv2
from ultralytics import YOLO
import os

print("Starting FINAL VIOLATION DETECTION...")

# Load models
vehicle_model = YOLO("runs/detect/train2/weights/best.pt")
light_model = YOLO("yolov8n.pt")

# Folders
video_folder = "../test_videos"
output_folder = "../test_output"

os.makedirs(output_folder, exist_ok=True)

for video_file in os.listdir(video_folder):

    if not (video_file.endswith(".avi") or video_file.endswith(".mp4")):
        continue

    print(f"\nProcessing: {video_file}")

    cap = cv2.VideoCapture(os.path.join(video_folder, video_file))

    width, height = 480, 360
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    out = cv2.VideoWriter(
        os.path.join(output_folder, f"output_{video_file}"),
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (width, height)
    )

    line_y = 200

    vehicle_positions = {}
    violated_ids = set()
    violation_count = 0

    frame_count = 0
    last_vehicle_results = None
    red_light = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Finished: {video_file}")
            break

        frame_count += 1
        frame = cv2.resize(frame, (width, height))

        # Run detection every 10 frames
        if frame_count % 10 == 0:

            # Vehicle tracking
            last_vehicle_results = vehicle_model.track(frame, persist=True)

            # Traffic light detection
            light_results = light_model(frame)
            red_light = False

            for r in light_results:
                for box in r.boxes:
                    cls = int(box.cls[0])

                    if light_model.names[cls] == "traffic light":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # -------- RED LIGHT DETECTION --------
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        crop = cv2.resize(crop, (50, 100))
                        crop = cv2.GaussianBlur(crop, (5,5), 0)

                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        brightness = hsv[:, :, 2].mean()

                        red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
                        red_mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
                        green_mask = cv2.inRange(hsv, (40, 40, 40), (90, 255, 255))

                        red_pixels = cv2.countNonZero(red_mask1 + red_mask2)
                        green_pixels = cv2.countNonZero(green_mask)

                        # Adaptive logic
                        if brightness < 80:
                            red_light = red_pixels > 5
                        else:
                            red_light = (red_pixels > green_pixels) and (red_pixels > 15)

                        # Draw light result
                        if red_light:
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                            cv2.putText(frame, "RED LIGHT", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                        else:
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                            cv2.putText(frame, "NOT RED", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                        # Debug
                        cv2.putText(frame, f"R:{red_pixels} G:{green_pixels} B:{int(brightness)}",
                                    (x1, y2+20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        # -----------------------------------

        # Draw stop line
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0,255,255), 2)

        # Process vehicles
        if last_vehicle_results is not None:
            for r in last_vehicle_results:

                # Handle missing IDs
                if r.boxes.id is None:
                    ids = [None] * len(r.boxes)
                else:
                    ids = r.boxes.id

                for box, track_id in zip(r.boxes, ids):

                    if track_id is None:
                        continue

                    track_id = int(track_id)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_y = (y1 + y2) // 2

                    # Draw vehicle
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1,y1-25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

                    # -------- CORRECT VIOLATION LOGIC --------
                    prev_y = vehicle_positions.get(track_id, None)
                    vehicle_positions[track_id] = center_y

                    if (
                        prev_y is not None and
                        prev_y < line_y + 20 and
                        center_y >= line_y and
                        red_light and
                        track_id not in violated_ids
                    ):
                        violation_count += 1
                        violated_ids.add(track_id)

                        cv2.putText(frame, "VIOLATION!", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    # ----------------------------------------

        # Display info
        cv2.putText(frame, f"Violations: {violation_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.putText(frame, f"Red: {red_light}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        out.write(frame)

    cap.release()
    out.release()

cv2.destroyAllWindows()

print("\nAll videos processed SUCCESSFULLY!")