import cv2
import os

# Load the cascade
carPlatesCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

# Load the video
cap = cv2.VideoCapture("carnoplate.mp4")

# Create output folder for storing detected plates
output_folder = "detected_plates"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Check if video is successfully loaded
if not cap.isOpened():
    print('Error reading video')
    exit()

frame_count = 0  # Counter for saving unique plate images

while True:
    ret, frame = cap.read()  # Read a frame
    if not ret:  # Exit the loop if the video ends
        break

    # Convert the frame to grayscale for plate detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect car plates in the frame
    car_plates = carPlatesCascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(25, 25)
    )

    # Process detected car plates
    for (x, y, w, h) in car_plates:
        # Draw a rectangle around the detected plate
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the detected plate region
        plate = frame[y: y + h, x: x + w]

        # Save the extracted plate as an image
        file_name = os.path.join(output_folder, f"plate_{frame_count}.jpg")
        cv2.imwrite(file_name, plate)
        frame_count += 1

    # Show the frame
    cv2.imshow('Video', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
