import cv2

print("Starting camera test...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    input("Press ENTER to exit...")
    exit()

print("Camera opened successfully.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()