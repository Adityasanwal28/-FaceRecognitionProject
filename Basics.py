import cv2
import face_recognition


# Function to load and convert an image to RGB format
def load_and_convert_image(image_path):
    img = face_recognition.load_image_file(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Function to detect and encode a face in an image
def detect_and_encode_face(image):
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        raise Exception("No face detected in the image.")

    # Encode the detected face
    encode = face_recognition.face_encodings(image, [face_locations[0]])[0]
    return face_locations[0], encode


# Function to compare two face encodings and calculate the face distance
def compare_faces(known_face_encode, test_face_encode):
    inner_results = face_recognition.compare_faces([known_face_encode], test_face_encode)
    face_distance = face_recognition.face_distance([known_face_encode], test_face_encode)
    return inner_results, face_distance[0]


if __name__ == "__main__":
    try:
        # Load and convert images
        imgElon = load_and_convert_image('ImagesBasic/Elon Musk.jpg')
        imgTest = load_and_convert_image('ImagesBasic/Bill Gates.jpg')

        # Detect and encode faces in the images
        faceLocElon, encodeElon = detect_and_encode_face(imgElon)
        faceLocTest, encodeTest = detect_and_encode_face(imgTest)

        # Draw rectangles around detected faces
        cv2.rectangle(imgElon, (faceLocElon[3], faceLocElon[0]), (faceLocElon[1], faceLocElon[2]), (255, 0, 255), 2)
        cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

        # Compare the faces and display the result
        results, faceDis = compare_faces(encodeElon, encodeTest)
        print(f"Are they the same person? {results[0]} (Face distance: {round(faceDis, 2)})")

        # Display the result on the image
        cv2.putText(imgTest, f"Are they the same person? {results[0]}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 0, 255), 2)

        # Display the images
        cv2.imshow('Elon Musk', imgElon)
        cv2.imshow('Elon Test', imgTest)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")
