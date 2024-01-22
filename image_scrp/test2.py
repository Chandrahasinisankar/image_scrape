from PIL import Image, ExifTags
from test1 import detect_objects, get_location_info
import pyttsx3

def extract_exif_tags(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()

        if exif_data is not None:
            exif_tags = {ExifTags.TAGS[key]: exif_data[key] for key in exif_data.keys() if key in ExifTags.TAGS}
            return exif_tags
        else:
            print("No EXIF data found in the image.")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    speak_text("Enter the image path:")
    image_path = input("Enter the image path: ")
    
    speak_text("The information scrapped from the given image file is:")
    
    detected_objects_result = detect_objects(image_path)
    location_info_result = get_location_info(image_path)
    exif_tags = extract_exif_tags(image_path)

    if exif_tags:
        for tag, value in exif_tags.items():
            print(f"{tag}: {value}")

if __name__ == "__main__":
    main()