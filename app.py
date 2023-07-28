import streamlit as st
from mtranslate import translate
from PIL import Image, ImageDraw, ImageFont
import sys
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import io
import pdf2image
import os
import PIL.Image
import os
from fpdf import FPDF
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
model =ocr_predictor(pretrained=True)


class BackgroundColorDetector():
    def __init__(self, imageLoc):
        self.img = cv2.imread(imageLoc, 1)
        self.manual_count = {}
        self.w, self.h, self.channels = self.img.shape
        self.total_pixels = self.w*self.h

    def count(self):
        for y in range(0, self.h):
            for x in range(0, self.w):
                RGB = (self.img[x, y, 2], self.img[x, y, 1], self.img[x, y, 0])
                if RGB in self.manual_count:
                    self.manual_count[RGB] += 1
                else:
                    self.manual_count[RGB] = 1

    def average_colour(self):
        red = 0
        green = 0
        blue = 0
        sample = 10
        for top in range(0, sample):
            red += self.number_counter[top][0][0]
            green += self.number_counter[top][0][1]
            blue += self.number_counter[top][0][2]

        average_red = red / sample
        average_green = green / sample
        average_blue = blue / sample

        return (average_red, average_green, average_blue)

    def twenty_most_common(self):
        self.count()
        self.number_counter = Counter(self.manual_count).most_common(20)


    def detect(self):
        self.twenty_most_common()
        self.percentage_of_first = (
            float(self.number_counter[0][1])/self.total_pixels)

        if self.percentage_of_first > 0.5:

            return self.number_counter[0][0]
        else:
            return self.average_colour()


def convert_pdf_to_images(pdf_file):
    pages = pdf2image.convert_from_path(pdf_file)
    image_paths = []

    for i, page in enumerate(pages):
        filename = f"page_{i}.png"
        page.save(filename)
        image_paths.append(filename)

    return image_paths

def convert_images_to_pdf(image_paths, output_pdf):
    pdf = FPDF()

    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size

            # Assuming letter-sized portrait orientation (8.5 x 11 inches)
            pdf.add_page(orientation='P')
            pdf.image(image_path, x=0, y=0, w=210, h=297)  # Adjust w and h as needed

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    pdf.output(output_pdf)
    print(f"PDF created: {output_pdf}")



def get_number_of_pages(pdf_file):
    """
    Find the number of pages in a PDF file.

    Args:
        pdf_file (str): The path to the PDF file.

    Returns:
        int: The number of pages in the PDF file.
    """

    with open(pdf_file, "rb") as f:
        contents = f.read()

    pages = pdf2image.convert_from_bytes(contents)

    return len(pages)

def overlay_strip(image, bg_color, x1,y1,x2,y2):
  """Overlays the strip on the image.

  Args:
    image: The image to overlay the strip on.
    strip: The strip to overlay.
    top_left: The top-left corner of the overlay region.
    width: The width of the overlay region.
    height: The height of the overlay region.

  Returns:
    The image with the strip overlayed.
  """

  image = cv2.rectangle(image,(x1,y1),(x2,y2),(int(bg_color[0]),int(bg_color[1]),int(bg_color[2])),-1)

  return image

def perform_ocr(filename):
  doc = DocumentFile.from_images(filename)
# Analyze
  result = model(doc)
  img = cv2.imread(filename)
  json_output = result.export()
  words = []
  geometry = []
  blocks = []
  bboxes = []
  for i in range(len(json_output["pages"][0]["blocks"])):
    for j in range(len(json_output["pages"][0]["blocks"][i]["lines"])):
      for k in range(len(json_output["pages"][0]["blocks"][i]["lines"][j]["words"])):
        words.append(json_output["pages"][0]["blocks"][i]["lines"][j]["words"][int(k)]["value"])
        geometry.append(json_output["pages"][0]["blocks"][i]["lines"][j]["words"][int(k)]["geometry"])
  h = json_output["pages"][0]["dimensions"][0]
  w = json_output["pages"][0]["dimensions"][1]
  for bbox in geometry:
    bboxes.append([bbox[0][0]*w,bbox[0][1]*h,bbox[1][0]*w,bbox[1][1]*h])
  return words,bboxes,json_output

def remove_text(filename):
  img = cv2.imread(filename)
  words,bboxes,json_output = perform_ocr(filename)

  for i in range(len(bboxes)):
    x1 = int(bboxes[i][0])
    y1 = int(bboxes[i][1])
    x2 = int(bboxes[i][2])
    y2 = int(bboxes[i][3])
    print(words[i])
    cropped_img = img[y1:y2,x1:x2]
    cv2.imwrite("cropped_img.png",cropped_img)
    BackgroundColor = BackgroundColorDetector("cropped_img.png")
    os.remove("cropped_img.png")
    bg_color = BackgroundColor.detect()
    img = overlay_strip(img,bg_color,x1,y1,x2,y2)
  cv2.imwrite("img_without_txt.png",img)
  return img,json_output





def translate_sentence(text, target_language='es'):
    

    translated_text = translate(text, 'es', 'en')
    return translated_text

    # Return the translated text
    

def find_lines_and_top_left(json_output):
  lines = []
  top_left = []
  h = json_output["pages"][0]["dimensions"][0]
  w = json_output["pages"][0]["dimensions"][1]
  for i in range(len(json_output["pages"][0]["blocks"])):

    for j in range(len(json_output["pages"][0]["blocks"][i]["lines"])):
      string = ""
      for k in range(len(json_output["pages"][0]["blocks"][i]["lines"][j]["words"])):
        if k==0:
          top_left.append([json_output["pages"][0]["blocks"][i]["lines"][j]["words"][k]["geometry"][0][0]*w,json_output["pages"][0]["blocks"][i]["lines"][j]["words"][k]["geometry"][0][1]*h])
        string+=json_output["pages"][0]["blocks"][i]["lines"][j]["words"][k]["value"] + " "
      lines.append(string)
  return lines,top_left

def put_text_on_image(image_path,json_output,iter,font_size=20,text_color=(0,0,0)):
  translated_lines = []
  lines,top_left = find_lines_and_top_left(json_output)
  image = Image.open("img_without_txt.png")

    # Create a drawing object
  draw = ImageDraw.Draw(image)

    # Set the font

  font = ImageFont.truetype("arial.ttf", font_size)


    # Add the text to the image

  for line in lines:
    translated_lines.append(translate_sentence(line))
  for i in range(len(translated_lines)):
    draw.text((int(top_left[i][0]),int(top_left[i][1])), translated_lines[i], text_color,font)
  filename = f"final_img_{iter}.png"
  image.save(filename)
  # final_img = cv2.imread("final_img.png")
  # return final_img
  return filename

def final_output(input_filename,output_filename):
  convert_pdf_to_images(input_filename,"/content/")
  image_paths = []
  for i in range(get_number_of_pages(input_filename)):
    img_without_txt,json_output = remove_text(f"page_{i}.png")

    final_img = put_text_on_image("img_without_txt.png",json_output,i)
  
  for file in os.listdir():
    if file.startswith("final_img"):
      image_paths.append(file)
  convert_images_to_pdf(image_paths,output_filename)


def main():
    st.title("Document Translation with Background Removal")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        # Process the PDF file
        with open("input.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write("Converting PDF to images...")
        image_paths = convert_pdf_to_images("input.pdf")

        st.write("Removing text from images...")
        final_images = []
        for i,image_path in enumerate(image_paths):
            img_without_txt, json_output = remove_text(image_path)
            final_img = put_text_on_image(img_without_txt, json_output,i)
            final_images.append(final_img)

        st.write("Converting images back to PDF...")
        convert_images_to_pdf(final_images, "output.pdf")

        st.success("Translation and background removal completed!")
        if st.button("Download Translated PDF"):
            with open("output.pdf", "rb") as f:
                pdf_data = f.read()
            st.download_button(label="Click here to download", data=pdf_data, file_name="output.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
