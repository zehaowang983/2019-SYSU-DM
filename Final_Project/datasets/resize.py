
from PIL import Image
import os.path
import glob
def convertpng(pngfile,outdir,width=64,height=64):
    img=Image.open(pngfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)   
        new_img.save(os.path.join(outdir,os.path.basename(pngfile)))
    except Exception as e:
        print(e)

if __name__ == '__main__':
    for pngfile in glob.glob("crop/*.jpg"):
        convertpng(pngfile,"./resized")