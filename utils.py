from PIL import Image
import glob

for file in glob.glob('./data/pokemon/*.png'):
    img = Image.open(file)
    jpg = img.convert('RGB')
    # print(file)
    jpg.save('./data/' + file.split('\\')[-1].split('.')[0] + '.jpg')


