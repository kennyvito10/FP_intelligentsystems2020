#Dog Classification App
#Created By : Klementinus Kennyvito Salim, Wely Dharma Putra, Alessandro Christopher Leonardo
from csv import DictReader
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
import webbrowser
from tensorflow.keras.models import Model

with open('d.csv', 'r') as read_obj:
    dict_reader = DictReader(read_obj)
    list_of_dict = list(dict_reader)
# Recreate the exact same model, including its weights and the optimizer
model = load_model('2020-06-15_dog_breed_model.h5')

df = pd.read_csv('dog/labels.csv')
selected_breed_list = list(df.groupby('breed').count().sort_values(by='id', ascending=False).head(120).index)
df = df[df['breed'].isin(selected_breed_list)]
df['filename'] = df.apply(lambda x: ('train/' + x['id'] + '.jpg'), axis=1)

breeds = pd.Series(df['breed'])
#print("total number of breeds to classify",len(breeds.unique()))

df.head()

# Check its architecture
#model.summary()
def predict_from_image(img_path):
    global acc

    img = image.load_img(img_path, target_size=(299, 299))
    tensor_image = image.img_to_array(img)                    # (height, width, channels)
    tensor_image = np.expand_dims(tensor_image, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    tensor_image /= 255.                                      # imshow expects values in the range [0, 1]

    pred = model.predict(tensor_image)
    breedlist = sorted(selected_breed_list)
    predicted_class = breedlist[np.argmax(pred)]
    acc=np.max(pred)
    print(predicted_class)
    print(acc)
    return predicted_class

root = Tk()
root.geometry("1920x1080")
root.title('Dog Breed Classification App')
l2 = Label(root, text = "This is A Dog Breed Classification App")
l4 = Label(root, text = "You can know a dog's breed just by Uploading an Image\n To use: Just Press the button below and upload the image")
var_chk = IntVar()
l2.config(font=(30))

l2.grid(row=0)
l4.grid(row=1)



#Text box
txt = Text(root , width = 40 , height = 15 , wrap = WORD)
txt.grid(row = 5 ,column = 0, sticky = W, padx=550,pady=30)
def open():
    global myimage
    global result
    root.filename = filedialog.askopenfilename(initialdir="C:/Users/ASUS/PycharmProjects/Intelligent Systems LAB/dc/testdata", title="Select an Image",filetypes=(("jpg","*.jpg"),("png","*.png"),("All Files","*.*")))
    im = Image.open(root.filename)
    im = im.resize((250, 250), Image.ANTIALIAS)
    myimage = ImageTk.PhotoImage(im)
    #myimage = ImageTk.PhotoImage(Image.open(root.filename))
    imagelabel = Label(image=myimage)
    imagelabel.grid(row = 2,column =0)
    result = str(predict_from_image(root.filename))
    result = result.replace('_','-')
    for item in list_of_dict:
        if item['dog'] == result:
            my_item = item
            break
    else:
        my_item = None
    btn2 = Button(root,text="More Info",command=info)
    btn2.grid(row = 6, column = 0)
    txt.delete('1.0', END)
    txt.insert(0.0 , "")
    accuracy = round(float(str(acc))*100,2)

    sentence ="\n\nPrediction Accuracy :" + str(accuracy) + "%"
    txt.insert(0.1,sentence)
    sentence ="\n\nArea of Origin: " + my_item['areaOfOrigin']
    txt.insert(0.1,sentence)
    sentence ="\n\nLife Span: " + my_item['lifespan'] + " years"
    txt.insert(0.1,sentence)
    sentence ="\n\nEstimated Weight: " + my_item['weight'] + " kg"
    txt.insert(0.1,sentence)
    sentence ="\n\nEstimated Height: " + my_item['height'] + " cm"
    txt.insert(0.1,sentence)
    sentence = "Dog Breed Identified: " + result
    txt.insert(0.5,sentence)


def info():

    tab = 'https://dogell.com/en/dog-breed/{}'.format(result)
    webbrowser.open_new_tab(tab)

btn = Button(root,text="Upload Image",command=open)
btn.grid(row = 4, column = 0)

root.mainloop()
