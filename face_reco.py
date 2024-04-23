from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import os
import numpy as np
import face_reco

class Train:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("face recognition system")
        #title
        title_lbl = Label(self.root, text="TRAIN DATASET", font=("times new roman", 35, "bold"), bg="white", fg="black")
        title_lbl.place(x=0, y=0, width=1530, height=45)

        img_top = Image.open(r"images\train_img.jpg")
        img_top = img_top.resize((1530, 700))
        self.photoimg_top = ImageTk.PhotoImage(img_top)

        f_lbl = Label(self.root, image=self.photoimg_top)
        f_lbl.place(x=0, y=5, width=1530, height=700)

        btn = Button(self.root, text="TRAIN DATA", command=self.train_classifier, cursor="hand2", font=("times new roman", 15, "bold"), bg="darkblue", fg="white")
        btn.place(x=0, y=700, width=1534, height=60)

    def train_classifier(self):
        data_dir = "data"
        path = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]

        face_encodings = []
        ids = []

        for image in path:
            img = face_reco.load_image_file(image)
            encoding = face_reco.face_encodings(img)

            if len(encoding) > 0:
                face_encodings.append(encoding[0])
                ids.append(int(os.path.split(image)[1].split('.')[1]))

        if len(face_encodings) == 0:
            messagebox.showerror("Error", "No faces found in the dataset")
            return

        # Save the face encodings and corresponding IDs
        np.save("encodings.npy", np.array(face_encodings))
        np.save("ids.npy", np.array(ids))

        messagebox.showinfo("Result", "Training dataset completed!!!")

if __name__ == "__main__":
    root = Tk()
    obj = Train(root)
    root.mainloop()
