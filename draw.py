import io
import pickle
import tkinter as tk
from PIL import Image

import numpy as np

from log_methods import predict, loadImage
from neural_methods import feedForward

nn_theta1, nn_theta2, nn_bias1, nn_bias2 = None, None, None, None
nn3_theta1, nn3_theta2, nn3_bias1, nn3_bias2 = None, None, None, None
log3_theta = None

with open('theta_log_l3.pkl', 'rb') as f:
            log3_theta = pickle.load(f)

with open('neural.pkl', 'rb') as f:
            nn_theta1, nn_theta2, nn_bias1, nn_bias2 = pickle.load(f)
with open('neural_3.pkl', 'rb') as f:
            nn3_theta1, nn3_theta2, nn3_bias1, nn3_bias2 = pickle.load(f)


listbox_items = {'Нейронная сеть, lambda := 0':0,'Нейронная сеть, lambda := 3':1, 'Логистическая регрессия, lambda := 3':2}
class ExampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.selected = 1

        self.previous_x = self.previous_y = 0
        self.x = self.y = 0
        self.points_recorded = []

        self.canvas = tk.Canvas(self, width=600, height=600, bg = "white", cursor="cross")
        self.canvas.create_circle = self._create_circle
        self.canvas.pack(side="top", fill="both", expand=True)

        self.button_clear = tk.Button(self, text = "Очистить", command = self.clear_all)
        self.button_clear.pack(side="top", fill="both", expand=True)
        self.button_save = tk.Button(self, text = "Предсказать", command = self.save)
        self.button_save.pack(side="top", fill="both", expand=True)

        self.listbox = tk.Listbox(self, width=40, height=4)
        self.listbox.pack(side="top", fill="both", expand=True)
        for item in listbox_items.keys():
            self.listbox.insert(tk.END, item)

        self.canvas.bind("<Motion>", self.tell_me_where_you_are)
        self.canvas.bind("<B1-Motion>", self.draw_from_where_you_are)
        self.listbox.bind('<<ListboxSelect>>', self.select_function)
    
    def select_function(self, event):
        value = (self.listbox.get(self.listbox.curselection()))
        self.selected = listbox_items[value]
        print(self.selected, value)

    def clear_all(self):
        self.canvas.delete("all")

    def tell_me_where_you_are(self, event):
        self.previous_x = event.x
        self.previous_y = event.y

    def draw_from_where_you_are(self, event):
        self.x = event.x
        self.y = event.y
        #self.canvas.create_line(self.previous_x, self.previous_y, 
         #                       self.x, self.y,fill="black")
        self.canvas.create_circle(self.x, self.y, 20, fill="black", outline="black", width=4)
        
        self.previous_x = self.x
        self.previous_y = self.y

    def _create_circle(self, x, y, r, **kwargs):
        return self.canvas.create_oval(x-r, y-r, x+r, y+r, **kwargs)

    def save(self):
        ps = self.canvas.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img.save('num.jpg')
        img = loadImage('num.jpg', True)
        if self.selected == 0:
            _, _, a = feedForward(img, nn_theta1, nn_theta2, nn_bias1, nn_bias2)
            y = np.argmax(a, axis=1)
            print(y)
        if self.selected == 1:
            _, _, a = feedForward(img, nn3_theta1, nn3_theta2, nn3_bias1, nn3_bias2)
            y = np.argmax(a, axis=1)
            print(y)
        if self.selected == 2:
            print(predict(img, log3_theta, True))
            #print(self.theta.shape, img.shape)


if __name__ == "__main__":
    app = ExampleApp()
    app.mainloop()