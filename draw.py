import io
import pickle
import tkinter as tk
from PIL import Image

from log_methods import predict, loadImage

class ExampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.previous_x = self.previous_y = 0
        self.x = self.y = 0
        self.points_recorded = []
        self.canvas = tk.Canvas(self, width=600, height=600, bg = "white", cursor="cross")
        self.canvas.create_circle = self._create_circle
        self.canvas.pack(side="top", fill="both", expand=True)
        self.button_clear = tk.Button(self, text = "Очистить", command = self.clear_all)
        self.button_clear.pack(side="top", fill="both", expand=True)
        self.button_save = tk.Button(self, text = "Сохранить", command = self.save)
        self.button_save.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<Motion>", self.tell_me_where_you_are)
        self.canvas.bind("<B1-Motion>", self.draw_from_where_you_are)

        with open('theta_log_l3.pkl', 'rb') as f:
            self.theta = pickle.load(f)

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
        print(predict(img, self.theta, True))
        print(self.theta.shape, img.shape)

if __name__ == "__main__":
    app = ExampleApp()
    app.mainloop()