#!/usr/bin/env python3
# coding: utf-8

import random
import numpy as np
import copy
import sys
import math
import cv2
import tkinter as tk
from PIL import ImageTk, Image
import tkinter.filedialog


# ------------------------------------------------- DLT ---------------------------------------------------

def dlt_algorithm(old_points, new_points):
    old_points = np.array(old_points)
    new_points = np.array(new_points)
    n = len(new_points)
    matrix_a = []

    # Matrix A[2nx9]
    for i in range(n):
        cpoint = old_points[i]
        cpointp = new_points[i]
        mini_matrix1 = [0, 0, 0, -cpointp[2] * cpoint[0], -cpointp[2] * cpoint[1], -cpointp[2] * cpoint[2],
                        cpointp[1] * cpoint[0], cpointp[1] * cpoint[1], cpointp[1] * cpoint[2]]
        mini_matrix2 = [cpointp[2] * cpoint[0], cpointp[2] * cpoint[1], cpointp[2] * cpoint[2], 0, 0, 0,
                        -cpointp[0] * cpoint[0], -cpointp[0] * cpoint[1], -cpointp[0] * cpoint[2]]
        matrix_a.append(mini_matrix1)
        matrix_a.append(mini_matrix2)

    matrix_a = np.array(matrix_a)

    # SVD(Singular Value Decomposition)
    s, v, d = np.linalg.svd(matrix_a)
    last_d = d[-1]

    # Transforming last column of matrix D to 3x3 P matrix
    p_matrix = []
    for i in range(3):
        col = [last_d[3 * i], last_d[3 * i + 1], last_d[3 * i + 2]]
        p_matrix.append(col)

    # Round to 9 decimals
    for i in range(3):
        for j in range(3):
            p_matrix[i][j] = round(p_matrix[i][j], 9)

    p_matrix = np.array(p_matrix)
    return p_matrix


# ------------------------------------------------ DLT-M --------------------------------------------------


def dlt_algorithm_m(old_points, new_points):
    n = len(new_points)
    old_points = np.array(old_points)
    new_points = np.array(new_points)

    # Making 3rd coordinate equal to 1
    for i in range(n):
        # originals
        old_points[i][0] = old_points[i][0] / old_points[i][2]
        old_points[i][1] = old_points[i][1] / old_points[i][2]
        old_points[i][2] = old_points[i][2] / old_points[i][2]

        # images
        new_points[i][0] = new_points[i][0] / new_points[i][2]
        new_points[i][1] = new_points[i][1] / new_points[i][2]
        new_points[i][2] = new_points[i][2] / new_points[i][2]

    # Center of points (G,G')
    cx = sum([old_points[i][0] for i in range(n)]) / float(n)
    cy = sum([old_points[i][1] for i in range(n)]) / float(n)
    cyp = sum([new_points[i][1] for i in range(n)]) / float(n)
    cxp = sum([new_points[i][0] for i in range(n)]) / float(n)

    # Translating points
    for i in range(n):
        new_points[i][0] -= cxp
        new_points[i][1] -= cyp
        old_points[i][0] -= cx
        old_points[i][1] -= cy

    # Homothetic transformation (S,S')
    lambd = 0
    lambdp = 0
    for i in range(n):
        lambd = lambd + math.sqrt(old_points[i][0] ** 2 + old_points[i][1] ** 2)
        lambdp = lambdp + math.sqrt(new_points[i][0] ** 2 + new_points[i][1] ** 2)

    lambd = lambd / float(n)
    lambdp = lambdp / float(n)
    if lambd == 0 or lambdp == 0:
        print("Error: points are collinear!")
        sys.exit(1)
    k = math.sqrt(2) / lambd
    kp = math.sqrt(2) / lambdp

    # Using Homothety on points
    for i in range(n):
        old_points[i][0] *= k
        old_points[i][1] *= k
        new_points[i][0] *= kp
        new_points[i][1] *= kp

    # DLT on new points = P'
    matrix_pp = dlt_algorithm(old_points, new_points)

    # Calculating matrix T = S * G
    matrix1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    matrix2 = np.array([[k, 0, 0], [0, k, 0], [0, 0, 1]])
    matrix_t = matrix2.dot(matrix1)

    # Calculating matrix T' = S' * G'
    matrix1p = np.array([[1, 0, -cxp], [0, 1, -cyp], [0, 0, 1]])
    matrix2p = np.array([[kp, 0, 0], [0, kp, 0], [0, 0, 1]])
    matrix_tp = matrix2p.dot(matrix1p)

    # (T')^-1
    matrix_tp_inv = np.linalg.inv(matrix_tp)
    # P = (T')^-1 * P' * T
    matrix_p = matrix_tp_inv.dot(matrix_pp)
    matrix_p = matrix_p.dot(matrix_t)

    # Round to 9 decimals
    for i in range(3):
        for j in range(3):
            matrix_p[i][j] = round(matrix_p[i][j], 9)

    return matrix_p



# Distance between points
def distnace(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# ----------------------------------------------- RANSAC --------------------------------------------------

def RANSAC(old_points, new_points):
    best_model = None
    max_inlier = 0
    max_iter = 10000
    for i in range(max_iter):
        s = random.sample(range(len(old_points)), 4)
        n = [new_points[i] for i in s]
        o = [old_points[i] for i in s]
        matrix = dlt_algorithm(o,n)
        current_inliers = 0
        for j in range(len(old_points)):
            npoint = np.dot(matrix, old_points[j])
            npoint[0] /= npoint[2]
            npoint[1] /= npoint[2]
            npoint[2] /= npoint[2]
            dist = distnace(npoint, new_points[j])
            if dist < 5:
                current_inliers += 1

        if current_inliers > max_inlier:
            max_inlier = current_inliers
            best_model = copy.deepcopy(matrix)

    print(max_inlier)
    return best_model


def Usage():
    print("Usage: \nInput number of pictures you want to load.\nSelect them in order you want them to be merged (from left to right)")
    print("Use you left/right arrow to switch between pictures")
    print("You can select how many points you like but keep note that there need to be equal number of points on both pictures you currenty looking")
    print("If you want to use RANSAC, press 'R'")
    print("Final picture will be shown on screen right away")
    print("Enjoy!")
        



def initializing():
    # Number of points for creating polygon TODO: dont care about user input
    num_of_pictures  = 0
    
    Usage();
    
    while True:
        try:
            num_of_pictures = int(input("How many pictures you want to use?\n"))
            #pyAssert(1 < num_of_pictures,"You must choose at least 2 pictures")
            break
        except ValueError:
            print("Please enter the correct number")
            pass
        
    return num_of_pictures

# ------------------------------------------------- GUI -------------------------------------------------




def gui():

    num_of_pictures = initializing()
    
    # Delete this after fixing =4 problem
    num_of_points = 4

    # Making window
    window = tk.Tk()
    window.title("Panorama")
    window.configure(background='black')

    # Supported picture formats TODO: so far only this 4, add more
    file_types = ["*.jpeg", "*.jpg", "*.bmp", "*.png", "*.JPG", "*.JPEG", "*.BMP", "*.PNG"]

    width_wc = 1280
    height_wc = 670
    canvas_height = 720
    canvas_width = 640
    
    geom_string = "{}x{}+{}+{}".format(width_wc+50,height_wc , 0, 0)
    window.geometry(geom_string)
    
    
    loaded_pictures = []
    image_files = []
    
    usagel = tk.Label(window, text = "Usage:\n Input number of pictures you want to load. Select them in order you want them to be merged (from left to right)\nUse you left/right arrow to switch between pictures. You can select how many points you like but keep note that there need to be equal number of points on both pictures you currenty looking.\n To run panorama with more then 4 points and RANSAC press 'r'\n To run algorithm on 4, press ENTER. GLHF",bg = 'white',anchor = tk.NW)
    usagel.pack(side = tk.TOP)

    
    for i in range(num_of_pictures):
                    
        try:
            # Open file dialog
            image_file = tkinter.filedialog.askopenfilename(initialdir="./examplesForInput",
                                                            title="Select picture to upload:",
                                                            filetypes=(("Pictures", file_types),))
            image_files.append(image_file)
            # Window size = picture dimensions

            im = Image.open(image_file)
            picture_width,picture_height = im.size 

            # If image is larger then 1080p, scale it to 720p
            scale_ratio = 1
            if picture_width > canvas_width or picture_height > canvas_height:
                if picture_width > picture_height:
                    scale_ratio = picture_width / canvas_width
                else:
                    scale_ratio = picture_height/ canvas_height
                

                im = im.resize((int(picture_width/scale_ratio),int(picture_height/scale_ratio)),Image.ANTIALIAS)
            
            new_image = ImageTk.PhotoImage(im)
        
            loaded_pictures.append(new_image)

        except(SystemError, AttributeError):
            print("You must choose file to upload")
            sys.exit(1)

    
    
    
    current_pictures = [0,1]
    canvas1 = tk.Canvas(window, width=canvas_width, height=loaded_pictures[current_pictures[0]].height(), bg='red',bd = 0,highlightbackground = "white")
    canvas1.create_image(0, 0, image=loaded_pictures[current_pictures[0]], anchor=tk.NW)
    canvas1.pack(side = tk.LEFT)


    canvas2 = tk.Canvas(window, width=canvas_width, height=loaded_pictures[current_pictures[1]].height(), bg='red',bd = 0,highlightbackground = "white")
    canvas2.create_image(0, 0, image=loaded_pictures[current_pictures[1]], anchor=tk.NW)
    canvas2.pack(side = tk.LEFT)
    
    
    # Mouse motion function
    def motion(event):
        x0 = int(event.x)
        y0 = int(event.y)
        canvas1.itemconfig(text_id, text="X: {} Y: {}".format(x0, y0))
        canvas1.update()


    # Mouse position
    text_id = canvas1.create_text(100, 10, fill="red", font="Times 15 bold", text="X: {} Y: {}".format(0, 0))
    canvas1.bind("<Motion>", motion)
    canvas2.bind("<Motion>",motion)

    
    
    #MOY GOD
    # [ ..,[ [] [] ],.. ]
    old_points = [[] for i in range(num_of_pictures-1) ]
    new_points = [[] for i in range(num_of_pictures-1) ]
    rec_id = [[],[]]
    
    def click_original(eventorigin):
        # Take (x,y) coords
        x0 = int(eventorigin.x)
        y0 = int(eventorigin.y)
        
        old_points[current_pictures[0]].append([x0, y0, 1.0])
        rec_id[0].append(canvas1.create_rectangle(x0 - 5, y0 - 5, x0 + 5, y0 + 5, outline="red", width=3))

    def click_images(eventorigin):
        # Take (x,y) coords
        x0 = int(eventorigin.x)
        y0 = int(eventorigin.y)
        
        new_points[current_pictures[0]].append([x0, y0, 1.0])
        rec_id[1].append(canvas2.create_rectangle(x0 - 5, y0 - 5, x0 + 5, y0 + 5, outline="blue", width=3))
        

    


    canvas1.bind("<Button 1>",click_original)
    canvas2.bind("<Button 1>",click_images)
    
    def move_right(event):
        if current_pictures[1] == len(loaded_pictures) - 1:
            return
        
        for i in rec_id[0]:
            canvas1.delete(i)
        for i in rec_id[1]:
            canvas2.delete(i)
        
        current_pictures[0]  = current_pictures[1]
        current_pictures[1] += 1
        canvas1.itemconfig(1, image=loaded_pictures[current_pictures[0]])
        canvas2.itemconfig(1, image=loaded_pictures[current_pictures[1]])        
        if len(old_points[current_pictures[0]])>0:                  
            l = old_points[current_pictures[0]]
            for x0,y0,z in l:
                rec_id[0].append(canvas1.create_rectangle(x0 - 5, y0 - 5, x0 + 5, y0 + 5, outline="red", width=3))

        if len(new_points[current_pictures[0]])>0:
                                                    
            l = new_points[current_pictures[0]]
            for x0,y0,z in l:
                rec_id[1].append(canvas2.create_rectangle(x0 - 5, y0 - 5, x0 + 5, y0 + 5, outline="blue", width=3))

        tk.mainloop()

    def move_left(event):
        
        if current_pictures[0] == 0:
            return
        for i in rec_id[0]:
            canvas1.delete(i)
        for i in rec_id[1]:
            canvas2.delete(i)
        current_pictures[1]  = current_pictures[0]
        current_pictures[0] -= 1
        canvas1.itemconfig(1, image=loaded_pictures[current_pictures[0]])
        canvas2.itemconfig(1, image=loaded_pictures[current_pictures[1]])        

        if len(old_points[current_pictures[0]])>0:                  
            l = old_points[current_pictures[0]]
            for x0,y0,z in l:
                rec_id[0].append(canvas1.create_rectangle(x0 - 5, y0 - 5, x0 + 5, y0 + 5, outline="red", width=3))

        if len(new_points[current_pictures[0]])>0:
                                                    
            l = new_points[current_pictures[0]]
            for x0,y0,z in l:
                rec_id[1].append(canvas2.create_rectangle(x0 - 5, y0 - 5, x0 + 5, y0 + 5, outline="blue", width=3))
        tk.mainloop()


    if(num_of_pictures > 2):
        window.bind("<Right>",move_right)
        window.bind("<Left>",move_left)    
    
    def use_ransac(event):
        create_panorama(old_points[0],new_points[0],image_files[0],image_files[1],scale_ratio,True)
        window.unbind("r")
        window.unbind("<Return>")
        
        
    window.bind("r",use_ransac)
    
    
    def run_algorithm(event):
        create_panorama(old_points[0],new_points[0],image_files[0],image_files[1],scale_ratio,False)
        window.unbind("r")
        window.unbind("<Return>")
        

    
    window.bind("<Return>",run_algorithm)
    


    tk.mainloop()


# Points translation
def translate(points,width):
    new_points = []
    for i in points:
        point = copy.deepcopy(i)
        point[0] += width
        new_points.append(point)
        
    return new_points

        


# Panorama 
def create_panorama(old_points,new_points,image1,image2,scale_ratio,using_ransac):
        

    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    image1_width = img1.shape[1]
    image1_height = img1.shape[0]
    image2_width = img2.shape[1]
    image2_height = img2.shape[0]



    new_points  = [[int(new_points[i][j] * scale_ratio) for j in range(2)]+[int(new_points[i][2])] for i in range(len(new_points)) ]
    old_points  = [[int(old_points[i][j] * scale_ratio) for j in range(2)]+[int(old_points[i][2])] for i in range(len(old_points)) ]
    
    
    new_points = translate(new_points,image1_width)
    

    if using_ransac:
        M  = RANSAC(old_points,new_points)
    else:
        M = dlt_algorithm(old_points[:4],new_points[:4])
    print(M)


    result = cv2.warpPerspective(img1, M,(image1_width + image2_width, image1_height))
    print(result.shape)
    result[0:image2_height, image1_width:(image2_width+image1_width)] = img2
    
    cv2.imwrite("result.jpg",result)

    

# MAIN
def main():

    gui();

# GO GO GO
if __name__ == "__main__":
    main()
