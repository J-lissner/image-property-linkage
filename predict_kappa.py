import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import subfunctions as subf
from tkinter import *
from tkinter import filedialog , messagebox
from PIL import Image

####################################  initialize, load the ANN models in a list
flag=0
frame= Tk()
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
config = tf.ConfigProto(allow_soft_placement=True)

print('loading the three ANN....')
for version in range(3):
    parent="./model"
    path="%s/nn_v%i" %(parent,version+1)
    if version==0: #initialize the list
        graph=[tf.get_default_graph() ]
        sess = [tf.Session(config=config,graph=graph[version])]
        tf.saved_model.loader.load(sess[version],['tag'],path)  
        x_nn = [graph[version].get_tensor_by_name('input:0') ]
        model = [graph[version].get_tensor_by_name('mymodel:0')]
        h=[ int(np.loadtxt("%s/ints" %(path))[0] ) ]
    
        #load the shifts associated with the model
        shifts=np.load("%s/shifts.npz" %(path))
        x_m= [ shifts['x_m']  ]
        mu_x=[ shifts['mu_x'] ]
        f_m= [ shifts['f_m']  ]
        mu_f=[ shifts['mu_f'] ]
        mu_inv=[ np.diag(1/np.diag(mu_f[version])) ]
        #%% load the RB for the NN
        B= [np.loadtxt("%s/rb/rb_v%i" %(parent,version+1) ) ]
    
    else: #append to each list
        graph.append(tf.Graph())#tf.get_default_graph() )
        sess.append(tf.Session(config=config,graph=graph[version]) )
        tf.saved_model.loader.load(sess[version],['tag'],path)  
        x_nn.append( graph[version].get_tensor_by_name('input:0') )
        model.append( graph[version].get_tensor_by_name('mymodel:0') )
        h.append(int(np.loadtxt("%s/ints" %(path))[0] ) )
    
        shifts=( np.load("%s/shifts.npz" %(path)) )
        x_m.append(  shifts['x_m']  )
        mu_x.append( shifts['mu_x'] )
        f_m.append(  shifts['f_m']  )
        mu_f.append( shifts['mu_f'] )
        mu_inv.append(  np.diag(1/np.diag(mu_f[version]) )  )
        B.append(  np.loadtxt("%s/rb/rb_v%i" %(parent,version+1) )  )
    print('ANN nr %i/3 loaded'%(version+1) )

####################################  define the commands
def predict_img():
    global flag
    i=int(selected.get() ) # i is the version numbered from 0 to 2
    #%% convert the given image to NN input
    c11=np.transpose(subf.correlation_fct(img)).reshape(400**2,1)
    x=subf.NN_input(B[i],h[i],img,c11)
    
    x= np.matmul(mu_x[i], x-x_m[i])
    x= np.transpose(x)
    if i==0:
       s1='circle trained model'
    elif i==1:
       s1='rectangle trained model'
    elif i==2:
       s1='mixed trained model'
    s2= 'prediction for kappa'
    
    # predict the set and rescale it
    fpre = sess[i].run(model[i],{x_nn[i]:x})
    fpre=(  np.matmul(mu_inv[i],np.transpose(fpre))+f_m[i]   )  
    fpre[2]=fpre[2]/np.sqrt(2)
    kappa= np.array( [ ( fpre[0] , fpre[2] ), (fpre[2],fpre[1] ) ] )
    print("%s with %s:" %(s2,s1) )
    np.set_printoptions(precision=3,suppress=True)
    print( np.matrix(kappa) )
    print("\n")
    if flag==1:
        txt.create_rectangle(1,1 , canvas_width,canvas_height/2,fill='#ffffff')
        txt.create_text(canvas_width/2, canvas_height/4, text="%s\n%s: \n [ %10.3f      %10.3f ]\n [ %10.3f      %10.3f ]" % (s1,s2,fpre[0],fpre[2],fpre[2],fpre[1] ) )
    else: 
        txt.create_rectangle(1,canvas_height/2 , canvas_width,canvas_height,fill='#ffffff')
        txt.create_text(canvas_width/2, 3*canvas_height/4, text="%s\n%s: \n [ %10.3f      %10.3f ]\n [ %10.3f      %10.3f ]" % (s1,s2,fpre[0],fpre[2],fpre[2],fpre[1] ) )
    flag *= -1

def load_txt():
    global img
    fname=filedialog.askopenfilename(initialdir = "./",title = "Select TIFF file",filetypes = (("txt files", "*.txt"),("all files","*")) )  
    img=np.loadtxt(fname )
    #convert the given image to 1 and 0
    tmp=np.array(img).reshape(400**2)
    color1=np.max(img)
    img= np.array([1 if x==color1 else 0 for x in tmp ]).reshape(400,400)
    plot_img()
    clear_txtcanvas()
    print('Image loaded, filename: %s' %(fname))

def load_tif():
    global img
    fname=filedialog.askopenfilename(initialdir = "./",title = "Select ascii file",filetypes = (("tif files", "*.tif"),("all files","*")) )  
    #convert the given image to 1 and 0
    img= Image.open(fname)
    tmp=np.array(img).reshape(400**2)
    color1=np.max(img)
    img= np.array([1 if x==color1 else 0 for x in tmp ]).reshape(400,400)
    plot_img()
    clear_txtcanvas()
    print('Image loaded, filename: %s' %(fname))
    

def plot_img():
    fig= Figure(figsize=(1.8,1.8),dpi=100)
    fig.add_subplot(111).imshow(img,interpolation='nearest')#=plt.matshow(img)
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0,wspace=0,hspace=0)
    
    canvas = FigureCanvasTkAgg(fig, master=frame)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().grid(column=0,row=7,columnspan=2)
    plt.show()

def infoascii():
    global flag
    print("Input: binarized matrix of dimension 400x400")
    if flag==1:
        txt.create_rectangle(1,1 , canvas_width,canvas_height/2,fill='#ffffff')
        txt.create_text(canvas_width/2, canvas_height/4, text="Input: binarized matrix\nof dimension 400x400" )
    else: 
        txt.create_rectangle(1,canvas_height/2 , canvas_width,canvas_height,fill='#ffffff')
        txt.create_text(canvas_width/2, 3*canvas_height/4,  text="Input: binarized matrix\nof dimension 400x400" )
    flag *= -1

def infotif():
    global flag
    print("Input: binarized .tif image of resolution/dimensions 400x400") 
    if flag==1:
        txt.create_rectangle(1,1 , canvas_width,canvas_height/2,fill='#ffffff')
        txt.create_text(canvas_width/2, canvas_height/4, text="Input: binarized .tif image \nof resolution 400x400" )
    else: 
        txt.create_rectangle(1,canvas_height/2 , canvas_width,canvas_height,fill='#ffffff')
        txt.create_text(canvas_width/2, 3*canvas_height/4,  text="Input: binarized .tif image \nof resolution 400x400" )
    flag *= -1

def ms_info():
    global flag
    print("Specific heat conductivity of matrix phase (dark blue) \t K=1 [W/mK]" )
    print("Specific heat conductivity of inclusions (yellow) \t K=0.2 [W/mK]\n" )
    if flag==1:
        txt.create_rectangle(1,1 , canvas_width,canvas_height/2,fill='#ffffff')
        txt.create_text(canvas_width/2, canvas_height/4, text="Specific heat conductivity [W/mK]\n matrix phase \t K=1\n inclusion \t K=0.2")
    else: 
        txt.create_rectangle(1,canvas_height/2 , canvas_width,canvas_height,fill='#ffffff')
        txt.create_text(canvas_width/2, 3*canvas_height/4,  text="Specific heat conductivity [W/mK]\n matrix phase \t K=1 \n inclusion \t K=0.2")
    flag *= -1

def clear_txtcanvas():
    global flag
    txt.create_rectangle(1,canvas_height/2 , canvas_width,canvas_height,fill='#ffffff')
    txt.create_rectangle(1,1 , canvas_width,canvas_height/2,fill='#ffffff')
    flag=1


#%%get the canvas image printing the output

canvas_width=220
canvas_height=180
txt=Canvas(frame,width=canvas_width,height=canvas_height)
txt.grid(column=2,row=7,columnspan=2)#,rowspan=2)
clear_txtcanvas()
txt.create_text(canvas_width/2, canvas_height/4, text=' finished loading \n showing "circle_example.txt"')
flag=-1

#%% assign the image canvas, load and plot the example image
print('loading and showing " circle_example.txt" ')
img= np.loadtxt('examples/circle_example.txt' )
plot_img()

################################# assign the commands to each button




#%% assign the rest of the buttons

frame.title("Predict heat conduction tensor with image")

label_in= Label (frame, text="Choose the model")
label_in.grid(column=1,row=0)

selected= IntVar()
ver1 = Radiobutton(frame,text='Circle training', value=0, variable=selected)#, command=load_model)
ver2 = Radiobutton(frame,text='Rectangle training', value=1, variable=selected)#, command=load_model)
ver3 = Radiobutton(frame,text='Mixed training', value=2, variable=selected)#, command=load_model)

ver1.grid(column=0,row=1)
ver2.grid(column=1,row=1)
ver3.grid(column=2,row=1)

info_ascii= Button(frame, text="Info ascii", command= infoascii )
info_ascii.grid(column=2,row=3)

info_tif= Button(frame, text="Info *.tif", command= infotif)
info_tif.grid(column=0,row=3)

load_ascii= Button(frame, text="Load ascii", command= load_txt)
load_ascii.grid(column=2,row=4)

load_tif= Button(frame, text="Load *.tif", command= load_tif)
load_tif.grid(column=0,row=4)

predict_nn=Button(frame,text="Predict Kappa",command=predict_img)
predict_nn.grid(column=1,row=6)

clear_text=Button(frame,text="Clear text",command=clear_txtcanvas)
clear_text.grid(column=2,row=8,columnspan=2)

quit=Button(frame,text="QUIT",command=frame.quit)
quit.grid(column=8,row=8)

info=Button(frame,text="Info microstructure",command=ms_info)
info.grid(column=0,row=8,columnspan=2)


frame.mainloop()
frame.destroy
