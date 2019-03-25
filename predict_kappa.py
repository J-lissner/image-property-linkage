import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import subfunctions as subf
from tkinter import *
from tkinter import filedialog , messagebox
from PIL import Image

####################################  initialize, load the ANN models in a list
frame= Tk()
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
    i=int(selected.get() ) # i is the version numbered from 0 to 2
    #%% convert the given image to NN input
    c11=np.transpose(subf.correlation_fct(img)).reshape(400**2,1)
    x=subf.NN_input(B[i],h[i],img,c11)
    
    x= np.matmul(mu_x[i], x-x_m[i])
    x= np.transpose(x)
    if i==0:
       print('prediction for the circle trained ANN') 
    elif i==1:
       print('prediction for the rectangle trained ANN') 
    elif i==2:
       print('prediction for the mixed trained ANN') 
    
    # predict the set and rescale it
    fpre = sess[i].run(model[i],{x_nn[i]:x})
    fpre=(  np.matmul(mu_inv[i],np.transpose(fpre))+f_m[i]   )  
    fpre[2]=fpre[2]/np.sqrt(2)
    kappa= np.array( [ ( fpre[0] , fpre[2] ), (fpre[2],fpre[1] ) ] )
    print("prediction for Kappa:")
    np.set_printoptions(precision=3,suppress=True)
    print( np.matrix(kappa) )
    print("\n")

def load_txt():
    global img
    fname=filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("txt files", "*.txt"),("all files","*")) )  
    img=np.loadtxt(fname )
    #convert the given image to 1 and 0
    tmp=np.array(img).reshape(400**2)
    color1=np.max(img)
    img= np.array([1 if x==color1 else 0 for x in tmp ]).reshape(400,400)
    print('Image loaded, filename: %s' %(fname))

def load_tif():
    global img
    fname=filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("tif files", "*.tif"),("all files","*")) )  
    #convert the given image to 1 and 0
    img= Image.open(fname)
    tmp=np.array(img).reshape(400**2)
    color1=np.max(img)
    img= np.array([1 if x==color1 else 0 for x in tmp ]).reshape(400,400)
    print('Image loaded, filename: %s' %(fname))

def plot_img():
    fig=plt.matshow(img)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()

def infoascii():
    print("Input: binarized matrix of dimension 400x400")
def infotif():
    print("Input: binarized .tif image of resolution/dimensions 400x400") 
def ms_info():
    print("Specific heat conductivity of matrix phase (dark blue) \t K=1 [W/mK]" )
    print("Specific heat conductivity of inclusions (yellow) \t K=0.2 [W/mK]" )


################################# assign the commands to each button

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

predict_nn=Button(frame,text="Plot RVE",command=plot_img)
predict_nn.grid(column=1,row=5)

predict_nn=Button(frame,text="Predict Kappa",command=predict_img)
predict_nn.grid(column=1,row=6)

quit=Button(frame,text="QUIT",command=frame.quit)
quit.grid(column=8,row=8)

info=Button(frame,text="INFO",command=ms_info)
info.grid(column=0,row=8)


frame.mainloop()
frame.destroy
