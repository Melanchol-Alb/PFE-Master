import numpy as np
import cv2
import pickle
from scipy import interpolate
from math import pi, cos, sin,atan
import math
from keras.applications.vgg16 import VGG16

SIZE = 32


VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

for layer in VGG_model.layers:
    layer.trainable = False
    
RF_model = pickle.load(open("PFEInterface/RFModel.sav", 'rb'))
le = pickle.load(open("PFEInterface/Encoder.sav", 'rb'))

class Ellipse:
    def __init__(self,clus,H1,H2,S1,S2,T):
        self.cluster = clus
        self.HMoy = H1
        self.SMoy = S1
        self.HEct = H2
        self.SEct = S2
        self.thet = 0 if np.isnan(T) else T
        self.Th = 2
    def __str__(self):
        return "Of cluster:"+str(self.cluster)+" with Th "+str(self.Th)+" At("+str(self.HMoy)+","+str(self.SMoy)+"),with("+str(self.HEct)+","+str(self.SEct)+"), theta "+str(self.thet)
    def is_it_in(self,H,S):
        return (pow((cos(self.thet)*(H-self.HMoy) + sin(self.thet)*(S-self.SMoy)),2) / pow(self.Th*self.HEct,2)) + (pow((-sin(self.thet)*(H-self.HMoy) + cos(self.thet)*(S-self.SMoy)),2) / pow(self.Th*self.SEct,2)) <= 1
    def are_they_in(self,H,S):
        return (np.power((cos(self.thet)*(H-self.HMoy) + sin(self.thet)*(S-self.SMoy)),2) / pow(self.Th*self.HEct,2)) + (np.power((-sin(self.thet)*(H-self.HMoy) + cos(self.thet)*(S-self.SMoy)),2) / pow(self.Th*self.SEct,2)) <= 1
    def F(self):
        print(Points)

ELLIPSES = {}
for i in range(256):
    ELLIPSES[i] = []
    
def getMask(img,ELLIPSES):
    I = img.reshape((img.shape[0]*img.shape[1],3))
    msk = np.zeros((img.shape[0],img.shape[1],3))
    for i in range(256):
        P = np.argwhere(img[:,:,2]==i)
        
        R = np.zeros((P.shape[0],1))
        for e in ELLIPSES[i]:
            R [e.are_they_in(img[P[:,0],P[:,1],0],img[P[:,0],P[:,1],1]).reshape((-1,1))] = 255
        
        for b in range(3):
            msk[P[:,0],P[:,1],b] = R.reshape((-1,))
        
    return msk
    
def classifier (img,mod_VGG,mod_RF,Encod,espace):
    image = img
    with open('PFEInterface/Detector/'+espace+'.pkl', 'rb') as f:
        ELLIPSES = pickle.load(f)
    if(espace=='YCbCr'):
        msk = getMask(cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB ) ,cv2.COLOR_BGR2RGB),ELLIPSES)
    if(espace=='YUV'):
        msk = getMask(cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2YUV ) ,cv2.COLOR_BGR2RGB),ELLIPSES)
    if(espace=='LAB'):
        msk = getMask(cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2LAB ) ,cv2.COLOR_BGR2RGB),ELLIPSES)
    if(espace=='HSV'):
        msk = getMask(cv2.cvtColor(image,cv2.COLOR_BGR2HSV ),ELLIPSES)
    msk = msk[:,:,0]
    msk[msk!=0] = 1
    
    SHAPE = 32
    Seuil = SHAPE*SHAPE-1
    
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            
    border1 = int(image.shape[0]/SHAPE)
    border2 = int(image.shape[1]/SHAPE)
            
    Data = []
    for i in range(border1):
        for j in range(border2):
            if(msk[i*SHAPE:(i+1)*SHAPE,j*SHAPE:(j+1)*SHAPE].sum()>Seuil):
                Data.append([image[i*SHAPE:(i+1)*SHAPE,j*SHAPE:(j+1)*SHAPE]/255.0])
    nb = len(Data)
    if(nb==0):
        print('Image vide.')
        return 'None',"0","0","0","0"
    if(nb<10):
        print('Trop peu d\'echantillons extraits.')
    pred = mod_VGG.predict(np.array(Data)[:,0,:,:,:])
    pred = pred.reshape((pred.shape[0],-1))
            
    prediction = mod_RF.predict(pred)
    prediction = Encod.inverse_transform(prediction)
    
    a = (prediction=='Asian').sum()
    b = (prediction=='Black').sum()
    w = (prediction=='White').sum()
    
    if(max(a,b,w)==a):
        return 'Asian',str(a),str(b),str(w),str(nb)
    if(max(b,w)==b):
        return 'Black',str(a),str(b),str(w),str(nb)
    return 'White',str(a),str(b),str(w),str(nb)


import pygame
import easygui

pygame.init()
Bounds = [700,700]
screen = pygame.display.set_mode((Bounds[0],Bounds[1]))
pygame.display.set_caption("Détection de pixels peau")

FPS = 60 # frames per second
fpsClock = pygame.time.Clock()

font = pygame.font.SysFont("tahoma", 28, True)
font2 = pygame.font.SysFont("tahoma", 22, False)

Background = pygame.image.load("PFEInterface/assets/skinBckg.jpg")

st = cv2.cvtColor(cv2.imread("PFEInterface/images/Base.jpg"),cv2.COLOR_BGR2RGB)
st2 = cv2.cvtColor(cv2.imread("PFEInterface/images/Base.jpg"),cv2.COLOR_BGR2RGB)
if(st.shape[0]>st.shape[1]):
    st = cv2.resize(st,(int(st.shape[1]*200/st.shape[0]),200))
    st2 = cv2.resize(st2,(int(st2.shape[1]*240/st2.shape[0]),240))
else:
    st = cv2.resize(st,(200,int(st.shape[0]*200/st.shape[1])))
    st2 = cv2.resize(st2,(int(st.shape[1]*240/st.shape[0]),240))
BaseImage = pygame.surfarray.make_surface(cv2.flip(cv2.rotate(st, cv2.ROTATE_90_COUNTERCLOCKWISE),0))
mskHSV = pygame.surfarray.make_surface(cv2.rotate(st, cv2.ROTATE_90_COUNTERCLOCKWISE))
mskYCbCr = pygame.surfarray.make_surface(cv2.rotate(st, cv2.ROTATE_90_COUNTERCLOCKWISE))
mskYUV = pygame.surfarray.make_surface(cv2.rotate(st, cv2.ROTATE_90_COUNTERCLOCKWISE))
mskLAB = pygame.surfarray.make_surface(cv2.rotate(st, cv2.ROTATE_90_COUNTERCLOCKWISE))

mskHSV2 = pygame.surfarray.make_surface(cv2.flip(cv2.rotate(cv2.resize(st2,(st2.shape[1],st2.shape[0])), cv2.ROTATE_90_COUNTERCLOCKWISE),0))
mskYCbCr2 = pygame.surfarray.make_surface(cv2.flip(cv2.rotate(cv2.resize(st2,(st2.shape[1],st2.shape[0])), cv2.ROTATE_90_COUNTERCLOCKWISE),0))
mskYUV2 = pygame.surfarray.make_surface(cv2.flip(cv2.rotate(cv2.resize(st2,(st2.shape[1],st2.shape[0])), cv2.ROTATE_90_COUNTERCLOCKWISE),0))
mskLAB2 = pygame.surfarray.make_surface(cv2.flip(cv2.rotate(cv2.resize(st2,(st2.shape[1],st2.shape[0])), cv2.ROTATE_90_COUNTERCLOCKWISE),0))
                    
xBase = 350 - st.shape[1]/2;BoundsX = [xBase,xBase + st.shape[1]]
yBase = 125 - st.shape[0]/2;BoundsY = [yBase,yBase + st.shape[0]]
xeBase = 0;yeBase = 0

IMAGE = 0
IMAGEHSV = 0
IMAGEYCBCR = 0
IMAGEYUV = 0
IMAGELAB = 0
    
def drawSkin():
    screen.blit(BaseImage, (xBase,yBase))
    screen.blit(mskHSV, (xBase-350+200,yBase-125+350))
    screen.blit(mskYCbCr, (xBase-350+500,yBase-125+350))
    screen.blit(mskYUV, (xBase-350+200,yBase-125+575))
    screen.blit(mskLAB, (xBase-350+500,yBase-125+575))
    if(extended):
        screen.blit(extendedPic, (xeBase,yeBase))

Res =  {"HSV":{"asian":"0","black":"0","white":"0","R":"AA","nb":"0"},"YCbCr":{"asian":"0","black":"0","white":"0","R":"AA","nb":"0"},
        "YUV":{"asian":"0","black":"0","white":"0","R":"AA","nb":"0"},  "LAB":{"asian":"0","black":"0","white":"0","R":"AA","nb":"0"}}
def drawRaces():
    screen.blit(font.render(""+str(Res["HSV"]["nb"]), 1, (0,0,0)), (150, 420-10))
    screen.blit(font.render(""+str(Res["HSV"]["asian"]), 1, (0,0,0)), (130, 450))
    screen.blit(font.render(""+str(Res["HSV"]["black"]), 1, (0,0,0)), (130, 485))
    screen.blit(font.render(""+str(Res["HSV"]["white"]), 1, (0,0,0)), (130, 520+5))
    screen.blit(font.render(""+str(Res["HSV"]["R"]), 1, (0,0,0)), (130, 560))
    
    screen.blit(font.render(""+str(Res["YCbCr"]["nb"]), 1, (0,0,0)), (150+250+20, 420-10))
    screen.blit(font.render(""+str(Res["YCbCr"]["asian"]), 1, (0,0,0)), (130+250+20, 450))
    screen.blit(font.render(""+str(Res["YCbCr"]["black"]), 1, (0,0,0)), (130+250+20, 485))
    screen.blit(font.render(""+str(Res["YCbCr"]["white"]), 1, (0,0,0)), (130+250+20, 520+5))
    screen.blit(font.render(""+str(Res["YCbCr"]["R"]), 1, (0,0,0)), (130+250+20, 560))
    
    screen.blit(font.render(""+str(Res["YUV"]["nb"]), 1, (0,0,0)), (150+500+40, 420-10))
    screen.blit(font.render(""+str(Res["YUV"]["asian"]), 1, (0,0,0)), (130+500+40, 450))
    screen.blit(font.render(""+str(Res["YUV"]["black"]), 1, (0,0,0)), (130+500+40, 485))
    screen.blit(font.render(""+str(Res["YUV"]["white"]), 1, (0,0,0)), (130+500+40, 520+5))
    screen.blit(font.render(""+str(Res["YUV"]["R"]), 1, (0,0,0)), (130+500+40, 560))
    
    screen.blit(font.render(""+str(Res["LAB"]["nb"]), 1, (0,0,0)), (150+750+60, 420-10))
    screen.blit(font.render(""+str(Res["LAB"]["asian"]), 1, (0,0,0)), (130+750+60, 450))
    screen.blit(font.render(""+str(Res["LAB"]["black"]), 1, (0,0,0)), (130+750+60, 485))
    screen.blit(font.render(""+str(Res["LAB"]["white"]), 1, (0,0,0)), (130+750+60, 520+5))
    screen.blit(font.render(""+str(Res["LAB"]["R"]), 1, (0,0,0)), (130+750+60, 560))
    
    screen.blit(mskHSV2, (xBase-350+10+125,yBase-125+215))
    screen.blit(mskYCbCr2, (xBase-350+30+250+125,yBase-125+215))
    screen.blit(mskYUV2, (xBase-350+50+500+125,yBase-125+215))
    screen.blit(mskLAB2, (xBase-350+70+750+125,yBase-125+215))

    
extended = False
extendedPic = BaseImage
launched = True
picked = False
skin = True
extending = False
unextending = False
retour = False
current = ""
while launched:
    if(extending):
        if(Bounds[0]<1100):
            Bounds[0]+=10+10*retour
            pygame.display.set_mode((Bounds[0],Bounds[1]))
        else:
            extending = False
            retour = False
    if(unextending):
        if(Bounds[0]>700):
            Bounds[0]-=20+20*retour
            pygame.display.set_mode((Bounds[0],Bounds[1]))
        else:
            unextending = False
            extended = False
            retour = False
            
    screen.blit(Background, (0,0))
    if(skin):
        drawSkin()
    else:
        drawRaces()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            launched = False
        if skin and event.type == pygame.MOUSEBUTTONDOWN:
            mouse = pygame.mouse.get_pos()
            if(10<mouse[0]<250 and 0 <mouse[1]<40):
                skin = False
                Background = pygame.image.load("PFEInterface/assets/ethnicBckg.jpg")
                extending = True
                extended = False
                retour = True
            if(10<mouse[0]<120 and 90 <mouse[1]<130):
                path = easygui.fileopenbox()
                if((not path is None)and(path.endswith(".jpg")or path.endswith(".png"))):
                    IMAGE = cv2.imread(path)
                    
                    for E in ['HSV','YCbCr','YUV','LAB']:
                        with open('PFEInterface/Detector/'+E+'.pkl', 'rb') as f:
                            ELLIPSES = pickle.load(f)
                            image = IMAGE
                            if(E=='YCbCr'):
                                IMAGEYCBCR = getMask(cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB ) ,cv2.COLOR_BGR2RGB),ELLIPSES)
                            if(E=='YUV'):
                                IMAGEYUV = getMask(cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2YUV ) ,cv2.COLOR_BGR2RGB),ELLIPSES)
                            if(E=='LAB'):
                                IMAGELAB = getMask(cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2LAB ) ,cv2.COLOR_BGR2RGB),ELLIPSES)
                            if(E=='HSV'):
                                IMAGEHSV = getMask(cv2.cvtColor(image,cv2.COLOR_BGR2HSV ),ELLIPSES)
                        b = ["0","0","0","0"]
                        a,b[0],b[1],b[2],b[3] = classifier(IMAGE,VGG_model,RF_model,le,E)
                        Res[E]["R"] = a
                        Res[E]["asian"] = b[0]
                        Res[E]["black"] = b[1]
                        Res[E]["white"] = b[2]
                        Res[E]["nb"] = b[3]
                    st = cv2.cvtColor(IMAGE,cv2.COLOR_BGR2RGB)
                    st2 = cv2.cvtColor(IMAGE,cv2.COLOR_BGR2RGB)
                    if(st.shape[0]>st.shape[1]):
                        st = cv2.resize(st,(int(st.shape[1]*200/st.shape[0]),200))
                        st2 = cv2.resize(st2,(int(st2.shape[1]*240/st2.shape[0]),240))
                    else:
                        st = cv2.resize(st,(200,int(st.shape[0]*200/st.shape[1])))
                        st2 = cv2.resize(st2,(int(st2.shape[1]*240/st2.shape[0]),240))
                    
                    BaseImage = pygame.surfarray.make_surface(cv2.flip(cv2.rotate(st, cv2.ROTATE_90_COUNTERCLOCKWISE),0))
                    mskHSV = pygame.surfarray.make_surface(cv2.flip(cv2.rotate(cv2.resize(IMAGEHSV,(st.shape[1],st.shape[0])), cv2.ROTATE_90_COUNTERCLOCKWISE),0))
                    mskYCbCr = pygame.surfarray.make_surface(cv2.flip(cv2.rotate(cv2.resize(IMAGEYCBCR,(st.shape[1],st.shape[0])), cv2.ROTATE_90_COUNTERCLOCKWISE),0))
                    mskYUV = pygame.surfarray.make_surface(cv2.flip(cv2.rotate(cv2.resize(IMAGEYUV,(st.shape[1],st.shape[0])), cv2.ROTATE_90_COUNTERCLOCKWISE),0))
                    mskLAB = pygame.surfarray.make_surface(cv2.flip(cv2.rotate(cv2.resize(IMAGELAB,(st.shape[1],st.shape[0])), cv2.ROTATE_90_COUNTERCLOCKWISE),0))
                    
                    mskHSV2 = pygame.surfarray.make_surface(cv2.flip(cv2.rotate(cv2.resize(IMAGEHSV,(st2.shape[1],st2.shape[0])), cv2.ROTATE_90_COUNTERCLOCKWISE),0))
                    mskYCbCr2 = pygame.surfarray.make_surface(cv2.flip(cv2.rotate(cv2.resize(IMAGEYCBCR,(st2.shape[1],st2.shape[0])), cv2.ROTATE_90_COUNTERCLOCKWISE),0))
                    mskYUV2 = pygame.surfarray.make_surface(cv2.flip(cv2.rotate(cv2.resize(IMAGEYUV,(st2.shape[1],st2.shape[0])), cv2.ROTATE_90_COUNTERCLOCKWISE),0))
                    mskLAB2 = pygame.surfarray.make_surface(cv2.flip(cv2.rotate(cv2.resize(IMAGELAB,(st2.shape[1],st2.shape[0])), cv2.ROTATE_90_COUNTERCLOCKWISE),0))
                    
                    xBase = 350 - st.shape[1]/2;BoundsX = [xBase,xBase + st.shape[1]]
                    yBase = 125 - st.shape[0]/2;BoundsY = [yBase,yBase + st.shape[0]]
                    
                    picked = True
                    unextending = True
                
            loa = False
            if(picked and BoundsX[0]<mouse[0]<BoundsX[1] and BoundsY[0]<mouse[1]<BoundsY[1]):
                if(not extending and not unextending):
                    if(not extended):
                        extending = True
                        extended = True
                        current = "ORI"
                        st = cv2.cvtColor(IMAGE,cv2.COLOR_BGR2RGB)
                        loa = True
                    else:
                        if(current == "ORI"):
                            unextending = True
                        else:
                            loa = True
                            current = "ORI"
                            st = cv2.cvtColor(IMAGE,cv2.COLOR_BGR2RGB)
            if(picked and BoundsX[0]-150<mouse[0]<BoundsX[1]-150 and BoundsY[0]+225<mouse[1]<BoundsY[1]+225):
                if(not extending and not unextending):
                    if(not extended):
                        extending = True
                        extended = True
                        current = "HSV"
                        ii = IMAGE.copy()
                        ii[IMAGEHSV==0] = 0
                        st = cv2.cvtColor(ii,cv2.COLOR_BGR2RGB)
                        loa = True
                    else:
                        if(current == "HSV"):
                            unextending = True
                        else:
                            loa = True
                            current = "HSV"
                            ii = IMAGE.copy()
                            ii[IMAGEHSV==0] = 0
                            st = cv2.cvtColor(ii,cv2.COLOR_BGR2RGB)
            if(picked and BoundsX[0]-150<mouse[0]<BoundsX[1]-150 and BoundsY[0]+450<mouse[1]<BoundsY[1]+450):
                 if(not extending and not unextending):
                    if(not extended):
                        extending = True
                        extended = True
                        current = "YUV"
                        ii = IMAGE.copy()
                        ii[IMAGEYUV==0] = 0
                        st = cv2.cvtColor(ii,cv2.COLOR_BGR2RGB)
                        loa = True
                    else:
                        if(current == "YUV"):
                            unextending = True
                        else:
                            loa = True
                            current = "YUV"
                            ii = IMAGE.copy()
                            ii[IMAGEYUV==0] = 0
                            st = cv2.cvtColor(ii,cv2.COLOR_BGR2RGB)
            if(picked and BoundsX[0]+150<mouse[0]<BoundsX[1]+150 and BoundsY[0]+225<mouse[1]<BoundsY[1]+225):
                 if(not extending and not unextending):
                    if(not extended):
                        extending = True
                        extended = True
                        current = "YCBCR"
                        ii = IMAGE.copy()
                        ii[IMAGEYCBCR==0] = 0
                        st = cv2.cvtColor(ii,cv2.COLOR_BGR2RGB)
                        loa = True
                    else:
                        if(current == "YCBCR"):
                            unextending = True
                        else:
                            loa = True
                            current = "YCBCR"
                            ii = IMAGE.copy()
                            ii[IMAGEYCBCR==0] = 0
                            st = cv2.cvtColor(ii,cv2.COLOR_BGR2RGB)
            if(picked and BoundsX[0]+150<mouse[0]<BoundsX[1]+150 and BoundsY[0]+450<mouse[1]<BoundsY[1]+450):
                 if(not extending and not unextending):
                    if(not extended):
                        extending = True
                        extended = True
                        current = "LAB"
                        ii = IMAGE.copy()
                        ii[IMAGELAB==0] = 0
                        st = cv2.cvtColor(ii,cv2.COLOR_BGR2RGB)
                        loa = True
                    else:
                        if(current == "LAB"):
                            unextending = True
                        else:
                            loa = True
                            current = "LAB"
                            ii = IMAGE.copy()
                            ii[IMAGELAB==0] = 0
                            st = cv2.cvtColor(ii,cv2.COLOR_BGR2RGB)
            if(loa):
                if(st.shape[0]>st.shape[1]):
                    st = cv2.resize(st,(int(st.shape[1]*375/st.shape[0]),375))
                else:
                    st = cv2.resize(st,(375,int(st.shape[0]*375/st.shape[1])))
                extendedPic = pygame.surfarray.make_surface(cv2.flip(cv2.rotate(st, cv2.ROTATE_90_COUNTERCLOCKWISE),0))
                xeBase = 900 - st.shape[1]/2
                yeBase = 350 - st.shape[0]/2
            continue
        if not skin and event.type == pygame.MOUSEBUTTONDOWN:
            mouse = pygame.mouse.get_pos()
            if(0<mouse[0]<200 and 0 <mouse[1]<40):
                skin = True
                Background = pygame.image.load("PFEInterface/assets/skinBckg.jpg")
                unextending = True
                retour = True
    pygame.display.update()
    fpsClock.tick(FPS)
