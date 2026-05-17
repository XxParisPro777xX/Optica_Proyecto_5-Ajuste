## LIBRERIA PARA CALCULO DE CURVAS DE LUZ DE OCULTACIONES ESTELARES 
## JOEL CASTRO JULIO 2019

#get_ipython().run_line_magic('matplotlib', 'qt')
#inline
#get_ipython().run_line_magic('pylab', '')
import numpy as np
import pandas as pd
#import mpld3
#mpld3.disable_notebook()


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(phi, rho)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def pupilCO(M,D,d):
    '''Generar obstruccion circular'''
    #M>> tamaño matriz en pixeles
    #D>> tamaño de matriz en metros
    #d>> oscurecimiento central en metros
    m=np.linspace(-D/2,D/2,M)
    a,b=np.meshgrid(m,m)
    th,r=cart2pol(a,b)
    P=np.double(r>= d/2)
    
    return(P)

def trasladar(P,smx,smy):
    ''' Trasladar una matriz en direcciones X, Y según el numero de pixeles en cada coordenada  Xpx, Ypx respectivamente... Si mx > 0 Se mueve hacia la derecha, si my > 0 se mueve hacia abajo en las graficas... y VICEVERSA'''
    MM=np.zeros(P.shape)
    x,y=MM.shape
    x=int(x/2); y=int(y/2)
    mx=int(smx)#Para evitar errores en los indices
    my=int(smy)
    
    if my==0 or type(smy)==float:
        #print("Ojo Desplazamiento en Y es 0, o FLOAT")
        MM=P
    if my>0:
        MM[y+my:,:]=P[y:-my,:]; #mover y1
        MM[:my,:]=P[2*y-my:,:] #mover y2
        MM[my:y+my,:]=P[:y,:] #Mover y3
    if my<0:#Negativos
        MM[:y+my,:]=P[-my:y,:]; #mover y1
        MM[2*y+my:,:]=P[:-my,:] #mover y2
        MM[y+my:2*y+my,:]=P[y:,:] #Mover y3

    M2=MM.copy() #Hacer copia para hacer el mismo proceso en X
    
    if mx==0 or type(smx)==float:
        #print("Ojo Desplazamiento en X es 0, o FLOAT")
        M2=MM
    
    if mx>0:
        M2[:,x+mx:]=MM[:,x:-mx]; #mover x1
        M2[:,:mx]=MM[:,2*x-mx:] #mover x2
        M2[:,mx:x+mx]=MM[:,:x] #Mover x3
    if mx<0:#Negativos
        M2[:,:x+mx]=MM[:,-mx:x]; #mover x1
        M2[:,2*x+mx:]=MM[:,:-mx] #mover x2
        M2[:,x+mx:2*x+mx]=MM[:,x:] #Mover x3
    
    return(M2)
       
    
def pupil_doble(M,D,d):
    '''Generar obstruccion tipo Binario de Contacto con la misma area de una obstruccion circular...'''
    #M>> tamaño matriz en pixeles (solo un lado)
    #D>> tamaño de matriz en metros
    #d>> oscurecimiento central en metros como si fuera circular
    r1=(d/2)*.65
    #r2=r1*0.82
    r2=np.sqrt((d/2)**2-(r1)**2)
    d1=r1*2
    d2=r2*2
    Dx=0.45*d1+0.45*d2 # Orientacion en X
    #print(Dx)
    Dy=0
    sepX=((Dx/2)/D)*M
    #print(sepX)
    sepY=((Dy/2)/D)*M
    m=np.linspace(-D/2,D/2,M)
    a,b=np.meshgrid(m,m)
    th,r=cart2pol(a,b)
    #Generar Objetos
    P1=np.double(r>=r1) # Obstruccion grande
    P2=np.double(r>=r2) # Obstruccion pequena
    #separar objetos simetricamente Usar funcion trasladar
    P=trasladar(P1,-sepX,sepY)+trasladar(P2,sepX,sepY)
    #Binarizar
    P=P==2;
    
    return(P)

def pupilCA(M,D,d):
    '''Generar apertura circular'''
    #M>> tamaño matriz en pixeles
    #D>> tamaño de matriz en metros
    #d>> oscurecimiento central en metros
    m=np.linspace(-D/2,D/2,M)
    a,b=np.meshgrid(m,m)
    th,r=cart2pol(a,b)
    P=np.double(r<= d/2)
    
    return(P)

def pupilSO(M,D,d):
    '''Generar obstruccion cuadrada'''
    #M>> tamaño matriz en pixeles
    #D>> tamaño de matriz en metros
    #d>> oscurecimiento central en metros
    t=(M*d/D)
    c=M/2
    P=np.ones((M,M))
    P[int(-t/2+c):int(t/2+c),int(-t/2+c):int(t/2+c)]=0
    #& A>=m/(d/2))
    
    return(P)

def pupilSA(M,D,d):
    '''Generar apertura cuadrada'''
    #M>> tamaño matriz en pixeles
    #D>> tamaño de matriz en metros
    #d>> oscurecimiento central en metros
    t=(M*d/D)
    c=M/2
    P=np.zeros((M,M))
    P[int(-t/2+c):int(t/2+c),int(-t/2+c):int(t/2+c)]=1 #& A>=m/(d/2))
    
    return(P)


def fresnel(U0,M,plano,z,lmda):
    '''Calcular el patron de difraccion en Intensidad de un objeto a  una distancia z'''
    k=2*np.pi/lmda
    nx,ny=np.shape(U0)
    x=(plano/M)*nx #ojo normalmente nx=M por lo tanto x=plano en metros
    y=(plano/M)*ny
    fx=1/x # frecuencia espacial en m**-1
    fy=1/y    
    
    u=np.ones((nx,1))*(np.arange(0,nx)-nx/2)*fx
    v=np.transpose((np.arange(0,ny)-ny/2)*np.ones((ny,1))*fy)
    
    O=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U0)))
    #O=fft2(U0)
    H=np.exp(1j*k*z)*np.exp(-1j*np.pi*(lmda*z)*(u**2+v**2))
    
    U=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(np.multiply(O,H))))
    #U=ifft2(O*H)
    I=np.abs(U)**2
    #import pdb; pdb.set_trace()
    return(I)

def spectra(U0,M,plano,z,nEst,nLmdas):
    '''listadat.txt--> A0=1;A1=2;A2=3;A3=4;A4=5;A5=6;A7=7;F0=8;F2=9;F3=10;F5=11;F6=12;F7=13;F8=14
G0=15;G1=16;G2=17;G5=18;G8=19;K0=20;K1=21;K2=22;K3=23;K4=24;K5=25;K7=26;
M0=27;M1=28;M2=29;M3=30;M4=31;M5=32;M6=33;M7=34;M8=35 '''
    
    fil=open('listadat.txt','r')#Abrir archivo de referencia para saber que estrella se eligió
    lista=fil.readlines()
    fil.close()
    #Direccion del archivo de datos de la convolucion FILTRO,ESTRELLA
    darch="spectra/"+lista[nEst-1][:-1]
    dat=pd.read_csv(darch,sep=',',header=None)#datos en formato panda
    dat1=np.array(dat)#datos en formato numpy
    a,b=dat1.shape
    ind=np.round(np.linspace(1,a-1,nLmdas))#indices para lambda y peso
    acc=0
    for k in ind:
        lamda=dat1[int(k)][0]*1e-10
        peso=dat1[int(k)][1]
        
        I=fresnel(U0,M,plano,z,lamda)*peso
        acc=acc+I
        
    In=np.abs(acc/acc[0][0])**2
    return(In)


def calc_rstar(mV,nEst,ua):
    ''' funcion para calcular los radios aparentes de estrellas
% mV--> magnitud Aparente
% nEst --> num de estrella
% ua --> Distancia al objeto ua
%Magnitudes absolutas en orden desde estrellas tipo A0 hasta M8
% M0=[1.5 1.7 1.8 2.0 2.1 2.2 2.4 3.0 3.3 3.5 3.7 4.0 4.3 4.4 4.7 4.9 5.0...
%     5.2 2.6 6.0 6.2 6.4 6.7 7.1 7.4 8.1 8.7 9.4 10.1 10.7 11.2 12.3 13.4...
%     13.9 14.4]; 
OUT--> tipo, R_star: tipo espectral elegido y radio de estrella calculado respectivamente'''
    ua=1.496e11*ua #distancia en metros
    stars=pd.read_csv('estrellas.dat',sep='\t',header=None)
    #PARAMETROS
    Tsol=5780 #Temperatura del SOl en grados Kelvin
    Rsol=6.96e8 #Radio del sol en mts
    
    T0=stars[1][nEst]#Temperatura
    M0=stars[2][nEst]#Magnitud absoluta
    L0=stars[3][nEst]#Luminosidad relativa al Sol
    
    d1=10**((mV-M0+5)/5)
    d=3.085e16*d1 #Convirtiendo de parsecs PCs a mts (distancia)
    Rst=(L0**0.5)/(T0/Tsol)**2 #Radio de la estrella en Rsol
    alfa=((Rsol*Rst)/d) #Tamano angular de la estrella en radianes
    R_star=(alfa)*ua #Tam de la estrella en mts, RADIO...sobre el objeto
    tipo=stars[0][nEst]
    
    return(tipo,R_star)
    


def promedio_PD(I,R_star,plano,M,d):
    '''I es la imagen del patron de difraccion en intensidad
R_star, es el radio aparente de la estrella mts--> Calcular con: calc_rstar()
plano, tamano de la pantalla blanca diametro mts
M,    tamano de la matriz en pixeles
d,    diametro del objeto en mts'''
    star_px=((R_star)/plano)*M
    obj_px=((d/4)/plano)*M
    div=np.ceil(star_px/obj_px)
    rr=star_px/div
    reso=np.arange(rr,star_px,rr) #arange(start,stop,step) Resolucion de paso
    kin=reso.size
    mu2=np.zeros((I.shape))
    co=1
    for k1 in range(kin):
        #calculo de desplazamiento en teta
        perim=2*np.pi*reso[k1] #perimetro en pixeles
        paso=np.ceil(perim/obj_px) #Num de veces que cabe el objeto en el perimetro
        resot=(2*np.pi)/paso #Paso en radianes
        k2=np.arange(0,2*np.pi,resot)
        for teta in k2:
            mu2=trasladar(I,reso[k1]*np.cos(teta),reso[k1]*np.sin(teta))+mu2
            co+=1
    
    #Ix=mu2+I
    Ix=(mu2+I)/co #normalizar
    #Ix=Ix/(Ix[M-np.int(M*0.1),M-np.int(M*0.1)]) #Normalizar
    return(Ix)


def extraer_perfil(I0,M,D,T,b):
    ''' Funcion que extrae el perfil de difraccion de un patron I0
    I0--> Patron de difraccion, matriz MxM
    M--> Num de pixeles en una dimension de la matriz del patron de difraccion
    D--> Tamanio en metros del plano donde se encuantra el objeto y el patron de difraccion
    T--> Angulo Tetha al cual sera extraido el perfil
    b--> parametro de impacto en metros
    OUT --> x,y vectores con los valores de x en metros y de las intensidades del patron'''
    #Funcion para extraer perfil de difraccion
    #T=0
    #b=1000
    m2p=M/D
    x=np.linspace(-D/2,D/2,M)
    #calcular los arreglos de coordenada X a extraer
    x1=x*np.cos(T*np.pi/180)-b*np.sin(T*np.pi/180)
    x2=x*np.sin(T*np.pi/180)+b*np.cos(T*np.pi/180)
    #Convertir a numero de pixeles
    hp=np.array(m2p*x1)+M/2 #ojo el +M/2 es para iniciar en positivos
    vp=np.array(m2p*x2)+M/2
    hp=hp.astype(int)
    vp=vp.astype(int)
    y=np.ones(x.shape)
    for k in range(M-1):
        y[k]=I0[vp[k],hp[k]] #Ojo Numpy invierte los ejes
    return(x,y)

def calc_plano(d,lmda,ua):
    '''Funcion para calcular el tamanio del plano (objeto y de difraccion) optimo para objetos pequenos (<10km)
    evitando el problema de escalamiento de la FFT
    d--> tam de objeto en metros diametro
    lmda --> long de onda en metros
    ua --> dist del objeto en UA
    OUT --> plano: tamanio del plano en metros (una dimension)'''
    z=ua*1.496e11 #dist en metros
    fscale=np.sqrt(lmda*z/2) #escala de fresnel
    Rho=d/(2*fscale)
    plano=(50*d)/Rho
    return(plano)
    

def add_ruido(I,mV):
    '''Anadir ruido de Poisson a una imagen 
    I--> matriz de la imagen
    mV--> magnitud aparente de la estrella
    OUT--> In: matriz con ruido anadido, asumiendo RUIDO=1/SNR calculada de TAOS-II'''
    ruido=1/SNR_TAOS2(mV)
    n_mask = np.random.poisson(I)
    n_mask=(n_mask/np.mean(n_mask))*ruido-ruido #pesando el ruido de acuerdo con TAOS-II y normalizando
    In=I+n_mask
    return(In)
    
    
def SNR_TAOS2(mV):
    ''' Ajuste polinomial para la curva de  SNR de TAOS-II
    mV-->Magnitud aparente de la estrella
    OUT--> SNR: valor de senal a ruido de TAOS-II'''
    x=mV
    p1 = 1.5792
    p2 = -57.045
    p3 = 515.04
    SNR = p1*x**2 + p2*x +p3
    return(SNR)

def muestreos(lc,D,vr,fps,toff,vE,opangle,ua):
    '''Funcion para muestrear el perfil de difraccion obteniendo el punto promedio  
    lc--> perfil de difraccion o curva de luz
    D--> tamaño del plano en metros
    vr--> velocidad del objeto ~5000 m/s (positiva si va en contra de la vel de la tierra)
    fps--> frames pos segundo de la camara, 20 para TAOS-2
    toff--> Tiempo de desfase dentro del periodo de muestreo
    vE--> velocidad traslacional de la tierra ==29800 m/s
    opangle--> angulo desde oposicion del objeto: O,S,E
    ua--> Distancia en Unidadades Astronomicas del objeto
        OUT--> s_lin,lc_lin,s_pun,lc_pun: vetores de tiempo para lineas, muestra en lineas, tiempo en puntos y muestra en puntos RESPECTIVAMENTE'''
    tam=lc.size
    T=1/fps #Tiempo de exposicion
    OA=opangle*np.pi/180 #Angulo desde oposicion en radianes
    Vt=vE*(np.cos(OA)-np.sqrt((1/ua)*(1-(1/ua**2)*np.sin(OA)**2)))+vr #Velocidad tangencial del obj. rel a tierra
    t=D/Vt #visibilidad del plano en segundos
    Nm=t/T # numero de muestras totales en el plano de observacion
    dpix=np.int(tam/Nm)
    pixoffset=np.int(toff)
    Xpx=tam
    curv=lc
    
    #partir en 2 la curva para comenzar a muestrear desde el centro, por eso uso fliplr 
    curv1=np.flip(curv[:np.int(Xpx/2)+pixoffset])
    curv2=curv[np.int(Xpx/2)+pixoffset:Xpx]
    mcurv1=np.ones(np.size(curv1))#vector de muestras lineas
    mcurv2=np.ones(np.size(curv2))#vector de muestras lineas
    cmuestras1=np.ones((np.int(np.floor((Xpx/2)/dpix+pixoffset/dpix))))#vector de muestras puntos
    cmuestras2=np.ones((np.int(np.floor((Xpx/2)/dpix-pixoffset/dpix))))#vector de muestras puntos

    n=0 #Muestrear curva 1
    for cu in range(cmuestras1.size):
        mcurv1[(cu)*dpix:(cu+1)*dpix]=np.mean(curv1[(cu)*dpix:(cu+1)*dpix])
        cmuestras1[n]=np.mean(curv1[(cu)*dpix:(cu+1)*dpix])
        n=n+1

    n=0 #Muestrear curva 2
    for cu in range(cmuestras2.size):
        mcurv2[(cu)*dpix:(cu+1)*dpix]=np.mean(curv2[(cu)*dpix:(cu+1)*dpix])
        cmuestras2[n]=np.mean(curv2[(cu)*dpix:(cu+1)*dpix])
        n=n+1
    
    lc_lin=np.append(np.flip(mcurv1), mcurv2) #Juntando curvas con lineas constantes
    lc_pun=np.append(np.flip(cmuestras1), cmuestras2)  #extraccion de puntos
    #Calculo de tiempos
    s_lin=np.linspace(-t/2,t/2,lc_lin.size);   #  vector de tiempo para lineas
    s_pun=np.linspace(-t/2,t/2,lc_pun.size);   #  vector de tiempo para puntos
    return(s_lin,lc_lin,s_pun,lc_pun)
    
    
   
    
    
    
    