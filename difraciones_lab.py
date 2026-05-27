## LIBRERIA PARA CALCULO DE CURVAS DE LUZ DE OCULTACIONES ESTELARES
## JOEL CASTRO JULIO 2019

#get_ipython().run_line_magic('matplotlib', 'qt')
#inline
#get_ipython().run_line_magic('pylab', '')
import numpy as np
import pandas as pd
import sys
import os
import time, datetime
#import mpld3
#mpld3.disable_notebook()

#Read File CSV from TANGRA
def readTangraCSV(archivo,exp,cent):
	''' 
	archivo-->archivo csv vreado con TANGRA: days with decimals, 
	signal- bkg
	exp--> exposure time in seconds
	cent--> number of central frame '''
	kk=np.loadtxt(archivo,delimiter=',',skiprows=14)
	flx=kk[:,2];normflx=flx/(np.mean(flx[-30:]))#Normalize curve
	num=kk[:,0];num=num-cent #centering curve
	secs=exp*num
	return secs,normflx
	
# Convert cartesian coordinates to polar coordinates
def cart2pol( x, y ):
    # Radius | Hypotenuse
    rho = np.sqrt( x ** 2 + y ** 2 )
    # Principal angle
    phi = np.arctan2( y, x )
    return phi, rho

# Convert polar coordinates to cartesian coordinates
def pol2cart( rho, phi ):
    x = rho * np.cos( phi )
    y = rho * np.sin( phi )
    return x, y

#Generate a circular obstruction
def pupilCO( M, D, d ):
    # M -> matrix dimension in pixels
    # D -> matrix size in meters
    # d -> central dimming

    # Coordinate vector between -D/2 and D/2 with M ( pixels ) samples
    m = np.linspace( -D / 2, D / 2, M )
    # Generate tow matrices; X with m as rows, Y with m as columns
    a, b = np.meshgrid( m, m )
    # Calculate angle ( th ) and radius ( r ) of each point
    th, r = cart2pol( a, b )
    # Matrix filled with
        # 1, if radius is greater than the obstruction radius,
        # and 0, if not
    P = np.double( r >= d / 2 )

    return P

# Translate a matrix by smx and smy in pixels
def trasladar( P, smx, smy ):
    # New matrix, filled with 0, and same size as P
    MM = np.zeros( P.shape )
    # Find the position of the central pixel
    x, y = MM.shape
    x = int( x / 2 )
    y = int( y / 2 )
    # X offset
    mx = int( smx )
    # Y offset
    my = int( smy )

    if my == 0 or type( smy ) == float:
        # No vertical translation
        MM = P
    if my > 0:
        # Positive vertical translation, move down
        MM[ y + my : , : ] = P[ y : -my, : ]; # Move y1
        MM[ : my, : ] = P[ 2 * y - my : , : ] # Move y2
        MM[ my : y + my, : ] = P[ : y, : ] # Move y3
    if my < 0:
        # Negative vertical translation, move up
        MM[ : y + my, : ] = P[ -my : y, : ]; # Move y1
        MM[ 2 * y + my : , : ] = P[ : -my, : ] # Mive y2
        MM[ y + my : 2 * y + my, : ] = P[ y : , : ] # Move y3

    # Makes a copy to replicate process with horizontal offset
    M2 = MM.copy()

    if mx == 0 or type( smx ) == float:
        # No horizontal translation
        M2 = MM

    if mx > 0:
        # Positive horizontal translation, move right
        M2[ : , x + mx : ] = MM[ : , x : -mx ]; # Move x1
        M2[ : , : mx ] = MM[ : , 2 * x - mx : ] # Move x2
        M2[ : , mx : x + mx ] = MM[ : , : x ] # Move x3
    if mx < 0:
        # Negative horizontal translation, move left
        M2[ : , : x + mx ] = MM[ : , -mx : x ]; # Move x1
        M2[ : , 2 * x +mx : ] = MM[ : , : -mx ] # Move x2
        M2[ : , x + mx :2 * x + mx ] = MM[ : , x : ] # Move x3

    return M2

# Generate a binary obstruction with same area as a circular obstruction
def pupil_doble( M, D, d  ):
    # M -> matrix size in pixels
    # D -> matriz size in meters
    # d -> central dimming as if it were circular

    r1 = ( d / 2 ) * .65
    #r2=r1*0.82
    r2 = np.sqrt( ( d / 2 ) ** 2 - ( r1 ) ** 2 )
    d1 = r1 * 2
    d2 = r2 * 2
    # X orientation
    Dx = 0.45 * d1 + 0.45 * d2 # Orientacion en X
    Dy = 0
    # X offset
    sepX = ( ( Dx / 2 ) / D ) * M
    # Y offset
    sepY = ( ( Dy / 2 ) / D ) * M
    m = np.linspace( -D / 2, D / 2, M )
    a, b = np.meshgrid( m, m )
    # Get angle and radius of each point
    th, r = cart2pol( a, b )
    # Generate objects
    # Matrices filled with
        # 1, if radius is greater than the obstruction radius,
        # and 0, if not
    P1 = np.double( r >= r1 ) # Big object obstruction
    P2 = np.double( r >= r2 ) # Small object obstruction
    #Combine objects by applying an offset
    P = trasladar( P1, -sepX, sepY ) + trasladar( P2, sepX, sepY )
    #Binarize ( in previous line, can use hadamard product ( * ) instead of sum to avoid this )
    P = P == 2;

    return P

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
    t=M*d/D
    c=M/2
    P=np.ones((M,M))
    P[-t/2+c:t/2+c,-t/2+c:t/2+c]=0
    #& A>=m/(d/2))

    return(P)

def pupilSA(M,D,d):
    '''Generar apertura cuadrada'''
    #M>> tamaño matriz en pixeles
    #D>> tamaño de matriz en metros
    #d>> oscurecimiento central en metros
    t=M*d/D
    c=M/2
    P=np.zeros((M,M))
    P[-t/2+c:t/2+c,-t/2+c:t/2+c]=1 #& A>=m/(d/2))

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

def spectra(U0,M,plano,z,nLmdas,lmda,ang):
    '''
    This is a modified version used for lab only...
    I average the number of wavelenghts around lmda 
    separated ang each
    '''
    kk=(nLmdas/2)
    ind=np.linspace(lmda-ang*kk,lmda+ang*kk,np.round(nLmdas))

    if nLmdas<=1:
        I=fresnel(U0,M,plano,z,lmda)
        return(I)
    
    acc=np.zeros((M,M))
    cont=0
    
    for k in ind:
        lamda=k
        peso=1

        I=fresnel(U0,M,plano,z,lamda)*peso
        #print(lamda)
        acc=acc+I
        #print(np.min(np.min(I)))
        cont=cont+1#Correcciones para probar errores en original matlab
    #acc=acc/cont   #OJO NO SE DEBERIA DIVIDIR
    In=(acc/acc[0][0])
    return(In)


def calc_rstar(mV,nEst,ua):
    ''' funcion para calcular los radios aparentes de estrellas
% mV--> magnitud Aparente
% nEst --> num de estrella
% ua --> Distancia al objeto metros
%Magnitudes absolutas en orden desde estrellas tipo A0 hasta M8
% M0=[1.5 1.7 1.8 2.0 2.1 2.2 2.4 3.0 3.3 3.5 3.7 4.0 4.3 4.4 4.7 4.9 5.0...
%     5.2 2.6 6.0 6.2 6.4 6.7 7.1 7.4 8.1 8.7 9.4 10.1 10.7 11.2 12.3 13.4...
%     13.9 14.4];
OUT--> tipo, R_star: tipo espectral elegido y radio de estrella calculado respectivamente'''
    #ua=1.496e11*ua #distancia en metros
    path = os.path.dirname( os.path.abspath( __file__ ) )
    stars=pd.read_csv( os.path.join( path, 'estrellas.dat' ), sep = '\t', header = None )
    #PARAMETROS
    Tsol=5780 #Temperatura del SOl en grados Kelvin
    Rsol=6.96e8 #Radio del sol en mts

    T0=stars[1][nEst-1]#Temperatura
    M0=stars[2][nEst-1]#Magnitud absoluta
    L0=stars[3][nEst-1]#Luminosidad relativa al Sol

    d1=10**((mV-M0+5)/5)
    d=3.085e16*d1 #Convirtiendo de parsecs PCs a mts (distancia)
    Rst=(L0**0.5)/(T0/Tsol)**2 #Radio de la estrella en Rsol
    alfa=((Rsol*Rst)/d) #Tamano angular de la estrella en radianes
    R_star=(alfa)*ua #Tam de la estrella en mts, RADIO...sobre el objeto
    tipo=stars[0][nEst-1]

    return(tipo,R_star)



def promedio_PD(I,R_star,plano,M,d):
    '''I es la imagen del patron de difraccion en intensidad
R_star, es el radio aparente de la estrella mts--> Calcular con: calc_rstar()
plano, tamano de la pantalla blanca diametro mts
M,    tamano de la matriz en pixeles
d,    diametro del objeto en mts'''
    
    if R_star ==0:
        return(I)
    
    star_px=((R_star)/plano)*M
    obj_px=((d/4)/plano)*M
    div=np.ceil(star_px/obj_px)
    rr=star_px/div
    reso=np.arange(rr,star_px+.0001,rr) #arange(start,stop,step) Resolucion de paso

    kin=len(reso)
    #print(star_px)
    #print(rr)
    mu2=np.zeros((I.shape))
    co=1
    for k1 in range(kin):
        #calculo de desplazamiento en teta
        perim=2*np.pi*reso[k1] #perimetro en pixeles
        paso=np.ceil(perim/obj_px) #Num de veces que cabe el objeto en el perimetro
        resot=(2*np.pi)/paso #Paso en radianes
        #print(resot)
        k2=np.arange(resot,2*np.pi+.0001,resot)#***OJO ESTO COMIENZA EN 0
        for teta in k2:
            mu2=trasladar(I,reso[k1]*np.cos(teta),reso[k1]*np.sin(teta))+mu2
            co+=1
            #print(np.min(np.min(mu2)))

    Ix=mu2+I
    #print(np.min(np.min(I)))
    Ix=(Ix/Ix[0][0])
    #Ix=(mu2+I)/co #normalizar
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
    m2p = M / D
    x = np.linspace(-D / 2, D / 2, M)
    
    # Calcular los arreglos de coordenada X a extraer
    x1 = x * np.cos(T * np.pi / 180) - b * np.sin(T * np.pi / 180)
    x2 = x * np.sin(T * np.pi / 180) + b * np.cos(T * np.pi / 180)
    
    # Convertir a número de píxeles
    hp = np.array(m2p * x1) + M / 2 # Ojo el +M/2 es para iniciar en positivos
    vp = np.array(m2p * x2) + M / 2
    
    hp = hp.astype(int)
    vp = vp.astype(int)
    y = np.ones(x.shape)
    
    for k in range(M):
        # --- VERIFICACIÓN DE LÍMITES ---
        # Aseguramos que los índices hp y vp estén dentro de los bordes 0 y M-1 de la matriz
        if 0 <= vp[k] < M and 0 <= hp[k] < M:
            y[k] = I0[vp[k], hp[k]] # Ojo Numpy invierte los ejes
        else:
            y[k] = 0.0 # Si la cámara sale de la imagen 2D simulada, asume oscuridad total (flujo 0)
            
    return x, y

def calc_plano(d,lmda,z):
    '''Funcion para calcular el tamanio del plano (objeto y de difraccion) optimo para objetos pequenos (<10km)
    evitando el problema de escalamiento de la FFT
    d--> tam de objeto en metros diametro
    lmda --> long de onda en metros
    z --> dist del objeto en m
    OUT --> plano: tamanio del plano en metros (una dimension)'''
    #z=ua#*1.496e11 #dist en metros
    fscale=np.sqrt(lmda*z/2) #escala de fresnel
    Rho=d/(2*fscale)
    plano=(20*d)/Rho
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

def muestreos( lc, D, vr, fps, toff, vE, ua ):
    '''Funcion para muestrear el perfil de difraccion obteniendo el punto promedio
    lc--> perfil de difraccion o curva de luz
    D--> tamaño del plano en metros
    vr--> velocidad del objeto 
    fps--> frames pos segundo de la camara, 20 para TAOS-2
    toff--> Tiempo de desfase dentro del periodo de muestreo
    vE--> velocidad traslacional de la CAMARA
    opangle--> angulo desde oposicion del objeto: O,S,E NOT USED FOR LAB 
    ua--> Distancia en Unidadades Astronomicas del objeto
        OUT--> s_lin,lc_lin,s_pun,lc_pun: vetores de tiempo para lineas, muestra en lineas, tiempo en puntos y muestra en puntos RESPECTIVAMENTE'''
    tam = lc.size
    T = 1 / fps #Tiempo de exposicion
    #OA = opangle * np.pi / 180 #Angulo desde oposicion en radianes
    #Vt = vE * ( np.cos( OA ) - np.sqrt( ( 1 / ua ) * ( 1 - ( 1 / ua ** 2 ) * np.sin( OA ) ** 2 ) ) ) + vr #Velocidad tangencial del obj. rel a tierra
    Vt=vE+vr
    t = D / Vt #visibilidad del plano en segundos
    Nm = t / T # numero de muestras totales en el plano de observacion
    dpix = np.int( tam / Nm )
    
    if dpix==0:
        dpix=1
        print('WARNING: Number of samples too high--> make mesh bigger')
    pixoffset = np.int( toff )
    Xpx = tam
    curv = lc

    #partir en 2 la curva para comenzar a muestrear desde el centro, por eso uso fliplr
    curv1 = np.flip( curv[ : np.int( Xpx / 2 ) + pixoffset ] )
    curv2 = curv[ np.int( Xpx / 2 ) + pixoffset : Xpx ]
    mcurv1 = np.ones( np.size( curv1 ) ) #vector de muestras lineas
    mcurv2 = np.ones( np.size( curv2 ) ) #vector de muestras lineas
    cmuestras1 = np.ones( ( np.int( np.floor( ( Xpx / 2 ) / dpix + pixoffset / dpix ) ) ) ) #vector de muestras puntos
    cmuestras2 = np.ones( ( np.int( np.floor( ( Xpx / 2 ) / dpix - pixoffset / dpix ) ) ) ) #vector de muestras puntos

    n = 0 #Muestrear curva 1
    for cu in range( cmuestras1.size ):
        mcurv1[ cu * dpix : ( cu + 1 ) * dpix ] = np.mean( curv1[ cu * dpix : ( cu + 1 ) * dpix ] )
        cmuestras1[ n ] = np.mean( curv1[ cu * dpix : ( cu + 1 ) * dpix ] )
        n = n + 1

    n = 0 #Muestrear curva 2
    for cu in range( cmuestras2.size ):
        mcurv2[ cu * dpix : ( cu + 1 ) * dpix ] = np.mean( curv2[ cu * dpix : ( cu + 1 ) * dpix ] )
        cmuestras2[ n ] = np.mean( curv2[ cu * dpix : ( cu + 1 ) * dpix ] )
        n = n + 1

    lc_lin = np.append( np.flip( mcurv1 ), mcurv2 ) #Juntando curvas con lineas constantes
    lc_pun = np.append( np.flip( cmuestras1 ), cmuestras2 )  #extraccion de puntos
    #Calculo de tiempos
    s_lin = np.linspace( -t / 2, t / 2, lc_lin.size );   #  vector de tiempo para lineas
    s_pun = np.linspace( -t / 2, t / 2, lc_pun.size );   #  vector de tiempo para puntos
    
    return s_lin, lc_lin, s_pun, lc_pun


def buscar_picos(x,y,D,fil=0.005):
    '''Funcion para buscar picos con la derivada de la ocultacion
    IN...
    x,y--> vectores con los datos de la ocultacion,distancia y amplitud
    D--> diametro del objeto [mts]
    fil--> valor de umbral para identificar los picos, DEFAULT=0.005
    OUT--> indices para ubicar los PICOS EN y,  también los valores.
    '''
    yp=np.diff(y)  #derivada de la ocultacion
    cyp=abs(yp)<fil # convertir 0s en 1s de la derivada, buscar valores cercanos a 0
    xin=np.where(abs(x)<(D/2)) # seleccionar solo la region de la ocultacion
    xin2=np.array(xin) # convertir los indices (tuple) en arreglo de numpy
    indx=np.where(cyp[xin]==1) # buscar los 1s en el rango establecido (xin)
    inpks=xin2[0,0]+indx # estos son los indices donde están los picos en la curva de luz original

    Y=np.array(y[inpks])#valores pico
    ban=0;inew=[];pY=[]
    for k in range(Y.size-1):#Eliminar Repetidos
        I=Y[0,k]
        J=Y[0,k+1]
        if np.abs(I-J)>fil or ban==0:#si NO esta repetido
            inew.append(inpks[0,k+1])#Indice del pico en la curva de luz
            pY.append(J)#Valor del pico en la curva de luz
            ban=1
    pY=np.array(pY)

    return inew, pY