#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 00:43:04 2018

@author: sebalander
"""

# %% imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal as sg
from scipy import optimize as op
import os

vS = 1.0 # parametro para pesar diferencia de velocidates


def alfas(params, positions, velocities):
    '''
    devuelve la clase que determina si los keypoints son fondo (cero) o están
    en el molino (uno) (es pura heurística): si la velocidad del keypoint es
    muy parecida a la velocidad que se estima como la del movimiento de la
    camara, entonces es fondo. la funcion es 1-exponencial cuadratica.
    '''
    Ux, Uy, w, Cx, Cy = params

    veldif = velocities - [Ux, Uy]

    alfas = 1 - np.exp(- np.sum((veldif / vS)**2, axis=1))

    return alfas


def Error(params, positions, velocities):
    '''
    funcion error, toma los cinco parametros Ux, Uy, w, Cx, Cy
    y los arrays de posiciones y velocidades de los keypoints y
    calcula la suma total del funcional propuesto
    '''
    Ux, Uy, w, Cx, Cy = params

    veldif = velocities - [Ux, Uy]

    # calculo una especie de funcion de clase difusa
    alfas = 1 - np.exp(- np.sum((veldif / vS)**2, axis=1))

    ex = veldif.T[0] - alfas * w * (Cy - positions.T[1])
    ey = veldif.T[1] - alfas * w * (positions.T[0] - Cx)

    return np.sum(ex**2 + ey**2)




# %%
vidFileIn = "molino.mp4"

vidFileOut = "molinoW"
vidFileOutExt = "mp4"

datRaw = "molinoDatosRaw.txt"
datOpt = "molinoDatosOpt.txt"
graph1 = "molinoPlot1.png"
graph2 = "molinoPlot2.png"
graph3 = "molinoPlot3.png"

saveVidBool = True # True para que guarde el output de video

# params for ShiTomasi corner detection
feature_params = dict(maxCorners = 1000,
                       qualityLevel = 1e-3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# abrir captura
cap = cv2.VideoCapture(vidFileIn)


# Take first frame and find corners in it
ret, old_frame = cap.read()
old_frame = cv2.flip(cv2.rotate(old_frame, cv2.ROTATE_90_CLOCKWISE), 1)

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)


# valores iniciales de los parametros Ux, Uy, w, Cx, Cy
params = [1.0, 0.0, 1, 200, 500]

optList = list() # donde guardar los datos y parametros optimizados
datos = list()

frmCnt = cap.get(cv2.CAP_PROP_FRAME_COUNT) # nro total de frames del video
fps = cap.get(cv2.CAP_PROP_FPS) # fps


# %%

while True:
    ret, frame = cap.read()
    frame = cv2.flip(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), 1)
    iFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    if not ret or not cap.isOpened():
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # saco las velocidades y ajusto los parametros
    deltas = good_new - good_old
    ret = op.minimize(Error, params, args=(good_new, deltas), method='Powell')
    params = ret.x
    #guardo datos crudos y optimizacion en las listas correspondientes
    datosFrame = np.concatenate(([[iFrame]]*good_old.shape[0], good_old, good_new), axis=1)
    datos.append(datosFrame)
    optList.append([params])

    print('avi ratio %.2f'%(100 * iFrame / frmCnt),
          '\tvel ang', ret.x[2])

    al = alfas(params, good_new, deltas)
    toDraw = 0.99 < al

    ptCen = np.float32(params[3:])


    for pt1, pt2 in zip(good_new[toDraw], good_old[toDraw]):
        img	= cv2.line(frame, tuple(pt1), tuple(pt2), (0,0,255), 3)

    ptCen = tuple(np.float32(params[3:]))
    img	= cv2.circle(img, ptCen, 10, (255, 0, 0), -1)


    if saveVidBool:
        # guardo cada frame individual como imagen png
        filename = vidFileOut + "%05d.png"%iFrame
        cv2.imwrite(filename, img)

    cv2.imshow('frame', img)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = cv2.goodFeaturesToTrack(old_gray, **feature_params) # good_new.reshape(-1,1,2)

print('fin procesamiento, a guardar')


if saveVidBool:
    # hago un video mp4 con los frame sindividiales
    cmd = "ffmpeg -y -r %f -f image2  -i "%fps
    cmd += vidFileOut + "%05d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p "
    cmd += vidFileOut + "." + vidFileOutExt

    os.system(cmd)

    os.system("rm " + vidFileOut + "*.png")


cap.release()
print('cap released')
cv2.destroyAllWindows()


# %% formateo y guardo los datos
datos = np.concatenate(datos, axis=0)

fmt = ("%d", "%d", "%d", "%f", "%f")
head = "NroFrame    xOld    yOld    xNew    yNew"
np.savetxt(datRaw, datos, fmt=fmt, header=head)

parOpt = np.concatenate(optList, axis=0)

head = "Vx    Vy    w    Cx    Cy"
np.savetxt(datOpt, parOpt, header=head)

# %%

wList = parOpt.T[2] * fps
t = np.arange(len(wList)) / fps

promedio = np.mean(wList[277:]) # hago un pormedio aprox en dos periodos completos
T = - 2 * np.pi / promedio

plt.figure()

plt.plot(t, wList)
plt.plot([0, t[-1]], [promedio, promedio], label="promedio %.1frad/s  ->  periodo %.1fs"%(promedio, T))
plt.xlabel("tiempo [s]")
plt.ylabel("velocidad angular[rad/s]")
plt.legend()
plt.savefig(graph1)

# %%
# suavizo con kernel gaussiano
k = sg.windows.gaussian(15, 2, sym=True)
k /= k.sum()
wSmooth = sg.convolve(wList, k, mode='same')

# integro para sacar el angulo
theta = np.cumsum(wSmooth) / fps
theta = theta - np.floor(theta / np.pi / 2) * np.pi * 2

# derivo para sacar aceleracion
acel = np.diff(wSmooth) * fps

plt.figure()
plt.scatter(theta[:-1], acel)
plt.ylim([-0.5, 0.5])
plt.xlabel('angulo [rad]')
plt.ylabel('aceleracion angular [rad/s²]')

# parametros a ojo para sinusoide de referencia
f0 = 0.06
aT = 0.25
acelSinT = aT * np.sin(theta[:-1]) + f0
plt.scatter(theta[:-1], acelSinT, label='0.25 * sin(angulo) + 0.06')
plt.legend()

plt.savefig(graph2)


plt.figure()
plt.scatter(theta, wSmooth)
plt.ylim([0.1, 1.1])
plt.ylabel('velocidad angular [rad/s]')
plt.xlabel('angulo [rad]')
plt.savefig(graph3)







