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
import time
import scipy as sc
from scipy import signal as sg
from scipy import optimize as op
from matplotlib import colors


vS = 1.0

def alfas(params, positions, velocities):
    Ux, Uy, w, Cx, Cy = params

    veldif = velocities - [Ux, Uy]

    alfas = 1 - np.exp(- np.sum((veldif / vS)**2, axis=1))

    return alfas

def Error(params, positions, velocities):
    Ux, Uy, w, Cx, Cy = params

    veldif = velocities - [Ux, Uy]

    alfas = 1 - np.exp(- np.sum((veldif / vS)**2, axis=1))

    ex = veldif.T[0] - alfas * w * (Cy - positions.T[1])
    ey = veldif.T[1] - alfas * w * (positions.T[0] - Cx)

    return np.sum(ex**2 + ey**2)



# %%
fileMP4 = "/home/sebalander/Code/molinoOscilante/molino.mp4"
fileOut = "/home/sebalander/Code/molinoOscilante/molinoW"
graph1 = "/home/sebalander/Code/molinoOscilante/molinoPlot1.png"
graph2 = "/home/sebalander/Code/molinoOscilante/molinoPlot2.png"
graph3 = "/home/sebalander/Code/molinoOscilante/molinoPlot3.png"

saveBool = True


# params for ShiTomasi corner detection
feature_params = dict(maxCorners = 1000,
                       qualityLevel = 1e-3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


cap = cv2.VideoCapture(fileMP4)



# Take first frame and find corners in it
ret, old_frame = cap.read()
old_frame = cv2.flip(cv2.rotate(old_frame, cv2.ROTATE_90_CLOCKWISE), 1)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# valores iniciales de los parametros
# Ux, Uy, w, Cx, Cy = params
params = [1.0, 0.0, 1, 200, 500]

paramsOptims = list()

frmCnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)

fps = cap.get(cv2.CAP_PROP_FPS)
if saveBool:
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
#    fourcc = -1
    out = cv2.VideoWriter(fileOut, fourcc, fps, old_frame.shape[:2])




# %%

while True:
    ret, frame = cap.read()
    frame = cv2.flip(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), 1)

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
    paramsOptims.append(params)

    print('avi ratio %.2f'%(100 * cap.get(cv2.CAP_PROP_POS_FRAMES) / frmCnt),
          '\tvel ang', ret.x[2])

    al = alfas(params, good_new, deltas)
    toDraw = 0.99 < al

    ptCen = np.float32(params[3:])


    for pt1, pt2 in zip(good_new[toDraw], good_old[toDraw]):
        img	= cv2.line(frame, tuple(pt1), tuple(pt2), (0,0,255), 3)

    ptCen = tuple(np.float32(params[3:]))
    img	= cv2.circle(img, ptCen, 10, (255, 0, 0), -1)


    if saveBool:
        # out.write(img)
        filename = fileOut + "%05d.png"%cap.get(cv2.CAP_PROP_POS_FRAMES)
        cv2.imwrite(filename, img)

    cv2.imshow('frame', img)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params) # good_new.reshape(-1,1,2)

print('fin procesamiento, a guardar')


if saveBool:
    out.release()
    print('out released')

cap.release()
print('cap released')
cv2.destroyAllWindows()

paramsOptims = np.array(paramsOptims)

# %%
cmd = '''ffmpeg -r 30 -f image2  -i molinoW%05d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p molinoW.mp4'''


# %%

if __name__ != '__main__':

    wList = paramsOptims.T[2] * fps
    t = np.arange(len(wList)) / fps

    promedio = np.mean(wList[277:])
    T = - 2 * np.pi / promedio

    plt.figure()

    plt.plot(t, wList)
    plt.plot([0, t[-1]], [promedio, promedio], label="promedio %.1frad/s  ->  periodo %.1fs"%(promedio, T))
    plt.xlabel("tiempo [s]")
    plt.ylabel("velocidad angular[rad/s]")
    plt.legend()
    plt.savefig(graph1)

# %%

    k = sg.windows.gaussian(15, 2, sym=True)
    k /= k.sum()
    wSmooth = sg.convolve(wList, k, mode='same')

    theta = np.cumsum(wSmooth) / fps
    theta = theta - np.floor(theta / np.pi / 2) * np.pi * 2


    acel = np.diff(wSmooth) * fps

    plt.figure()
    plt.scatter(theta[:-1], acel)
    plt.ylim([-0.5, 0.5])
    plt.xlabel('angulo [rad]')
    plt.ylabel('aceleracion angular [rad/s²]')

    f0 = 0.06
    aT = 0.25
    acelSinT = aT * np.sin(theta[:-1]) + f0
    plt.scatter(theta[:-1], acelSinT, label='0.25 * sin(angulo) + 0.06')
    plt.legend()

    plt.savefig(graph2)

#
#
#    plt.figure()
#    plt.scatter(wSmooth[:-1], acel - acelSinT)
#    plt.ylim([-0.5, 0.5])
#    plt.xlabel('velocidad angular [rad/s]')
#    plt.ylabel('aceleracion angular [rad/s²]')
#    plt.savefig(graph3)




    plt.figure()
    plt.scatter(theta, wSmooth)
    plt.ylim([0.1, 1.1])
    plt.ylabel('velocidad angular [rad/s]')
    plt.xlabel('angulo [rad]')
    plt.savefig(graph3)







