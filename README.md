# molinoOscilante
[Ejercicio didáctico] En un video de un molino de agua se vé que rota pero de manera curiosa. ¿se podrá medir su movimiento y averiguar algo sobre su dinámica?

## Archivos

extractW.py : script de python que procesa los frames

LICENSE : GPL

README.md : este archivo

molinoDatosOpt.txt : parametros estimados para cada frame. cada columna corresponde a Vx, Vy, w, Cx, Cy. Donde  Vx, Vy es el desplazamiento de los keypoints de fondo respecto al frame anterior, es decir el movimiento de la camara; w es la velocidad angular del molino; Cx, Cy es el centro de rotacion del molino

molinoDatosRaw.txt : Las columnas de este archivo son cinco: NroFrame, xOld, yOld, xNew, yNew. Donde NroFrame es el indice del fotograma procesado; xOld, yOld son las coordenadas de un keypoint en el fotogrma anterior; xNew, yNew son las coordenadas de ese keypoint en ese frame. Cada linea corresponde a un keypoint. 

molino.mp4 : video original

molinoW.mp4 : Video con los keypoints marcados

molinoPlot1.png : grafico de velocidad angular vs tiempo

molinoPlot2.png : grafico de aceleracion angular vs angulo

molinoPlot3.png : grafico de velocidad angular vs angulo

sampleFrame.png : un frame de muestra con los keypoints marcados

## keypoints y estimacion de velocidad angular

Se calcularon las coordenadas de los keypoints en un frame anterior con la funcion de openCV ![goodFeaturesToTrack](https://docs.opencv.org/ref/master/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541). En el frame actual se encuentran sus nuevas posiciones cn ![calcOpticalFlowPyrLK](https://docs.opencv.org/ref/master/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323). En la figura se muestran los desplazamientos de los keypoints que no son fondo como ejemplo:

![frame ejemplo](/sampleFrame.png)


Es decir que se tienen las coordenadas de los keypoints en el frame anterior y en el nuevo: (xOld, yOld) y (xNew, yNew)

### Modelo del movimiento de los keypoints

Si es un keypoint que no pertenece al molino, es del fondo, debería cuplir

xNew = xOld + Vx

yNew = yOld + Vy

para todos los keypoints del frame, o sea (Vx, Vy) es el desplazamiento global del frame

Pero si tengo que describir los puntos en el molino que gira, además de desplazarse la camara tengo que pensar que es como si estuvieran en un disco de velocidad angular w con centro en (Cx, Cy):

xNew = xOld + Vx - w (yNew - Cy)

yNew = yOld + Vy + w (xNew - Cx)

Peeeero el tema es que yo no se si un keypoint es del fondo o no. Asi que inventé una función de membresía alfa que tienda a 0 (cero) si el keypoint es del fondo y se acerque a 1 (uno) si es del molino. Y para no complicarme la vida la definí con la siguiente idea: "si la velocidad del keypoint es muy cercana a la velocidad del fondo es que pertence al fondo". En simbolos

Ux = xNew - xOld

Uy = yNew - yOld

alfa = 1 - exp( - ( (Ux - Vx)^2 - (Uy - Vy)^2 ) / vS^2 )

el parametro vS me permite controlar que tan parecidas tienen que ser las velocidades, lo deje vS=1 despues de un par de pruebas, mirar histogramas de las velocidades y demás.
Finalmente el modelo del movimiento del keypoint es 

Ux = Vx - alfa w (yNew - Cy)

Uy = Vy + alfa w (xNew - Cx)


### estimando w
ahora que tengo un modelo que describe el movimiento de los keypoints con los parametros (Vx, Vy, w, Cx, Cy) y una lista de datos (xNew, yNew, Ux, Uy) de cada keypoint es cuestion de encontrar cuanto valen esos parametros. Recordar que los 5 parametros, desplazamiento de la camara, velocidad angular del molino y centro del molino, son comunes a todos los keypoints. 

Defino una función de optimizacion al viejo estilo de error cuadrático entre lo que dice el modelo y lo que miden los datos

Ux = xNew - xOld

Uy = yNew - yOld

alfa = 1 - exp( - (Ux-Vx)^2 - (Uy - Vy)^2 )

ex = Vx - alfa w (yNew - Cy) - Ux

ey = Vy + alfa w (xNew - Cx) - Uy

E = ex^2 + ey^2  [suma para todos los keypoints]

Y para cada frame minimizo el escalar E obteniendo los 5 parametros con el ![metodo Powell disponible en scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize)

## Resultados

tengo la velocidad angular en funcion del tiempo

![velocidad angular vs tiempo](/molinoPlot1.png)

me interesa ver la aceleracion angular, antes de derivar suavizo la velocidad angular con un kernel gaussiano. Una vez suavizada la derivo para sacar la aceleracion y la integro para sacar el angulo. obtengo:

![aceleracion vs angulo](/molinoPlot2.png)

y la velocidad angulas vs angulo da:

![velocidad vs angulo](/molinoPlot3.png)




