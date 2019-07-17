import cv2
import imutils
import numpy as np

imagem = cv2.imread('dados.jpg')

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
cv2.imshow("Mayelle - imagem (cinza)", imagem_cinza)
cv2.waitKey(0)
cv2.imwrite("dados_cinza.jpg", imagem_cinza)

imagem_suave = cv2.blur(imagem_cinza, (7,7))
cv2.imshow("Mayelle - imagem (blur)", imagem_suave)
cv2.waitKey(0)
cv2.imwrite("dados_suave.jpg", imagem_suave)

imagem_bin = cv2.threshold(imagem_suave, 110, 250, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Mayelle - Binarização", imagem_bin)
cv2.waitKey(0)
cv2.imwrite("dados_binarizada.jpg", imagem_bin)

imagem_borda = cv2.Canny(imagem_bin, 70,150)
cv2.imshow("Mayelle - Bordas", imagem_borda)
cv2.waitKey(0)
cv2.imwrite("dados_bordas.jpg", imagem_borda)

qtdd_objetos = cv2.findContours(imagem_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
qtdd_objetos = imutils.grab_contours(qtdd_objetos)
img_final = imagem.copy()

for n in qtdd_objetos:
    cv2.drawContours(img_final, [n], -1, (85,200,100), 3)

resultado = "{} dados!".format(len(qtdd_objetos))
cv2.putText(img_final, resultado, (10,25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (85,200,100),2)

cv2.imshow("Mayelle - resultado", img_final)
cv2.waitKey(0)
cv2.imwrite("dados_resultado.jpg", img_final)