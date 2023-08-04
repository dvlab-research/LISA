import cv2

img = cv2.imread("/data/xinlai/LISA2/imgs/teaser.png")
cv2.imwrite("./teaser.jpg", img)
