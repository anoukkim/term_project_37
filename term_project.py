import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

image = cv.imread("shapes.png")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
ret, thresh = cv.threshold(blur, 200, 255,cv.THRESH_BINARY_INV)

cv.imwrite("thresh.png", thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

blank = np.zeros(thresh.shape[:2], dtype='uint8')
cv.drawContours(blank, contours, -1, (255, 0, 0), 1)

cv.imwrite("Contours.png", blank)

print("coordinates of centers:")
for i in contours:
    M = cv.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv.drawContours(image, [i], -1, (0, 255, 0), 2)
        cv.circle(image, (cx, cy), 7, (0, 0, 255), -1)
        cv.putText(image, "center", (cx - 70, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        print(f"x: {cx} y: {cy}")
print("number of shapes detected: ", len(contours))

cv.imwrite("image.png", image)

'''
image1 = cv.imread("image.png")
cv.imshow("Centers", image1)
cv.waitKey(0)
cv.destroyAllWindows()
'''

plt.imshow(image)
plt.show()


