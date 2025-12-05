import cv2

image_path="/Users/jcy/Library/Mobile Documents/com~apple~CloudDocs/CODE/opencv/1.png"
image=cv2.imread(image_path)
cv2.namedWindow("display image",cv2.WINDOW_NORMAL) #调整
cv2.resizeWindow("display image",800,600)  #（width,height)改变出现的尺寸
cv2.imshow("display image",image)
pixel_value=image[100,50]
key=cv2.waitKey(0)  #表示无限等待 
if key== ord('s'):
    """
    如果我想要用任意键就可以保存或者默认保存
    """
    output_path=(" ",image_path)
    cv2.imwrite(output_path,image)
    print(f'save as {output_path}')
elif key !=-1:
    output_path=(" ",image_path)
    cv2.imwrite(output_path,image)
    print(f'save as {output_path}') 
else:
    print('false')
cv2.destroyAllWindows()