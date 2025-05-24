import numpy as np
from PIL import Image

def rgb_to_grayscale(image_path, output_path):
    """Конвертирует цветное изображение в градации серого методом взвешенного усреднения."""
    img = Image.open(image_path).convert('RGB') 
    pixels = np.array(img)  
    
    
    grayscale = 0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]
    grayscale = grayscale.astype(np.uint8)  
    
    Image.fromarray(grayscale).save(output_path)  
    return grayscale

def wolf_thresholding(image, window_size=7, k=0.3):
    """Адаптивная бинаризация Вульфа для изображения в градациях серого."""
    height, width = image.shape 
    output = np.zeros_like(image)  
    
    S = np.std(image)  
    
    half_window = window_size // 2  
    padded_image = np.pad(image, half_window, mode='reflect')  
    
    
    for y in range(height):
        for x in range(width):
            
            region = padded_image[y:y+window_size, x:x+window_size]
            mean = np.mean(region)  
            std_dev = np.std(region)  
            
            
            threshold = mean * (1 + k * (std_dev / S - 1))
            
            
            output[y, x] = 255 if image[y, x] > threshold else 0
    
    return output



gray_img = rgb_to_grayscale("image.png", "grayscale.bmp")

mapp = rgb_to_grayscale("map.png", "map_gray.bmp")
binary_map = wolf_thresholding(mapp)
Image.fromarray(binary_map).save("binary_map.bmp")



xray = rgb_to_grayscale("x-ray.png", "xray_gray.bmp")
binary_xray = wolf_thresholding(xray)
Image.fromarray(binary_xray).save("binary_xray.bmp")


cartoon = rgb_to_grayscale("cartoon.png", "cartoon_gray.bmp")
binary_cartoon = wolf_thresholding(cartoon)
Image.fromarray(binary_cartoon).save("binary_cartoon.bmp")


image = rgb_to_grayscale("image.png", "image_gray.bmp")
binary_image = wolf_thresholding(image)
Image.fromarray(binary_image).save("binary_image.bmp")


finger = rgb_to_grayscale("finger.png", "finger_gray.bmp")
binary_finger = wolf_thresholding(finger)
Image.fromarray(binary_finger).save("binary_finger.bmp")


text = rgb_to_grayscale("text.png", "text_gray.bmp")
binary_text = wolf_thresholding(text)
Image.fromarray(binary_text).save("binary_text.bmp")

