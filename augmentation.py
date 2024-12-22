import cv2 as cv
import numpy as np
import os

ITEMS = ['breasts','butterfliedDrumsticks','drumsticks','wholeLeg','wings']

#INPUT_DIR = rf'data/originalData/{ITEM}'
#OUTPUT_DIR = rf'data/augmentedData/{ITEM}'

def augmentImages():
    name = 0

    for item in ITEMS:
        INPUT_DIR = rf'data/train/{item}'
        OUTPUT_DIR = rf'data/train/{item}'

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        for idx, images in enumerate(os.listdir(INPUT_DIR)):
            if idx >= 5:
                break

            image_path = os.path.join(INPUT_DIR, images)
            image = cv.imread(image_path)
            if image is None:
                print(f"Warning: {image_path} okunamadı, bu dosya atlandı.")
                continue


            input_height, input_width = image.shape[:2]

            horizontal_flip = cv.flip(image, 1)
            cv.imwrite(os.path.join(OUTPUT_DIR, f"{name}_horizontal_flip.jpg"), horizontal_flip)
            name += 1


            vertical_flip = cv.flip(image, 0)
            cv.imwrite(os.path.join(OUTPUT_DIR, f"{name}_vertical_flip.jpg"), vertical_flip)
            name += 1


            high_contrast = cv.convertScaleAbs(image, alpha=1.5, beta=0)  # alpha artırılarak kontrast yükseltilir
            cv.imwrite(os.path.join(OUTPUT_DIR, f"{name}_high_contrast.jpg"), high_contrast)
            name += 1

            low_contrast = cv.convertScaleAbs(image, alpha=0.5, beta=0)  # alpha düşürülerek kontrast azaltılır
            cv.imwrite(os.path.join(OUTPUT_DIR, f"{name}_low_contrast.jpg"), low_contrast)
            name += 1


            hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            hsv_image[:, :, 0] = cv.add(hsv_image[:, :, 0], 10)  # Hue değerini artır
            high_hue = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)
            cv.imwrite(os.path.join(OUTPUT_DIR, f"{name}_high_hue.jpg"), high_hue)
            name += 1


            hsv_image[:, :, 0] = cv.subtract(hsv_image[:, :, 0], 20)  # Hue değerini azalt
            low_hue = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)
            cv.imwrite(os.path.join(OUTPUT_DIR, f"{name}_low_hue.jpg"), low_hue)
            name += 1


            mean = 0
            for std_dev in [20, 30]:  # Farklı seviyelerde gürültü
                gaussian_noise = np.random.normal(mean, std_dev, image.shape).astype(np.int16)
                noisy_image = cv.add(image.astype(np.int16), gaussian_noise)
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
                cv.imwrite(os.path.join(OUTPUT_DIR, f"{name}_gaussian_noise_{std_dev}.jpg"), noisy_image)
                name += 1


            b, g, r = cv.split(image)
            r = cv.add(r, 20)
            warm_image = cv.merge((b, g, r))
            cv.imwrite(os.path.join(OUTPUT_DIR, f"{name}_warm.jpg"), warm_image)
            name += 1


            b = cv.add(b, 20)
            cool_image = cv.merge((b, g, r))
            cv.imwrite(os.path.join(OUTPUT_DIR, f"{name}_cool.jpg"), cool_image)
            name += 1
