import cv2
import glob
import os
import numpy as np
import pydicom
from PIL import Image
import concurrent.futures


# DICOMをスケーリング処理をかけて読み込む⇒LとRの文字を埋める⇒端に黒い影がある場合はそれを埋め込むように処理する（画像のサイズが変更しないように）⇒ノイズを除去する⇒ヒストグラムの最大ピークをシフトして輝度値を合わせる
class ImagePreprocessor:
    def __init__(self, dicom_dir, output_dir, preprocessed_dir):
        self.dicom_dir = dicom_dir
        self.output_dir = output_dir
        self.preprocessed_dir = preprocessed_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        self.reference_histogram_peak = None

    # スケーリング処理をする
    def dicom_to_png(self, dicom_path):
        dcm_file = pydicom.dcmread(dicom_path)
        dcm_img = dcm_file.pixel_array
        wc = dcm_file.WindowCenter
        ww = dcm_file.WindowWidth
        ri = dcm_file.RescaleIntercept
        rs = dcm_file.RescaleSlope

        img = dcm_img * rs + ri
        vmax = wc + ww / 2
        vmin = wc - ww / 2
        img = 255 * (img - vmin) / (vmax - vmin)
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)

        dicom_image = Image.fromarray(img)

        file_name = os.path.splitext(os.path.basename(dicom_path))[0] + ".png"
        output_path = os.path.join(self.output_dir, file_name)
        dicom_image.save(output_path)

    # DICOMを変換する処理
    def convert_dicom_to_png(self):
        dicom_files = [
            os.path.join(self.dicom_dir, file_name)
            for file_name in os.listdir(self.dicom_dir)
            if file_name.endswith(".dcm")
        ]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.dicom_to_png, dicom_files)

    # 画像を左端から探索して、黒い影を検出する
    def find_x_coordinate_from_left(self, image, threshold):
        width = image.shape[1]
        for x in range(width):
            column = image[:, x]
            if all(pixel > threshold for pixel in column):
                return x
        return None

    # 画像を右端から探索して、黒い影を検出する
    def find_x_coordinate_from_right(self, image, threshold):
        width = image.shape[1]
        for x in range(width - 1, -1, -1):
            column = image[:, x]
            if all(pixel > threshold for pixel in column):
                return x
        return None

    # 画像を左右から探索して、黒い影をトリミングする
    def trim_image(self, image, threshold=100):
        # 輝度値threshold以下を黒いピクセルと判定
        left_x = self.find_x_coordinate_from_left(image=image, threshold=threshold)
        right_x = self.find_x_coordinate_from_right(image=image, threshold=threshold)

        # トリミングする範囲を計算
        if left_x is not None and right_x is not None:
            trimmed_image = image[:, left_x:right_x]
            return trimmed_image
        else:
            # トリミングが不要な場合は元の画像を返す
            return image

    def process_images(self):
        for i, image_path in enumerate(
            sorted(glob.glob(os.path.join(self.output_dir, "*.png")))
        ):
            print(image_path)

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # inpaintを使用して影やRとL文字を埋め込む
            inpaint_radius = 10
            mask = np.zeros_like(image)
            mask[(image == 0) | (image == 255)] = 255
            inpainted_image = cv2.inpaint(
                image, mask, inpaint_radius, cv2.INPAINT_TELEA
            )

            trim_image = self.trim_image(image=inpainted_image)

            basename = os.path.basename(image_path)
            cv2.imwrite(os.path.join(self.preprocessed_dir, basename), trim_image)


if __name__ == "__main__":
    dicom_dir = "detect_hallux_valgus/src/dicom"
    output_dir = "detect_hallux_valgus/src/images"
    preprocessed_dir = "detect_hallux_valgus/src/preprocessed"
    preprocessor = ImagePreprocessor(dicom_dir, output_dir, preprocessed_dir)
    preprocessor.convert_dicom_to_png()
    preprocessor.process_images()
