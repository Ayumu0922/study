import os
import cv2
import numpy as np
import torch
import os
import time
import glob
import sys

sys.path.append("/home/kubota/study/detect_hallux_valgus")


from segment_anything import sam_model_registry, SamPredictor
from get_prompt_position import GetPromptPosition


class SegmentAnythingProcess:
    def __init__(self, image, model_checkpoint_path, model_type, device):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam = sam_model_registry[model_type](checkpoint=model_checkpoint_path)
        sam = sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def make_mask(
        self,
        prompt_lists,
    ):
        self.predictor.set_image(self.image)

        # 各座標を2次元numpy配列に変換
        prompt_lists = [
            coord for sublist in prompt for pair in sublist for coord in pair
        ]
        prompt_lists = np.array(prompt_lists).reshape(-1, 2)

        print(prompt_lists)

        input_label = np.array([1, 1, 0, 0, 0, 0])

        masks, scores, logits = self.predictor.predict(
            point_coords=prompt_lists,
            point_labels=input_label,
            multimask_output=False,
        )

        # スコアが最大のマスクを取得
        max_mask = self.get_mask_with_max_score(masks, scores)

        return max_mask

    # scoreが最大のmaskを取得する
    def get_mask_with_max_score(self, masks, scores):
        scores = scores.tolist()  # numpy.ndarray をリストに変換
        max_score = max(scores)  # スコアの最大値を取得
        max_index = scores.index(max_score)  # 最大スコアのインデックスを取得
        max_mask = masks[max_index]  # 最大スコアに対応するマスクを取得
        return max_mask


if __name__ == "__main__":
    sam_checkpoint = "detect_hallux_valgus/my_module/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    input_dir = "detect_hallux_valgus/src/preprocessed"
    output_dir = "detect_hallux_valgus/output/mask"
    os.makedirs(output_dir, exist_ok=True)

    for image_path in glob.glob(os.path.join(input_dir, "*.png")):
        print(
            "====================================",
            image_path,
            "================================================",
        )
        image = cv2.imread(image_path)
        get_prompt_position = GetPromptPosition(image=image)

        segment_anything_process = SegmentAnythingProcess(
            image=image,
            model_checkpoint_path=sam_checkpoint,
            model_type=model_type,
            device=device,
        )

        # promptの座標位置取得
        M1, M2, P1, M5, background1, background3 = get_prompt_position.get_prompt()

        # cv2.circle(
        #     img=image,
        #     center=M1[0],
        #     radius=10,
        #     color=(0, 0, 255),
        #     thickness=-1,
        # )

        # cv2.circle(
        #     img=image,
        #     center=M1[1],
        #     radius=10,
        #     color=(0, 0, 255),
        #     thickness=-1,
        # )
        # cv2.circle(
        #     img=image,
        #     center=M2[0],
        #     radius=10,
        #     color=(0, 0, 255),
        #     thickness=-1,
        # )

        # cv2.circle(
        #     img=image,
        #     center=M2[1],
        #     radius=10,
        #     color=(0, 0, 255),
        #     thickness=-1,
        # )

        # cv2.circle(
        #     img=image,
        #     center=P1[0],
        #     radius=10,
        #     color=(0, 0, 255),
        #     thickness=-1,
        # )

        # cv2.circle(
        #     img=image,
        #     center=P1[1],
        #     radius=10,
        #     color=(0, 0, 255),
        #     thickness=-1,
        # )

        # cv2.circle(
        #     img=image,
        #     center=M5[0],
        #     radius=10,
        #     color=(0, 0, 255),
        #     thickness=-1,
        # )

        # cv2.circle(
        #     img=image,
        #     center=M5[1],
        #     radius=10,
        #     color=(0, 0, 255),
        #     thickness=-1,
        # )

        # cv2.circle(
        #     img=image,
        #     center=background[0],
        #     radius=10,
        #     color=(0, 255, 0),
        #     thickness=-1,
        # )

        # cv2.circle(
        #     img=image,
        #     center=background[1],
        #     radius=10,
        #     color=(0, 255, 0),
        #     thickness=-1,
        # )

        prompt = M1, background1, background3

        mask = segment_anything_process.make_mask(prompt_lists=prompt)

        # マスク画像を保存
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        # cv2.imwrite(output_path, image)
        cv2.imwrite(output_path, np.uint8(mask) * 255)
