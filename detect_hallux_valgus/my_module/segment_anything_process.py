import os
import cv2
import numpy as np
import torch
import os
import time
from detect_HV_points import DetectHVPoints
from detect_M1M5_points import DetectM1M5Point
from get_background_prompt import GetBackGroundPrompt
from segment_anything import sam_model_registry, SamPredictor


class SegmentAnythingProcess:
    def __init__(self, model_checkpoint_path, model_type, device):
        sam = sam_model_registry[model_type](checkpoint=model_checkpoint_path)
        sam = sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def process_image(
        self,
        file_name,
        output_folderHV,
        output_folderM1M5,
        prompt_folder_HV,
        prompt_folder_M1M5,
        prompt_coords_list,
        back_ground_prompt,
        input,
    ):
        start = time.time()
        input_image_path = os.path.join(input_folder_1, file_name)
        image = cv2.imread(input_image_path)
        color = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
        self.predictor.set_image(image)

        ################################################# M1M5角の検出 #################################################

        if input == "1":
            print(
                "-------------------------------input1-------------------------------"
            )
            # プロンプトとなる座標をNumpy配列に変換
            prompt_coords_np = np.array(prompt_coords_list)
            input_label = np.array([1, 0, 0])

            back_R1 = back_ground_prompt[0]
            back_L1 = back_ground_prompt[1]
            back_R2 = back_ground_prompt[2]
            back_L2 = back_ground_prompt[3]

            for idx, _ in enumerate(prompt_coords_np):
                ########### 右足の検出 ###########
                if idx == 0:
                    print("idx0")
                    back_R1 = np.array(back_R1)
                    back_R2 = np.array(back_R2)

                    prompt_coords = np.vstack((prompt_coords_np[0], back_R1, back_R2))
                    print(prompt_coords)

                    masks, scores, logits = self.predictor.predict(
                        point_coords=prompt_coords,
                        point_labels=input_label,
                        multimask_output=False,
                    )

                    # スコアが最大のマスクを取得
                    max_mask = self.get_mask_with_max_score(masks, scores)

                    output_file = os.path.join(
                        output_folderM1M5,
                        f"{file_name.replace('images/', '').replace('.png','')}_right_M1M5.png",
                    )

                ########### 左足の検出 ###########
                elif idx == 1:
                    print("idx1")
                    back_L1 = np.array(back_L1)
                    back_L2 = np.array(back_L2)

                    prompt_coords = np.vstack((prompt_coords_np[1], back_L1, back_L2))
                    print(prompt_coords)

                    masks, scores, logits = self.predictor.predict(
                        point_coords=prompt_coords,
                        point_labels=input_label,
                        multimask_output=False,
                    )

                    # スコアが最大のマスクを取得
                    max_mask = self.get_mask_with_max_score(masks, scores)

                    output_file = os.path.join(
                        output_folderM1M5,
                        f"{file_name.replace('images/', '').replace('.png','')}_left_M1M5.png",
                    )
                    output_file_dst = os.path.join(
                        prompt_folder_M1M5,
                        f"{file_name.replace('images/', '').replace('.png','')}_M1M5.png",
                    )

                    end = time.time()
                    print("検出時間：", end - start)

        ################################################# HV角の検出 #################################################

        elif input == "2":
            # # リストにすることで要素を追加可能
            # prompt_coords_list = list(prompt_coords_HV)
            # プロンプトとなる座標をNumpy配列に変換
            prompt_coords_np = np.array(prompt_coords_list)

            input_label = np.array([1, 1, 0])

            # back ground prompt
            back_R1 = back_ground_prompt[0]
            back_L1 = back_ground_prompt[1]

            # 右足を検出したのち左足を検出する(左右にそれぞれ二つのセグメンテーション対象)
            for idx, prompt_coords in enumerate(prompt_coords_np):
                ###########　右足の検出 ###########

                if idx == 0:
                    back_R1 = np.array(back_R1)
                    prompt_coords = np.vstack((prompt_coords, back_R1))

                    masks, scores, logits = self.predictor.predict(
                        point_coords=prompt_coords,
                        point_labels=input_label,
                        multimask_output=True,
                    )

                    # スコアが最大のマスクを取得
                    max_mask = self.get_mask_with_max_score(masks, scores)

                    output_file = os.path.join(
                        output_folderHV,
                        f"{file_name.replace('images/', '').replace('.png','')}_right_HV.png",
                    )

                ########### 左足の検出 ###########

                elif idx == 1:
                    back_L1 = np.array(back_L1)
                    prompt_coords = np.vstack((prompt_coords, back_L1))

                    masks, scores, logits = self.predictor.predict(
                        point_coords=prompt_coords,
                        point_labels=input_label,
                        multimask_output=True,
                    )

                    # スコアが最大のマスクを取得
                    max_mask = self.get_mask_with_max_score(masks, scores)
                    output_file = os.path.join(
                        output_folderHV,
                        f"{file_name.replace('images/', '').replace('.png','')}_left_HV.png",
                    )
                    output_file_dst = os.path.join(
                        prompt_folder_HV,
                        f"{file_name.replace('images/', '').replace('.png','')}_HV.png",
                    )

                    end = time.time()

    # scoreが最大のmaskを取得する
    def get_mask_with_max_score(self, masks, scores):
        scores = scores.tolist()  # numpy.ndarray をリストに変換
        max_score = max(scores)  # スコアの最大値を取得
        max_index = scores.index(max_score)  # 最大スコアのインデックスを取得
        max_mask = masks[max_index]  # 最大スコアに対応するマスクを取得
        return max_mask


if __name__ == "__main__":
    import time

    input_folder_1 = "./src/images/"
    input_folder_2 = "./src/median"
    output_folderHV = "./assets/MASKHV"
    output_folderM1M5 = "./assets/MASKM1M5"
    prompt_folderHV = "./assets/promptHV"
    prompt_folderM1M5 = "./assets/promptM1M5"

    if not os.path.exists(output_folderHV):
        os.makedirs(output_folderHV)
    if not os.path.exists(output_folderM1M5):
        os.makedirs(output_folderM1M5)
    if not os.path.exists(prompt_folderHV):
        os.makedirs(prompt_folderHV)
    if not os.path.exists(prompt_folderM1M5):
        os.makedirs(prompt_folderM1M5)

    sam_checkpoint = "./study/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    if torch.cuda.is_available():
        device = "cuda"

    segment_anything_process = SegmentAnythingProcess(
        sam_checkpoint, model_type, device
    )
    image_files = os.listdir(input_folder_1)
    for file_name in image_files:
        if file_name.endswith("png"):
            input_image_path = os.path.join(input_folder_1, file_name)

            print(
                "---------------", input_image_path, "--------------------------------"
            )

            start = time.time()

            # append back ground prompt(HV角用)
            get_back_ground_prompt = GetBackGroundPrompt(input_image_path)
            (
                background_R_HV,
                background_L_HV,
                _,
            ) = get_back_ground_prompt.detect_back_ground_point_HV()
            background_point_HV = (background_R_HV, background_L_HV)

            # HV角の検出に必要な座標の取得（input = 2）
            detect_HV = DetectHVPoints(input_image_path)
            prompt_coords_HV = detect_HV.process_image()
            segment_anything_process.process_image(
                file_name,
                output_folderHV,
                output_folderM1M5,
                prompt_folderHV,
                prompt_folderM1M5,
                prompt_coords_HV,
                background_point_HV,
                input="2",
            )

            # append back ground prompt(M1-M5角用)
            (
                background_R_M1M5_1,
                background_L_M1M5_1,
                background_R_M1M5_2,
                background_L_M1M5_2,
                _,
            ) = get_back_ground_prompt.detect_back_ground_point_M1M5()

            background_point_M1M5 = (
                background_R_M1M5_1,
                background_L_M1M5_1,
                background_R_M1M5_2,
                background_L_M1M5_2,
            )

            print("M1M5検出の背景プロンプト", background_point_M1M5)

            # M1-M5角の検出に必要な座標の取得(input = 1)
            detect_M1M5 = DetectM1M5Point(input_image_path)
            prompt_coords_M1M5 = detect_M1M5.process_image()
            print("M1M5検出の背景ではないプロンプト", prompt_coords_M1M5)

            # 以下の画像には処理を行ったのちの画像を使用する
            segment_anything_process.process_image(
                file_name,
                output_folderHV,
                output_folderM1M5,
                prompt_folderHV,
                prompt_folderM1M5,
                prompt_coords_M1M5,
                background_point_M1M5,
                input="1",
            )

            end = time.time()
            print(f"{file_name}の検出時間は{start-end}です。")
