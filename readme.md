# 위성영상을 활용한 선박 탐지 AI 경진대회



## 목차

1. [전처리](#전처리)
2. [네거티브 샘플 추가](#네거티브-샘플-추가)
3. [이미지 복사](#이미지-복사)
4. [CSV 파일 생성](#csv-파일-생성)
5. [라벨 데이터 생성(JSON -> CSV)](#라벨-데이터-생성json---csv)
6. [데이터셋 병합(AIHub + DataON)](#데이터셋-병합aihub--dataon)
7. [Yaml 파일 생성](#yaml-파일-생성)
8. [파인튜닝](#파인튜닝)
9. [학습](#학습)
10. [TASK 데이터 전처리, 추론 및 후처리](#task-데이터-전처리-추론-및-후처리)
11. [Submission 파일 생성](#submission-파일-생성)
12. [제출 코드](#제출-코드)

## 전처리

JSON 파일을 필터링하고 변환하여 새로운 JSON 파일을 생성합니다. 주요 함수는 다음과 같습니다:

- `convert_imcoords_to_bbox(imcoords)`: 이미지 좌표를 바운딩 박스로 변환합니다.
- `filter_and_modify_json(directory, output_directory)`: 지정된 디렉토리 내의 JSON 파일을 필터링하고 변환합니다.
- `check_for_other_classes(directory)`: JSON 파일 내에 특정 클래스 외의 다른 클래스가 있는지 확인합니다.

## 네거티브 샘플 추가

배가 없는 빈 JSON 파일을 생성합니다. 주요 함수는 다음과 같습니다:

- `create_empty_ship_json(input_directory, output_directory, num_empty_files=80)`: 지정된 수만큼 빈 JSON 파일을 생성합니다.

## 이미지 복사

JSON 파일을 기반으로 해당 이미지 파일을 복사합니다. 주요 함수는 다음과 같습니다:

- `get_and_copy_img(json_path, img_path, save_dir)`: JSON 파일을 기반으로 이미지를 찾아 복사합니다.

## CSV 파일 생성

JSON 파일과 이미지 파일 경로를 CSV 파일로 저장합니다. 주요 함수는 다음과 같습니다:

- `get_img_json_list(json_path, img_path)`: JSON 파일과 이미지 파일 경로를 리스트로 반환합니다.
- `save_to_csv(data, output_csv)`: 경로 정보를 CSV 파일로 저장합니다.

## 라벨 데이터 생성(JSON -> CSV)

JSON 파일에서 바운딩 박스 좌표를 추출하여 CSV 파일로 저장합니다. 주요 함수는 다음과 같습니다:

- `get_bbox_from_json(json_file)`: JSON 파일에서 바운딩 박스 좌표를 추출합니다.
- `save_bbox_to_txt(bbox_list, output_file, img_path)`: 바운딩 박스 정보를 txt 파일로 저장합니다.
- `visualize_image_with_coords(image_path, coords_list)`: 이미지와 바운딩 박스를 시각화합니다.

## 데이터셋 병합(AIHub + DataON)

기존 클래스 ID를 새 클래스 ID로 매핑하여 라벨 파일과 이미지 파일을 병합합니다. 주요 함수는 다음과 같습니다:

- `update_label_file(file_path, new_file_path)`: 라벨 파일을 업데이트하여 새 디렉터리에 저장합니다.

## Yaml 파일 생성

데이터셋 경로와 클래스 정보를 포함한 YAML 파일을 생성합니다.

## 파인튜닝

YOLO11x-obb 모델을 사용하여 하이퍼파라미터 튜닝을 수행합니다.

## 학습

YOLO 모델을 사용하여 객체 탐지 모델을 학습합니다.

## TASK 데이터 전처리, 추론 및 후처리

이미지를 클러스터링하고 크롭하여 모델에 입력한 후, 예측 결과를 후처리합니다. 주요 함수는 다음과 같습니다:

- `get_imglist(dir)`: 이미지 파일 목록을 가져옵니다.
- `create_temp_dir(base_dir)`: 임시 폴더를 생성합니다.
- `cluster_and_crop(image_path, k, size, crop_num, threshold)`: 이미지를 클러스터링하고 크롭합니다.
- `save_cropped_patches_as_numpy(image_path, crop_size, resize_size, cluster_img_size, cluster_threshold, save_dir, is_cluster)`: 크롭된 패치를 NumPy 배열로 저장합니다.
- `run_model(image_names, images, positions, model, scale_factor, device, result)`: 모델을 실행하고 결과를 저장합니다.

## Submission 파일 생성

예측 결과를 CSV 파일로 저장합니다.

