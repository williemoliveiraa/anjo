import json
import time

import cv2  # type:ignore
import numpy as np  # type:ignore
import PIL.Image
import torch  # type:ignore
import torch2trt  # type:ignore
import torchvision.transforms as transforms  # type:ignore
import trt_pose.coco  # type:ignore
import trt_pose.models  # type:ignore
from jetcam.csi_camera import CSICamera  # type:ignore
from torch2trt import TRTModule  # type:ignore
from trt_pose.draw_objects import DrawObjects  # type:ignore
from trt_pose.parse_objects import ParseObjects  # type:ignore

from storage_face import (
    storage_face_by_get_face,
    storage_face_by_hands_sanitation,
    send_image_to_server,
    send_request_to_led,
    save_face_imagem_in_local,
    save_hands_sanitation_imagem_in_local,
)

with open("human_pose.json", "r") as f:
    human_pose = json.load(f)

TOPOLOGY = trt_pose.coco.coco_category_to_topology(human_pose)
NUM_PARTS = len(human_pose["keypoints"])
NUM_LINKS = len(human_pose["skeleton"])
MODEL = trt_pose.models.resnet18_baseline_att(NUM_PARTS, 2 * NUM_LINKS).cuda().eval()
MODEL_WEIGHTS = "resnet18_baseline_att_224x224_A_epoch_249.pth"
MODEL.load_state_dict(torch.load(MODEL_WEIGHTS))
WIDTH = 224
HEIGHT = 224
DATA = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
TRT_MODEL = torch2trt.torch2trt(
    MODEL, [DATA], fp16_mode=True, max_workspace_size=1 << 25
)
OPTIMIZED_MODEL = "resnet18_baseline_att_224x224_A_epoch_249_trt.pth"
torch.save(TRT_MODEL.state_dict(), OPTIMIZED_MODEL)

TRT_MODEL = TRTModule()
TRT_MODEL.load_state_dict(torch.load(OPTIMIZED_MODEL))
MEAN = torch.Tensor([0.485, 0.456, 0.406]).cuda()
STD = torch.Tensor([0.229, 0.224, 0.225]).cuda()
DEVICE = torch.device("cuda")
CAMERA = CSICamera(
    width=WIDTH, height=HEIGHT, capture_width=3264, capture_height=2464, capture_fps=21
)
CAMERA.running = True
WRIST_IN_BBOX = []
ACCEPTABLE_DISTANCE = []
VERIFIED_WRIST = []

PARSE_OBJECTS = ParseObjects(TOPOLOGY)
DRAW_OBJECTS = DrawObjects(TOPOLOGY)

def get_fps():
    t0 = time.time()
    torch.cuda.current_stream().synchronize()
    for _ in range(50):
        _ = TRT_MODEL(DATA)
    torch.cuda.current_stream().synchronize()
    t1 = time.time()
    fps = (50.0 / (t1 - t0))
    print(f"FPS: {fps}")
    return int(fps)

APPLICATION_FPS = get_fps()
APPLICATION_FPS = get_fps()

def preprocess(image):
    global DEVICE
    DEVICE = torch.device("cuda")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(DEVICE)
    image.sub_(MEAN[:, None, None]).div_(STD[:, None, None])
    return image[None, ...]

def seek_peaks(image, persons, normalized_peaks):
    normalized_peaks = normalized_peaks[0]
    persons = persons[0]
    left_eye = 1
    right_eye = 2
    left_wrist = 9
    right_wrist = 10
    left_shoulder = 5
    right_shoulder = 6
    height = image.shape[0]
    width = image.shape[1]
    amount_peaks = 0
    wrist_coordinates = [0, 0, 0, 0]
    face_coordinates = [
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ]  # [left_shoulder, left_eye, right_eye, right_shoulder]
    person_number = 0
    person = persons[person_number]
    for landmark in range(person.shape[0]):
        peak_number = int(person[landmark])
        if peak_number < 0:
            continue
        else:
            amount_peaks += 1
        peak = normalized_peaks[landmark][peak_number]
        x = round(float(peak[1]) * width)
        y = round(float(peak[0]) * height)
        if landmark == left_shoulder:
            face_coordinates[0] = (x, y)
        if landmark == left_eye:
            face_coordinates[1] = (x, y)
        if landmark == right_eye:
            face_coordinates[2] = (x, y)
        if landmark == right_shoulder:
            face_coordinates[3] = (x, y)
        if landmark == left_wrist:
            wrist_coordinates[0] = x
            wrist_coordinates[1] = y
        if landmark == right_wrist:
            wrist_coordinates[2] = x
            wrist_coordinates[3] = y
    return wrist_coordinates, face_coordinates, amount_peaks

# BBOX: [A B
#       D C]
def join_bbox(bbox_max, bbox_min):
    add = 5
    bbox_max_points = bbox_max.tolist()
    bbox_min_points = bbox_min.tolist()
    contour_points = bbox_max_points + bbox_min_points
    x_points: list[int] = [point[0] for point in contour_points]
    y_points: list[int] = [point[1] for point in contour_points]

    min_x = min(x_points)
    min_y = min(y_points)
    max_x = max(x_points)
    max_y = max(y_points)

    min_x -= add * 3
    min_y -= add * 3
    max_x += add * 3
    max_y += add * 3

    bbox = np.int0(
        [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]  # type:ignore
    )
    return bbox

def identify_lines(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 150, 150])
    upper = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    red_contours = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    try:
        ordered_red_contours = sorted(red_contours, reverse=True, key=cv2.contourArea)
        max_red_box = cv2.boxPoints(cv2.minAreaRect(ordered_red_contours[0]))
        min_red_box = cv2.boxPoints(cv2.minAreaRect(ordered_red_contours[1]))
        max_red_box = np.int0(max_red_box)
        min_red_box = np.int0(min_red_box)
        red_bbox = join_bbox(max_red_box, min_red_box)
        cv2.drawContours(image, [red_bbox], 0, (255, 0, 255), 2)
    except:
        red_bbox = np.int0([[0, 0], [0, 0], [0, 0], [0, 0]])
    return red_bbox

def verify_wrist_in_bbox(wrist, bbox):
    return wrist[0] in range(bbox[0][0] + 1, bbox[1][0]) and wrist[1] in range(
        bbox[0][1] + 1, bbox[2][1]
    )

def verify_wrists_in_bbox(image, wrist_coordinates):
    bbox = identify_lines(image)
    left_wrist = np.int0([wrist_coordinates[0], wrist_coordinates[1]])  # type:ignore
    right_wrist = np.int0([wrist_coordinates[2], wrist_coordinates[3]])  # type:ignore
    if verify_wrist_in_bbox(left_wrist, bbox) or verify_wrist_in_bbox(
        right_wrist, bbox
    ):
        WRIST_IN_BBOX.append(True)
        if WRIST_IN_BBOX.count(True) == 3:
            print("pulso por 3x seguidas")
            VERIFIED_WRIST.append(True)
            del WRIST_IN_BBOX[:]
    else:
        del VERIFIED_WRIST[:]
        del WRIST_IN_BBOX[:]

def verify_hands_sanitation(acceptable_distance, image_face):
    if acceptable_distance.count(True) >= int(len(acceptable_distance) * 0.8):
        print(
            str(acceptable_distance.count(True))
            + " >= "
            + str(int(len(acceptable_distance) * 0.8))
        )
        print("Higienização")
        storage_face_by_hands_sanitation(image_face)
        send_request_to_led("hands")
    else:
        print(
            str(acceptable_distance.count(True))
            + " < "
            + str(int(len(acceptable_distance) * 0.8))
        )
        print("NÃO higienização")
    del acceptable_distance[:]

def verify_wrist_union(wrist_coordinates, image_face):
    diff_axis_ok = 30
    left_wrist = np.int0([wrist_coordinates[0], wrist_coordinates[1]])  # type:ignore
    right_wrist = np.int0([wrist_coordinates[2], wrist_coordinates[3]])  # type:ignore

    x_distance = abs(right_wrist[0] - left_wrist[0])  # type:ignore
    y_distance = abs(right_wrist[1] - left_wrist[1])  # type:ignore
    if (
        not 0 in (x_distance, y_distance)
        and x_distance <= diff_axis_ok
        and y_distance <= diff_axis_ok
    ):
        ACCEPTABLE_DISTANCE.append(True)
    else:
        ACCEPTABLE_DISTANCE.append(False)

    if len(ACCEPTABLE_DISTANCE) >= 5 * APPLICATION_FPS:
        verify_hands_sanitation(ACCEPTABLE_DISTANCE, image_face)
        del VERIFIED_WRIST[:]

def get_face(image, face_coordinates):
    required_size = (160, 160)
    if face_coordinates[1] != (0, 0) != face_coordinates[2]:
        start_x = min(face_coordinates[0][0], face_coordinates[3][0])
        end_x = max(face_coordinates[0][0], face_coordinates[3][0])

        start_y = (min(face_coordinates[1][1], face_coordinates[2][1])) - 30
        end_y = max(face_coordinates[0][1], face_coordinates[3][1])
        image_face = image[start_y:end_y, start_x:end_x]
        try:
            image_face = cv2.resize(
                image_face, required_size, interpolation=cv2.INTER_AREA
            )
            storage_face_by_get_face(image_face)
            send_request_to_led("bed_entry")
        except:
            pass
    else:
        image_face = image[0, 0]
    return image_face

def verify_people_on_local(amount_peaks):
    amount_peaks_necessary = 3
    if amount_peaks >= amount_peaks_necessary:
        print("Há pessoa no local")
        send_request_to_led("person")

def execute(change):
    wrist_coordinates = [0, 0, 0, 0]  # x_left, y_left, x_right, y_right
    face_coordinates = [
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ]  # Ombro esquerdo, olho esquerdo, olho direito, ombro direito
    amount_peaks = 0
    image = change["new"]
    data = preprocess(image)
    cmap, paf = TRT_MODEL(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    _, persons, peaks = PARSE_OBJECTS(cmap, paf)
    wrist_coordinates, face_coordinates, amount_peaks = seek_peaks(
        image, persons, peaks
    )
    verify_people_on_local(amount_peaks)
    image_face = get_face(image, face_coordinates)
    if VERIFIED_WRIST.count(True) >= 1:
        verify_wrist_union(wrist_coordinates, image_face)
    else:
        verify_wrists_in_bbox(image, wrist_coordinates)

if __name__ == "__main__":
    print("-" * 5 + " Iniciando App SENFIO - Jetson " + "-" * 5)
    send_request_to_led("init")
    execute({"new": CAMERA.value})
    CAMERA.observe(execute, names="value")
    try:
        while True:
            #save_hands_sanitation_imagem_in_local()
            #save_face_imagem_in_local()
            send_image_to_server("face")
            send_image_to_server("sanitation")
    except:
        print("ERROR")
        CAMERA.unobserve_all()