import base64
import json
from datetime import datetime
import time

import cv2  # type:ignore
from requests import post
from httpx import get
from uuid import getnode as get_mac

SERVER_URL_SANITATION = "http://higienize.senfio.com.br:8080/senfio/HM"
SERVER_URL_FACE = "http://higienize.senfio.com.br:8080/senfio/leito"
FACES_FOUND_HANDS_SANITATION = {}
FACES_FOUND = {}
PATH_TO_SAVE_BY_HANDS_SANITATION = "./hands/"
PATH_TO_SAVE_BY_GET_FACE = "./get_face/"

def storage_face_by_hands_sanitation(image_face):    
    global FACES_FOUND_HANDS_SANITATION
    FACES_FOUND_HANDS_SANITATION.update({get_timestamp():image_face})

def storage_face_by_get_face(image_face):    
    global FACES_FOUND
    FACES_FOUND.update({get_timestamp():image_face})

def send_image_to_server(action):
    global FACES_FOUND_HANDS_SANITATION
    global FACES_FOUND
    dict_to_send = {}
    url_to_send = SERVER_URL_SANITATION
    if action == "sanitation":
        dict_to_send = FACES_FOUND_HANDS_SANITATION
    else:
        dict_to_send = FACES_FOUND
        url_to_send = SERVER_URL_FACE
    if not len(dict_to_send) > 0:
        return None
    for timestamp, image_face in dict_to_send.items():
        break
    _, encoded_image = cv2.imencode(".jpg", image_face)
    encoded_image = encoded_image.tobytes()
    image_64 = cv2_to_base64(encoded_image)
    data = {"images":[image_64], "timestamp":str(timestamp), "mac":str(get_mac_str())}
    try:
        response = post(
            url=url_to_send,
            headers={"Content-type": "application/json"},
            data=json.dumps(data),
            timeout=10,
        ).json()
        if response.ok:
            print(f"Imagem enviada! Status code: {response.status_code}")
        else:
            print(f"Imagem não enviada! Status code: {response.status_code}")
    except:
        pass
    del dict_to_send[timestamp]
    if action == "sanitation":
        FACES_FOUND_HANDS_SANITATION = dict_to_send
    else:
        FACES_FOUND = dict_to_send

def send_request_to_led(function):
    server = "http://localhost:8080/" + str(function)
    try:
        get(server, timeout = 0.01)
    except:
        pass

def cv2_to_base64(image) -> str:
    return base64.b64encode(image).decode("utf8")

def save_hands_sanitation_imagem_in_local():
    global FACES_FOUND_HANDS_SANITATION
    if not len(FACES_FOUND_HANDS_SANITATION) > 0:
        return None     
    for timestamp, image_to_save in FACES_FOUND_HANDS_SANITATION.items():
        break
    cv2.imwrite(f"{PATH_TO_SAVE_BY_HANDS_SANITATION}{str(timestamp)}.jpg", image_to_save)
    print("Salvando imagem em: "+f"{PATH_TO_SAVE_BY_HANDS_SANITATION}{str(timestamp)}.jpg")
    del FACES_FOUND_HANDS_SANITATION[timestamp]

def save_face_imagem_in_local():
    global FACES_FOUND
    if not len(FACES_FOUND) > 0:
        return None
    for timestamp, image_to_save in FACES_FOUND.items():
        break
    cv2.imwrite(f"{PATH_TO_SAVE_BY_GET_FACE}{str(timestamp)}.jpg", image_to_save)
    print("Salvando imagem em: "+f"{PATH_TO_SAVE_BY_GET_FACE}{str(timestamp)}.jpg")
    del FACES_FOUND[timestamp]

def get_datetime():
    return datetime.now()

def get_timestamp():
    return time.time()

def get_mac_str():
    mac = get_mac()
    mac = ':'.join(("%012X" % mac)[i:i+2] for i in range(0, 12, 2))
    return mac