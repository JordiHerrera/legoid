import cv2
import requests
from rapidfuzz.fuzz import token_set_ratio
import re

def predict_lego_part(cv2_image, debug=False):

    if debug:
        cv2.imshow('Imatge original', cv2_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    api_url = "https://api.brickognize.com/predict/parts/"

    success, encoded_image = cv2.imencode(".jpg", cv2_image)
    if not success:
        print("Failed to encode image.")
        return None

    files = {
        "query_image": ("image.jpg", encoded_image.tobytes(), "image/jpeg")
    }

    headers = {
        "accept": "application/json"
    }

    response = requests.post(api_url, headers=headers, files=files)

    if response.status_code == 200:
        json = response.json()
        if len(json['items']) > 0:
            if 'round' in json['items'][0]['name'].lower() and '1 x 1' in json['items'][0]['name'].lower():
                return None
        return json
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def strings_match(api_piece, alg_piece, threshold: float = 90.0, debug=False):
    match = re.search(r'(\d+)\s*x\s*(\d+)', api_piece.given_name)
    if match:
        num1, num2 = map(int, match.groups())
        api_piece.grid_pattern = [num1, num2]
    else:
        if debug:
            print('given name sense grid')
        return False

    if debug:
        print(55 * ':·')
        print('api:', api_piece.grid_pattern)
        print('alg:', alg_piece.grid_pattern)
        print(55 * ':·')


    if api_piece.grid_pattern != alg_piece.grid_pattern:
        if abs(alg_piece.grid_pattern[0]-api_piece.grid_pattern[0]) <= 2 and abs(alg_piece.grid_pattern[1]-api_piece.grid_pattern[1]) <= 2:
            return True
        elif api_piece.grid_pattern[0]*api_piece.grid_pattern[1] == alg_piece.grid_pattern[0]*alg_piece.grid_pattern[1]:
            return True
        return False

    api_string = api_piece.given_name
    alg_string = alg_piece.given_name

    similarity = token_set_ratio(api_string.lower(), alg_string)
    if similarity >= threshold:
        return similarity >= threshold
    else:
        if 'plate' in alg_string:
            alg_string = alg_string.replace('plate', 'brick')
        else:
            alg_string = alg_string.replace('brick', 'plate')

        similarity = token_set_ratio(api_string, alg_string)

        return similarity >= threshold

