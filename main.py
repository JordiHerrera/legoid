from classes.legoPiece import LegoPiece, print_piece
from funcions.readImages import get_subimages, threshold_v2, select_image_file, correccio_calid
from funcions.findStuds import regions, filtrar_regions
from funcions.gridPatterns import find_grid_patterns_aprox, find_closest_pair
from funcions.centroids import calculate_centroids
from funcions.color import get_dominant_hex, get_mean_hex, closest_color_name, hex2rgb, double_check_gray
from funcions.thickness import brick_or_plate
from funcions.apiCall import predict_lego_part, strings_match

from funcions_set.findSet import find_matching_sets, generate_combinations

import cv2
from collections import Counter
import json
import pandas as pd

print(
    r'''
 .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
| |   _____      | || |  _________   | || |    ______    | || |     ____     | || |     _____    | || |  ________    | |
| |  |_   _|     | || | |_   ___  |  | || |  .' ___  |   | || |   .'    `.   | || |    |_   _|   | || | |_   ___ `.  | |
| |    | |       | || |   | |_  \_|  | || | / .'   \_|   | || |  /  .--.  \  | || |      | |     | || |   | |   `. \ | |
| |    | |   _   | || |   |  _|  _   | || | | |    ____  | || |  | |    | |  | || |      | |     | || |   | |    | | | |
| |   _| |__/ |  | || |  _| |___/ |  | || | \ `.___]  _| | || |  \  `--'  /  | || |     _| |_    | || |  _| |___.' / | |
| |  |________|  | || | |_________|  | || |  `._____.'   | || |   `.____.'   | || |    |_____|   | || | |________.'  | |
| |              | || |              | || |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'                                                   
    '''
)

debugs_text = False

# root_dir = os.getcwd()
# imgs_dir = f'{root_dir}/images/' debug c trencada
root_dir = 'C:/lego/'

data_dir = f'{root_dir}data/'
imgs_dir = f''

print('Carregant bases de dades')

print('\tCombinacions de peces', end=" ")
with open(f'{data_dir}piece_combinations.json', 'r') as file:
    grid_patterns = json.load(file)
print('✓')

print('\tLlista completa de colors format HEX', end=" ")
df_colors_hex = pd.read_csv(f'{data_dir}colors_hex.csv')
df_colors_hex['hex'] = df_colors_hex['hex'].astype(str)
df_colors_hex['rgb'] = df_colors_hex['hex'].apply(hex2rgb)

df_colors_hex = df_colors_hex[~df_colors_hex['name'].str.contains('Chrome', case=False, na=False)]
df_colors_hex = df_colors_hex[df_colors_hex['is_trans'] == 'f']
# chrome i transparent poc detectables amb imatge, possibles confusions

llista_colors = df_colors_hex['simplified'].str.lower().unique().tolist()

print('✓')

image_subject = select_image_file('Imatge de peces estandard')
api_subject = select_image_file('Imatge de peces no estandard')

sbj_path = f'{imgs_dir}{image_subject}'

start_image = cv2.imread(sbj_path)
api_cv2 = cv2.imread(f'{imgs_dir}{api_subject}')

og_image = correccio_calid(start_image, False)
api_cv2 = correccio_calid(api_cv2, False)

h_og, w_og, c_og = og_image.shape

piece_array = get_subimages(og_image, debug=False)

test_array = []
# test_array.append(piece_array[3])  # 2X2 BROWN BRICK
# test_array.append(piece_array[2])  # 1X6 GRAY PLATE
# test_array.append(piece_array[4])  # 1X6 GRAY PLATE

defined_pieces = []
api_validation = []
print('')
counter = 0

print(55*'=+')
print('Iniciant fase de peces estandard')
print(55*'=+')

for current_piece in piece_array:
    api_res = False

    current_piece.thresholded_image = threshold_v2(current_piece.base_image)
    res = regions(current_piece.thresholded_image, False)
    studs = filtrar_regions(res, 7, 65, debugs_text)

    current_piece.contours = studs
    current_piece.stud_count = len(studs)
    centroids = calculate_centroids(studs)

    h, w, c = current_piece.base_image.shape

    if len(centroids) > 0:
        try:
            ret = predict_lego_part(current_piece.base_image, debug=False)


            score = float(ret['items'][0]['score'])
            if score > 0.75:
                final_piece = LegoPiece(current_piece.base_image, current_piece.base_image)
                api_res = True
                ret_code = ret['items'][0]['id']
                ret = ret['items'][0]['name']
                #print(f'{ret} : {ret_code} ---> {score}')

                final_piece.given_name = ret
                api_validation.append(final_piece)
        except:
            #print('sense resposta de la api')
            continue




        # thersholding per centroides i per dubimatges diferents

        current_piece.centroids = centroids
        # print(current_piece.centroids)

        combinacio_ret = find_grid_patterns_aprox(centroids)  # acabr guardant a grid_pattern !!
        combinacio_pre = [min(combinacio_ret), max(combinacio_ret)]

        # print(f'Combinacio sense tractar: {combinacio_pre[0]}x{combinacio_pre[1]}')
        current_count = min(grid_patterns.keys(), key=lambda k: abs(int(k) - current_piece.stud_count))
        existing_patterns = grid_patterns[str(current_count)]

        combinacio = find_closest_pair(combinacio_pre, existing_patterns)
        current_piece.grid_pattern = combinacio

        # print(f'Combinacio obtinguda: {current_piece.grid_pattern[0]}x{current_piece.grid_pattern[1]}\n')

        hex_detectat = get_dominant_hex(current_piece.base_image, debug=False)
        #print(hex_detectat, 33*'=')
        current_piece.color_hex = hex_detectat

        color_match = closest_color_name(hex_detectat, df_colors_hex, debug=debugs_text)
        current_piece.color_simple = color_match

        isPlate = brick_or_plate(current_piece, debug=False)
        current_piece.is_plate = isPlate

        current_piece.given_name = f'{current_piece.is_plate} {current_piece.grid_pattern[0]} x {current_piece.grid_pattern[1]}'

        if api_res:
            #print(f'validant: {current_piece.given_name.lower()} amb {final_piece.given_name.lower()}')

            val = strings_match(final_piece, current_piece)

            if val:
                if debugs_text:
                    print(f'Peça estandard identificada: {final_piece.given_name}')
                current_piece.grid_pattern = final_piece.grid_pattern
                current_piece.given_name = final_piece.given_name.lower()
                current_piece.is_uneven = False
                #current_piece.given_name = f'{current_piece.is_plate} {current_piece.grid_pattern[0]} x {current_piece.grid_pattern[1]}'
                defined_pieces.append(current_piece)

    else:
        a = 1
        # print(f'Subimatge sense centroides\n')

    #print('')

print('')
print('Completada ✓\n')
if debugs_text:
    for i in range(len(defined_pieces)):
        print(f'Piece {i}')
        print_piece(defined_pieces[i])
        print('')


# input('Enter to continue to uneven pieces ')
# ======================================================================
# API CALL PER PECES RARES
# ======================================================================

print(55*'=+')
print('Iniciant fase de peces no estandard')
print(55*'=+')

api_array = get_subimages(api_cv2)
#api_array = get_subimages(cv2.imread(f'{imgs_dir}{image_subject}'))


api_pieces = []

for current_piece in api_array:
    ret = predict_lego_part(current_piece.base_image, debug=False)
    try:
        score = float(ret['items'][0]['score'])
        if score > 0.75:

            '''cv2.imshow('SENSE FONS', current_piece.nobg_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''

            final_piece = LegoPiece(current_piece.base_image, current_piece.base_image)

            #hex_aprox = get_dominant_hex(current_piece.base_image)
            hex_aprox = get_dominant_hex(current_piece.nobg_image)

            color = closest_color_name(hex_aprox, df_colors_hex, debugs_text)


            final_piece.color_simple = color
            final_piece.color_hex = hex_aprox

            ret_code = ret['items'][0]['id']
            ret = ret['items'][0]['name']
            # print(f'{color} : {hex_aprox} | {ret} : {ret_code} ---> {score}')

            final_piece.given_name = ret

            final_piece.is_uneven = True
            api_pieces.append(final_piece)
        else:
             # print('passing, score too low')
            pass
    except:
        # print('passing, not printable')
        # print(e)
        pass

print('\n\n\n')

print('Completada ✓\n')

print('Peces identificades')

for p in defined_pieces:
    print(p.given_name, p.color_simple)

print('')

for p in api_pieces:
    print(p.given_name, p.color_simple)

# primer peces standard
# seguir amb peces uneven

df_database = pd.read_csv(f'{data_dir}complete_ref.csv')
df_database['part_name'] = df_database['part_name'].str.lower()
df_database['part_name'] = df_database['part_name'].str.lower()
df_database['part_name'] = df_database['part_name'].str.replace(r'\[.*?\]', '', regex=True)

# print('\n\n')
# print(f'Mida original: {len(df_database)}')

complete_piecs = defined_pieces + api_pieces


cross_list = list(generate_combinations(complete_piecs, 0.6))
total_iters = len(cross_list)

results = []

counter = 0
total_iters = len(cross_list)

print('')
print(55*'=+')
print(f'Iniciant cerca de sets ({df_database["set_num"].nunique()} possibles)')
print(55*'=+', '\n\n')

for current_combination in cross_list:
    counter += 1
    df_iter = df_database.copy()
    for current_piece in current_combination:
        name = current_piece.given_name
        color = current_piece.color_simple
        even_type = current_piece.is_uneven

        #print(name, color, even_type)

        df_filter = find_matching_sets(name.lower(), color, df_iter, 90, is_uneven=even_type)
        filter_list = list(df_filter['set_num'])

        df_iter = df_iter[df_iter['set_num'].isin(filter_list)]

        #print(f"{len(df_iter)} --> {df_iter['set_num'].nunique()} sets unics")
        #print(df_iter['set_num'].unique(), '\n')

    info = f'{round(counter/total_iters*100, 2)}%'
    print(info, end="\r")
    #print(df_iter['set_num'].unique().tolist())
    results = results + df_iter['set_num'].unique().tolist()

print('Completada ✓\n')

counts = Counter(results)
# duplicates = [item for item, count in counts.items() if count > 1]
duplicates = {item: count for item, count in counts.items() if count > 1}

detectat = max(duplicates.values())

# Collect all keys with the maximum value
duplicates_list = [key for key, value in duplicates.items() if value == detectat]

print('')
print("Sets identificats:", duplicates, '\n')

print(f'Instruccions de les millor coincidencies disponibles a:')
for lego_set in duplicates_list:
    print(f'{lego_set} | https://www.lego.com/en-us/service/building-instructions/{lego_set.split("-")[0]}')

exit_q = input('Prem enter per sortir')
