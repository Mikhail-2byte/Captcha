import json
import time
import base64
import requests
from bs4 import BeautifulSoup
from flask import Flask, request
from loguru import logger
from datetime import date
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import onnxruntime as ort  # Работа с ONNX-моделью
from collections import defaultdict
import logging

# Константы
MAX_ATTEMPTS = 1
HTML_PARSER = 'lxml'
DOMAIN = 'https://pb.nalog.ru'
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0'
MODEL_PATH = 'C:/Captcha/model.onnx'  # Укажите путь к вашей модели

app = Flask(__name__)
today = date.today()
logFileName = 'main ' + str(today)
logger.add(logFileName)

# Загрузка модели ONNX
logger.info("Загрузка модели ONNX...")
model = ort.InferenceSession(MODEL_PATH)
logger.info("Модель загружена.")


def do_http_request(
    session: requests.Session,
    url: str,
    method: str = 'GET',
    dont_log: bool = False,
    no_proxy: bool = False,
    **kwargs
) -> requests.Response:
    # это если будут прокси, пока их нет
    # proxies = {} if no_proxy else random.choice(proxies_list)
    proxies = None

    if 'headers' in kwargs:
        kwargs['headers']['User-Agent'] = USER_AGENT
    else:
        kwargs['headers'] = {'User-Agent': USER_AGENT}

    try:
        response = session.request(method, url, proxies=proxies, **kwargs)
        logger.info(f'[{response.status_code}] {url}')
        # if not dont_log:
        #     logger.info(f'[{response.status_code}] {url}')
        # else:
        #     logger.info(f'[{response.status_code}] {url}')
        return response
    except Exception as e:
        logger.error(f'Ошибка во время HTTP запроса к {url}: {e}')
        return do_http_request(session, url, method, dont_log, no_proxy, **kwargs)

def clean_image(image_path):
    logger.debug(f"Чистка изображения: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error(f"Изображение не найдено: {image_path}")
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    logger.debug("Начало очистки изображения...")
    _, binary = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(binary)

    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < 10 or w < 10:
            cv2.drawContours(clean, [contour], -1, 0, thickness=cv2.FILLED)

    result = cv2.bitwise_not(clean)
    logger.debug("Очистка изображения завершена.")
    return result

def predict_digits(image_path, model):
    cleaned_image = clean_image(image_path)
    cleaned_path = 'cleaned_image.jpg'
    cv2.imwrite(cleaned_path, cleaned_image)

    digits, _ = find_contours(cleaned_path)

    predictions = []
    for (x, y, w, h) in digits:
        img = Image.open(cleaned_path).crop((x, y, x + w, y + h)).resize((24, 44))
        img_array = np.array(img) / 255.0 
        img_array = img_array[np.newaxis, ..., np.newaxis]

        prediction = model.run(None, {"input": img_array.astype(np.float32)})[0]
        predicted_label = np.argmax(prediction)
        predictions.append(predicted_label)

    result = ''.join(map(str, predictions))
    logger.info(f'Распознанная последовательность: {result}')
    return result



def find_contours(image_path):
    logger.debug(f"Поиск контуров в изображении: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 10 < h < 60 and 10 < w < 50:  # Фильтр по размеру 
            digits.append((x, y, w, h))

    digits = sorted(digits, key=lambda b: b[0])  # Сортировка по X
    logger.debug(f"Найдено контуров: {len(digits)}")
    return digits, thresh

def solve_captcha_with_model(captcha_image: bytes, model) -> str:
    logger.debug("Решение капчи с использованием модели...")
    try:
        with open('captcha.jpg', 'wb') as f:
            f.write(captcha_image)
        captcha_code = predict_digits('captcha.jpg', model)
        if not captcha_code:
            raise ValueError("Не удалось распознать капчу")
        logger.info(f"Решенная капча: {captcha_code}")
        return captcha_code
    except Exception as e:
        logger.error(f'Ошибка при решении капчи: {e}')
        return ""




def get_and_solve_captcha(session: requests.Session) -> str:
    logger.info("Запрос страницы капчи...")
    response = do_http_request(session, f'{DOMAIN}/captcha-dialog.html')

    if response is None or response.status_code != 200:
        logger.error(f"Ошибка при получении страницы капчи: {response}")
        return ""

    logger.debug("Анализ HTML-страницы для получения изображения и токена капчи...")
    soup = BeautifulSoup(response.text, 'html.parser')

    captcha_image_url = DOMAIN + soup.find('img').get('src')
    captcha_token = soup.find('input', {'name': 'captchaToken'}).get('value')
    logger.debug(f'URL изображения капчи: {captcha_image_url}')
    logger.debug(f'Токен капчи: {captcha_token}')
    captcha_image = do_http_request(session, captcha_image_url, dont_log=True).content
    captcha_code = solve_captcha_with_model(captcha_image, model)

    if not captcha_code:
        logger.error("Не удалось распознать капчу, пробуем снова...")
        return get_and_solve_captcha(session)

    logger.info(f"Распознанная капча: {captcha_code}")

    response = do_http_request(
        session, 'https://pb.nalog.ru/captcha-proc.json', 'POST',
        data={
            'captcha': captcha_code,
            'captchaToken': captcha_token
        }
    ).json()
    print("код капчи в get_and_solve_captcha", captcha_code)
    if 'ERRORS' in response:
        logger.error(f"Капча решена неверно: {response['ERRORS']}, пробуем снова...")
        return get_and_solve_captcha(session)

    logger.info("Капча успешно решена.")
    return response


def get_request_id(session: requests.Session, inn: int | str, captcha_code: str) -> str:
    logger.info(f"Получение request_id для ИНН: {inn}")
    # Получаем капчу перед отправкой запроса
    captcha_code = get_and_solve_captcha(session)
    captcha_code = captcha_code.strip()

    headers = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'X-Requested-With': 'XMLHttpRequest',
        'Origin': 'https://pb.nalog.ru',
        'Connection': 'keep-alive',
        'Referer': 'https://pb.nalog.ru/search.html',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
    }


    data = {
        'mode': 'search-all',
        'queryAll': f'{inn}',
        'queryUl': '',
        'okvedUl': '',
        'regionUl': '',
        'statusUl': '',
        'isMspUl': '',
        'mspUl1': '1',
        'mspUl2': '1',
        'mspUl3': '1',
        'queryIp': '',
        'okvedIp': '',
        'regionIp': '',
        'statusIp': '',
        'isMspIp': '',
        'mspIp1': '1',
        'mspIp2': '1',
        'mspIp3': '1',
        'queryUpr': '',
        'uprType1': '1',
        'uprType0': '1',
        'queryRdl': '',
        'dateRdl': '',
        'queryAddr': '',
        'regionAddr': '',
        'queryOgr': '',
        'ogrFl': '1',
        'ogrUl': '1',
        'ogrnUlDoc': '',
        'ogrnIpDoc': '',
        'npTypeDoc': '1',
        'nameUlDoc': '',
        'nameIpDoc': '',
        'formUlDoc': '',
        'formIpDoc': '',
        'ifnsDoc': '',
        'dateFromDoc': '',
        'dateToDoc': '',
        'page': '1',
        'pageSize': '10',
        'pbCaptchaToken': captcha_code.strip(),
        'token': '',
    }

    logger.debug(f"Отправка запроса для ИНН {inn} с данными: {data}")
    response = do_http_request(
        session, 'https://pb.nalog.ru/search-proc.json', 'POST',
        headers=headers, data=data
    ).json()

    logger.info(f"Отправляется капча в get_request_id:{captcha_code}")
    logger.info(f"Ответ от API для ИНН {inn}: {response}")

    if 'ERRORS' in response:
        captcha_code = get_and_solve_captcha(session)
        return get_request_id(session, inn, captcha_code)
    
    logger.info(f"Получен request_id: {id}")
    return response['id']


    # if 'ERRORS' in response:
    #     if 'Требуется ввести цифры с картинки' in response['ERRORS']:
    #         logger.error("Капча решена неверно, пробуем снова...")
    #         return get_request_id(session, inn)  # Рекурсивный вызов для новой капчи
    #     else:
    #         logger.error(f"Ошибка от API: {response['ERRORS']}")
    #         return None

    # request_id = response.get('id')
    # logger.info(f"Получен request_id: {request_id}")
    # return request_id



def get_response(session: requests.Session, request_id: str) -> dict:
    logger.info(f"Запрос результата для request_id: {request_id}")
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'X-Requested-With': 'XMLHttpRequest',
        'Origin': 'https://pb.nalog.ru',
        'Connection': 'keep-alive',
        'Referer': 'https://pb.nalog.ru/search.html',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
    }

    data = {
        'id': request_id,
        'method': 'get-response',
    }

    logger.debug(f"Отправка запроса для получения данных по request_id: {request_id}")
    return do_http_request(
        session,
        'https://pb.nalog.ru/search-proc.json',
        'POST',
        headers=headers,
        data=data,
    ).json()


def get_request_id_for_company(
    session: requests.Session,
    token: str,
    referer: str,
    captcha_code: str
) -> tuple[str, str]:
    logger.info("Запрос request_id для компании...")
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'X-Requested-With': 'XMLHttpRequest',
        'Origin': 'https://pb.nalog.ru',
        'Connection': 'keep-alive',
        'Referer': referer,
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
    }

    data = {
        'token': token,
        'method': 'get-request',
        'pbCaptchaToken': captcha_code,
    }

    logger.debug(f"Отправляем данные: {data}")

    try:
        response = do_http_request(
            session, 'https://pb.nalog.ru/company-proc.json', 'POST',
            headers=headers, data=data
        ).json()

        if 'ERRORS' in response:
            logger.warning(f"Получена ошибка: {response['ERRORS']}. Пробуем снова...")
            captcha_code = get_and_solve_captcha(session)
            return get_request_id_for_company(session, token, referer, captcha_code)

        logger.info(f"Получены request_id и token: {response['id']}, {response['token']}")
        return response['id'], response['token']

    except Exception as e:
        logger.exception(f"Ошибка при получении request_id: {e}")
        return None, None



def get_rsmpcategory(catid: float) -> str:
    logger.debug(f"Получение категории РСМП для ID: {catid}")

    if not catid:
        return ''

    if catid == 1.0:
        return 'Микропредприятие'
    if catid == 2.0:
        return 'Малое предприятие'
    else:
        return 'Среднее предприятие'

def get_response_company(session: requests.Session, rid: str, token: str, referer: str) -> dict:
    logger.info(f"Запрос данных для компании с request_id: {rid}")
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'X-Requested-With': 'XMLHttpRequest',
        'Origin': 'https://pb.nalog.ru',
        'Connection': 'keep-alive',
        'Referer': referer
    }
    # print (token)
    data = {
        'token': token,
        'id': rid,
        'method': 'get-response',
    }

    logger.debug(f"Отправляем запрос с данными: {data}")

    try:
        response = do_http_request(
            session, 'https://pb.nalog.ru/company-proc.json', 'POST',
            headers=headers, data=data
        ).json()

        logger.info(f"Получен ответ: {response}")

        # Сохраняем ответ в файл для отладки
        with open('proc.json', 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=4, ensure_ascii=False)

        vyp = response.get('vyp', {})

        # Извлекаем данные о руководителе и адресе
        gen_dir, gen_dir_pos, gen_dir_dost = None, None, None
        directors = vyp.get('masruk')
        if directors:
            gen_dir = directors[0].get('name')
            gen_dir_pos = directors[0].get('position')
            gen_dir_dost = directors[0].get('invalid')

        ur_address_dost = None
        ur_address_dostf = vyp.get('СвНедАдресЮЛ')
        if ur_address_dostf:
            ur_address_dost = ur_address_dostf[0].get('ТекстНедАдресЮЛ')

        result = {
            'token': token,
            'egrul_status': vyp.get('НаимСтатусЮЛ'),
            'ur_address': vyp.get('АдресРФ'),
            'ur_address_dost': ur_address_dost,
            'gen_dir': gen_dir,
            'gen_dir_pos': gen_dir_pos,
            'gen_dir_dost': gen_dir_dost,
            'human_amount': vyp.get('sschr'),
            'nalog_sum': vyp.get('taxpaysum'),
            'nalog_year': vyp.get('taxpay_yearcode'),
            'revenue_sum': vyp.get('revenuesum'),
            'revenue_year': vyp.get('form1_yearcode'),
            'nalog_debt_sum': vyp.get('totalarrearsum'),
            'nalog_debt_year': vyp.get('arrear_yearcode'),
            'mspstatus': get_rsmpcategory(vyp.get('rsmpcategory')),
            'mspupdate': vyp.get('rsmpdate'),
        }

        logger.info(f"Результат обработки: {result}")
        return result

    except Exception as e:
        logger.exception(f"Ошибка при получении ответа для компании: {e}")
        return {}

def get_extra_data(session: requests.Session, token: str) -> dict:
    referer = f'https://pb.nalog.ru/company.html?token={token}'
    logger.info(f"Запрос дополнительных данных для token: {token}")
    
    do_http_request(session, referer, 'GET', True)
    rid, token = get_request_id_for_company(session, token, referer, '')
    logger.info(f"Получены request_id: {rid} и token: {token}")

    logger.debug("Спим 1 секунду...")
    time.sleep(1)

    try:
        return get_response_company(session, rid, token, referer)
    except Exception as e:
        logger.exception(f"Ошибка при получении данных для token: {token}")
        return get_extra_data(session, token)


def prepare_response(session: requests.Session, nalog_response: dict, inn: str | int):
    inn = str(inn)
    logger.info(f"Подготовка ответа для ИНН: {inn}")

    uls = nalog_response.get('ul', {}).get('data', [])
    for ul in uls:
        if ul['inn'] == inn:
            try:
                logger.info(f"Получение дополнительных данных для ИНН: {inn}")
                extra_data = get_extra_data(session, ul['token'])
            except (AttributeError, Exception) as e:
                logger.exception(f"Не удалось получить статус по ЕГРЮЛ, ИНН: {inn}")
                extra_data = {'egrul_status': 'Не удалось получить'}

            response = {
                'success': True,
                'status': ul['sulst_name_ex'],
                'full_name': ul['namep'],
                'short_name': ul.get('namec', ul['namep']),
                'inn': inn,
                'registration_date': ul['dtreg'],
                **extra_data
            }
            logger.debug(f"Ответ сформирован: {response}")
            return response

    logger.warning(f"ИНН {inn} не найден")
    return {'success': False, 'error': 'По данному ИНН ничего не найдено'}

def get_company_info(inn: int | str) -> dict:
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})

    logger.info(f"Запрос информации о компании с ИНН: {inn}")
    rid = get_request_id(session, inn, '')
    logger.debug(f"Получен request_id: {rid}")

    logger.debug("Спим 1 секунду...")
    time.sleep(1)

    nalog_response = get_response(session, rid)
    response = prepare_response(session, nalog_response, inn)

    if response['success']:
        logger.info(f"ИНН {inn}: Данные найдены")
    else:
        logger.info(f"ИНН {inn}: Данные не найдены")

    return response


def try_to_get_company_info(inn: int | str, attempt: int = 0) -> tuple[int, dict]:
    logger.info(f"Попытка получить информацию по ИНН: {inn}, Попытка: {attempt}")

    if not inn or not inn.strip().isdigit():
        logger.warning(f"Некорректный ИНН: {inn}")
        return {'success': False, 'error': 'ИНН должен состоять из цифр'}, 200

    if attempt > MAX_ATTEMPTS:
        logger.error(f"Превышено количество попыток для ИНН: {inn}")
        return {'success': False, 'error': 'Ошибка сайта, попробуйте еще раз'}, 500

    try:
        logger.info(f"Попытка #{attempt}: Получаем данные для ИНН {inn}")
        return get_company_info(inn), 200
    except AttributeError as e:
        logger.warning(f"Получен пустой ответ для ИНН {inn}, повторная попытка")
        return try_to_get_company_info(inn, attempt + 1)

@app.route('/get_company_data_by_inn', methods=['GET', 'POST'])
def api_request():
    inn = request.args.get('inn')
    logger.info(f"Получен запрос на получение данных для ИНН: {inn}")
    
    response, code = try_to_get_company_info(inn)
    logger.debug(f"Ответ API: {response}, Код: {code}")

    return json.dumps(response, indent=2, ensure_ascii=False), code

if __name__ == "__main__":
    logger.info("Стартую сервер...")
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
