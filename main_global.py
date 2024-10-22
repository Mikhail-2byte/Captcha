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

MAX_ATTEMPTS = 1
HTML_PARSER = 'lxml'
DOMAIN = 'https://pb.nalog.ru'
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0'
RUCAPTCHA_KEY = 'a705d169b195c6f79af08777de5bf77a'


app = Flask(__name__)
today = date.today()
logFileName = 'main' + ' ' + str(today)
logger.add(logFileName)




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


def solve_captcha(captcha_image: str) -> tuple[str, str]:
    logger.info(' * Решаем капчу')

    response = do_http_request(
        requests.Session(),
        'http://rucaptcha.com/in.php',
        'POST',
        dont_log=True,
        no_proxy=True,
        data={
            'method': 'base64',
            'key': RUCAPTCHA_KEY,
            'json': 1,
            'soft_id': '3343',
            'body': captcha_image
        }
    ).json()

    request_id = response['request']
    logger.debug(f'Получен ID запроса от RuCaptcha: {request_id}')

    while True:
        response = do_http_request(
            requests.Session(),
            'http://rucaptcha.com/res.php',
            dont_log=True,
            no_proxy=True,
            params={
                'key': RUCAPTCHA_KEY,
                'json': 1,
                'action': 'get',
                'id': request_id
            }
        ).json()
        code = response.get('request')

        if code and code != 'CAPCHA_NOT_READY':
            logger.info(f' * Капча решена: {code}')
            return str(request_id), code
        logger.debug('Капча не готова, ждем 1 секунду...')
        time.sleep(1)


def report_captcha(request_id: str, ok: bool = True):
    report = 'reportgood' if ok else 'reportbad'
    response = do_http_request(
        requests.Session(),
        'http://rucaptcha.com/res.php',
        no_proxy=True,
        params={
            'key': RUCAPTCHA_KEY,
            'action': report,
            'id': request_id
        }
    )
    # logger.debug(f'Отправлен отчет в рукапчу: [{report}] {response.text}')


def get_and_solve_captcha(session: requests.Session) -> str:
    logger.info('Получаем капчу с сервера...')
    response = do_http_request(
        session,
        f'{DOMAIN}/captcha-dialog.html',
        # params={
        #     'aver': '2.8.15',
        #     'sver': '4.39.5',
        #     'pageStyle': 'GM2',
        # }
    )

    soup = BeautifulSoup(response.text, 'html.parser')

    captcha_image_url = DOMAIN + soup.find('img').get('src')
    captcha_token = soup.find('input', {'name': 'captchaToken'}).get('value')

    logger.debug(f'URL изображения капчи: {captcha_image_url}')
    logger.debug(f'Токен капчи: {captcha_token}')

    captcha_image = do_http_request(session, captcha_image_url, dont_log=True).content
    img = Image.open(BytesIO(captcha_image)) # Load the image

    captcha_image = base64.b64encode(captcha_image)
    logger.debug('Кодируем изображение капчи в base64')

    captcha_id, captcha_code = solve_captcha(captcha_image)
    logger.info('Отправляем решение капчи на сервер...')
    response = do_http_request(
        session,
        'https://pb.nalog.ru/captcha-proc.json',
        'POST',
        data={
            'captcha': captcha_code,
            'captchaToken': captcha_token,
        }
    ).json()
    print("код капчи в get_and_solve_captcha", captcha_code)

    report_captcha(captcha_id, 'ERRORS' not in response)
    if 'ERRORS' in response:
        logger.error('Капча решена неверно')
        return get_and_solve_captcha(session)
    else:
        logger.info(f'Капча решена верно')
        img.save(f"images/{captcha_code}.jpg") # Save the image
        logger.info(f'Изображение капчи сохранено как images/{captcha_code}.jpg')
    return response


def get_request_id(session: requests.Session, inn: int | str, captcha_code: str) -> str:
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
        'pbCaptchaToken': captcha_code,
        'token': '',
    }

    logger.info(f'Полученный код капчи: {captcha_code}')

    response = do_http_request(
        session,
        'https://pb.nalog.ru/search-proc.json',
        'POST',
        headers=headers,
        data=data
    ).json()

    logger.info(f'Ответ от API для ИНН {inn}: {response}')  # Логируем ответ

    if 'ERRORS' in response:
        captcha_code = get_and_solve_captcha(session)
        return get_request_id(session, inn, captcha_code)
    
    return response['id']


def get_response(session: requests.Session, request_id: str) -> dict:
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

    logger.info(f'Запрашиваем ответ для ID: {request_id}')

    response = do_http_request(
        session,
        'https://pb.nalog.ru/search-proc.json',
        'POST',
        headers=headers,
        data=data,
    ).json()

    logger.info(f'Ответ от API для ID {request_id}: {response}')  # Логируем ответ 
    return response

def get_request_id_for_company(
    session: requests.Session,
    token: str,
    referer: str,
    captcha_code: str
) -> tuple[str, str]:
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
    
    logger.info(f'Запрашиваем ID компании с токеном: {token} и кодом капчи: {captcha_code}')

    response = do_http_request(
        session,
        'https://pb.nalog.ru/company-proc.json',
        'POST',
        headers=headers,
        data=data
    ).json()

    if 'ERRORS' in response:
        logger.error(f'Ошибка при получении ID компании: {response["ERRORS"]}')
        captcha_code = get_and_solve_captcha(session)
        return get_request_id_for_company(session, token, referer, captcha_code)
    
    return response['id'], response['token']

def get_rsmpcategory(catid: float) -> str:
    if not catid:
        return ''

    if catid == 1.0:
        return 'Микропредприятие'
    if catid == 2.0:
        return 'Малое предприятие'
    else:
        return 'Среднее предприятие'

def get_response_company(session: requests.Session, rid: str, token: str, referer: str) -> dict:
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

    logger.info(f'Запрашиваем информацию о компании с RID: {rid} и токеном: {token}')

    response = do_http_request(
        session,
        'https://pb.nalog.ru/company-proc.json',
        'POST',
        headers=headers,
        data=data
    ).json()

    logger.debug(f'Ответ от API для компании: {response}')

    import json
    with open('proc.json', 'w', encoding='utf-8') as f:
        json.dump(response, f, indent=4, ensure_ascii=False)
    
    vyp = response.get('vyp', {})
    # print (vyp)

    gen_dir = None
    gen_dir_pos = None
    gen_dir_dost = None
    directors = vyp.get('masruk')
    if directors:
        gen_dir = directors[0].get('name', None)
        gen_dir_pos = directors[0].get('position', None)
        gen_dir_dost = directors[0].get('invalid', None)

    ur_address_dost = None
    ur_address_dostf = vyp.get('СвНедАдресЮЛ')
    if ur_address_dostf:
        ur_address_dost = ur_address_dostf[0].get('ТекстНедАдресЮЛ', None)
     


    # for person in vyp.get('masruk', []):
    #     if 'директор' in person.get('position', '').lower():
    #         gen_dir = person.get('name', None)

    return {
        'token': token,
        'egrul_status': vyp.get('НаимСтатусЮЛ', None),
        'ur_address': vyp.get('АдресРФ', None),
        'ur_address_dost': ur_address_dost,
        'gen_dir': gen_dir,
        'gen_dir_pos': gen_dir_pos,
        'gen_dir_dost': gen_dir_dost,
        'human_amount': vyp.get('sschr', None),
        'nalog_sum': vyp.get('taxpaysum', None),
        'nalog_year': vyp.get('taxpay_yearcode', None),
        'revenue_sum': vyp.get('revenuesum', None),
        'revenue_year': vyp.get('form1_yearcode', None),
        'nalog_debt_sum': vyp.get('totalarrearsum', None),
        'nalog_debt_year': vyp.get('arrear_yearcode', None),
        'mspstatus': get_rsmpcategory(vyp.get('rsmpcategory', None)),
        'mspupdate': vyp.get('rsmpdate', None),
    }


def get_extra_data(session: requests.Session, token: str) -> dict:
    referer = f'https://pb.nalog.ru/company.html?token={token}'
    logger.info(f'Запрашиваем доплнительные данные с токеном: {token}')
    do_http_request(session, referer, 'GET', True)
    rid, token = get_request_id_for_company(session, token, referer, '')
    logger.info(f'Получен RID: {rid} для дополнительных данных')
    print("sleeping2")
    time.sleep(1)
    

    try:
        return get_response_company(session, rid, token, referer)
    except Exception as e:
        logger.error(e.with_traceback(None))
        return get_extra_data(session, token)


def prepare_response(session: requests.Session, nalog_response: dict, inn: str | int):
    inn = str(inn)
    logger.info(f'Подготавливаем ответ для ИНН: {inn}')
    # формируем ответ
    uls = nalog_response.get('ul', {}).get('data', [])
    for ul in uls:
        # ищем подходящий инн
        if ul['inn'] == inn:
            try:
                # print (ul)
                extra_data = get_extra_data(session, ul['token'])
                # egrul_status = get_egrul_status(session, ul['token'])
            except (AttributeError, Exception):
                logger.error(f'Не удалось получить статус по егрюл, ИНН: {inn}')
                extra_data = {'egrul_status': 'Не удалось получить'}
                # egrul_status = 'Не удалось получить'

            return {
                'success': True,
                'status': ul['sulst_name_ex'],
                'full_name': ul['namep'],
                'short_name': ul.get('namec', ul['namep']),
                'inn': inn,
                'registration_date': ul['dtreg'],
                **extra_data
            }
    
    # ничего не найдено
    logger.warning(f'По данному ИНН {inn} ничего не найдено')
    return {'success': False, 'error': 'По данному ИНН ничего не найдено'}


def get_company_info(inn: int | str) -> dict:
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})

    rid = get_request_id(session, inn, '')
    logger.info(f'Ищем информацию для ИНН: {inn}')
    #print("sleeping3")
    time.sleep(1)
    nalog_response = get_response(session, rid)

    response = prepare_response(session, nalog_response, inn)
    if response['success']:
        logger.debug(f'Инн: {inn}. Данные найдены')
    else:
        logger.debug(f'Инн: {inn}. Данные не найдены')

    return response


def try_to_get_company_info(inn: int | str, attempt: int = 0) -> tuple[int, dict]:
    if not inn or not inn.strip().isdigit():
        logger.error('ИНН должен состоять из цифр')
        return {'success': False, 'error': 'Инн должен состоять из цифр'}, 200

    # иногда сайт возвращает пустой ответ, эта функция нужна
    # в том числе для того, чтобы пробовать получить данные снова
    if attempt > MAX_ATTEMPTS:
        logger.error('Достигнуто максимальное количество попыток')
        return {'success': False, 'error': 'Ошибка сайта, попробуйте еще раз'}, 500
    try:
        return get_company_info(inn), 200
    except AttributeError:
        # когда сайт вернул пустой ответ
        logger.warning('Сайт вернул пустой ответ, пробуем снова')
        return try_to_get_company_info(inn, attempt + 1)


@app.route('/get_company_data_by_inn', methods=['GET', 'POST'])
def api_request():
    response, code = try_to_get_company_info(request.args.get('inn'))
    return json.dumps(response, indent=2, ensure_ascii=False), code


# if __name__ == '__main__':
#     app.run('0.0.0.0', 5000, True)


    
if __name__ == "__main__":
    logger.debug(f'Стартую сервер')
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
