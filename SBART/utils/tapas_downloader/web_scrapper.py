import ftplib
import gzip
import json
import os
import time
from io import BytesIO

import requests
from loguru import logger

from SBART import SBART_LOC


def login(user, password):
    values = {"login": user, "pwd": password}
    url = "http://cds-espri.ipsl.fr/tapas/project?methodName=login"

    login_response = requests.post(url, data=values)

    if login_response.status_code != 200:
        raise Exception("Tapas login failed")

    return login_response.cookies


def Tapas_login(username):
    """Login to TAPAS and move to 'username' directory"""
    url = "ftp.ipsl.fr"
    ftp = ftplib.FTP(url)
    ftp.login("tapas", "tapas")
    ftp.cwd(username)
    return ftp


def get_folder_info(tapas_ftp):
    """Find the folders that exist on the ftp server"""
    x = tapas_ftp.mlsd(facts=[])
    folder_names = []
    for i in x:
        folder_names.append(i[0])
    return set(folder_names)


def download_IPAC_file(tapas_ftp, folder_name, destination):

    tapas_ftp.cwd(folder_name)

    r = BytesIO()  # to avoid writting to disk
    tapas_ftp.retrbinary("RETR tapas_000001.ipac.gz", r.write)

    r.seek(0)  # change the pointer of the BytesIO object to the beginning of the file

    decompressed_data = gzip.GzipFile(fileobj=r)  # decompress the data

    if os.path.isdir(destination):
        storing_path = os.path.join(destination, "tapas_000001.ipac")
    else:
        storing_path = destination

    with open(storing_path, "wb") as fp:
        fp.write(decompressed_data.read())

    return storing_path


def update_request(template, new_values):
    """
    For now only download data from ESPRESSO

    template:
        header file present in the .json file
    new_values:
        dictionary with the new values. Description:
            - RA : right ascension J2000
            - DEC: declination J2000
            - date: astropy.time date to request
    """

    template["requests"][0]["observation"]["los"]["raJ2000"] = new_values["RA"]
    template["requests"][0]["observation"]["los"]["decJ2000"] = new_values["DEC"]
    # standard_headers['requests'][0]['observation']['observatory']['name'] =

    date_time = new_values["mjd_time"].unix
    epoch = int(date_time) * 1000

    template["requests"][0]["observation"]["date"] = epoch

    inst_names = {
        "ESPRESSO": r"ESO%20Paranal%20Chile%20(VLT)",
        "HARPS": r"ESO%20La%20Silla%20Chile",
        "CARMENES": r"Calar%20Alto%20German-Spanish%20Ast.%20Center%20Spain",
    }
    template["requests"][0]["observation"]["observatory"]["name"] = inst_names[
        new_values["instrument"]
    ]

    template["requests"][0]["observation"]["instrument"]["spectralRange"] = "{}%20{}".format(
        *new_values["spectralRange"]
    )

    new_values["mjd_time"].format = "fits"

    return template


def get_TAPAS_data(
    user, password, request_data, storing_destination, timeout=5, request_interval=60
):
    """

    timeout:
        Maximum number of minutes to wait for Tapas to answer
    request_interval:
        Number of seconds to wait before sending next request to TAPAS
    """

    # Login to ftp server and to the web-interface
    cookies = login(user, password)
    tapas_ftp = Tapas_login(user)
    logger.debug("TAPAS login successful")

    # Check initial information in the ftp server
    initial_folders = get_folder_info(tapas_ftp)

    # Load the header to request new data from the web interface and update the values to the desired ones
    dir_path = os.path.join(SBART_LOC, "resources/header_file.json")

    with open(dir_path) as json_data:
        standard_headers = json.load(json_data)

    headers = update_request(standard_headers, request_data)
    request_url = (
        "http://cds-espri.ipsl.fr/tapas/data?methodName=createUserRequest&jsonTapas="
        + json.dumps(headers).replace('"', "%22")
    )
    logger.debug("Sending the TAPAS request... {}", headers)
    outputs = requests.get(request_url, cookies=cookies)
    logger.debug("Sent the request. Output: {}", outputs.text)

    # wait a little bit to allow calculations to happen
    logger.debug("Waiting for TAPAS model")
    time.sleep(2 * request_interval)

    # wait for an update on the ftp server
    t0 = time.time()
    error_flag = False
    errors_msg = ""

    logger.debug("Entering request cycle")
    while time.time() - t0 <= timeout * 60:

        existing_folders = get_folder_info(tapas_ftp)

        new_folder = existing_folders - initial_folders

        if len(new_folder) == 1:
            # found one new file; can proceed to download it
            tapas_template_name = list(new_folder)[0]
            logger.debug("Found a new file; Preparing to download")
            break
        elif len(new_folder) > 1:
            error_flag = True
            errors_msg = "More than one new TAPAS template; problems"
            break
        elif len(new_folder) < 0:
            error_flag = True
            errors_msg = "Less folders now than at the start ... "
            break

        logger.debug("TAPAS model still not calculated")
        time.sleep(request_interval)

    if len(new_folder) == 0:
        error_flag = True
        errors_msg = "No folder was added"
    if not error_flag:
        # Download the data from the ftp link
        storing_path = download_IPAC_file(tapas_ftp, tapas_template_name, storing_destination)

    # close connection
    tapas_ftp.quit()

    if error_flag:
        raise Exception(errors_msg)

    else:
        return storing_path
