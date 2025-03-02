import utaupy
import importlib.util
import requests
from bs4 import BeautifulSoup
import re


def import_module_from_path(module_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bowlroll_file_download(file_id: int):
    with requests.Session() as session:
        session.headers.update(
            {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            }
        )
        response = session.get(f"https://bowlroll.net/file/{file_id}")
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        csrf_token_element = soup.find("div", {"id": "initialize"})
        if csrf_token_element is None:
            raise Exception("CSRF token not found")
        csrf_token = csrf_token_element.get("data-csrf_token")
        if csrf_token is None:
            raise Exception("CSRF token not found")
        data = {
            "download_key": "bowlroll_download_control_mischievous",
            "csrf_token": csrf_token,
        }
        response = session.post(
            f"https://bowlroll.net/api/file/{file_id}/download-check", data=data
        )
        response.raise_for_status()
        response = session.get(response.json()["url"], stream=True)
        response.raise_for_status()
        return response.content


def remove_specific_consecutive_duplicates(
    input_list: list[str], specific_elements: list[str]
):
    if not input_list:
        return []

    # Initialize the result list with the first element
    result = [input_list[0]]

    # Iterate through the input list starting from the second element
    for item in input_list[1:]:
        # If the current item is different from the last item in the result list
        # or if it is not in the specific elements list, add it
        if item != result[-1] or item not in specific_elements:
            result.append(item)

    return result


def remove_duplicate_otos(otos: list[utaupy.otoini.Oto]):
    unique_otos: list[utaupy.otoini.Oto] = []
    for oto in otos:
        for unique_oto in unique_otos:
            if (
                oto.filename == unique_oto.filename
                and oto.offset == unique_oto.offset
                and oto.consonant == unique_oto.consonant
                and oto.cutoff == unique_oto.cutoff
                and oto.preutterance == unique_oto.preutterance
                and oto.overlap == unique_oto.overlap
            ):
                break
        else:
            unique_otos.append(oto)
    return unique_otos


def convert_sharp_flat_in_notes(text: str) -> str:
    """
    音名の # を ♯ に、b を ♭ に変換する関数。
    関係ない # や b はそのまま維持する。

    Args:
        text (str): 変換対象の文字列

    Returns:
        str: 変換後の文字列
    """

    def replace_match(match: re.Match) -> str:
        note: str = match.group(1)
        accidental: str = match.group(2)
        octave: str = match.group(3)

        if accidental == "#":
            accidental = "♯"
        elif accidental == "b":
            accidental = "♭"

        return note + accidental + octave

    return re.sub(r"([A-G])([#b])(\d)", replace_match, text)
