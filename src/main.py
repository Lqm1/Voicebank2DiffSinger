import sys

sys.path.append("src/SOFA")
sys.path.append("src/SOFA/modules")
sys.path.append("src/MakeDiffSinger/acoustic_forced_alignment")
sys.path.append("src/MakeDiffSinger/variance-temp-solution")
import tempfile
import pathlib
import tqdm
import re
import tomllib
import soundfile
import librosa
import librosa.effects
import librosa.util
import SOFA.modules.AP_detector
import SOFA.modules.g2p
from SOFA.modules.utils.export_tool import Exporter
from SOFA.modules.utils.post_processing import post_processing
from SOFA.train import LitForcedAlignmentTask
from g2p import PyOpenJTalkG2P
from utils import (
    import_module_from_path,
    bowlroll_file_download,
    remove_specific_consecutive_duplicates,
    remove_duplicate_otos,
)
import pyopenjtalk
import torch
import lightning as pl
import click
from MakeDiffSinger.acoustic_forced_alignment.build_dataset import build_dataset
import datetime
import shutil
import os
import subprocess
import utaupy
import time
import textgrid


add_ph_num: click.Command = import_module_from_path(
    "src/MakeDiffSinger/variance-temp-solution/add_ph_num.py", "add_ph_num"
).add_ph_num
estimate_midi: click.Command = import_module_from_path(
    "src/MakeDiffSinger/variance-temp-solution/estimate_midi.py", "estimate_midi"
).estimate_midi
csv2ds: click.Command = import_module_from_path(
    "src/MakeDiffSinger/variance-temp-solution/convert_ds.py", "convert_ds"
).csv2ds


if not pathlib.Path("src/Moresampler").exists():
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = pathlib.Path(temp_dir_str)
        file_id = 139123
        file_name = pathlib.Path("Moresampler.zip")
        content = bowlroll_file_download(file_id)
        with open(temp_dir / file_name, "wb") as f:
            f.write(content)
        shutil.unpack_archive(temp_dir / file_name, temp_dir / "Moresampler")
        shutil.move(temp_dir / "Moresampler", "src/Moresampler")


with open("pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)

HIRAGANA_REGEX = re.compile(r"([あ-ん][ぁぃぅぇぉゃゅょ]|[あ-ん])")
KATAKANA_REGEX = re.compile(r"([ア-ン][ァィゥェォャュョ]|[ア-ン])")


def validate_directories(
    ctx: click.Context, param: click.Parameter, value: tuple[str, ...]
) -> tuple[str, ...]:
    if not value:
        raise click.BadParameter("At least one directory path must be specified.")
    for path in value:
        if not os.path.isdir(path):
            raise click.BadParameter(f"'{path}' is not a directory.")
    return value


@click.command()
@click.version_option(version=pyproject["project"]["version"])
@click.argument("voicebank_dir_strs", nargs=-1, callback=validate_directories)
def main(voicebank_dir_strs: list[str]):
    print(
        f"Voicebank to DiffSinger {pyproject['project']['version']} - Convert the UTAU Voicebank to a configuration compatible with DiffSinger Dataset"
    )
    print()
    print("Select the forced aligner to use:")
    print("1: SOFA")
    print("2: Moresampler")
    forced_aligner_type = input("Enter the number of the forced aligner to use: ")
    if forced_aligner_type == "1":
        forced_aligner = "SOFA"
    elif forced_aligner_type == "2":
        forced_aligner = "Moresampler"
    else:
        print("Invalid input.")
        sys.exit(1)
    normalize_flag = False
    if forced_aligner == "SOFA" or forced_aligner == "Moresampler":
        normalize_flag = input("Do you want to normalize the volume? (y/n): ") == "y"
    detect_nonslicent_flag = False
    if forced_aligner == "SOFA":
        detect_nonslicent_flag = (
            input("Do you want to perform silence trimming? (y/n): ") == "y"
        )
    print()

    voicebank_dirs = [
        pathlib.Path(voicebank_dir_str) for voicebank_dir_str in voicebank_dir_strs
    ]

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = pathlib.Path(temp_dir_str)
        if forced_aligner == "SOFA":
            print("Phase 1: Merge voicebanks...")
            print()

            with tqdm.tqdm(total=len(voicebank_dirs)) as pbar:
                for voicebank_dir in voicebank_dirs:
                    for wav_file in voicebank_dir.glob("*.wav"):
                        shutil.copy(
                            wav_file,
                            temp_dir / f"{wav_file.stem}_{voicebank_dir.stem}.wav",
                        )
                    pbar.update(1)

            wav_files = list(temp_dir.glob("*.wav"))

            print()
            print("Phase 1: Done.")
            print()

            if normalize_flag:
                print("Phase 1-1: Normalizing volume...")
                print()

                with tqdm.tqdm(total=len(wav_files)) as pbar:
                    for wav_file in wav_files:
                        y, sr = librosa.load(wav_file, sr=None)
                        y = librosa.util.normalize(y)
                        soundfile.write(wav_file, y, sr)
                        pbar.update(1)

                print()
                print("Phase 1-1: Done.")
                print()

            if detect_nonslicent_flag:
                print("Phase 1-1: Performing silence trimming...")
                print()

                with tqdm.tqdm(total=len(wav_files)) as pbar:
                    for wav_file in wav_files:
                        y, sr = librosa.load(wav_file, sr=None)
                        y = librosa.effects.trim(y, top_db=30)[0]
                        soundfile.write(wav_file, y, sr)
                        pbar.update(1)

                print()
                print("Phase 1-1: Done.")
                print()

            print("Phase 2: Generating text files...")
            print()

            with tqdm.tqdm(total=len(wav_files)) as pbar:
                for wav_file in wav_files:
                    file_name = pathlib.Path(wav_file).stem
                    words = file_name[1:]
                    graphemes = remove_specific_consecutive_duplicates(
                        [
                            *HIRAGANA_REGEX.findall(words),
                            *KATAKANA_REGEX.findall(words),
                        ],
                        ["あ", "い", "う", "え", "お", "ん"],
                    )
                    with open(
                        str(temp_dir) + "/" + file_name + ".txt",
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(" ".join(graphemes))
                    pbar.update(1)

            print()
            print("Phase 2: Done.")
            print()
            print("Phase 3: Generating TextGrids...")
            print()

            AP_detector_class = (
                SOFA.modules.AP_detector.LoudnessSpectralcentroidAPDetector
            )
            get_AP = AP_detector_class()

            g2p_class = PyOpenJTalkG2P
            grapheme_to_phoneme = g2p_class()

            torch.set_grad_enabled(False)

            model = LitForcedAlignmentTask.load_from_checkpoint(
                "src/ckpt/step.100000.ckpt"
            )
            model.set_inference_mode("force")

            trainer = pl.Trainer(logger=False)

            dataset = grapheme_to_phoneme.get_dataset(wav_files)

            predictions = trainer.predict(
                model, dataloaders=dataset, return_predictions=True
            )

            predictions = get_AP.process(predictions)
            predictions, log = post_processing(predictions)

            exporter = Exporter(predictions, log)
            exporter.export(["textgrid"])

            print()
            print("Phase 3: Done.")
            print()
        elif forced_aligner == "Moresampler":
            print("Phase 1: Generating oto.ini file...")
            print()

            for voicebank_dir in voicebank_dirs:
                temp_voicebank_dir = temp_dir / voicebank_dir.stem
                temp_voicebank_dir.mkdir()
                wav_files = list(voicebank_dir.glob("*.wav"))
                with tqdm.tqdm(total=len(wav_files)) as pbar:
                    for wav_file in voicebank_dir.glob("*.wav"):
                        shutil.copy(
                            wav_file,
                            temp_voicebank_dir / wav_file.name,
                        )
                        pbar.update(1)
                print()
                process = subprocess.Popen(
                    [
                        "src/Moresampler/moresampler.exe",
                        str(temp_voicebank_dir),
                    ],
                    stdin=subprocess.PIPE,
                    text=True,
                )
                process.stdin.write("1\n")
                process.stdin.flush()
                process.stdin.write("y\n")
                process.stdin.flush()
                process.stdin.write("n\n")
                process.stdin.flush()
                process.stdin.write("1\n")
                process.stdin.flush()
                process.stdin.write("n\n")
                process.stdin.flush()
                process.stdin.write("\n")
                process.stdin.flush()
                while process.poll() is None:
                    process.stdin.write("\n")
                    process.stdin.flush()
                    time.sleep(0.1)
                print()

            print()
            print("Phase 1: Done.")
            print()
            print("Phase 2: Merge voicebanks...")
            print()

            merged_oto_ini = utaupy.otoini.OtoIni()
            with tqdm.tqdm(total=len(voicebank_dirs)) as pbar:
                for voicebank_dir in voicebank_dirs:
                    temp_voicebank_dir = temp_dir / voicebank_dir.stem
                    oto_ini = utaupy.otoini.load(str(temp_voicebank_dir / "oto.ini"))
                    for wav_file in (temp_voicebank_dir).glob("*.wav"):
                        shutil.move(
                            wav_file,
                            temp_dir / f"{wav_file.stem}_{voicebank_dir.stem}.wav",
                        )
                        otos: list[utaupy.otoini.Oto] = list(
                            filter(lambda oto: oto.filename == wav_file.name, oto_ini)
                        )
                        for oto in otos:
                            oto.filename = f"{wav_file.stem}_{voicebank_dir.stem}.wav"
                            merged_oto_ini.append(oto)
                    shutil.rmtree(temp_voicebank_dir)
                    pbar.update(1)
            merged_oto_ini.write(str(temp_dir / "oto.ini"))

            print()
            print("Phase 2: Done.")
            print()
            print("Phase 3: Convert oto.ini to TextGrid...")
            print()

            textgrid_dir = temp_dir / "TextGrid"
            textgrid_dir.mkdir()
            oto_ini = utaupy.otoini.load(str(temp_dir / "oto.ini"))
            wav_files = list(temp_dir.glob("*.wav"))
            with tqdm.tqdm(total=len(wav_files)) as pbar:
                for wav_file in wav_files:
                    otos: list[utaupy.otoini.Oto] = remove_duplicate_otos(
                        list(filter(lambda oto: oto.filename == wav_file.name, oto_ini))
                    )
                    sorted_otos = sorted(otos, key=lambda oto: oto.offset)
                    if any(
                        [
                            len(pyopenjtalk.g2p(oto.alias.split()[1], join=False)) > 2
                            for oto in sorted_otos
                            if oto.alias.split()[1] != "-"
                        ]
                    ):
                        wav_file.unlink()
                        pbar.update(1)
                        continue
                    y, sr = librosa.load(wav_file, sr=None)
                    duration_seconds = librosa.get_duration(y=y, sr=sr)
                    tg = textgrid.TextGrid()
                    grapheme_tier = textgrid.IntervalTier(
                        name="graphemes", minTime=0, maxTime=duration_seconds
                    )
                    phoneme_tier = textgrid.IntervalTier(
                        name="phonemes", minTime=0, maxTime=duration_seconds
                    )
                    for i, oto in enumerate(sorted_otos[:-1]):
                        splitted_alias = oto.alias.split()
                        next_splitted_alias = sorted_otos[i + 1].alias.split()
                        phs = (
                            pyopenjtalk.g2p(splitted_alias[1], join=False)
                            if splitted_alias[1] != "-"
                            else []
                        )
                        next_phs = (
                            pyopenjtalk.g2p(next_splitted_alias[1], join=False)
                            if next_splitted_alias[1] != "-"
                            else []
                        )
                        if i == 0:
                            if len(next_phs) == 0:
                                if len(phs) == 1:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = 0
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # phoneme.end = audio_length
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        0,
                                        (oto.offset + oto.preutterance) / 1000,
                                        "AP",
                                    )
                                    grapheme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        duration_seconds,
                                        splitted_alias[1],
                                    )
                                    grapheme_tier.add(
                                        0,
                                        (oto.offset + oto.preutterance) / 1000,
                                        "SP",
                                    )
                                    phoneme_tier.add(
                                        0,
                                        (oto.offset + oto.preutterance) / 1000,
                                        "AP",
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        phs[0],
                                    )
                                    phoneme_tier.add(
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        duration_seconds,
                                        "SP",
                                    )
                                elif len(phs) == 2:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = 0
                                    # phoneme.end = (oto.offset + oto.overlap) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.overlap) * 1000
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[1]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # phoneme.end = audio_length
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        0,
                                        (oto.offset + oto.overlap) / 1000,
                                        "AP",
                                    )
                                    grapheme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    grapheme_tier.add(
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        duration_seconds,
                                        "SP",
                                    )
                                    phoneme_tier.add(
                                        0,
                                        (oto.offset + oto.overlap) / 1000,
                                        "AP",
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (oto.offset + oto.preutterance) / 1000,
                                        phs[0],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        phs[1],
                                    )
                                    phoneme_tier.add(
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        duration_seconds,
                                        "SP",
                                    )
                                else:
                                    raise ValueError("Invalid phoneme length.")
                            elif len(next_phs) == 1:
                                if len(phs) == 1:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = 0
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        0,
                                        (oto.offset + oto.preutterance) / 1000,
                                        "AP",
                                    )
                                    grapheme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    phoneme_tier.add(
                                        0,
                                        (oto.offset + oto.preutterance) / 1000,
                                        "AP",
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        phs[0],
                                    )
                                elif len(phs) == 2:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = 0
                                    # phoneme.end = (oto.offset + oto.overlap) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.overlap) * 1000
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[1]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        0,
                                        (oto.offset + oto.overlap) / 1000,
                                        "AP",
                                    )
                                    grapheme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    phoneme_tier.add(
                                        0,
                                        (oto.offset + oto.overlap) / 1000,
                                        "AP",
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (oto.offset + oto.preutterance) / 1000,
                                        phs[0],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        phs[1],
                                    )
                                else:
                                    raise ValueError("Invalid phoneme length.")
                            elif len(next_phs) == 2:
                                if len(phs) == 1:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = 0
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].overlap) * 1000
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        0,
                                        (oto.offset + oto.preutterance) / 1000,
                                        "AP",
                                    )
                                    grapheme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].overlap)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    phoneme_tier.add(
                                        0,
                                        (oto.offset + oto.preutterance) / 1000,
                                        "AP",
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].overlap)
                                        / 1000,
                                        phs[0],
                                    )
                                elif len(phs) == 2:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = 0
                                    # phoneme.end = (oto.offset + oto.overlap) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.overlap) * 1000
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[1]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].overlap) * 1000
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        0,
                                        (oto.offset + oto.overlap) / 1000,
                                        "AP",
                                    )
                                    grapheme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].overlap)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    phoneme_tier.add(
                                        0,
                                        (oto.offset + oto.overlap) / 1000,
                                        "AP",
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (oto.offset + oto.preutterance) / 1000,
                                        phs[0],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].overlap)
                                        / 1000,
                                        phs[1],
                                    )
                                else:
                                    raise ValueError("Invalid phoneme length.")
                            else:
                                raise ValueError("Invalid phoneme length.")
                        else:
                            if len(next_phs) == 0:
                                if len(phs) == 1:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # phoneme.end = audio_length
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    grapheme_tier.add(
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        duration_seconds,
                                        "SP",
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        phs[0],
                                    )
                                    phoneme_tier.add(
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        duration_seconds,
                                        "SP",
                                    )
                                elif len(phs) == 2:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.overlap) * 1000
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[1]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = "SP"
                                    # phoneme.start = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # phoneme.end = audio_length
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    grapheme_tier.add(
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        duration_seconds,
                                        "SP",
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (oto.offset + oto.preutterance) / 1000,
                                        phs[0],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        phs[1],
                                    )
                                    phoneme_tier.add(
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        duration_seconds,
                                        "SP",
                                    )
                                else:
                                    raise ValueError("Invalid phoneme length.")
                            elif len(next_phs) == 1:
                                if len(phs) == 1:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        phs[0],
                                    )
                                elif len(phs) == 2:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.overlap) * 1000
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[1]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].preutterance) * 1000
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (oto.offset + oto.preutterance) / 1000,
                                        phs[0],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].preutterance)
                                        / 1000,
                                        phs[1],
                                    )
                                else:
                                    raise ValueError("Invalid phoneme length.")
                            elif len(next_phs) == 2:
                                if len(phs) == 1:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].overlap) * 1000
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].overlap)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].overlap)
                                        / 1000,
                                        phs[0],
                                    )
                                elif len(phs) == 2:
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[0]
                                    # phoneme.start = (oto.offset + oto.overlap) * 1000
                                    # phoneme.end = (oto.offset + oto.preutterance) * 1000
                                    # label.append(phoneme)
                                    # phoneme = utaupy.label.Phoneme()
                                    # phoneme.symbol = phs[1]
                                    # phoneme.start = (oto.offset + oto.preutterance) * 1000
                                    # phoneme.end = (otos[i + 1].offset + otos[i + 1].overlap) * 1000
                                    # label.append(phoneme)
                                    grapheme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].overlap)
                                        / 1000,
                                        splitted_alias[1],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.overlap) / 1000,
                                        (oto.offset + oto.preutterance) / 1000,
                                        phs[0],
                                    )
                                    phoneme_tier.add(
                                        (oto.offset + oto.preutterance) / 1000,
                                        (otos[i + 1].offset + otos[i + 1].overlap)
                                        / 1000,
                                        phs[1],
                                    )
                                else:
                                    raise ValueError("Invalid phoneme length.")
                            else:
                                raise ValueError("Invalid phoneme length.")
                    tg.append(grapheme_tier)
                    tg.append(phoneme_tier)
                    tg.write(str(textgrid_dir / f"{wav_file.stem}.TextGrid"))
                    pbar.update(1)

            print()
            print("Phase 3: Done.")
            print()

            if normalize_flag:
                print("Phase 3-1: Normalizing volume...")
                print()

                wav_files = list(temp_dir.glob("*.wav"))
                with tqdm.tqdm(total=len(wav_files)) as pbar:
                    for wav_file in wav_files:
                        y, sr = librosa.load(wav_file, sr=None)
                        y = librosa.util.normalize(y)
                        soundfile.write(wav_file, y, sr)
                        pbar.update(1)

                print()
                print("Phase 3-1: Done.")
                print()
        else:
            print("Invalid forced aligner.")
            sys.exit(1)
        print("Phase 4: Build dataset...")
        print()

        ctx = click.Context(build_dataset)
        with ctx:
            build_dataset.parse_args(
                ctx,
                [
                    "--wavs",
                    str(temp_dir),
                    "--tg",
                    str(temp_dir / "TextGrid"),
                    "--dataset",
                    str(temp_dir / "Dataset"),
                ],
            )
            build_dataset.invoke(ctx)

        outputs_path = pathlib.Path("src/outputs")
        output_path = outputs_path / datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_path.mkdir()
        shutil.move(temp_dir / "Dataset" / "transcriptions.csv", output_path)
        shutil.move(temp_dir / "Dataset" / "wavs", output_path)
        output_wavs_path = output_path / "wavs"

    print()
    print("Phase 4: Done.")
    print()
    print("Phase 5: Add phoneme number...")
    print()

    ctx = click.Context(add_ph_num)
    with ctx:
        add_ph_num.parse_args(
            ctx,
            [
                str(output_path / "transcriptions.csv"),
                "--dictionary",
                "src/dictionaries/japanese-extension-sofa.txt",
            ],
        )
        add_ph_num.invoke(ctx)

    print()
    print("Phase 5: Done.")
    print()
    print("Phase 6: Estimate MIDI...")
    print()

    ctx = click.Context(estimate_midi)
    with ctx:
        estimate_midi.parse_args(
            ctx,
            [
                str(output_path / "transcriptions.csv"),
                str(output_wavs_path),
            ],
        )
        estimate_midi.invoke(ctx)

    print()
    print("Phase 6: Done.")
    print()
    print("Phase 7: Convert CSV to DiffSinger...")
    print()

    ctx = click.Context(csv2ds)
    with ctx:
        csv2ds.parse_args(
            ctx,
            [
                str(output_path / "transcriptions.csv"),
                str(output_wavs_path),
            ],
        )
        csv2ds.invoke(ctx)

    print()
    print("Phase 7: Done.")
    print()
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
