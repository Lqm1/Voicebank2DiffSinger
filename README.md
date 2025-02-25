# Voicebank2DiffSinger
UTAU音源からSOFAとMakeDiffSingerを用いて、DiffSinger用の学習用データセットを作成する

## 前提要件
- Windows
- C++ によるデスクトップ開発 (Visual Studio)
- CMake
- Python 3.12未満 (3.11.11にてテスト済み)

## インストール方法 (uv (高速) ) 
1. uvをセットアップ (オプション)
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
2. このリポジトリをsubmoduleを含めcloneし、ディレクトリに移動
    ```powershell
    git clone --recursive
    cd Voicebank2DiffSinger
    ```
3. 必要なモジュールをインストールする
    ```powershell
    uv sync
    ```
4. [日本語のSOFAモデル](https://github.com/Greenleaf2001/SOFA_Models/releases/tag/JPN_Test2)から「step.100000.ckpt」と「japanese-extension-sofa.txt
」をダウンロードし、「step.100000.ckpt」を「src/ckpt」に配置し、「japanese-extension-sofa.txt
」を「src/dictionaries」に配置する

## インストール方法 (pip)
1. このリポジトリをsubmoduleを含めcloneし、ディレクトリに移動
    ```powershell
    git clone --recursive
    cd Voicebank2DiffSinger
    ```
2. 仮想環境を構築し、入る
    ```powershell
    python -m venv .venv
    .venv/scripts/activate
    ```
3. 必要なモジュールをインストールする
    ```powershell
    pip install -r requirements.txt
    ```
4. [日本語のSOFAモデル](https://github.com/Greenleaf2001/SOFA_Models/releases/tag/JPN_Test2)から「step.100000.ckpt」と「japanese-extension-sofa.txt
」をダウンロードし、「step.100000.ckpt」を「src/ckpt」に配置し、「japanese-extension-sofa.txt
」を「src/dictionaries」に配置する

## 使用方法
1. 仮想環境に入る (オプション)
    ```powershell
    .venv/scripts/activate
    ```
2. src/main.py の args に音源 (音階) フォルダを一つ(もしくは複数)渡し起動する
    ```powershell
    python src/main.py example/A3 example/A2 example/A4
    ```
