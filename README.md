# Voicebank2DiffSinger

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3111/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Voicebank2DiffSinger** は、UTAU音源から **SOFA** と **MakeDiffSinger** を用いて、DiffSinger 用の学習データセットを自動生成するツールです。

## 目次

- [概要](#概要)
- [機能](#機能)
- [ディレクトリ構造](#ディレクトリ構造)
- [前提条件](#前提条件)
- [インストール方法](#インストール方法)
  - [uv を利用する方法 (高速インストール)](#uv-を利用する方法-高速インストール)
  - [pip を利用する方法](#pip-を利用する方法)
- [使用方法](#使用方法)
- [注意事項](#注意事項)
- [貢献](#貢献)
- [ライセンス](#ライセンス)
- [連絡先](#連絡先)

## 概要

本ツールは、UTAU音源データを解析し、DiffSinger 用の学習データセットに変換します。内部では、**SOFA モデル** と **MakeDiffSinger** の仕組みを活用し、音声データの前処理・変換を自動で行います。

## 機能

- **音源解析:** UTAU音源から音素や単語のシーケンスを抽出します。
- **SOFA モデル活用:** 高精度な音声処理を実現するために、日本語用SOFAモデルを利用します。
- **DiffSinger用データ生成:** MakeDiffSingerとの連携で、DiffSinger用の学習データセットを生成します。

## ディレクトリ構造

```
Directory structure:
└── Voicebank2DiffSinger/
    ├── README.md
    ├── LICENSE
    ├── pyproject.toml
    ├── requirements.txt
    ├── uv.lock
    ├── .python-version
    └── src/
        ├── g2p.py
        ├── main.py
        ├── utils.py
        ├── MakeDiffSinger/
        ├── SOFA/
        ├── ckpt/
        │   └── .gitkeep
        ├── dictionaries/
        │   └── .gitkeep
        └── outputs/
            └── .gitkeep
```

## 前提条件

- **OS:** Windows
- **開発環境:** C++（Visual Studioを用いたデスクトップ開発）、CMake
- **Python:** 3.11（3.12未満、3.11.11でテスト済み）

## インストール方法

### uv を利用する方法 (高速インストール)

1. **uv のセットアップ（オプション）**

   以下のコマンドを PowerShell で実行してください：

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **リポジトリのクローン**

   サブモジュールも含めてリポジトリをクローンし、ディレクトリに移動します：

   ```powershell
   git clone --recursive https://github.com/Lqm1/Voicebank2DiffSinger.git
   cd Voicebank2DiffSinger
   ```

3. **必要モジュールのインストール**

   ```powershell
   uv sync
   ```

4. **日本語 SOFA モデルの導入**

   [日本語のSOFAモデル](https://github.com/Greenleaf2001/SOFA_Models/releases/tag/JPN_Test2) から以下のファイルをダウンロードしてください：
   
   - `step.100000.ckpt` を `src/ckpt` フォルダへ配置
   - `japanese-extension-sofa.txt` を `src/dictionaries` フォルダへ配置

### pip を利用する方法

1. **リポジトリのクローン**

   サブモジュールも含めてリポジトリをクローンし、ディレクトリに移動します：

   ```powershell
   git clone --recursive https://github.com/Lqm1/Voicebank2DiffSinger.git
   cd Voicebank2DiffSinger
   ```

2. **仮想環境の構築とアクティベート**

   ```powershell
   python -m venv .venv
   .venv/scripts/activate
   ```

3. **必要モジュールのインストール**

   ```powershell
   pip install -r requirements.txt
   ```

4. **日本語 SOFA モデルの導入**

   [日本語のSOFAモデル](https://github.com/Greenleaf2001/SOFA_Models/releases/tag/JPN_Test2) から以下のファイルをダウンロードしてください：
   
   - `step.100000.ckpt` を `src/ckpt` フォルダへ配置
   - `japanese-extension-sofa.txt` を `src/dictionaries` フォルダへ配置

## 使用方法

1. **仮想環境のアクティベート（pip インストールの場合）**

   ```powershell
   .venv/scripts/activate
   ```

2. **実行方法**

   `src/main.py` に対して、音源（音階）フォルダを1つまたは複数引数として指定して実行します。例：

   ```powershell
   python src/main.py example/A3 example/A2 example/A4
   ```

   ※ 各フォルダ内に対象の音源ファイルと、同名の `.txt` ファイル（ラベル情報）が必要です。

## 注意事項

- **ファイル配置:**  
  `src/ckpt` および `src/dictionaries` に日本語SOFAモデルのファイルが正しく配置されていない場合、実行時にエラーが発生します。

- **依存関係:**  
  本プロジェクトは多くの外部パッケージに依存しています。インストール時にエラーが発生した場合は、Pythonのバージョンや各パッケージのバージョンに注意してください。

- **詳細設定:**  
  各モジュールの詳細な設定やカスタマイズ方法については、ソースコード内のコメントおよび各ディレクトリ内のドキュメントをご参照ください。

## 貢献

バグ報告、機能追加の提案、プルリクエストなど、どなたからの貢献も大歓迎です。まずは [Issue](https://github.com/[ユーザー名]/Voicebank2DiffSinger/issues) をご利用ください。

## ライセンス

このプロジェクトは [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0) のもとで公開されています。

## 連絡先

ご質問やご提案は、[info@lami.zip](mailto:info@lami.zip) までご連絡ください。
