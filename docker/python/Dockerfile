# python version number:3.9.16
FROM python:3.9.16

# 作業ディレクトリ
WORKDIR /python/src/app/

# ローカルからコンテナへコピー
COPY ./src .

# パッケージリストの更新とインストール
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# upgrade pip コマンド
RUN pip install --upgrade pip 

# python ライブラリのインストール
RUN pip install -r lib/requirements.txt

# アプリケーションの実行
CMD ["python3", "main.py"]