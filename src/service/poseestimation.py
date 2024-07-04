import csv
import os
import zipfile
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from ahrs.filters import Madgwick
from fastapi.responses import Response


def PoseEstimation(gyroUrl:str,accUrl:str):
	print(gyroUrl,accUrl)
	# 保存先ディレクトリ
	save_dir = "./download"
	result_dir="./result"
	os.makedirs(save_dir, exist_ok=True)  # ディレクトリが存在しない場合は作成

	# ジャイロデータの保存ファイル名とパス
	gyro_file_name = get_filename_from_url(gyroUrl)
	gyro_save_path = os.path.join(save_dir, gyro_file_name)
	response =requests.get(gyroUrl)
	# レスポンスが成功したか確認
	if response.status_code == 200:
		# ファイルを保存
		with open(gyro_save_path, "wb") as file:
			file.write(response.content)
		print("File downloaded successfully")
	else:
		print("Failed to download gyro file. Status code:", response.status_code)
	
	acc_file_name = get_filename_from_url(accUrl)
	acc_save_path = os.path.join(save_dir, acc_file_name)

	response =requests.get(accUrl)
	# レスポンスが成功したか確認
	if response.status_code == 200:
		# ファイルを保存
		with open(acc_save_path, "wb") as file:
			file.write(response.content)
		print("File downloaded successfully")
	else:
		print("Failed to download acc file. Status code:", response.status_code)
	# 角速度データを読み込み
	gyro_data = pd.read_csv(gyro_save_path)
	# 加速度データを読み込み
	acc_data = pd.read_csv(acc_save_path)
	#共通のタイムスタンプを作成
	common_timestamps = np.union1d(gyro_data['time'], acc_data['time'])
	# 線形補間を使用してデータを補間
	gyro_data = gyro_data.set_index('time').reindex(common_timestamps).interpolate().reset_index()
	acc_data = acc_data.set_index('time').reindex(common_timestamps).interpolate().reset_index()
	# サンプリングレートの計算
	time_diffs = np.diff(common_timestamps) / 1000.0  # タイムスタンプの差を秒単位に変換
	mean_dt = np.mean(time_diffs)
	sampling_rate = 1.0 / mean_dt

	# Madgwickフィルタの初期化
	quaternion = [1.0, 0.0, 0.0, 0.0]  # 初期の四元数
	madgwick = Madgwick(frequency=sampling_rate,gain_imu=0.33)

	filtered_orientation=[]
	filtered_orientation_csv=[]

	# フィルタの適用と出力
	for i in range(len(gyro_data)):
		gyr = [gyro_data['x'][i], gyro_data['y'][i], gyro_data['z'][i]]
		acc = [acc_data['x'][i], acc_data['y'][i], acc_data['z'][i]]
		# フィルタの更新
		quaternion=madgwick.updateIMU(q=quaternion,gyr=gyr, acc=acc)

		filtered_orientation_csv.append([common_timestamps[i],quaternion[0],quaternion[1],quaternion[2],quaternion[3]])
		filtered_orientation.append({"time": int(common_timestamps[i]), "w": float(quaternion[0]), "x": float(quaternion[1]), "y": float(quaternion[2]), "z": float(quaternion[3])})




	with open(result_dir+'/result.csv', 'w', newline='') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(['time', 'w', 'x', 'y','z'])  # ヘッダーを書き込む
		csvwriter.writerows(filtered_orientation_csv)

	try:
		os.remove(gyro_save_path)
		os.remove(acc_save_path)
		print("Files deleted successfully.")
	except Exception as e:
		print("Error while deleting files:", e)

	zip_csv_file(result_dir,"result")
	body,boundary=create_multipart(result_dir)
	return Response(body, media_type=f"multipart/form-data; boundary={boundary}")

def get_filename_from_url(url):
		parsed_url = urlparse(url)
		return os.path.basename(parsed_url.path)

def zip_csv_file(result_dir,filename):
    with zipfile.ZipFile(result_dir+"/"+filename+".zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(result_dir+"/"+filename+".csv", arcname=filename+".csv")

def create_multipart(result_dir):
	boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
	filename = "result.zip"
	filepath = os.path.join(result_dir, filename)

	
	# zipファイルをバイナリモードで読み込む
	with open(filepath, "rb") as file:
		zip_data = file.read()

	# マルチパートフォームデータの構築
	content = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f'Content-Type: application/zip\r\n\r\n'
    ).encode('utf-8') + zip_data + f"\r\n--{boundary}--".encode('utf-8')
	return content, boundary