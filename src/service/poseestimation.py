import csv
import os
import zipfile
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from ahrs.filters import Madgwick
from fastapi.responses import Response
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R


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
	# 重複を削除
	gyro_data = gyro_data.drop_duplicates(subset=['time'])
	acc_data = acc_data.drop_duplicates(subset=['time'])

	# 線形補間を使用してデータを補間
	gyro_data = gyro_data.set_index('time').reindex(common_timestamps).interpolate().reset_index()
	acc_data = acc_data.set_index('time').reindex(common_timestamps).interpolate().reset_index()

	interpolated_data_path = os.path.join(result_dir, 'interpolated_data.csv')
	with open(interpolated_data_path, 'w', newline='') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(['time', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z'])  # ヘッダーを書き込む
		for i in range(len(common_timestamps)):
			csvwriter.writerow([
				common_timestamps[i],
				gyro_data['x'][i],
				gyro_data['y'][i],
				gyro_data['z'][i],
				acc_data['x'][i],
				acc_data['y'][i],
				acc_data['z'][i]
			])
	# サンプリングレートの計算
	time_diffs = np.diff(common_timestamps) / 1000.0  # タイムスタンプの差を秒単位に変換
	mean_dt = np.mean(time_diffs)
	sampling_rate = 1.0 / mean_dt

	# Madgwickフィルタの初期化
	quaternion = [1.0, 0.0, 0.0, 0.0]  # 初期の四元数
	madgwick = Madgwick(frequency=sampling_rate,gain_imu=0.33)

	filtered_orientation=[]
	filtered_orientation_csv=[]
	gravity_data=[]
	linear_acc_data=[]
	#重力加速度
	g=9.81
	# フィルタの適用と出力
	linearAccData = pd.DataFrame(columns=["time", "x", "y", "z"])

	for i in range(len(gyro_data)):
		gyr = [gyro_data['x'][i], gyro_data['y'][i], gyro_data['z'][i]]
		acc = [acc_data['x'][i], acc_data['y'][i], acc_data['z'][i]]
		raw_acc = np.array([acc_data['x'][i], acc_data['y'][i], acc_data['z'][i]])
		# フィルタの更新
		quaternion=madgwick.updateIMU(q=quaternion,gyr=gyr, acc=acc)

		gravity = np.array([
			2 * (quaternion[1] * quaternion[3] - quaternion[0] * quaternion[2]),
			2 * (quaternion[0] * quaternion[1] + quaternion[2] * quaternion[3]),
			quaternion[0]**2 - quaternion[1]**2 - quaternion[2]**2 + quaternion[3]**2
		])
		# 重力ベクトルの大きさを考慮
		gravity *= g
		gravity_data.append([common_timestamps[i],gravity[0],gravity[1],gravity[2]])
		linear_acc=raw_acc-gravity

		# クォータニオンを回転行列に変換
		rotation_matrix = R.from_quat(quaternion).as_matrix()

		# 地球座標系での加速度
		earth_acc = rotation_matrix @ linear_acc
		#linear_acc_data.append([common_timestamps[i],linear_acc[0],linear_acc[1],linear_acc[2]])
		linear_acc_data.append([common_timestamps[i],earth_acc[0],earth_acc[1],earth_acc[2]])
		# 新しい行を作成
		new_row = pd.DataFrame({"time": [common_timestamps[i]], "x": [earth_acc[0]], "y": [earth_acc[1]], "z": [earth_acc[2]]})
		linearAccData = pd.concat([linearAccData, new_row], ignore_index=True)
		filtered_orientation_csv.append([common_timestamps[i],quaternion[0],quaternion[1],quaternion[2],quaternion[3]])
		filtered_orientation.append({"time": int(common_timestamps[i]), "w": float(quaternion[0]), "x": float(quaternion[1]), "y": float(quaternion[2]), "z": float(quaternion[3])})
		
	
	# 閾値の設定
	threshold = 0.1  # 必要に応じて調整

	# 各軸で閾値以下となる最初のインデックスを取得
	condition = (
		(linearAccData["x"].abs() <= threshold) &
		(linearAccData["y"].abs() <= threshold) &
		(linearAccData["z"].abs() <= threshold)
	)

	# 条件を満たす最初のインデックスを取得
	start_index = linearAccData[condition].index.min()
	linearAccData = linearAccData.loc[start_index:].reset_index(drop=True)
	linear_acc_data=linear_acc_data[start_index:]
	filtered_orientation_csv = filtered_orientation_csv[start_index:]
	time = (linearAccData["time"] - linearAccData["time"][0]) / 1000

	def butter_lowpass_filter(data, cutoff, fs, order=5):
		nyquist = 0.5 * fs
		normal_cutoff = cutoff / nyquist
		b, a = butter(order, normal_cutoff, btype='low', analog=False)
		y = filtfilt(b, a, data)
		return y

	# サンプリングレートとカットオフ周波数の設定
	fs = 1 / (time[1] - time[0])
	cutoff = 1.0  # 低周波成分を残すためのカットオフ周波数

	# # フィルタリング
	linearAccData["x"] = butter_lowpass_filter(linearAccData["x"], cutoff, fs)
	linearAccData["y"] = butter_lowpass_filter(linearAccData["y"], cutoff, fs)
	linearAccData["z"] = butter_lowpass_filter(linearAccData["z"], cutoff, fs)
	# 速度データの初期化
	speedData = pd.DataFrame(columns=["time", "vx", "vy", "vz"])

	# 速度を計算するためのループ
	vx = 0
	vy = 0
	vz = 0
	for i in range(len(linearAccData)):
		if i == 0:
			dt = 0
		else:
			dt = (linearAccData["time"][i] - linearAccData["time"][i - 1])/1000

		# 台形積分を使用して速度を更新
		if i > 0:
			avg_acc_x = (linearAccData["x"][i] + linearAccData["x"][i - 1]) / 2
			avg_acc_y = (linearAccData["y"][i] + linearAccData["y"][i - 1]) / 2
			avg_acc_z = (linearAccData["z"][i] + linearAccData["z"][i - 1]) / 2
			
			vx += avg_acc_x * dt
			vy += avg_acc_y * dt
			vz += avg_acc_z * dt

		# 新しい行を作成
		new_row = pd.DataFrame({"time": [linearAccData["time"][i]], "vx": [vx], "vy": [vy], "vz": [vz]})
		speedData = pd.concat([speedData, new_row], ignore_index=True)

	# 距離を計算するためのループ
	distanceData = pd.DataFrame(columns=["time", "dx", "dy", "dz"])
	dx = 0
	dy = 0
	dz = 0
	for i in range(len(speedData)):
		if i == 0:
			dt = 0
		else:
			dt = (speedData["time"][i] - speedData["time"][i - 1])/1000
		
		dx += speedData["vx"][i] * dt
		dy += speedData["vy"][i] * dt
		dz += speedData["vz"][i] * dt

		# 新しい行を作成
		new_row = pd.DataFrame({"time": [speedData["time"][i]], "dx": [dx], "dy": [dy], "dz": [dz]})
		distanceData = pd.concat([distanceData, new_row], ignore_index=True)

	def butter_highpass_filter(data, cutoff, fs, order=5):
		nyquist = 0.5 * fs
		normal_cutoff = cutoff / nyquist
		b, a = butter(order, normal_cutoff, btype='high', analog=False)
		y = filtfilt(b, a, data)
		return y

	# カットオフ周波数の設定（ここでは0.1 Hzを例にしていますが、必要に応じて変更してください）
	cutoff_high = 0.1

	# 速度データのフィルタリング
	speedData["vx"] = butter_highpass_filter(speedData["vx"], cutoff_high, fs)
	speedData["vy"] = butter_highpass_filter(speedData["vy"], cutoff_high, fs)
	speedData["vz"] = butter_highpass_filter(speedData["vz"], cutoff_high, fs)

	speedData["vx"] = butter_lowpass_filter(speedData["vx"], cutoff, fs)
	speedData["vy"] = butter_lowpass_filter(speedData["vy"], cutoff, fs)
	speedData["vz"] = butter_lowpass_filter(speedData["vz"], cutoff, fs)


	# 距離データのフィルタリング
	distanceData["dx"] = butter_highpass_filter(distanceData["dx"], cutoff_high, fs)
	distanceData["dy"] = butter_highpass_filter(distanceData["dy"], cutoff_high, fs)
	distanceData["dz"] = butter_highpass_filter(distanceData["dz"], cutoff_high, fs)

	distanceData["dx"] = butter_lowpass_filter(distanceData["dx"], cutoff, fs)
	distanceData["dy"] = butter_lowpass_filter(distanceData["dy"], cutoff, fs)
	distanceData["dz"] = butter_lowpass_filter(distanceData["dz"], cutoff, fs)
			


	with open(result_dir+'/result.csv', 'w', newline='') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(['time', 'w', 'x', 'y','z'])  # ヘッダーを書き込む
		csvwriter.writerows(filtered_orientation_csv)

	with open(result_dir+'/gravity.csv', 'w', newline='') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(['time','x', 'y','z'])  # ヘッダーを書き込む
		csvwriter.writerows(gravity_data)

	with open(result_dir+'/liner.csv', 'w', newline='') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(['time','x', 'y','z'])  # ヘッダーを書き込む
		csvwriter.writerows(linear_acc_data)

	csv_file_path = result_dir+'/earth.csv'
	distanceData.to_csv(csv_file_path, index=False)


	try:
		os.remove(gyro_save_path)
		os.remove(acc_save_path)
		print("Files deleted successfully.")
	except Exception as e:
		print("Error while deleting files:", e)

	zip_csv_files(result_dir,["result","earth"])
	body,boundary=create_multipart(result_dir)
	return Response(body, media_type=f"multipart/form-data; boundary={boundary}")

def get_filename_from_url(url):
		parsed_url = urlparse(url)
		return os.path.basename(parsed_url.path)

def zip_csv_files(result_dir, filenames):
    zip_filename = os.path.join(result_dir, "result.zip")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in filenames:
            csv_file_path = os.path.join(result_dir, filename + ".csv")
            if os.path.exists(csv_file_path):
                zipf.write(csv_file_path, arcname=filename + ".csv")
            else:
                print(f"File {csv_file_path} does not exist and will not be added to the archive.")
    
    print(f"Created zip file: {zip_filename}")


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