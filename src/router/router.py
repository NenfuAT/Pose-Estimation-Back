import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from service import poseestimation


def Init():
	app = FastAPI()


	@app.get("/")
	async def root():
		return {"message": "Hello World"}

	@app.get("/api")
	async def root():
		return {"message": "It'sAPI"}
	

	class URL(BaseModel):
		gyro_url: str
		acc_url: str
		

	@app.post("/api/estimation")
	def poseEstimation(urls:URL):
		response=poseestimation.PoseEstimation(urls.gyro_url,urls.acc_url)
		return response


	

	uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")