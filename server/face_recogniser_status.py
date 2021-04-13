from face_recogniser import face_recogniser 
import logging 
log = logging.getLogger()

class face_recogniser_status(face_recogniser):
	def __init__(self):
		super().__init__()

	def update_status(self, state):
		import requests as req
		with req.get(f"http://localhost:4242/setState/{state}") as r:
			log.info(r.status_code)

	def process(self, frame):
		frame = super().add_face_rectangle(frame)
		if self.nFaces > 0:
			self.update_status(1)
		else:
			self.update_status(0)
		print(self.nFaces)
		return frame
