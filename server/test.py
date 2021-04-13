class mother():
	def __init__(self):
		print("mother init")
		self.name="mamma"

	def run(self):
		print("mother run")
		print(self.name)

class child(mother):
	def __init__(self):
		super().__init__()
		print("child init")
	def run(self):
		super().run()
		print("run child")

if __name__ == "__main__":
	m = mother()
	m.run()

	c=child()
	c.run()
