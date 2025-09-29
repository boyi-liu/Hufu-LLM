from alg.fedft.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record

class Client(FTBaseClient):
    @time_record
    def run(self):
        self.run()

class Server(FTBaseServer):
    def run(self):
        self.sample()
        self.downlink()
        self.local_run()
        self.uplink()
        self.aggregate()