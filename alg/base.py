class BaseClient:
    def __init__(self, id, args):
        self.id = id
        self.args = args
        self.dataset_train = None
        self.dataset_test = None
        self.server = None

    def load_data(self):
        pass

    def run(self):
        raise NotImplementedError

    def local_test(self):
        raise NotImplementedError


class BaseServer(BaseClient):
    def __init__(self, id, args, clients):
        super().__init__(id, args)
        self.id = id
        self.args = args
        self.clients = clients

        self.round = 0

    def run(self):
        self.sample()
        self.downlink()
        self.local_run()
        self.uplink()
        self.aggregate()

    def sample(self):
        raise NotImplementedError

    def downlink(self):
        raise NotImplementedError

    def local_run(self):
        raise NotImplementedError

    def uplink(self):
        raise NotImplementedError

    def aggregate(self):
        raise NotImplementedError

    def test_all(self):
        raise NotImplementedError