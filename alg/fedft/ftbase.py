from alg.base import BaseClient, BaseServer


class FTBaseClient(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)


class FTBaseServer(BaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)