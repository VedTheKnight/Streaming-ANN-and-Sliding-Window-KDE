class Bucket:
    def __init__(self,t):
        self.size=1
        self.tst=t
    def get_size(self):
        return self.size
    def get_timestamp(self):
        return self.tst
    def set_size(self,s):
        self.size=s
    def set_timestamp(self,t):
        self.tst=t
    