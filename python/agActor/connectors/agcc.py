from agActor.base import BaseConnector


class Agcc(BaseConnector):

    @property
    def filepath(self):
        return self._getValues('agc_fitsfile')['filename']

    @property
    def frameId(self):
        return self._getValues('agc_frameid')

    @property
    def dataTime(self):
        return self._getValues('agc_fitsfile')['timestamp']
